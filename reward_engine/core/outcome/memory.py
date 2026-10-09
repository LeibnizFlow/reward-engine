"""Battle Attention for Outcome experience, independent of Policy Critic releases.

Query = current Task/requirements/gaps; K = applicability; V = recorded effects.
Default is sparse cosine retrieval, not a trained Transformer. Optional
embedding provider implements embed(texts, text_type=...), Keys only.
"""

import json
import math
import re
from collections import Counter
from copy import deepcopy
from uuid import uuid4


def tokens(text):
    text = text.lower()
    return Counter(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text))


def cosine(a, b):
    if isinstance(a, dict):
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        aa, bb = sum(v * v for v in a.values()), sum(v * v for v in b.values())
    else:
        if len(a) != len(b) or any(not math.isfinite(v) for v in [*a, *b]):
            raise ValueError("Invalid embedding vectors")
        dot, aa, bb = (
            sum(x * y for x, y in zip(a, b, strict=True)),
            sum(x * x for x in a),
            sum(x * x for x in b),
        )
    return dot / math.sqrt(aa * bb) if aa and bb else 0.0


class Memory:
    def __init__(self, rows=()):
        self.rows = deepcopy(list(rows))
        if len({r["memory_id"] for r in self.rows}) != len(self.rows):
            raise ValueError("Duplicate memory identity")

    def observe_jury(self, task, jury, *, source):
        """Jury may write a grounded gap, NOT a made-up successful method."""
        for gap in jury.get("gaps", []):
            key = "\n".join(
                [task["prompt"], "Outcome gap: " + gap["gap"], gap["expected"], gap["verification"]]
            )
            previous = next(
                (r for r in self.rows if r["task_id"] == task["task_id"] and r["key"] == key), None
            )
            entry = {
                "memory_id": previous["memory_id"] if previous else uuid4().hex,
                "task_id": task["task_id"],
                "key": key,
                "value": {
                    "applicability": gap["gap"],
                    "method": None,
                    "gap_status": gap["status"],
                    "expected": gap["expected"],
                    "observed": gap["observed"],
                    "verification": gap["verification"],
                    "product_ids": deepcopy(gap["product_ids"]),
                    "expected_basis": gap["expected_basis"],
                    "expected_refs": deepcopy(gap["expected_refs"]),
                    "evidence_refs": deepcopy(gap["evidence_refs"]),
                    "source": source,
                    "status": "jury_observation_not_verified_method",
                },
            }
            if previous:
                previous.update(
                    entry
                )  # New observation replaces stale suspicion; source reports remain on disk.
            else:
                self.rows.append(entry)

    def record(self, task, method, before, after, *, source, context_note):
        """Record improvements AND regressions; observation is not causal proof."""
        before_rows = {r["test_id"]: r for r in before["results"]}
        changes = []
        if before["pack_id"] != after["pack_id"]:
            return None  # A changed definition invalidates a direct check comparison.
        for row in after["results"]:
            previous = before_rows.get(row["test_id"])
            if not previous:
                continue
            for scope in ("standard", "inquiry"):
                old, new = previous, row
                if scope == "inquiry":
                    if not row.get("measurement_inquiry") or row["measurement_inquiry"] != previous.get("measurement_inquiry"):
                        continue  # Different questions are not a comparable before/after pair.
                    old, new = previous.get("inquiry_result"), row.get("inquiry_result")
                if not old or not new or old["status"] == "unknown" or new["status"] == "unknown":
                    continue
                changes.append(
                    {
                        "test_id": row["test_id"],
                        "before": old["status"],
                        "after": new["status"],
                        "gap_before": old.get("gap") or "",
                        "gap_after": new.get("gap") or "",
                        "before_evidence_refs": deepcopy(old["evidence_refs"]),
                        "after_evidence_refs": deepcopy(new["evidence_refs"]),
                        **({"scope": "inquiry", "measurement_inquiry": deepcopy(row["measurement_inquiry"])}
                           if scope == "inquiry" else {}),
                    }
                )
        if not changes:
            return None
        key = "\n".join(
            [
                task["prompt"],
                context_note,
                *[r["gap_before"] for r in changes],
                *[r["gap_after"] for r in changes],
            ]
        )
        entry = {
            "memory_id": uuid4().hex,
            "task_id": task["task_id"],
            "key": key,
            "value": {
                "applicability": context_note,
                "method": method,
                "observed_changes": changes,
                "source": source,
                "pack_id": after["pack_id"],
                "status": "observed_not_causal_or_official",
                "limitation": "Local check changes do not prove overall improvement; use terminal Ranking. "
                "Requirement changes, source changes and evaluator noise may explain differences.",
            },
        }
        self.rows.append(entry)
        return deepcopy(entry)

    async def retrieve(self, context, *, top_k=3, embedding=None, cross_task=False, query_extra=None):
        if type(top_k) is not int or top_k < 0:
            raise ValueError("top_k is a nonnegative hyperparameter")
        candidates = [r for r in self.rows if cross_task or r["task_id"] == context["task"]["task_id"]]
        if not candidates or top_k == 0:
            return []
        query = json.dumps(
            {
                "task": context["task"],
                "requirements": [p["user_updates"] for p in context["products"]],
                "checks": context["checks"],
                "gaps": context.get("gaps", []),
                **({"query_extra": query_extra} if query_extra is not None else {}),
            },
            ensure_ascii=False,
        )
        if embedding:
            q = await embedding.embed([query], text_type="query")
            keys = await embedding.embed([r["key"] for r in candidates], text_type="document")
            if len(q.vectors) != 1 or len(keys.vectors) != len(candidates):
                raise ValueError("Embedding must preserve Query/Key count")
            scores = [cosine(q.vectors[0], key) for key in keys.vectors]
        else:
            scores = [cosine(tokens(query), tokens(r["key"])) for r in candidates]
        chosen = sorted(((score, i) for i, score in enumerate(scores) if score > 0), reverse=True)[:top_k]
        return [
            {
                "memory_id": candidates[i]["memory_id"],
                "key": candidates[i]["key"],
                "relevance": score,
                "value": deepcopy(candidates[i]["value"]),
            }
            for score, i in chosen
        ]
