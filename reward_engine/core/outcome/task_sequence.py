"""Public multi-turn Task scopes; internal critiques never create user requirements."""

import asyncio
from copy import deepcopy
from time import monotonic
from urllib.parse import quote

from reward_engine.core.outcome.construction import fingerprint


class TaskSequence:
    def __init__(self, task):
        self.initial = deepcopy(task)
        self.events = []
        self.documents = {d["path"]: deepcopy(d) for d in task.get("documents", [])}
        self.deliveries = []

    @property
    def task(self):
        value = deepcopy(self.initial)
        if self.events:
            value["prompt"] += (
                "\n\nPublic user follow-ups, in arrival order. Preserve earlier requirements "
                "unless the user changes them; a later conflicting user instruction supersedes "
                "the earlier one. These are actual user messages, not evaluator suggestions.\n"
                + "\n".join(
                    f"[User follow-up {i}]\n{e['text']}"
                    for i, e in enumerate(self.events, 1)
                )
            )
        if self.documents:
            value["documents"] = deepcopy(list(self.documents.values()))
        return value

    @property
    def scope(self):
        return fingerprint(self.task)

    def advance(self, text, *, event_id, documents=()):
        """Called only at the adapter's real public-user/step boundary."""
        if (
            not isinstance(text, str)
            or not text.strip()
            or not isinstance(event_id, str)
            or not event_id
        ):
            raise ValueError(
                "A public turn needs an actual message and a stable native event ID"
            )
        event = {"id": event_id, "text": text, "documents": deepcopy(list(documents))}
        if any(
            not isinstance(d, dict)
            or not isinstance(d.get("path"), str)
            or not d["path"]
            for d in event["documents"]
        ):
            raise ValueError("Public step inputs need explicit paths")
        existing = next((e for e in self.events if e["id"] == event_id), None)
        if existing:
            if event != existing:
                raise ValueError("A native public event changed after admission")
            return False
        self.events.append(event)
        for document in event["documents"]:
            self.documents[document["path"]] = deepcopy(document)
        return True

    def observe(self, candidate):
        result = deepcopy(candidate)
        result["task_scope_sha256"] = self.scope
        result["artifact"]["task_scope_sha256"] = self.scope
        if self.events and "public_consultation" not in result["artifact"]:
            artifact = result["artifact"]
            artifact["public_consultation"] = {
                "earlier_exchanges": [
                    {"speaker": "user", "text": e["text"]} for e in self.events
                ]
            }
        self.deliveries.append(result)
        return result

    def bind(self, pack):
        if (
            pack.get("task_id") != self.task["task_id"]
            or pack.get("task_prompt") != self.task["prompt"]
        ):
            raise ValueError(
                "Feedback Pack does not belong to the currently revealed Task"
            )
        value = {**deepcopy(pack), "task_scope_sha256": self.scope}
        if value.get("feedback_policy") == "evidence_admitted":
            from reward_engine.core.outcome.admission import publication_digest

            value["validation"]["publication_sha256"] = publication_digest(value)
        return value

    def rebase(self, pack):
        """Old estimates remain fallible hints; old case support never transfers.

        A public follow-up can add, amend or withdraw requirements. The complete
        current Task drives review; an old estimated constraint cannot override it.
        No claim is made that this cheap rebind re-estimated or validated new rules.
        """
        from reward_engine.core.outcome.admission import validation_payload
        from reward_engine.core.outcome.validated_construction import publish_validation

        value = validation_payload(pack)
        value["task_prompt"] = self.task["prompt"]
        value["pack_id"] = "task_scope_" + self.scope[:24]
        value["task_scope_sha256"] = self.scope
        value["scope_rebind"] = {
            "source_pack_sha256": fingerprint(pack),
            "status": "unverified_priors",
            "new_validation_calls": 0,
        }
        # These ledgers describe the previous public requirement set. Keeping
        # their task-origin labels would silently reintroduce withdrawn duties.
        for key in (
            "independent_expectations",
            "proposal_expectations",
            "expectation_coverage",
        ):
            value[key] = []
        sources = value.setdefault("measurement_sources", {})
        for key in list(sources):
            if key.startswith("task_input:"):
                del sources[key]
        sources.update(
            {
                "task_input:" + quote(path, safe=""): deepcopy(document)
                for path, document in self.documents.items()
            }
        )
        for criterion in value["criteria"]:
            criterion["kind"], criterion["origin"] = "soft", "domain"
            criterion["scope_note"] = (
                "Estimate from an earlier public Task scope. The latest public requirements decide "
                "applicability; this criterion cannot impose a superseded obligation."
            )
            for clause in criterion.get("clauses", []):
                if clause.get("source_quotes"):
                    clause["historical_source_quotes"] = clause["source_quotes"]
                    clause["source_quotes"] = []
                if clause["use"] == "task_constraint":
                    clause["use"] = "quality_judgment"
                    clause["scope_note"] = (
                        "Previous user constraint; applicability must be reassessed."
                    )
        for test in value["tests"]:
            test["question"] = (
                "First check applicability against the latest public Task and follow-ups. "
                "A superseded requirement is not a defect; unknown applicability remains unknown. "
                "Earlier-scope measurement question: " + test.get("question", "")
            )
        value = publish_validation(
            value,
            {
                "status": "inconclusive",
                "artifact_sha256": fingerprint(value),
                "records": [],
            },
        )
        return value


class ScopedFeedback:
    """Invalidate stale guidance before rebuilding for a newly revealed user turn.

    Only public Task state and previously observed artifacts enter refresh.
    The adapter owns one native deadline; a new turn never resets that budget.
    """

    def __init__(
        self, sequence, refresh, *, deadline, record=lambda *_: None, clock=monotonic
    ):
        self.sequence, self.refresh = sequence, refresh
        self.deadline, self.record, self.clock = deadline, record, clock
        self.current = None
        self.lock = asyncio.Lock()

    async def advance(self, text, *, event_id, documents=()):
        async with self.lock:
            changed = self.sequence.advance(
                text, event_id=event_id, documents=documents
            )
            if not changed:
                return self.current
            self.current = None
            scope = self.sequence.scope
            self.record(
                "task_scope/" + scope,
                {
                    "task": self.sequence.task,
                    "event_id": event_id,
                    "status": "refreshing",
                    "stale_guidance_disabled": True,
                },
            )
            remaining = max(0, self.deadline - self.clock())
            if remaining <= 0:
                self.record(
                    "task_scope/" + scope,
                    {"status": "unavailable", "reason": "native_deadline"},
                )
                return None
            try:
                value = await asyncio.wait_for(
                    self.refresh(
                        self.sequence.task, deepcopy(self.sequence.deliveries)
                    ),
                    timeout=remaining,
                )
                value["pack"] = self.sequence.bind(value["pack"])
                self.current = value
                self.record(
                    "task_scope/" + scope,
                    {"status": "ready", "task_scope_sha256": scope},
                )
            except Exception as error:
                self.record(
                    "task_scope/" + scope,
                    {
                        "status": "unavailable",
                        "error_type": type(error).__name__,
                        "stale_guidance_disabled": True,
                    },
                )
            return self.current


def require_version_scope(versions, candidate):
    expected = candidate.get("task_scope_sha256")
    if expected and any(
        v["candidate"].get("task_scope_sha256") != expected for v in versions
    ):
        raise ValueError(
            "Cannot select or restore a delivery from an earlier public Task scope"
        )
