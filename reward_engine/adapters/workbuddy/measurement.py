"""Read parsed Office evidence by file path, without host IO or hidden scores."""

import fnmatch
import json
import re
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from uuid import uuid4

from reward_engine.core.outcome.measurement_requests import (
    MeasurementRequestRejected,
    match_path,
)
from reward_engine.core.outcome.measurement_requests import (
    enabled as recovery_enabled,
)
from reward_engine.core.outcome.measurement_tools import MeasurementTools


def workspace_records(evidence):
    groups = defaultdict(list)
    for ref, value in evidence.items():
        if (
            ref.startswith("product_1:source:")
            and isinstance(value, dict)
            and all(key in value for key in ("source", "offset", "total_chars", "text"))
        ):
            groups[value["source"]].append((ref, value))
    records = []
    for chunks in groups.values():
        chunks.sort(key=lambda row: row[1]["offset"])
        text, refs = "", []
        for ref, chunk in chunks:
            if (
                chunk["offset"] != len(text)
                or chunk["total_chars"] != chunks[0][1]["total_chars"]
            ):
                raise ValueError("Incomplete or ambiguous workspace evidence chunks")
            text += chunk["text"]
            refs.append(ref)
        if len(text) != chunks[0][1]["total_chars"]:
            raise ValueError("Incomplete workspace evidence")
        files = json.loads(text)
        if not isinstance(files, list) or any(
            not isinstance(f, dict) or "path" not in f for f in files
        ):
            raise ValueError("Workspace evidence is not a file inventory")
        records.extend({**file, "evidence_refs": refs} for file in files)
    if not groups:
        raise ValueError("No collected WorkBuddy workspace evidence")
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Ambiguous workspace paths")
    expected_count = evidence.get("product_1:environment", {}).get("file_count")
    if type(expected_count) is int and len(records) != expected_count:
        raise ValueError("Incomplete workspace inventory")
    return records


def workspace_response(request, evidence, *, canonical):
    """Build the deterministic collected-data view shared by admission and reads."""
    tool, args = request["tool"], request.get("arguments", {})
    paths = args.get("paths", ["**"] if canonical else ["*"])
    matches = match_path if canonical else fnmatch.fnmatchcase
    if (
        not isinstance(paths, list)
        or not 1 <= len(paths) <= 32
        or any(not isinstance(p, str) for p in paths)
    ):
        raise ValueError("Supply one to 32 path patterns")
    records = workspace_records(evidence)
    matched = [
        r
        for r in records
        if any(matches(r["path"], p) for p in paths)
    ]
    missing = [
        p
        for p in paths
        if not any(matches(r["path"], p) for r in records)
    ]
    files = matched
    if tool == "workspace.inspect":
        files = []
        for row in matched:
            item = {
                k: row[k]
                for k in ("path", "sha256", "bytes", "changed", "evidence_refs")
                if k in row
            }
            content = row.get("content", {})
            if row["path"].endswith(".json") and "text" in content:
                try:
                    item["json_value"] = json.loads(content["text"])
                except ValueError as error:
                    item["json_error"] = str(error)
            if "sheets" in content:
                item["sheets"] = []
                for sheet in content["sheets"]:
                    rows = {
                        int(re.search(r"\d+$", c["cell"])[0])
                        for c in sheet["cells"]
                    }
                    item["sheets"].append(
                        {
                            "name": sheet["name"],
                            "populated_row_count": len(rows),
                            "max_populated_row": max(rows, default=0),
                            "first_row_cells": [
                                c
                                for c in sheet["cells"]
                                if rows
                                and int(re.search(r"\d+$", c["cell"])[0])
                                == min(rows)
                            ],
                        }
                    )
            if content.get("status") == "unknown":
                item["parse_error"] = content
            files.append(item)
    text = json.dumps(
        {"files": files, "missing_paths": missing}, ensure_ascii=False
    )
    return matched, missing, text


class WorkBuddyTools(MeasurementTools):
    def validate_request(self, request, evidence, *, require_available=True, check_program=True):
        bindings = super().validate_request(request, evidence, require_available=require_available,
                                            check_program=check_program)
        if request.get("tool") in {"workspace.read", "workspace.inspect"}:
            try:
                _, _, text = workspace_response(request, evidence, canonical=recovery_enabled(self))
            except (ValueError, KeyError, TypeError, IndexError):
                # Invalid supplied content is a real read failure, not proof
                # that the caller's request could be rejected before execution.
                return bindings
            offset = request.get("arguments", {}).get("offset", 0)
            if offset > len(text):
                raise MeasurementRequestRejected("Offset exceeds workspace response length",
                                                 field="arguments.offset", actual=offset)
        return bindings

    def validate_historical(self, receipt, evidence):
        if (not recovery_enabled(self) or receipt.get("contract_policy") == self.role_contract_policy
                or receipt.get("tool") not in {"workspace.read", "workspace.inspect"}
                or receipt.get("status") != "observed"):
            return
        # Prove selection equivalence without reexecuting a successful read.
        # A legacy glob that omitted a root file cannot be relabelled as the
        # current recursive query merely because its arguments are unchanged.
        records = workspace_records(evidence)
        args = receipt.get("arguments", {})
        before = args.get("paths", ["*"])
        after = args.get("paths", ["**"])
        old_matches = [r["path"] for r in records if any(fnmatch.fnmatchcase(r["path"], p) for p in before)]
        new_matches = [r["path"] for r in records if any(match_path(r["path"], p) for p in after)]
        old_missing = [p for p in before if not any(fnmatch.fnmatchcase(r["path"], p) for r in records)]
        new_missing = [p for p in after if not any(match_path(r["path"], p) for r in records)]
        if old_matches != new_matches or old_missing != new_missing:
            raise ValueError("Historical workspace glob selected different files under the current contract")

    def catalog_for(self, evidence):
        catalog = self.catalog()
        # An absent input inventory is detectable before inspection. Malformed
        # or incomplete supplied chunks still reach the real validator and fail.
        supplied = any(ref.startswith("product_1:source:") and isinstance(value, dict)
                       and all(key in value for key in ("source", "offset", "total_chars", "text"))
                       for ref, value in evidence.items())
        if not supplied:
            for tool in ("workspace.inspect", "workspace.read"):
                catalog[tool].update(available=False, unavailable_reason="No collected workspace file inventory is supplied; use the available evidence handles")
        return catalog

    def text_evidence(self, arguments, evidence):
        ref = arguments.get('ref')
        value = evidence.get(ref) if isinstance(ref, str) else None
        if isinstance(value, str):
            return super().text_evidence(arguments, evidence)
        if not isinstance(value, list) or not value or any(not isinstance(r, dict) for r in value):
            raise ValueError('Text checker requires exact text or a file inventory')
        path = arguments.get('path')
        choices = [r for r in value if isinstance(r.get('path'), str)
                   and (r['path'] == path if path is not None else r['path'].lower().endswith('.json'))]
        if len(choices) != 1:
            raise ValueError('Ambiguous or missing JSON file; supply arguments.path within the cited inventory')
        metadata = choices[0]
        records = [r for r in workspace_records(evidence) if r['path'] == metadata['path']]
        if len(records) != 1 or any(metadata.get(k) != records[0].get(k) or k not in metadata
                                    for k in ('sha256', 'bytes')):
            raise ValueError('File inventory does not match collected content identity')
        record = records[0]
        text = record.get('content', {}).get('text')
        if (not isinstance(text, str) or len(text.encode('utf-8')) != metadata['bytes']
                or sha256(text.encode('utf-8')).hexdigest() != metadata['sha256']):
            raise ValueError('Collected text is missing or not exact file bytes')
        return {'text': text, 'path': record['path'],
                'evidence_refs': [ref, *record['evidence_refs']]}

    def catalog(self):
        catalog = {
            **super().catalog(),
            "workspace.inspect": {
                "available": True,
                "arguments": {
                    "paths": [
                        "optional exact relative paths or glob patterns; default *"
                    ]
                },
                "meaning": "Actual file inventory, parsed JSON keys/values, XLSX sheet names and populated row counts/first-row cells. Reads collected evidence only.",
            },
            "workspace.read": {
                "available": True,
                "arguments": {
                    "paths": [
                        "relative paths or glob patterns, e.g. input/* or output/*"
                    ],
                    "offset": 0,
                },
                "meaning": "Read actual parsed file contents across evidence chunks, with next_offset for pagination. Missing paths and incomplete excerpts are explicit. No host IO.",
            },
        }
        if recovery_enabled(self):
            for tool in ("workspace.inspect", "workspace.read"):
                catalog[tool]["arguments"]["paths"] = [
                    "relative paths/globs: * matches one segment, ** matches zero or more directories; default ** (all files)"
                ]
        return catalog

    async def run(self, request, evidence):
        if request.get("tool") not in {"workspace.read", "workspace.inspect"}:
            return await super().run(request, evidence)
        tool, args = request["tool"], request.get("arguments", {})
        receipt = {"tool": tool, "arguments": deepcopy(args), "status": "unknown"}
        try:
            receipt.update(self.execution_metadata(request, evidence))
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
            matched, missing, text = workspace_response(request, evidence, canonical=recovery_enabled(self))
            offset = args.get("offset", 0)
            if type(offset) is not int or not 0 <= offset <= len(text):
                if recovery_enabled(self):
                    raise MeasurementRequestRejected("Offset exceeds workspace response length",
                                                     field="arguments.offset", actual=offset)
                raise ValueError("Invalid workspace read offset")
            excerpt = text[offset : offset + 36000]
            while True:
                end = offset + len(excerpt)
                result = {
                    "text": excerpt,
                    "offset": offset,
                    "next_offset": end if end < len(text) else None,
                    "total_chars": len(text),
                    "excerpt_only": offset != 0 or end < len(text),
                    "evidence_refs": sorted(
                        {ref for r in matched for ref in r["evidence_refs"]}
                    ),
                }
                # Server-authored provenance, never a model's assertion that a
                # read happened. Keep mixed/input-only inventories conservative:
                # they cannot establish that the delivered artifact was read.
                if (
                    tool == "workspace.read"
                    and len(matched) == 1
                    and not missing
                    and matched[0].get("changed") is True
                    and excerpt
                    and matched[0].get("content", {}).get("status") != "unknown"
                ):
                    result["observed_product_refs"] = matched[0]["evidence_refs"]
                if (
                    len(excerpt.encode()) <= 40000
                    and len(json.dumps(result, ensure_ascii=False).encode()) <= 48000
                ):
                    break
                if not excerpt:
                    raise ValueError("Workspace provenance exceeds receipt budget")
                excerpt = excerpt[: len(excerpt) * 3 // 4]
            receipt.update(status="observed", result=result)
        except Exception as error:
            receipt.update(error_type=type(error).__name__, reason=str(error)[:300])
            if recovery_enabled(self):
                receipt["request_rejected"] = isinstance(error, MeasurementRequestRejected)
                if receipt["request_rejected"]:
                    receipt.update(execution_phase="preflight", diagnostics=error.diagnostics)
        self.record("measurement_tool_" + uuid4().hex, receipt)
        return receipt
