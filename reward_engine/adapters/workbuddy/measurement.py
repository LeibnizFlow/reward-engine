"""Read parsed Office evidence by file path, without host IO or hidden scores."""

import fnmatch
import json
import re
from collections import defaultdict
from copy import deepcopy
from uuid import uuid4

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


class WorkBuddyTools(MeasurementTools):
    def catalog(self):
        return {
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

    async def run(self, request, evidence):
        if request.get("tool") not in {"workspace.read", "workspace.inspect"}:
            return await super().run(request, evidence)
        tool, args = request["tool"], request.get("arguments", {})
        receipt = {"tool": tool, "arguments": deepcopy(args), "status": "unknown"}
        try:
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
            paths = args.get("paths", ["*"])
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
                if any(fnmatch.fnmatchcase(r["path"], p) for p in paths)
            ]
            missing = [
                p
                for p in paths
                if not any(fnmatch.fnmatchcase(r["path"], p) for r in records)
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
            offset = args.get("offset", 0)
            if type(offset) is not int or not 0 <= offset <= len(text):
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
        self.record("measurement_tool_" + uuid4().hex, receipt)
        return receipt
