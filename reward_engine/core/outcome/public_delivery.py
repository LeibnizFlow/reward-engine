"""Opt-in, explicitly supplied public path prerequisites; never infer task semantics.

The executor translates quoted PUBLIC instructions into this typed contract. Exact
quote/hash validation proves provenance, not that the translation covers all natural
language requirements. Content quality remains the responsibility of semantic checks.
"""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath

SCHEMA = "public_delivery.contract.v1"
RECEIPT = "public_delivery.gate.v1"
INVENTORY = "public_delivery.inventory.v1"
CHECKER = "public_delivery.paths.v1"
CRITERION = "public_delivery_prerequisite_v1"
TEST = "public_delivery_paths_v1"
REF = "public_delivery:contract"
BOUNDARY = "Exact public paths only; neither semantic correctness nor overall quality."


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or PurePosixPath(value).is_absolute()
        or any(p in {"", ".", ".."} for p in value.split("/"))
        or ":" in value
    ):
        raise ValueError(
            "Delivery path must be an exact safe workspace-relative POSIX path"
        )
    return value


def _keys(row, names):
    if not isinstance(row, dict) or set(row) != set(names):
        raise ValueError("Unexpected public delivery schema fields")


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected SHA-256")
    return value


@dataclass(frozen=True)
class RequiredFile:
    id: str
    path: str
    source_quote_indices: tuple[int, ...]


@dataclass(frozen=True)
class PublicDeliveryContract:
    task_id: str
    public_prompt_sha256: str
    requirements: tuple[RequiredFile, ...]

    @classmethod
    def validate(cls, value, task):
        _keys(
            value,
            (
                "schema",
                "task_id",
                "public_prompt_sha256",
                "source",
                "translation",
                "requirements",
            ),
        )
        if value["schema"] != SCHEMA or value["task_id"] != task["task_id"]:
            raise ValueError("Public delivery Task/schema mismatch")
        if (
            _sha(value["public_prompt_sha256"])
            != hashlib.sha256(task["prompt"].encode()).hexdigest()
        ):
            raise ValueError("Stale public instruction binding")
        source = value["source"]
        _keys(source, ("kind", "path", "quotes"))
        if source["kind"] != "public_task_instruction":
            raise ValueError("Only explicit public instruction provenance is supported")
        safe_path(source["path"])
        lines = task["prompt"].splitlines()
        quotes = source["quotes"]
        if not isinstance(quotes, list) or not quotes:
            raise ValueError("Exact public instruction quotes required")
        for quote in quotes:
            _keys(quote, ("start_line", "end_line", "text"))
            start, end = quote["start_line"], quote["end_line"]
            if (
                type(start) is not int
                or type(end) is not int
                or not 1 <= start <= end <= len(lines)
                or not quote["text"]
                or quote["text"] != "\n".join(lines[start - 1 : end])
            ):
                raise ValueError("Public source quote does not match exact lines")
        translation = value["translation"]
        _keys(translation, ("method", "authority", "scope"))
        if translation["method"] != "explicit_executor_translation" or not all(
            isinstance(translation[k], str) and translation[k].strip()
            for k in ("authority", "scope")
        ):
            raise ValueError("Explicit executor translation provenance required")
        rows = value["requirements"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Nonempty typed public requirements required")
        requirements = []
        for row in rows:
            _keys(row, ("id", "path", "kind", "source_quote_indices"))
            refs = row["source_quote_indices"]
            if (
                not isinstance(row["id"], str)
                or not re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"])
                or row["kind"] != "regular_file"
                or not isinstance(refs, list)
                or not refs
                or any(type(i) is not int or not 0 <= i < len(quotes) for i in refs)
                or len(set(refs)) != len(refs)
            ):
                raise ValueError("Invalid typed required file or provenance reference")
            requirements.append(
                RequiredFile(row["id"], safe_path(row["path"]), tuple(refs))
            )
        if len({r.id for r in requirements}) != len(requirements) or len(
            {r.path for r in requirements}
        ) != len(requirements):
            raise ValueError("Duplicate public delivery requirements")
        return cls(value["task_id"], value["public_prompt_sha256"], tuple(requirements))


def validate(contract, task):
    PublicDeliveryContract.validate(contract, task)
    return deepcopy(contract)


def attach_context(contract, context, evidence):
    contract = validate(contract, context["task"])
    old = context["task"].get("public_delivery_contract")
    if old is not None and old != contract:
        raise ValueError("Conflicting public delivery contract")
    context["task"]["public_delivery_contract"] = contract
    evidence[REF] = deepcopy(contract)
    context["available_evidence_refs"] = list(evidence)
    return context, evidence


def criterion(contract):
    paths = [r["path"] for r in contract["requirements"]]
    return {
        "id": CRITERION,
        "kind": "hard",
        "origin": "task",
        "priority": "critical",
        "standard": "Explicit public delivery prerequisite: exact regular files "
        + json.dumps(paths, ensure_ascii=False),
        "measurement_question": "Does the collected workspace contain every explicitly bound exact file path?",
        "evidence_refs": ["task", REF],
        "anchors": {
            "fail": "A required exact file is absent from a complete relevant inventory or is a non-regular path.",
            "partial": "Collection cannot establish a required path; unresolved, never acceptable.",
            "full": "All required exact regular files have collected byte/hash evidence; no content-quality claim.",
        },
        "public_delivery_binding": fingerprint(contract),
    }


def bind_definition(definition, contract):
    value = {
        k: deepcopy(v)
        for k, v in definition.items()
        if not k.startswith("public_delivery")
    }
    if any(c["id"] == CRITERION for c in value["criteria"]):
        raise ValueError("Reserved public delivery criterion collision")
    if len(value["criteria"]) >= 20:
        raise ValueError(
            "Public prerequisite requires a free criterion slot within the existing audit limit"
        )
    value["criteria"].append(criterion(contract))
    value["public_delivery_contract"] = deepcopy(contract)
    value["public_delivery_origin"] = (
        "Explicit executor binding; model-authored definition retained separately"
    )
    value["public_delivery_projection"] = {
        "schema": "public_delivery.projection.v1",
        "origin": "explicit_executor_public_binding",
        "raw_sha256": fingerprint(definition),
        "contract_sha256": fingerprint(contract),
        "ignored_untrusted_metadata": sorted(
            k for k in definition if k.startswith("public_delivery")
        ),
        "model_criteria": len(definition["criteria"]),
        "bound_prerequisites": 1,
        "model_checks": len(definition.get("tests", [])),
        "bound_checks": 1 if "tests" in definition else 0,
    }
    return value


def bind_pack(pack, contract):
    task = {"task_id": pack["task_id"], "prompt": pack["task_prompt"]}
    validate(contract, task)
    value = bind_definition(pack, contract)
    if any(t["test_id"] == TEST or t["checker"] == CHECKER for t in pack["tests"]):
        raise ValueError("Reserved public delivery checker collision")
    value["tests"].append(
        {
            "test_id": TEST,
            "criterion_id": CRITERION,
            "checker": CHECKER,
            "binding_status": "callable",
            "question": criterion(contract)["measurement_question"],
            "rubric": criterion(contract),
            "arguments": {"contract_sha256": fingerprint(contract)},
            "public_delivery_mandatory": True,
        }
    )
    value["pack_id"] = fingerprint(
        {"semantic_pack_id": pack["pack_id"], "contract": contract}
    )
    value["public_delivery_semantic_pack_id"] = pack["pack_id"]
    return value


def semantic_pack(pack):
    if not pack or "public_delivery_contract" not in pack:
        return deepcopy(pack)
    value = deepcopy(pack)
    value["pack_id"] = value.pop("public_delivery_semantic_pack_id")
    for key in (
        "public_delivery_contract",
        "public_delivery_origin",
        "public_delivery_projection",
    ):
        value.pop(key, None)
    value["criteria"] = [c for c in value["criteria"] if c["id"] != CRITERION]
    value["tests"] = [t for t in value["tests"] if t["test_id"] != TEST]
    return value


def mandatory_tests(pack):
    if not pack.get("public_delivery_contract"):
        return []
    expected = bind_pack(semantic_pack(pack), pack["public_delivery_contract"])
    if pack != expected:
        raise ValueError("Public delivery published Pack binding changed")
    return [TEST]


def actor_instruction(contract):
    return (
        "Explicit public delivery prerequisite (executor translation of the quoted public Task; "
        "not additional semantic or private grading criteria):\n"
        + json.dumps(
            {
                "required_regular_files_relative_to_task_workspace": [
                    r["path"] for r in contract["requirements"]
                ],
                "public_quotes": contract["source"]["quotes"],
                "contract_sha256": fingerprint(contract),
            },
            ensure_ascii=False,
        )
        + "\nUse each exact declared path. Another directory or a matching basename is not an alias. "
        "File presence alone does not establish content correctness. Follow the original public Task."
    )


def validate_inventory(inventory):
    _keys(
        inventory,
        (
            "schema",
            "phase",
            "complete",
            "files",
            "blocked",
            "excluded_components",
            "inventory_sha256",
        ),
    )
    if (
        inventory["schema"] != INVENTORY
        or inventory["phase"] != "before_official_verifier"
        or type(inventory["complete"]) is not bool
        or not isinstance(inventory["files"], list)
        or not isinstance(inventory["blocked"], list)
        or not isinstance(inventory["excluded_components"], list)
    ):
        raise ValueError("Invalid public delivery inventory")
    if inventory["inventory_sha256"] != fingerprint(inventory["files"]):
        raise ValueError("Public delivery inventory hash mismatch")
    seen = set()
    for row in inventory["files"]:
        _keys(row, ("path", "sha256", "bytes", "changed"))
        name = safe_path(row["path"])
        _sha(row["sha256"])
        if (
            name in seen
            or type(row["bytes"]) is not int
            or row["bytes"] < 0
            or type(row["changed"]) is not bool
        ):
            raise ValueError("Invalid/duplicate collected file")
        seen.add(name)
    for row in inventory["blocked"]:
        _keys(row, ("path", "reason"))
        if row["path"] != ".":
            safe_path(row["path"])
        if row["reason"] not in {
            "symlink",
            "special",
            "directory",
            "read_error",
            "scan_error",
            "changed_during_collection",
        }:
            raise ValueError("Unknown collection limitation")
    for part in inventory["excluded_components"]:
        if safe_path(part) != PurePosixPath(part).name:
            raise ValueError("Excluded component must be a single name")
    return inventory


def evaluate(contract, candidate_sha256, inventory):
    """Consume only collected observations; exact presence can pass despite unrelated gaps."""
    _sha(candidate_sha256)
    if inventory is not None:
        validate_inventory(inventory)
    files = {r["path"]: r for r in inventory["files"]} if inventory else {}
    rows = []
    for required in contract["requirements"]:
        name = required["path"]
        blocked = [
            r
            for r in (inventory or {}).get("blocked", [])
            if r["path"] == "." or name == r["path"] or name.startswith(r["path"] + "/")
        ]
        excluded = bool(
            set(name.split("/")) & set((inventory or {}).get("excluded_components", []))
        )
        invalid = any(
            r["reason"] in {"symlink", "special"}
            or (r["reason"] == "directory" and r["path"] == name)
            for r in blocked
        )
        obstructed = [r for r in blocked if r["reason"] != "directory"]
        if invalid:
            status, reason = (
                "fail",
                "Required path is not a safely collected regular file",
            )
        elif excluded or obstructed or inventory is None:
            status, reason = (
                "unknown",
                "Required path was excluded or collection could not establish it",
            )
        elif name in files:
            status, reason = (
                "pass",
                "Exact regular file present with collected hash and size",
            )
        elif inventory["complete"]:
            status, reason = (
                "fail",
                "Exact required file absent from complete relevant workspace inventory",
            )
        else:
            status, reason = (
                "unknown",
                "Incomplete workspace inventory cannot establish absence",
            )
        rows.append(
            {
                "requirement_id": required["id"],
                "path": name,
                "status": status,
                "reason": reason,
                "file": deepcopy(files.get(name)) if status == "pass" else None,
            }
        )
    status = (
        "fail"
        if any(r["status"] == "fail" for r in rows)
        else "unknown"
        if any(r["status"] == "unknown" for r in rows)
        else "pass"
    )
    return {
        "schema": RECEIPT,
        "contract_sha256": fingerprint(contract),
        "public_prompt_sha256": contract["public_prompt_sha256"],
        "candidate_sha256": candidate_sha256,
        "inventory_sha256": (inventory or {}).get("inventory_sha256"),
        "inventory_receipt_sha256": fingerprint(inventory),
        "status": status,
        "requirements": rows,
        "boundary": BOUNDARY,
    }


def candidate_gate(contract, candidate):
    return evaluate(
        contract,
        fingerprint(candidate),
        candidate.get("environment", {}).get("public_delivery_inventory"),
    )


async def check(test, context, evidence):
    contract = validate(context["task"]["public_delivery_contract"], context["task"])
    if (
        test.get("arguments") != {"contract_sha256": fingerprint(contract)}
        or len(context["products"]) != 1
    ):
        raise ValueError("Public delivery checker input mismatch")
    product = context["products"][0]
    receipt = evaluate(
        contract,
        product["candidate_sha256"],
        product["environment"].get("public_delivery_inventory"),
    )
    gaps = [
        f"{r['path']}: {r['reason']}"
        for r in receipt["requirements"]
        if r["status"] != "pass"
    ]
    return {
        "status": receipt["status"],
        "reason": "; ".join(gaps) or BOUNDARY,
        "gap": "; ".join(gaps) if receipt["status"] == "fail" else "",
        "evidence_refs": [product["environment_ref"], REF],
        "public_delivery_receipt": receipt,
    }


def install(registry):
    if CHECKER not in registry.checkers:
        registry.register(CHECKER, check, BOUNDARY)
    elif registry.checkers[CHECKER][0] is not check:
        raise ValueError("Public delivery checker cannot be replaced")
