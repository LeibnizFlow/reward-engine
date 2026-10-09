"""Opt-in paired public-Task facts inside the existing independent Overall call.

Only structure and evidence ownership are checked here, never factual correctness,
Task satisfaction or the winner. The complete model response survives in reason.
"""

import json
import re
from copy import deepcopy

PROTOCOL_ID = "task-evidence-matrix-v1"
SCHEMAS = {
    "reward_builder_v4.final_outcome_battle.v1",
    "reward_builder_v4.overall_calibration_judge_request.v1",
}
SLOTS = ("candidate_1", "candidate_2")


def _factor_shared_measurements(body):
    """Factor exact repeated large values, never tests, facts or source citations."""
    if (
        body.get("schema_version")
        != "reward_builder_v4.overall_calibration_judge_request.v1"
    ):
        return
    groups = {}
    for row in body.get("harness_evidence_by_criterion", ()):
        for test in row["harness_evidence"]["test_results"]:
            for name, value in test["measurements"].items():
                # Do not introduce an ambiguous reserved shape into arbitrary JSON.
                if isinstance(value, dict) and "shared_measurement_ref" in value:
                    return
                encoded = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                if len(encoded.encode()) > 1024:
                    groups.setdefault(encoded, []).append(
                        (test["measurements"], name, value)
                    )
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    if not repeated:
        return
    if "shared_measurements" in body:
        raise ValueError("Conflicting shared_measurements in Overall request")
    body["shared_measurements"] = {}
    for index, rows in enumerate(repeated, 1):
        reference = f"measurement-{index}"
        body["shared_measurements"][reference] = rows[0][2]
        for values, name, _ in rows:
            values[name] = {"shared_measurement_ref": reference}
    body["instruction"] += (
        " A measurement value containing only shared_measurement_ref is an exact reference to"
        " shared_measurements[that reference]. Expand it to the complete original JSON value before"
        " interpreting the test. Each test retains its own contract, status and evidence citations;"
        " shared values do not imply shared success or semantic judgments."
    )


def _task_units(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Task matrix requires the original public Task prompt")
    # Keep the introduction and all continuation lines, not just numbered titles.
    parts = re.split(r"(?m)(?=^[ \t]*\d+[.)][ \t]+)", prompt)
    if len(parts) == 1:
        parts = re.split(r"\n[ \t]*\n", prompt)
    return [
        {"requirement_id": f"task-{index}", "text": part}
        for index, part in enumerate((p.strip() for p in parts if p.strip()), 1)
    ]


def prepare_matrix_request(body):
    """Preserve original inputs; provide deterministic Task IDs and paired cells."""
    result = deepcopy(body)
    if body.get("schema_version") not in SCHEMAS:
        return result
    protocol = {"id": PROTOCOL_ID, "task_units": _task_units(body["task"]["prompt"])}
    if "outcome_comparison_protocol" in body:
        if body["outcome_comparison_protocol"] != protocol:
            raise ValueError("Conflicting Outcome comparison protocol")
        return result
    _factor_shared_measurements(result)
    result["outcome_comparison_protocol"] = protocol
    decision = "verdict" if "verdict" in body["response"] else "preference"
    cell = {
        "fact": "Brief observed fact, contradiction or missing evidence; no invented completion.",
        "evidence_refs": [
            "Only this candidate's evidence aliases; [] when evidence is missing."
        ],
    }
    result["response"] = {
        "requirement_comparisons": [
            {
                "requirement_id": "One supplied task_units ID; cover every ID once.",
                **{slot: deepcopy(cell) for slot in SLOTS},
            }
        ],
        "action_format_check": {
            "requested_boundary": "What actions, terminal state and delivery format does the public Task request?",
            **{slot: deepcopy(cell) for slot in SLOTS},
        },
        "reason": "Brief decisive Task-grounded difference or equivalence after the paired facts.",
        "evidence_refs": body["response"]["evidence_refs"],
        decision: body["response"][decision],
    }
    result["instruction"] += (
        " Return requirement_comparisons, action_format_check, reason, evidence_refs, then the final label."
        " Compare the same supplied public Task unit for both candidates in each row; keep each fact brief."
        " Task units are verbatim context, not a new rubric or a point-counting scheme."
        " Separate what products claim from independently observed supporting facts; absence of evidence"
        " is not proof of absence. In action_format_check distinguish requested preparation/recommendation"
        " from observed execution. Extra actions or stronger commitments receive no automatic credit."
        " Do not invent a required file, storage location or format absent from the Task; equivalent"
        " delivery forms are acceptable. Cite each candidate's own evidence aliases only, or [] for"
        " missing evidence. Then independently judge the complete Task outcome, not the number of rows,"
        " input position or obedience to a proposed method. A faithful tie is allowed; retain uncertainty."
        " This is one response, not a request for another judge or any tool call."
    )
    return result


def _fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} fields differ from Task matrix schema")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _refs(value, allowed, label):
    if (
        not isinstance(value, list)
        or any(not isinstance(ref, str) for ref in value)
        or len(value) != len(set(value))
        or not set(value) <= allowed
    ):
        raise ValueError(
            f"{label} must cite unique known evidence aliases from the correct candidate"
        )


def normalize_matrix_reply(request, response):
    """Check shape/aliases, preserving every model fact and the unchanged verdict."""
    protocol = request.get("outcome_comparison_protocol")
    if protocol is None:
        return response, []
    if (
        request.get("schema_version") not in SCHEMAS
        or protocol.get("id") != PROTOCOL_ID
    ):
        raise ValueError("Unsupported Outcome comparison protocol")
    decision = "verdict" if "verdict" in request["response"] else "preference"
    _fields(
        response,
        (
            "requirement_comparisons",
            "action_format_check",
            "reason",
            "evidence_refs",
            decision,
        ),
        "Outcome comparison",
    )
    _text(response["reason"], "Outcome reason")
    choices = {
        "first",
        "second",
        "tie",
        "unknown" if decision == "verdict" else "abstain",
    }
    if response[decision] not in choices:
        raise ValueError("Invalid independent Outcome label")
    aliases = {
        slot: {row["evidence_ref"] for row in request[slot]["evidence"]}
        for slot in SLOTS
    }
    allowed = set(
        request.get("evidence_contract", {}).get(
            "allowed_refs", request.get("evidence_refs", [])
        )
    )
    _refs(response["evidence_refs"], allowed, "Outcome evidence_refs")
    rows = response["requirement_comparisons"]
    if not isinstance(rows, list):
        raise ValueError("requirement_comparisons must be a list")
    expected = {row["requirement_id"] for row in protocol["task_units"]}
    seen = set()
    for row in rows:
        _fields(row, ("requirement_id", *SLOTS), "Requirement comparison")
        requirement = row["requirement_id"]
        if (
            not isinstance(requirement, str)
            or requirement not in expected
            or requirement in seen
        ):
            raise ValueError("Task matrix requires unique known requirement IDs")
        seen.add(requirement)
    if seen != expected:
        raise ValueError("Task matrix must cover every public Task unit")
    boundary = response["action_format_check"]
    _fields(boundary, ("requested_boundary", *SLOTS), "action_format_check")
    _text(boundary["requested_boundary"], "requested_boundary")
    for row in (*rows, boundary):
        for slot in SLOTS:
            _fields(row[slot], ("fact", "evidence_refs"), f"{slot} fact")
            _text(row[slot]["fact"], f"{slot} fact")
            _refs(row[slot]["evidence_refs"], aliases[slot], f"{slot} evidence_refs")
    normalized = {
        "reason": json.dumps(response, ensure_ascii=False, separators=(",", ":")),
        "evidence_refs": deepcopy(response["evidence_refs"]),
        decision: response[decision],
    }
    return normalized, [
        {
            "operation": "preserve_task_evidence_matrix_in_reason",
            "protocol_id": PROTOCOL_ID,
        }
    ]
