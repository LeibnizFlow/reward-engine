"""Source facts project into existing knowledge fields without new repair advice."""

import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome import task_review
from reward_engine.core.outcome.knowledge_evidence import spans

VERIFICATION = (
    "Check these source statements and their applicability against the actual Task evidence; "
    "no task-specific repair is established here."
)
UNSUPPORTED_ADVICE = "UNSUPPORTED_REPAIR: replace every frame salt without checking generation or checksums"


def base():
    bodies = [
        "Frame salts must equal the WAL header salts. Compare the two salt fields before interpreting the frame.",
        "Preserve the original bytes. Check header and frame salts from the same WAL generation; "
        "a mismatched generation is not established merely by a damaged frame.",
    ]
    batch = {"complete": True, "total": 2, "succeeded": 2, "items": []}
    return {
        "sources": [
            {
                "source_id": f"source_{i}",
                "title": f"Specification {i}",
                "url": f"https://docs.example/spec/{i}",
                "text": text,
                "offset": 0,
                "total_chars": len(text),
                "source_kind": "web_document",
                "read_kind": "body",
            }
            for i, text in enumerate(bodies)
        ],
        "task_needs": [
            {
                "need_id": "need_0",
                "requirement": "Explain why the supplied WAL frames are ignored",
                "task_basis": "BASIS_NOT_A_GENERATED_CONNECTION",
                "next_action": "NEXT_ACTION_NOT_SOURCE_FACT",
            },
            {
                "need_id": "need_1",
                "requirement": "Preserve the original evidence during inspection",
                "task_basis": "Retain the input",
                "next_action": "Inspect a copy",
            },
            {
                "need_id": "need_2",
                "requirement": "Report limitations of the diagnosis",
                "task_basis": "Do not overstate conclusions",
                "next_action": "Explain uncertainty",
            },
        ],
        "search_batch": deepcopy(batch),
        "read_batch": deepcopy(batch),
        "research": {"history": [], "stop_reason": "model_stop"},
        "skill_search": False,
    }


def citation(source):
    return {
        "source_id": source["source_id"],
        "span_id": spans(source["text"])[0]["span_id"],
    }


def finding(original, i=0, **changes):
    return {
        "id": f"finding_{i}",
        "finding": original["sources"][i]["text"],
        "applicability": f"Applies only to the documented WAL format and generation {i}.",
        "need_ids": [f"need_{i}"],
        "citations": [citation(original["sources"][i])],
        "disposition": "retained",
        "direction_id": "knowledge_0",
        "reason": "AUDIT_ONLY: retain this observed source statement",
        **changes,
    }


def brief(original):
    return {
        "source_contract": "source-facts-v1",
        "source_findings": [finding(original)],
        "directions": [{"id": "knowledge_0", "direction": "WAL validation mechanisms"}],
    }


def admitted(original, draft):
    normalized = task_review.normalize_source_brief(original, draft)
    return normalized, task_review.admit_brief(
        {**original, "source_contract": "source-facts-v1"}, normalized, span_only=True
    )


def test_projection_drops_model_connections_and_repair_fields_without_mutating_inputs():
    original, draft = base(), brief(base())
    for row in draft["source_findings"] + draft["directions"]:
        row.update(
            relevance=UNSUPPORTED_ADVICE,
            verification=UNSUPPORTED_ADVICE,
            proposal=UNSUPPORTED_ADVICE,
            advice=UNSUPPORTED_ADVICE,
            arbitrary_payload={"repair": UNSUPPORTED_ADVICE},
        )
    draft["directions"][0]["applicability"] = UNSUPPORTED_ADVICE
    before_base, before_draft = deepcopy(original), deepcopy(draft)
    canonical, result = admitted(original, draft)
    mechanism, direction = canonical["source_findings"][0], canonical["directions"][0]
    requirement = original["task_needs"][0]["requirement"]
    assert mechanism["relevance"] == direction["relevance"] == requirement
    assert mechanism["verification"] == direction["verification"] == VERIFICATION
    assert mechanism["finding"] == original["sources"][0]["text"]
    assert mechanism["reason"] == draft["source_findings"][0]["reason"]
    assert "AUDIT_ONLY" not in json.dumps(direction)
    assert UNSUPPORTED_ADVICE not in json.dumps(canonical)
    assert "BASIS_NOT_A_GENERATED_CONNECTION" not in json.dumps(canonical)
    assert "NEXT_ACTION_NOT_SOURCE_FACT" not in json.dumps(canonical)
    assert (
        len(result["directions"]) == 1
        and len(result["directions"][0]["mechanisms"]) == 1
    )
    assert "AUDIT_ONLY" not in json.dumps(result["directions"])
    assert original == before_base and draft == before_draft


def test_direction_fields_are_assembled_only_from_retained_member_facts():
    original = base()
    draft = brief(original)
    first, second = (
        finding(original),
        finding(original, 1, need_ids=["need_1", "need_2"]),
    )
    draft["source_findings"] = [
        first,
        second,
        finding(
            original,
            id="deferred",
            disposition="deferred",
            applicability="DEFERRED_CONDITION",
        ),
        finding(
            original,
            id="excluded",
            disposition="excluded",
            applicability="EXCLUDED_CONDITION",
        ),
    ]
    draft["directions"][0].update(
        relevance=UNSUPPORTED_ADVICE,
        verification=UNSUPPORTED_ADVICE,
        applicability=UNSUPPORTED_ADVICE,
        citations=[{"source_id": "fabricated", "span_id": "fake"}],
    )
    canonical, result = admitted(original, draft)
    direction = canonical["directions"][0]
    for member in (first, second):
        assert member["applicability"] in direction["applicability"]
    assert "DEFERRED_CONDITION" not in direction["applicability"]
    assert "EXCLUDED_CONDITION" not in direction["applicability"]
    for need in original["task_needs"]:
        assert need["requirement"] in direction["relevance"]
    assert direction["verification"] == VERIFICATION
    assert {(c["source_id"], c["span_id"]) for c in direction["citations"]} == {
        (c["source_id"], c["span_id"])
        for row in (first, second)
        for c in row["citations"]
    }
    assert {r["id"] for r in result["directions"][0]["mechanisms"]} == {
        "finding_0",
        "finding_1",
    }
    assert {r["id"] for r in result["pending_findings"]} == {"deferred"}
    assert {r["id"] for r in result["source_findings"]} == {
        "finding_0",
        "finding_1",
        "deferred",
        "excluded",
    }


@pytest.mark.parametrize(
    "binding",
    [
        "missing",
        [],
        None,
        "need_0",
        ["unknown"],
        ["need_0", "unknown"],
        [True],
        [["need_0"]],
        [{"need_id": "need_0"}],
    ],
)
def test_malformed_or_missing_need_binding_defers_the_fact_instead_of_deleting_it(
    binding,
):
    original = base()
    draft = brief(original)
    if binding == "missing":
        draft["source_findings"][0].pop("need_ids")
    else:
        draft["source_findings"][0]["need_ids"] = binding
    canonical, result = admitted(original, draft)
    assert len(canonical["source_findings"]) == 1
    assert canonical["source_findings"][0]["disposition"] == "deferred"
    assert canonical["directions"] == [] and result["directions"] == []
    assert len(result["source_findings"]) == len(result["pending_findings"]) == 1
    pending = result["pending_findings"][0]
    assert pending["finding"] == original["sources"][0]["text"]
    assert pending["citations"][0]["quote"] == original["sources"][0]["text"]
    assert pending["reason"] == draft["source_findings"][0]["reason"]


def test_legitimate_source_backed_check_in_finding_is_not_stripped_as_free_verification():
    original = base()
    draft = brief(original)
    draft["source_findings"] = [finding(original, 1)]
    draft["source_findings"][0]["verification"] = UNSUPPORTED_ADVICE
    canonical, result = admitted(original, draft)
    preserved = result["directions"][0]["mechanisms"][0]
    assert (
        "Check header and frame salts from the same WAL generation"
        in preserved["finding"]
    )
    assert preserved["finding"] == original["sources"][1]["text"]
    assert preserved["citations"][0]["quote"] == original["sources"][1]["text"]
    assert preserved["verification"] == VERIFICATION
    assert UNSUPPORTED_ADVICE not in json.dumps(canonical)


def test_projection_does_not_repair_bad_spans_or_promote_navigation_to_evidence():
    original = base()
    original["sources"].append(
        {**original["sources"][0], "source_id": "navigation", "read_kind": "navigation"}
    )
    draft = brief(original)
    draft["source_findings"] = [
        finding(
            original,
            id="invalid_span",
            citations=[{"source_id": "source_0", "span_id": "not_observed"}],
        ),
        finding(
            original, id="navigation_only", citations=[citation(original["sources"][2])]
        ),
    ]
    canonical, result = admitted(original, draft)
    assert {r["id"] for r in canonical["source_findings"]} == {
        "invalid_span",
        "navigation_only",
    }
    assert result["directions"] == []
    assert {r["id"] for r in result["pending_findings"]} == {
        "invalid_span",
        "navigation_only",
    }
    assert all(r["grounding"] == "unverified" for r in result["pending_findings"])


def test_unused_topic_cannot_supply_its_own_citations_or_conditions():
    original = base()
    draft = brief(original)
    draft["directions"].append(
        {
            "id": "orphan",
            "direction": "Unbacked repair topic",
            "citations": [citation(original["sources"][1])],
            "relevance": UNSUPPORTED_ADVICE,
            "applicability": UNSUPPORTED_ADVICE,
            "verification": UNSUPPORTED_ADVICE,
        }
    )
    canonical, result = admitted(original, draft)
    assert [r["id"] for r in canonical["directions"]] == ["knowledge_0"]
    assert [r["id"] for r in result["directions"]] == ["knowledge_0"]
    assert UNSUPPORTED_ADVICE not in json.dumps(canonical)


def test_bad_member_cannot_smuggle_conditions_into_an_otherwise_valid_group():
    original = base()
    draft = brief(original)
    draft["source_findings"].append(
        finding(
            original,
            1,
            applicability=UNSUPPORTED_ADVICE,
            citations=[{"source_id": "absent", "span_id": "invented"}],
        )
    )
    canonical, result = admitted(original, draft)
    assert UNSUPPORTED_ADVICE not in json.dumps(result["directions"])
    assert result["pending_findings"][0]["applicability"] == UNSUPPORTED_ADVICE
    assert canonical["source_findings"][1]["disposition"] == "deferred"


def test_new_projection_does_not_change_direct_legacy_admission_contract():
    original = base()
    legacy = {
        "directions": [
            {
                "id": "old",
                "direction": "Inspect documented WAL fields",
                "relevance": "Legacy task connection",
                "applicability": "Matching format only",
                "verification": "Legacy check text",
                "citations": [citation(original["sources"][0])],
            }
        ]
    }
    result = task_review.admit_brief(original, legacy, span_only=True)
    assert result["directions"][0]["verification"] == "Legacy check text"
    assert result["directions"][0]["relevance"] == "Legacy task connection"
