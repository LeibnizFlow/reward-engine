"""Offline actual-Role replay of Claw v9 partition repair; no native regrading."""

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome import jury
from reward_engine.core.outcome.context import (
    EvidenceReadLimit,
    ResponseContractError,
    ask,
    build_context,
    inline_recent_sources,
)
from tests.unit.test_workbuddy_stream_terminal_replay import encode, event
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

# Exact parsed replies retained from Claw v9 R1 jury_member_1. Tests below use
# synthetic source content with the same handle inventory, never claim factual
# validation of these native observations or acceptance of the stopped run.
CAPTURE_FIRST = {
    "products": [
        {"id": "product_5", "groups": [["product_5"]]},
        {"id": "product_3", "groups": [["product_3"]]},
        {"id": "product_6", "groups": [["product_6"]]},
        {"id": "product_2", "groups": [["product_2"]]},
        {"id": "product_4", "groups": [["product_4"]]},
        {"id": "product_1", "groups": [["product_1"]]},
    ],
    "unknown": [],
    "observations": [
        {
            "product_id": "product_5",
            "reason": "Passes C1–C7 with the deepest task-relevant treatment of the two "
            "anomaly cases (INV-002 → A/B/C verification branches tied to invoice "
            "key, TXN-703 pending → names approver 张总监 and follow-up route) plus "
            "explicit annotations for msg_705, TXN-705/706 exclusions and "
            "explicit rule declarations (matching key, currency, unmatched row "
            "template, bounded date-gap hypothesis). All seven checks pass and "
            "the extra depth is task-relevant rather than unrequested extras.",
            "evidence_refs": [
                "product_5:artifact:final_response",
                "product_5:source:2",
                "product_5:source:3",
                "product_5:source:8",
                "check:4:0",
                "check:4:1",
                "check:4:2",
                "check:4:3",
                "check:4:4",
                "check:4:5",
                "check:4:6",
            ],
        },
        {
            "product_id": "product_3",
            "reason": "Passes C1–C7 with full coverage, correct three-category "
            "classification, sourced amounts/status, pending described as "
            "未审批/尚未完成 without unsupported failure labels, and explicit footnote "
            "exclusions for msg_705 and TXN-705/TXN-706. No self-contradictory "
            "draft inside the final response.",
            "evidence_refs": [
                "product_3:artifact:final_response",
                "product_3:source:2",
                "product_3:source:3",
                "check:2:0",
                "check:2:1",
                "check:2:2",
                "check:2:3",
                "check:2:4",
                "check:2:5",
                "check:2:6",
            ],
        },
        {
            "product_id": "product_6",
            "reason": "Passes C1–C7 with full coverage, correct classification, sourced "
            "IDs, and explicit footnote exclusions for msg_705 and "
            "TXN-705/TXN-706. Usability defect only: the '匹配正常 (3 笔)' table "
            "initially lists msg_702 (with the wrong amount) before a "
            "self-correction, which is a presentation flaw rather than a "
            "substantive miscategorization of C3's evidence requirement.",
            "evidence_refs": [
                "product_6:artifact:final_response",
                "product_6:source:2",
                "product_6:source:3",
                "product_6:source:8",
                "check:5:0",
                "check:5:1",
                "check:5:2",
                "check:5:3",
                "check:5:4",
                "check:5:5",
                "check:5:6",
            ],
        },
        {
            "product_id": "product_2",
            "reason": "Passes C1–C7; finance-side noise (TXN-705/706) is explicitly "
            "footnoted and classified correctly. Minor completeness gap: the "
            "single non-reimbursement inbox email msg_705 is not individually "
            "footnoted (only implicit via the '4 封报销邮件' scope), so the inbox-side "
            "exclusion annotation is less explicit than product_3/5/6.",
            "evidence_refs": [
                "product_2:artifact:final_response",
                "product_2:source:2",
                "product_2:source:7",
                "check:1:0",
                "check:1:1",
                "check:1:2",
                "check:1:3",
                "check:1:4",
                "check:1:5",
                "check:1:6",
            ],
        },
        {
            "product_id": "product_4",
            "reason": "Passes C1–C6 but fails C7: TXN-705 and TXN-706 are not identified or "
            "annotated as excluded, leaving finance-channel noise exclusion "
            "incomplete despite msg_705 being noted. Otherwise correct coverage, "
            "IDs, and classification.",
            "evidence_refs": [
                "product_4:artifact:final_response",
                "product_4:source:2",
                "product_4:source:3",
                "check:3:0",
                "check:3:1",
                "check:3:2",
                "check:3:3",
                "check:3:4",
                "check:3:5",
                "check:3:6",
            ],
        },
        {
            "product_id": "product_1",
            "reason": "Passes C1–C6 but fails C7: neither msg_705 nor TXN-705/TXN-706 is "
            "mentioned or given an exclusion rationale. Classification and "
            "cross-check content are otherwise correct.",
            "evidence_refs": [
                "product_1:artifact:final_response",
                "product_1:source:2",
                "product_1:source:3",
                "check:0:0",
                "check:0:1",
                "check:0:2",
                "check:0:3",
                "check:0:4",
                "check:0:5",
                "check:0:6",
            ],
        },
    ],
    "reason": "Task requires both reading the inbox and the finance system, then producing a "
    "three-category reconciliation report. Ranking is driven by: (a) correctness of "
    "extraction, finance lookup, and amount/status cross-check (all six products do this "
    "correctly per C1–C3/C4–C6); (b) completeness of the cross-channel scope statement "
    "required by C7 — identifying and explaining msg_705, TXN-705, and TXN-706 exclusions. "
    "Product_1 and product_4 fail C7 (product_4 fails the finance side only; product_1 "
    "fails both sides), placing them last. Product_2 partially satisfies C7 (finance "
    "footnoted, inbox only implicit), below product_3 and product_6 which annotate both "
    "sides. Product_6 is correct but has a self-contradictory draft presentation ('匹配正常 (3 "
    "笔)' row for msg_702 with a follow-up correction), so it sits below product_3. "
    "Product_5 adds task-relevant depth (named approver for pending, A/B/C decision "
    "branches on INV-002, explicit unmatched-row template, bounded date-gap explanation) on "
    "the same fully-correct factual base as product_3, so it leads.",
}

CAPTURE_REPAIR = {
    "groups": [
        ["product_5"],
        ["product_3"],
        ["product_6"],
        ["product_2"],
        ["product_4"],
        ["product_1"],
    ],
    "unknown": [],
    "observations": [
        {
            "product_id": "product_5",
            "reason": "Passes C1–C7 and adds task-relevant depth beyond minimum: names "
            "approver 张总监 for TXN-703 pending with follow-up route, supplies "
            "A/B/C decision branches for INV-002 keyed on the invoice, declares "
            "matching key/currency rules, gives an explicit unmatched-row "
            "template, and provides a bounded hypothesis for the "
            "email–transaction date gap. Exclusions of msg_705, TXN-705, and "
            "TXN-706 are all annotated with reasons. No substantive defects.",
            "evidence_refs": [
                "product_5:artifact:final_response",
                "product_5:source:2",
                "product_5:source:3",
                "product_5:source:8",
                "check:4:0",
                "check:4:1",
                "check:4:2",
                "check:4:3",
                "check:4:4",
                "check:4:5",
                "check:4:6",
            ],
        },
        {
            "product_id": "product_3",
            "reason": "Passes C1–C7 with full coverage, correct three-category "
            "classification, sourced amounts/status, pending described as "
            "未审批/尚未完成 without unsupported failure labels, and explicit footnote "
            "exclusions for msg_705 and TXN-705/TXN-706. No self-contradictory "
            "content in the final response.",
            "evidence_refs": [
                "product_3:artifact:final_response",
                "product_3:source:2",
                "product_3:source:3",
                "product_3:source:8",
                "check:2:0",
                "check:2:1",
                "check:2:2",
                "check:2:3",
                "check:2:4",
                "check:2:5",
                "check:2:6",
            ],
        },
        {
            "product_id": "product_6",
            "reason": "Passes C1–C7 with full coverage, correct classification, sourced "
            "IDs, and explicit footnote exclusions for msg_705 and "
            "TXN-705/TXN-706. Usability defect only: the '匹配正常 (3 笔)' table "
            "initially lists msg_702 with the wrong system amount before a "
            "self-correction, which is a presentation flaw rather than a "
            "substantive miscategorization of any criterion's evidence "
            "requirement.",
            "evidence_refs": [
                "product_6:artifact:final_response",
                "product_6:source:2",
                "product_6:source:3",
                "product_6:source:8",
                "check:5:0",
                "check:5:1",
                "check:5:2",
                "check:5:3",
                "check:5:4",
                "check:5:5",
                "check:5:6",
            ],
        },
        {
            "product_id": "product_2",
            "reason": "Passes C1–C7; finance-side noise (TXN-705/TXN-706) is explicitly "
            "footnoted and classification is correct. Minor completeness gap: the "
            "non-reimbursement inbox email msg_705 is not individually footnoted "
            "(only implicit via the '4 封报销邮件' scope), so the inbox-side exclusion "
            "annotation is less explicit than in product_3/5/6.",
            "evidence_refs": [
                "product_2:artifact:final_response",
                "product_2:source:2",
                "product_2:source:7",
                "check:1:0",
                "check:1:1",
                "check:1:2",
                "check:1:3",
                "check:1:4",
                "check:1:5",
                "check:1:6",
            ],
        },
        {
            "product_id": "product_4",
            "reason": "Passes C1–C6 but fails C7: TXN-705 and TXN-706 are not identified or "
            "annotated as excluded, leaving finance-channel noise exclusion "
            "incomplete despite msg_705 being noted. Coverage, IDs, and "
            "classification are otherwise correct.",
            "evidence_refs": [
                "product_4:artifact:final_response",
                "product_4:source:2",
                "product_4:source:3",
                "check:3:0",
                "check:3:1",
                "check:3:2",
                "check:3:3",
                "check:3:4",
                "check:3:5",
                "check:3:6",
            ],
        },
        {
            "product_id": "product_1",
            "reason": "Passes C1–C6 but fails C7: neither msg_705 nor TXN-705/TXN-706 is "
            "mentioned or given an exclusion rationale, so both channels of the "
            "C7 scope requirement are unmet. Classification and cross-check "
            "content are otherwise correct.",
            "evidence_refs": [
                "product_1:artifact:final_response",
                "product_1:source:2",
                "product_1:source:3",
                "check:0:0",
                "check:0:1",
                "check:0:2",
                "check:0:3",
                "check:0:4",
                "check:0:5",
                "check:0:6",
            ],
        },
    ],
    "reason": "The Task requires reading both the inbox and the finance system and producing a "
    "three-category reconciliation report. All six products correctly perform extraction "
    "(C1), finance lookup (C2), amount/status cross-check (C3), report structure (C4), "
    "uncertainty handling (C5), and provenance IDs (C6). Ranking therefore turns on "
    "completeness of the cross-channel scope statement under C7 — identifying and "
    "explaining msg_705 (inbox), TXN-705 and TXN-706 (finance) exclusions. Product_1 and "
    "product_4 fail C7 (product_4 fails the finance side only; product_1 fails both sides) "
    "and rank last. Product_2 partially satisfies C7 (finance footnoted, inbox only "
    "implicit), placing it below the products that annotate both sides. Product_6 is "
    "correct but has a self-contradictory draft presentation ('匹配正常 (3 笔)' row for msg_702 "
    "with a follow-up correction), so it sits below product_3. Product_5 adds task-relevant "
    "depth (named approver for pending, A/B/C decision branches on INV-002, explicit "
    "unmatched-row template, bounded date-gap explanation) on the same fully-correct "
    "factual base as product_3, so it leads.",
}


def captured_context(*, previews=False):
    context, evidence = build_context(
        {"task_id": "offline-replay", "prompt": "Compare the supplied products."},
        [
            {
                "artifact": {"final_response": f"Offline product {i + 1}"},
                "source_documents": [
                    {
                        "text": "Synthetic observed source. "
                        * (
                            100
                            if previews and j in ({2, 7} if i == 1 else {2, 3})
                            else 1
                        )
                    }
                    for j in range(count)
                ],
            }
            for i, count in enumerate([8, 8, 8, 12, 9, 9])
        ],
    )
    for i in range(6):
        for j in range(7):
            evidence[f"check:{i}:{j}"] = {"product_id": f"product_{i + 1}"}
    return context, evidence


def correction():
    # Explicit scripted model response, never inferred by the host under test.
    return {
        "field_corrections": {
            "groups": [
                ["product_3", "product_5"],
                ["product_6"],
                ["product_2"],
                ["product_4"],
                ["product_1"],
            ],
            "unknown": [],
        }
    }


def corrected():
    return {**deepcopy(CAPTURE_FIRST), **correction()["field_corrections"]}


def scripted(transport, replies, *, bounded=True):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": bounded}, "judge", original.output
    )
    for reply in replies:
        text = (
            reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        )
        responses.append(
            httpx.Response(200, text=encode([event({"content": text}, "stop")]))
        )
    return role, calls


async def replay(
    transport,
    replies,
    *,
    context=None,
    evidence=None,
    bounded=True,
    expansion_limit=None,
    **limits,
):
    if context is None:
        context, evidence = captured_context()
    role, calls = scripted(transport, replies, bounded=bounded)
    if expansion_limit is not None:
        role.evidence_expansion_limit = expansion_limit
    receipts = []
    repair = jury._RankingRepair(
        role,
        context,
        evidence,
        record=lambda rows: receipts.__setitem__(slice(None), rows),
    )
    try:
        value = await ask(
            repair,
            "outcome_ranking_map",
            context,
            evidence,
            "Compare products.",
            {"groups": [], "unknown": [], "observations": [], "reason": "text"},
            validate=repair.validate,
            verify_used_previews=True,
            max_contract_retries=1,
            max_syntax_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
            **limits,
        )
        return value, receipts, calls, role
    except Exception as exc:
        exc.test_receipts, exc.test_calls, exc.test_role = receipts, calls, role
        raise


def request(calls, index):
    return json.loads(calls[index]["messages"][1]["content"])


def test_portable_captures_and_handle_inventory():
    for row, expected in [
        (
            CAPTURE_FIRST,
            "8e1dac4add57b4eac3f4cc5b78cbdd8063d1e8d9ab3a320d2d6347387efb89e1",
        ),
        (
            CAPTURE_REPAIR,
            "c3d5a4a0b03051095c973d2cdea572686dc6281fda1bc93950d6affb961f64f4",
        ),
    ]:
        assert (
            hashlib.sha256(
                json.dumps(
                    row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
            == expected
        )
    ctx, evidence = captured_context()
    with pytest.raises(ResponseContractError) as exc:
        jury.validate_ranking(CAPTURE_FIRST, ctx, evidence)
    assert [i["field"] for i in exc.value.details["issues"]] == ["groups/unknown"]
    with pytest.raises(ResponseContractError, match="product_3:source:8"):
        jury.validate_ranking(CAPTURE_REPAIR, ctx, evidence)


@pytest.mark.asyncio
async def test_actual_role_partition_patch_preserves_exact_fields_and_provenance(
    transport,
):
    first, patch = deepcopy(CAPTURE_FIRST), correction()
    before = deepcopy((first, patch))
    result, receipts, calls, role = await replay(transport, [first, patch])
    assert result == corrected()
    assert (first, patch) == before
    assert len(calls) == 2
    recovery = request(calls, 1)["response_recovery"]
    offer = recovery["field_correction_reply"]
    assert offer["schema"] == "ranking-field-correction/v1"
    assert offer["required_fields"] == offer["allowed_fields"] == ["groups", "unknown"]
    assert offer["product_ids"] == [f"product_{i}" for i in range(1, 7)]
    assert "Preferred repair reply" in recovery["instruction"]
    assert recovery["rejected_response"] == first
    assert receipts[0]["response"] == first and not receipts[0]["contract_valid"]
    assert receipts[1]["response"] == patch
    assert receipts[1]["composed_response"] == result
    provenance = receipts[1]["field_provenance"]
    assert provenance["base_attempt"] == 1 and provenance["correction_attempt"] == 2
    for field in result:
        replaced = field in ("groups", "unknown")
        assert provenance["fields"][field] == {
            "kind": "replacement" if replaced else "preserved",
            "source_attempt": 2 if replaced else 1,
            "source_field": "field_corrections." + field if replaced else field,
        }
    raw = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(raw) == 2
    assert {r["raw_output"] for r in raw} == {
        json.dumps(x, ensure_ascii=False) for x in [first, patch]
    }
    assert not any(r["answer"] == result for r in raw)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["captured", "rewrite", "delete", "reorder", "refs", "reason"]
)
async def test_full_reply_cannot_change_valid_assessment(transport, change):
    value = corrected()
    if change == "captured":
        value = deepcopy(CAPTURE_REPAIR)
    elif change == "rewrite":
        value["observations"][0]["reason"] = "New judgment from the same evidence"
    elif change == "delete":
        value["observations"].pop()
    elif change == "reorder":
        value["observations"].reverse()
    elif change == "refs":
        value["observations"][1]["evidence_refs"].append("product_3:source:8")
    else:
        value["reason"] = "New reason"
    with pytest.raises(
        ResponseContractError, match="preserve the already-valid"
    ) as exc:
        await replay(transport, [CAPTURE_FIRST, value])
    assert len(exc.value.test_calls) == 2
    assert exc.value.test_receipts[-1]["response"] == value
    assert not exc.value.test_receipts[-1]["contract_valid"]


@pytest.mark.asyncio
async def test_full_reply_compatibility_preserves_valid_fields(transport):
    value = corrected()
    value.pop("products")  # Invalid alternate partition is not a required field.
    result, _, calls, _ = await replay(transport, [CAPTURE_FIRST, value])
    assert result == value and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "observation",
        "path",
        "missing",
        "sibling",
        "type",
        "unknown_product",
        "duplicate",
        "empty_group",
        "missing_product",
    ],
)
async def test_patch_allowlist_and_full_partition_validation(transport, fault):
    patch = correction()
    fields = patch["field_corrections"]
    if fault == "extra":
        fields["reason"] = "Override"
    elif fault == "observation":
        fields["observations"] = CAPTURE_REPAIR["observations"]
    elif fault == "path":
        fields["observations[0].reason"] = "Override"
    elif fault == "missing":
        fields.pop("unknown")
    elif fault == "sibling":
        patch["reason"] = "Override"
    elif fault == "type":
        patch["field_corrections"] = []
    elif fault == "unknown_product":
        fields["unknown"] = ["product_7"]
    elif fault == "duplicate":
        fields["unknown"] = ["product_1"]
    elif fault == "empty_group":
        fields["groups"].append([])
    else:
        fields["groups"].pop()
    with pytest.raises(ResponseContractError) as exc:
        await replay(transport, [CAPTURE_FIRST, patch])
    assert len(exc.value.test_calls) == 2
    assert exc.value.test_receipts[-1]["response"] == patch
    assert not exc.value.test_receipts[-1]["contract_valid"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
async def test_no_unadvertised_patch_or_legacy_offer(transport, bounded):
    replies = [correction()] if bounded else [CAPTURE_FIRST, correction()]
    # The first unsolicited patch consumes the one contract repair; repeating it fails.
    if bounded:
        replies.append(correction())
    with pytest.raises(ResponseContractError, match="active offer") as exc:
        await replay(transport, replies, bounded=bounded)
    assert (
        "field_correction_reply"
        not in request(exc.value.test_calls, 1)["response_recovery"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("last", ["patch", "rewrite", "tools"])
async def test_syntax_retry_keeps_partition_offer_and_preservation(transport, last):
    value = correction() if last == "patch" else deepcopy(CAPTURE_REPAIR)
    if last == "tools":
        value = {"need_evidence": ["product_1:source:0"]}
    replies = [CAPTURE_FIRST, '{"field_corrections":', value]
    if last == "patch":
        result, receipts, calls, _ = await replay(transport, replies)
        assert result == corrected()
        assert receipts[1]["raw_response"] == replies[1]
        assert receipts[2]["field_provenance"]["base_attempt"] == 1
        assert receipts[2]["field_provenance"]["correction_attempt"] == 3
    else:
        with pytest.raises(ResponseContractError) as exc:
            await replay(transport, replies)
        receipts, calls = exc.value.test_receipts, exc.value.test_calls
    assert len(calls) == 3
    assert (
        request(calls, 2)["response_recovery"]["field_correction_reply"]["base_attempt"]
        == 1
    )
    assert (
        "Preferred repair reply"
        in request(calls, 2)["response_recovery"]["instruction"]
    )
    assert receipts[1]["error_type"] == "JSONDecodeError"


@pytest.mark.asyncio
async def test_corrected_partition_does_not_bypass_unread_preview_failure(transport):
    ctx, evidence = captured_context(previews=True)
    with pytest.raises(EvidenceReadLimit) as exc:
        await replay(
            transport,
            [CAPTURE_FIRST, correction(), corrected(), corrected()],
            context=ctx,
            evidence=evidence,
            expansion_limit=6,
        )
    # Six-source cap does not cover the twelve cited previews; keep the failure.
    calls = exc.value.test_calls
    assert len(calls) == 4
    assert len(request(calls, 2)["requested_evidence"]) == 6
    assert request(calls, 3)["evidence_access_closed"]
    assert exc.value.test_receipts[1]["composed_response"] == corrected()


@pytest.mark.asyncio
async def test_genuine_preview_expansion_requires_full_fresh_assessment(transport):
    ctx, evidence = captured_context(previews=True)
    fresh = deepcopy(CAPTURE_REPAIR)
    # A fresh scripted model assessment cites only fully supplied artifact evidence.
    for row in fresh["observations"]:
        row["evidence_refs"] = [row["product_id"] + ":artifact:final_response"]
    result, _, calls, _ = await replay(
        transport, [CAPTURE_FIRST, correction(), fresh], context=ctx, evidence=evidence
    )
    assert result == fresh
    assert "response_recovery" not in request(calls, 2)
    assert request(calls, 2)["requested_evidence"]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_stale_patch_cannot_replace_fresh_assessment_after_expansion(transport):
    ctx, evidence = captured_context(previews=True)
    with pytest.raises(ResponseContractError, match="active offer") as exc:
        await replay(
            transport,
            [CAPTURE_FIRST, correction(), correction(), correction()],
            context=ctx,
            evidence=evidence,
        )
    assert (
        len(exc.value.test_calls) == 3
    )  # Stage-wide contract budget was already used.
    assert "response_recovery" not in request(exc.value.test_calls, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["task", "inline", "duplicate"])
async def test_redundant_read_does_not_clear_preservation(transport, kind):
    ctx, evidence = captured_context(previews=True)
    if kind == "inline":
        ctx = inline_recent_sources(ctx, evidence)
    role, calls = scripted(transport, [CAPTURE_FIRST, CAPTURE_REPAIR])
    repair = jury._RankingRepair(role, ctx, evidence)
    ref = "task" if kind == "task" else "product_1:source:3"
    req = {"context": ctx}
    if kind == "inline":
        assert ctx["products"][0]["sources"][3]["preview_only"] is False
    if kind == "duplicate":
        req["requested_evidence"] = {ref: evidence[ref]}
    value = await repair(deepcopy(req))
    with pytest.raises(ResponseContractError):
        repair.validate(value)
    # Directly exercise duplicate/inline attachment tracking; ordinary partition
    # repair rejects such model read requests even earlier.
    req["requested_evidence"] = {ref: evidence[ref]}
    req["evidence_access_closed"] = True
    value = await repair(req)
    with pytest.raises(ResponseContractError, match="preserve the already-valid"):
        repair.validate(value)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_rank_reducer_publishes_separate_response_receipts(transport):
    ctx, evidence = captured_context()
    role, calls = scripted(transport, [CAPTURE_FIRST, correction()])

    async def member(_):
        return corrected()

    records = {}
    value = await jury.rank(
        ctx,
        evidence,
        {str(i): member for i in range(3)},
        role,
        record=records.__setitem__,
    )
    assert value == {**corrected(), "status": "compared"}
    assert records["ranking_map"]["succeeded"] == 3
    assert records["ranking_reduce_responses"][1]["response"] == correction()
    assert records["ranking_reduce_responses"][1]["composed_response"] == corrected()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_enveloped_base_provenance_points_to_actual_response_fields(transport):
    from reward_engine.core.outcome.construction import fingerprint

    original = {"response": deepcopy(CAPTURE_FIRST)}
    result, receipts, calls, role = await replay(transport, [original, correction()])
    assert result == corrected()
    assert receipts[0]["response"] == original
    assert receipts[1]["field_provenance"]["fields"]["observations"] == {
        "kind": "preserved",
        "source_attempt": 1,
        "source_field": "response.observations",
    }
    stored = [json.loads(p.read_text()) for p in role.output.glob("*/request.json")]
    assert {fingerprint(r) for r in stored} == {r["request_sha256"] for r in receipts}
    assert all(
        r["request_sha256"] == fingerprint(request(calls, i))
        for i, r in enumerate(receipts)
    )


@pytest.mark.asyncio
async def test_model_can_explicitly_assign_all_products_unknown(transport):
    patch = {
        "field_corrections": {
            "groups": [],
            "unknown": [f"product_{i}" for i in range(1, 7)],
        }
    }
    result, _, calls, _ = await replay(transport, [CAPTURE_FIRST, patch])
    assert result["groups"] == []
    assert result["unknown"] == patch["field_corrections"]["unknown"]
    assert result["observations"] == CAPTURE_FIRST["observations"]
    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["map", "reduce"])
async def test_rank_enforces_all_twelve_preview_reads_with_production_role_limit(
    transport, phase
):
    ctx, evidence = captured_context(previews=True)
    role, calls = scripted(transport, [CAPTURE_FIRST, correction(), corrected()])
    assert role.evidence_expansion_limit == 64  # Actual adapter limit, unchanged.

    async def member(_):
        value = corrected()
        for row in value["observations"]:
            row["evidence_refs"] = [row["product_id"] + ":artifact:final_response"]
        return value

    members = {str(i): member for i in range(3)}
    reducer = role
    if phase == "map":
        members["0"], reducer = role, member
    records = {}
    value = await jury.rank(ctx, evidence, members, reducer, record=records.__setitem__)
    assert value["status"] == "compared"
    assert len(calls) == 3  # Partition correction was only provisional.
    assert len(request(calls, 2)["requested_evidence"]) == 12
    assert "response_recovery" not in request(calls, 2)
    key = "ranking_member_0_responses" if phase == "map" else "ranking_reduce_responses"
    assert records[key][1]["response_kind"] == "field_correction"
    assert records[key][2]["response_kind"] == "full_response"


@pytest.mark.asyncio
async def test_transport_failure_keeps_original_and_does_not_restore_it(transport):
    original, responses, calls = transport
    role, calls = scripted(transport, [CAPTURE_FIRST])
    responses.append(httpx.Response(400, json={"error": "offline transport rejection"}))
    ctx, evidence = captured_context()
    records = []
    repair = jury._RankingRepair(
        role, ctx, evidence, record=lambda rows: records.__setitem__(slice(None), rows)
    )
    with pytest.raises(httpx.HTTPStatusError):
        await ask(
            repair,
            "outcome_ranking_map",
            ctx,
            evidence,
            "Compare",
            {},
            validate=repair.validate,
        )
    assert len(calls) == 2
    assert records[0]["response"] == CAPTURE_FIRST
    assert records[1]["error_type"] == "HTTPStatusError"
    assert "response" not in records[1] and "composed_response" not in records[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("syntax", [False, True])
async def test_repair_read_control_cannot_escape_in_response_envelope(
    transport, wrapped, syntax
):
    ctx, evidence = captured_context(previews=True)
    read = {"need_evidence": ["product_1:source:2"]}
    if wrapped:
        read = {"response": read}
    replies = [CAPTURE_FIRST] + (['{"field_corrections":'] if syntax else []) + [read]
    with pytest.raises(
        ResponseContractError, match="Evidence and tools are closed"
    ) as exc:
        await replay(transport, replies, context=ctx, evidence=evidence)
    assert len(exc.value.test_calls) == (3 if syntax else 2)
    # Role unwraps its supported read-only envelope before returning the object;
    # its provider receipt still retains the exact raw model response.
    assert exc.value.test_receipts[-1]["response"] == (
        read["response"] if wrapped else read
    )
    native = [
        json.loads(p.read_text())
        for p in exc.value.test_role.output.glob("*/response.json")
    ]
    assert any(r["raw_output"] == json.dumps(read, ensure_ascii=False) for r in native)
    assert not any(
        "requested_evidence" in request(exc.value.test_calls, i)
        for i in range(len(exc.value.test_calls))
    )


def rr1_case():
    """Portable exact structure from review006/reproduce_RR1.py (simulation only)."""
    ctx, evidence = build_context(
        {"task_id": "RR1", "prompt": "Compare the supplied answers."},
        [
            {
                "artifact": {"final_response": str(i)},
                "source_documents": [{"text": "A previously unread source. " * 200}],
            }
            for i in (1, 2)
        ],
    )
    initial = {
        "groups": [["product_1"], ["product_2"]],
        "unknown": [],
        "reason": "Original model preference",
        "observations": [
            {
                "product_id": f"product_{i}",
                "reason": "Original observation",
                "evidence_refs": [f"product_{i}:artifact:final_response"],
            }
            for i in (1, 2)
        ],
    }
    bad = deepcopy(initial)
    bad["observations"][0].pop("reason")
    flipped = deepcopy(initial)
    flipped["groups"].reverse()
    flipped["reason"] = "Reversed after repair-time source read"
    return ctx, evidence, initial, bad, flipped


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("syntax", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
async def test_rr1_public_rank_rejects_repair_origin_read_without_partition_offer(
    transport, bounded, syntax, wrapped
):
    ctx, evidence, initial, bad, flipped = rr1_case()
    read = {"need_evidence": ["product_1:source:0"]}
    if wrapped:
        read = {"response": read}
    replies = [bad] + (['{"groups":'] if syntax else []) + [read, flipped]
    role, calls = scripted(transport, replies, bounded=bounded)
    records = {}

    async def member(_):
        return deepcopy(initial)

    with pytest.raises(ResponseContractError, match="Evidence and tools are closed"):
        await jury.rank(
            ctx,
            evidence,
            {str(i): member for i in range(3)},
            role,
            record=records.__setitem__,
        )
    assert "ranking" not in records
    assert len(calls) == 2 + int(syntax)
    assert len(transport[1]) == 1  # The scripted flipped ranking is never requested.
    assert records["ranking_map"]["succeeded"] == 3
    receipts = records["ranking_reduce_responses"]
    assert receipts[0]["response"] == bad and not receipts[0]["contract_valid"]
    assert receipts[-1]["error_type"] == "ResponseContractError"
    for i in range(1, len(calls)):
        req = request(calls, i)
        assert "requested_evidence" not in req
        assert "field_correction_reply" not in req["response_recovery"]
        assert (
            "Evidence and tools are closed" in req["response_recovery"]["instruction"]
        )
    assert request(calls, -1)["response_recovery"]["error_type"] == (
        "JSONDecodeError" if syntax else "ResponseContractError"
    )
    native = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert {r["raw_output"] for r in native} == {
        json.dumps(bad, ensure_ascii=False),
        json.dumps(read, ensure_ascii=False),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("syntax", [False, True])
async def test_rr1_valid_provisional_response_allows_host_preview_then_fresh_ranking(
    transport, bounded, syntax
):
    ctx, evidence, initial, bad, flipped = rr1_case()
    fixed = deepcopy(initial)
    fixed["observations"][0]["evidence_refs"] = ["product_1:source:0"]
    bad["observations"][0]["evidence_refs"] = ["product_1:source:0"]
    replies = [bad] + (['{"groups":'] if syntax else []) + [fixed, flipped]
    role, calls = scripted(transport, replies, bounded=bounded)

    async def member(_):
        return deepcopy(initial)

    records = {}
    result = await jury.rank(
        ctx,
        evidence,
        {str(i): member for i in range(3)},
        role,
        record=records.__setitem__,
    )
    assert result == {**flipped, "status": "compared"}
    assert len(calls) == 3 + int(syntax)
    final_request = request(calls, -1)
    assert "response_recovery" not in final_request
    assert final_request["provisional_check"] == fixed
    assert final_request["requested_evidence"] == {
        "product_1:source:0": evidence["product_1:source:0"]
    }
    assert records["ranking_reduce_responses"][-2]["response"] == fixed
    assert records["ranking_reduce_responses"][-1]["response"] == flipped
