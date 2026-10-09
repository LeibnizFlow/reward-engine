import asyncio
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ask, build_context
from reward_engine.core.outcome.harness import defaults


def fixture():
    context, evidence = build_context(
        {"task_id": "generic", "prompt": "Summarize the shipment status from records."},
        [
            {
                "artifact": {"final_response": "Shipment arrived."},
                "source_documents": [{"padding": "x" * 700, "status": "arrived"}],
            }
        ],
    )
    result = {
        "status": "fail",
        "reason": "No arrival in preview",
        "gap": "Verify arrival",
        "evidence_refs": ["product_1:artifact:final_response"],
        "claims": [
            {
                "claim": "Shipment arrived",
                "claim_refs": ["product_1:artifact:final_response"],
                "evidence_refs": ["product_1:source:0"],
                "judgment": "unverified",
                "observation": "Not visible in preview",
            }
        ],
    }
    return context, evidence, result


def test_cited_preview_in_claim_is_expanded_before_final_verdict():
    context, evidence, initial = fixture()
    original = deepcopy(context)
    calls = []

    async def call(request):
        calls.append(request)
        if len(calls) == 1:
            assert (
                "arrived"
                not in request["context"]["products"][0]["sources"][0]["preview"]
            )
            return initial
        assert (
            request["requested_evidence"]["product_1:source:0"]["status"] == "arrived"
        )
        row = deepcopy(initial)
        row.update(status="pass", reason="Full record confirms arrival", gap="")
        row["claims"][0].update(
            judgment="supported", observation="Full record: arrived"
        )
        return row

    result = asyncio.run(
        defaults(call).run(
            {
                "test_id": "read",
                "criterion_id": "grounding",
                "checker": "evidence.review",
            },
            context,
            evidence,
        )
    )
    assert result["status"] == "pass" and len(calls) == 2
    assert calls[1]["provisional_check"]["status"] == "fail"
    assert (
        context == original
    )  # No silent rewrite of the frozen input or first model answer.


def test_no_extra_call_without_used_preview_and_other_roles_unchanged():
    context, evidence, initial = fixture()
    for verify, refs in [(False, ["product_1:source:0"]), (True, ["task"])]:
        calls = []

        async def call(request, calls=calls, refs=refs):
            calls.append(request)
            return {"evidence_refs": refs}

        asyncio.run(
            ask(
                call,
                "test",
                context,
                evidence,
                "inspect",
                {},
                verify_used_previews=verify,
            )
        )
        assert len(calls) == 1


def test_delivery_claim_support_reference_also_expands_preview():
    context, evidence, _ = fixture()
    calls = []

    async def call(request):
        calls.append(request)
        if len(calls) == 2:
            assert (
                request["requested_evidence"]["product_1:source:0"]["status"]
                == "arrived"
            )
        return {
            "claim_checks": [
                {"claim": "Shipment arrived", "support_refs": ["product_1:source:0"]}
            ]
        }

    asyncio.run(
        ask(
            call,
            "outcome_delivery_review",
            context,
            evidence,
            "inspect",
            {},
            verify_used_previews=True,
        )
    )
    assert len(calls) == 2


def test_manual_read_and_automatic_read_share_one_followup_budget():
    context, evidence, initial = fixture()
    context["products"][0]["sources"].append(
        {"ref": "product_1:source:1", "preview": "partial", "preview_only": True}
    )
    evidence["product_1:source:1"] = {"full": "second source"}
    calls = []

    async def call(request):
        calls.append(request)
        if len(calls) == 1:
            return {"need_evidence": ["product_1:source:1"]}
        return initial  # Still relies on another unread preview, cannot finalize.

    result = asyncio.run(
        defaults(call).run(
            {
                "test_id": "read",
                "criterion_id": "grounding",
                "checker": "evidence.review",
            },
            context,
            evidence,
        )
    )
    assert result["status"] == "unknown" and result["error"] and len(calls) == 2


def test_full_source_cannot_bypass_request_budget():
    context, evidence, initial = fixture()
    evidence["product_1:source:0"] = {"text": "x" * 170000}
    calls = []

    async def call(request):
        calls.append(request)
        return initial

    asyncio.run(
        defaults(call).run(
            {
                "test_id": "read",
                "criterion_id": "grounding",
                "checker": "evidence.review",
            },
            context,
            evidence,
        )
    )
    # An oversized source is bounded before attachment: the request never carries
    # the full 170 KB and the model is told to page the full record via document.read.
    attached = calls[1]["requested_evidence"]["product_1:source:0"]
    assert attached["excerpt_only"] is True
    assert attached["total_bytes"] == len(
        json.dumps(evidence["product_1:source:0"], ensure_ascii=False).encode()
    )
    assert len(json.dumps(calls[1], ensure_ascii=False).encode()) < 170000


@pytest.mark.parametrize("explicit_flag", [True, False])
def test_catalog_citation_requires_actual_stored_passage_not_title(explicit_flag):
    context, evidence, _ = fixture()
    ref = "reference:task_review:manual"
    context["reference_documents"] = [
        {"ref": ref, "title": "Instructions", "url": "https://example.org"}
    ]
    if explicit_flag:
        context["reference_documents"][0]["preview_only"] = True
    evidence[ref] = {
        "text": "Record status is provisional, not delivered",
        "excerpt_only": True,
        "offset": 500,
        "total_chars": 30000,
    }
    context["available_evidence_refs"].append(ref)
    calls = []

    async def call(request):
        calls.append(request)
        if len(calls) == 1:
            assert "provisional, not delivered" not in str(request)
            return {
                "claim_checks": [{"support_refs": [ref, ref]}],
                "status": "target_satisfied",
            }
        assert request["requested_evidence"] == {ref: evidence[ref]}
        assert "excerpt_only" in request["evidence_followup"]
        return {
            "status": "unknown",
            "reason": "Passage does not establish delivery",
            "evidence_refs": [ref],
        }

    result = asyncio.run(
        ask(
            call,
            "outcome_delivery_review",
            context,
            evidence,
            "inspect",
            {},
            verify_used_previews=True,
        )
    )
    assert result["status"] == "unknown" and len(calls) == 2


def test_uncited_catalog_does_not_trigger_source_read():
    context, evidence, _ = fixture()
    context["reference_documents"] = [
        {"ref": "reference:unused", "title": "Unneeded manual"}
    ]
    calls = []

    async def call(request):
        calls.append(request)
        return {"evidence_refs": ["task"]}

    asyncio.run(
        ask(call, "test", context, evidence, "inspect", {}, verify_used_previews=True)
    )
    assert len(calls) == 1
