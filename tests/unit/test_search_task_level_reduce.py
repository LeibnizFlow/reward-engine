"""Task-wide coverage must be recomputed, not inherited from one source's absence."""

from copy import deepcopy

import pytest

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import task_review
from tests.support.search_source_map_reduce import draft, fixture
from tests.support.search_summary_fakes import keep_groups


@pytest.mark.asyncio
async def test_reduce_sees_all_evidence_but_not_stale_reader_absence():
    base, requests, records = fixture(), [], {}
    original = deepcopy(base)

    async def model(request):
        requests.append(request)
        if request["mode"] == "task_review_source_map":
            proposal = draft(base["sources"][0])
            proposal["task_need_coverage"] = [
                {
                    "need_id": base["task_needs"][0]["need_id"],
                    "status": "needs_material",
                    "finding_ids": [],
                    "materials_needed": ["STALE_MISSING_MATERIAL"],
                    "open_questions": [
                        {
                            "question": "STALE_SOURCE_LOCAL_ABSENCE",
                            "card_ids": [],
                            "next_action_kind": "inspect_material",
                        }
                    ],
                }
            ]
            return proposal
        assert "reader_questions" not in request
        assert "STALE_SOURCE_LOCAL_ABSENCE" not in str(request)
        assert "STALE_MISSING_MATERIAL" not in str(request)
        assert (
            request["task_needs"][0]["requirement"]
            == base["task_needs"][0]["requirement"]
        )
        assert request["groups"][0]["cards"][0]["source_statement"]
        assert request["sources"][0]["source_id"] == base["sources"][0]["source_id"]
        assert "application_check" not in request["response"]["group_decisions"][0]
        assert "ALL supplied passages" in request["instruction"]
        assert "defer disputed cards" in request["instruction"]
        return keep_groups(request)

    result = await task_review.summarize(
        base,
        model,
        policy=BatchPolicy(deadline_s=1),
        record=lambda key, value: records.__setitem__(key, deepcopy(value)),
    )
    assert result["finalization_status"] == "complete" and result["directions"]
    assert len(requests) == 2 and base == original
    # No destruction of source-local diagnostics: they remain replayable audit data.
    maps = [row for key, row in records.items() if key.startswith("summary_map/")]
    assert maps and "STALE_SOURCE_LOCAL_ABSENCE" in str(maps)
