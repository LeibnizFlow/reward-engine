"""Retention is an immutable artifact choice, never a native-state rollback."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from reward_engine.core.outcome import version_selection
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.harness import defaults
from reward_engine.core.outcome.runner import (
    DeliveryReview,
    review_result_fields,
    run_delivery,
)
from tests.support.delivery_retention import (
    TASK,
    ScriptedActor,
    prefer_regressed,
    published_pack,
    reviewed_row,
    scripted_review,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts,retained_id", [(("{}", "{"), "delivery_1"), (("{", "{}", "{"), "delivery_2")]
)
async def test_regression_exports_preceding_artifact_without_rebinding_native(
    texts, retained_id
):
    actor, review, row = await reviewed_row(texts)
    session = row["review_session"]
    assert actor.calls == ["start"] + ["resume"] * (len(texts) - 1)
    assert [
        v["candidate"]["artifact"]["final_response"] for v in session["versions"]
    ] == list(texts)
    assert session["retention"] == {
        "status": "artifact_retained",
        "reason": "critical_check_regression",
        "retained_version_id": retained_id,
        "rejected_version_id": f"delivery_{len(texts)}",
        "lost_checks": ["parse"],
        "native_state_restored": False,
    }
    assert review.status == "revision_regressed"
    assert row["candidate"] == actor.current
    assert row["candidate"]["artifact"]["final_response"] == "{"
    assert row["final_measurement"]["results"][0]["status"] == "fail"
    assert row["retained_candidate"]["artifact"]["final_response"] == "{}"
    assert row["retained_measurement"]["results"][0]["status"] == "pass"
    assert (
        row["retained_delivery"]["measurement_source"] == f"checks/{retained_id}.json"
    )
    assert row["retained_delivery"]["official_evaluation"] == "not_run"
    assert row["retained_delivery"]["native_state_restored"] is False
    assert "feedback_execution" not in row
    version_selection.validate_review_session(row, TASK, review.pack)


@pytest.mark.asyncio
async def test_recorder_actor_and_export_mutations_do_not_change_review_evidence():
    def corrupt_record(name, value):
        if name.startswith("delivery_"):
            value["candidate"]["artifact"]["final_response"] = "RECORDER MUTATION"
            value["measurement"]["results"].clear()
        if name == "revision_regression":
            value["retention"]["lost_checks"].clear()

    actor, review, row = await reviewed_row(record=corrupt_record)
    snapshot = review.export_session()
    row["review_session"]["versions"][0]["candidate"]["artifact"].clear()
    row["retained_measurement"]["results"].clear()
    actor.current["artifact"].clear()
    review.history[0]["results"].clear()
    assert review.export_session() == snapshot
    assert snapshot["retention"]["lost_checks"] == ["parse"]
    assert snapshot["versions"][0]["candidate"]["artifact"]["final_response"] == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["unknown", "error", "missing"])
async def test_unavailable_critical_recheck_retains_prior_evidence(
    failure, monkeypatch
):
    from reward_engine.core.outcome import delivery_triage

    real_review = delivery_triage.review
    calls = 0

    async def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = await real_review(*args, **kwargs)
        if calls == 2:
            if failure == "missing":
                result["results"] = []
            else:
                result["results"][0].update(status="unknown", error=failure == "error")
        return result

    monkeypatch.setattr(delivery_triage, "review", interrupted)
    _, review, row = await reviewed_row()
    assert review.status == "revision_regressed"
    assert row["review_session"]["retention"]["lost_checks"] == ["parse"]
    assert row["retained_measurement"]["results"][0]["status"] == "pass"
    assert row["final_measurement"]["results"] == review.history[-1]["results"]
    version_selection.validate_review_session(row, TASK, review.pack)


@pytest.mark.asyncio
async def test_changed_user_scope_records_unavailable_retention():
    class ChangedTaskActor(ScriptedActor):
        async def resume(self, feedback):
            candidate = await super().resume(feedback)
            candidate["artifact"]["public_consultation"] = {
                "earlier_exchanges": [{"speaker": "user", "text": "Use a new format."}]
            }
            return candidate

    actor = ChangedTaskActor()
    review = DeliveryReview(TASK, published_pack(), defaults(scripted_review))
    native = await run_delivery(actor, review)
    row = {
        "candidate": native,
        "review_status": review.status,
        **review.result_fields(native),
    }
    assert row["review_session"]["retention"]["status"] == "retention_unavailable"
    assert "retained_candidate" not in row
    with pytest.raises(ValueError, match="retention unavailable"):
        version_selection.validate_review_session(row, TASK, review.pack)


@pytest.mark.asyncio
async def test_real_selector_rejects_regression_despite_ranker_preference():
    _, review, row = await reviewed_row()
    native, measurement, session = deepcopy(
        (row["candidate"], row["final_measurement"], row["review_session"])
    )
    await version_selection.select_versions(
        TASK,
        review.pack,
        {"C": row},
        defaults(prefer_regressed),
        lambda _: prefer_regressed,
    )
    assert row["selected_candidate"] == row["retained_candidate"]
    assert (row["candidate"], row["final_measurement"], row["review_session"]) == (
        native,
        measurement,
        session,
    )
    decision = row["delivery_selection"]["decisions"][0]
    assert decision["strictly_preferred"] and not decision["accepted"]
    assert decision["lost_critical_checks"] == ["parse"]
    for version in row["version_checks"]:
        assert version["measurement_sha256"] == fingerprint(version["measurement"])
        assert (
            version["measurement_source"]
            == "version_checks/" + version["artifact_sha256"] + ".json"
        )


@pytest.mark.asyncio
async def test_result_export_rejects_historical_candidate_as_native():
    _, review, row = await reviewed_row()
    with pytest.raises(ValueError, match="Native candidate differs"):
        review.result_fields(row["retained_candidate"])


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize(
    "corruption",
    [
        "missing_session",
        "missing_versions",
        "missing_first",
        "reorder",
        "candidate",
        "source_documents",
        "measurement",
        "measurement_pack",
        "retained_id",
        "lost_checks",
        "native",
        "terminal_measurement",
        "retained_measurement",
        "scope",
        "physical_restore",
        "missing_retention",
        "invalid_results",
        "missing_source",
        "version_id",
    ],
)
async def test_corrupt_regression_history_fails_before_any_selection_work(
    corruption, scoped, monkeypatch
):
    _, review, row = await reviewed_row()
    session = row["review_session"]
    if scoped:
        row.update(effective_task=TASK, effective_pack=review.pack)
    first, last = session["versions"]
    if corruption == "missing_session":
        row.pop("review_session")
    elif corruption == "missing_versions":
        session["versions"] = []
    elif corruption == "missing_first":
        session["versions"].pop(0)
    elif corruption == "reorder":
        session["versions"].reverse()
    elif corruption == "candidate":
        first["candidate"]["artifact"]["final_response"] = "CHANGED"
    elif corruption == "source_documents":
        first["candidate"]["source_documents"] = ["CHANGED"]
    elif corruption == "measurement":
        first["measurement"]["results"].clear()
    elif corruption == "measurement_pack":
        first["measurement"]["pack_id"] = "wrong"
        first["measurement_sha256"] = fingerprint(first["measurement"])
    elif corruption == "retained_id":
        session["retention"]["retained_version_id"] = "delivery_99"
    elif corruption == "lost_checks":
        session["retention"]["lost_checks"] = []
    elif corruption == "native":
        row["candidate"] = deepcopy(first["candidate"])
    elif corruption == "terminal_measurement":
        row["final_measurement"] = deepcopy(first["measurement"])
    elif corruption == "retained_measurement":
        row["retained_measurement"] = deepcopy(last["measurement"])
    elif corruption == "scope":
        first["requirements_sha256"] = "different"
    elif corruption == "physical_restore":
        session["retention"]["native_state_restored"] = True
    elif corruption == "missing_retention":
        session["retention"] = None
    elif corruption == "invalid_results":
        last["measurement"]["results"] = None
        last["measurement_sha256"] = fingerprint(last["measurement"])
    elif corruption == "missing_source":
        first["measurement_source"] = ""
    elif corruption == "version_id":
        last["candidate_id"] = first["candidate_id"]
    measure, rank = AsyncMock(), AsyncMock()
    monkeypatch.setattr(version_selection, "measure", measure)
    monkeypatch.setattr(version_selection.jury, "rank", rank)
    with pytest.raises(ValueError, match="Invalid review session"):
        await version_selection.select_versions(
            TASK, review.pack, {"C": row}, None, lambda _: None
        )
    measure.assert_not_awaited()
    rank.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [("{}",), ("{}", '{"x":1}')])
async def test_no_regression_exports_history_without_retained_or_feedback_claims(texts):
    _, review, row = await reviewed_row(texts)
    assert row["review_session"]["retention"] is None
    assert "retained_candidate" not in row and "feedback_execution" not in row
    assert row["candidate"]["artifact"]["final_response"] == texts[-1]
    version_selection.validate_review_session(row, TASK, review.pack)


@pytest.mark.asyncio
async def test_generic_callback_and_absent_review_preserve_terminal_contract():
    calls = []

    async def callback(candidate):
        calls.append(deepcopy(candidate))
        return "revise" if len(calls) == 1 else None

    final = await run_delivery(ScriptedActor(), callback)
    assert final["artifact"]["final_response"] == "{"
    assert review_result_fields(callback, final) == {}
    initial = await run_delivery(ScriptedActor(), None)
    assert initial["artifact"]["final_response"] == "{}"
    assert review_result_fields(None, initial) == {}
