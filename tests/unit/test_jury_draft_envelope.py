from copy import deepcopy

from reward_engine.adapters.claw_eval.reply_format import normalize_reply


def test_named_jury_draft_envelope_preserves_metadata_and_inquiry():
    request = {
        "mode": "outcome_jury_map",
        "response": {"criteria": [{}]},
        "evidence_access": {},
    }
    payload = {
        "criteria": [{"id": "a", "clauses": [{"text": "Unverified claim"}]}],
        "task_anchors": {"intent": "Original"},
        "verdict": "Draft only",
        "need_evidence": ["source:1"],
    }
    source = {"outcome_jury": deepcopy(payload)}
    result, notes = normalize_reply(request, source)
    assert result == payload
    assert source == {"outcome_jury": payload}
    assert notes == [
        {"operation": "unwrap_jury_draft_envelope", "field": "outcome_jury"}
    ]
    for ambiguous in [
        {**source, "criteria": []},
        {**source, "other_answer": {}},
        {"outcome_jury": {"criteria": []}},
    ]:
        result, notes = normalize_reply(request, ambiguous)
        assert result == ambiguous
        assert not notes
    result, notes = normalize_reply({**request, "mode": "outcome_jury_reduce"}, source)
    assert result == source
    assert not notes


def test_jury_assessment_report_is_not_guessed_into_standard_draft():
    request = {"mode": "outcome_jury_map", "response": {"criteria": [{}]}}
    source = {
        "outcome_jury": {
            "verdict": "partially_meets",
            "sub_criteria": [{"id": "a", "score": 0.5}],
            "uncertainties": ["Not verified"],
        }
    }
    result, notes = normalize_reply(request, source)
    assert result == source
    assert "criteria" not in result
    assert not notes
