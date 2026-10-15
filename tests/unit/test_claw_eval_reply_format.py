from copy import deepcopy

import pytest

from reward_engine.adapters.claw_eval.reply_format import (
    normalize_confirmation,
    normalize_reply,
)


@pytest.mark.parametrize("alias", ["outcome_expectations", "outcome"])
def test_expectations_list_alias_preserves_all_clauses_and_rejects_ambiguous_envelopes(
    alias,
):
    request = {"mode": "outcome_expectations", "response": {"criteria": [{}]}}
    criterion = {
        "id": "a",
        "standard": "Keep exact",
        "kind": "hard",
        "origin": "task",
        "evidence_refs": ["task"],
        "measurement_question": "Check actual output",
        "clauses": [{"text": "Original unverified clause", "source_quotes": []}],
    }
    source = {alias: [criterion]}
    result, notes = normalize_reply(request, source)
    assert result == {"criteria": [criterion]}
    assert source == {alias: [criterion]}
    assert notes == [
        {"operation": "rename_declared_criteria_list", "from": alias, "to": "criteria"}
    ]
    for ambiguous in [
        {**source, "criteria": []},
        {**source, "extra_answer": "other"},
        {alias: {"verdict": "pass"}},
        {alias: [{"standard": "incomplete shape"}]},
    ]:
        result, notes = normalize_reply(request, ambiguous)
        assert result == ambiguous and notes == []
    result, notes = normalize_reply({**request, "mode": "outcome_jury_map"}, source)
    assert result == source and notes == []


@pytest.mark.parametrize("separator", ["", "\n", " \r\n "])
def test_declared_format_echo_preserves_complete_answer(separator):
    import json

    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    answer = {
        "status": "unknown",
        "reason": "Condition unverified; retain {braces}",
        "evidence_refs": ["source_1"],
        "nested": {"type": "json_object"},
    }
    prefix = '{"type":"json_object"}'
    raw = prefix + separator + json.dumps(answer, ensure_ascii=False)
    result, notes = decode_completed_reply(raw, response_format={"type": "json_object"})
    assert result == answer
    assert notes == [
        {"operation": "remove_echoed_response_format", "offset": len(prefix)}
    ]
    assert decode_completed_reply(
        json.dumps(answer), response_format={"type": "json_object"}
    ) == (answer, [])


@pytest.mark.parametrize(
    "raw",
    [
        '{"status":"pass"}{"status":"fail"}',
        '{"type":"json_object","note":"discard me"}{"ok":true}',
        '{"type":"json_object","type":"json_object"}{"ok":true}',
        '{"type":"json_object"}{"status":"fail","status":"pass"}',
        '{"type":"json_object"}{"x":{"a":1,"a":2}}',
        '{"type":"json_object"}{"ok":true}{"ok":false}',
        '{"type":"json_object"}{"ok":true} explanation',
        '{"type":"json_object"}{"ok":true',
        '{"type":"json_object"}{"ok":true,}',
        '{"type":"json_object"}{"type":"json_object"}',
        '{"type":"json_object"}[{"ok":true}]',
        '{"type":"json_object"}{"x":NaN}',
    ],
)
def test_format_echo_never_selects_repairs_or_merges_answers(raw):
    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    with pytest.raises(ValueError):
        decode_completed_reply(raw, response_format={"type": "json_object"})


@pytest.mark.parametrize("configured", [None, {}, {"type": "text"}])
def test_format_echo_requires_matching_explicit_configuration(configured):
    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    with pytest.raises(ValueError):
        decode_completed_reply(
            '{"type":"json_object"}{"ok":true}', response_format=configured
        )


@pytest.mark.parametrize("prefix", ["```json\n", ":\n```json\n", "```\n"])
@pytest.mark.parametrize("suffix", ["", "\n```"])
def test_complete_json_with_presentation_damage_preserves_verdict(prefix, suffix):
    import json

    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    verdict = {
        "status": "fail",
        "reason": "Keep `code`, : and </think> as evidence",
        "evidence_refs": ["source:1"],
        "gap": None,
    }
    raw = prefix + json.dumps(verdict) + suffix
    value, notes = decode_completed_reply(raw)
    assert value == verdict
    assert notes == [{"operation": "unwrap_complete_json_presentation"}]


@pytest.mark.parametrize(
    "raw",
    [
        ':\n```json\n{"status":',
        ':\n```json\n{"status":"pass"',
        '```json\n{"a":1}{"a":2}',
        '```json\n{"a":1}\n```\nUse this instead',
        'Maybe this:\n```json\n{"a":1}',
        '```yaml\n{"a":1}',
    ],
)
def test_presentation_recovery_cannot_complete_or_choose_an_answer(raw):
    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    with pytest.raises(ValueError):
        decode_completed_reply(raw)


def test_empty_reply_has_its_own_failure_not_an_invented_verdict():
    from reward_engine.adapters.claw_eval.reply_format import (
        EmptyModelReply,
        decode_completed_reply,
    )

    with pytest.raises(EmptyModelReply):
        decode_completed_reply("  ")


def test_task_qualified_ref_is_exact_identity_not_fuzzy_citation_repair():
    request = {
        "context": {
            "task": {"task_id": "generic-case"},
            "available_evidence_refs": ["task", "skill:0"],
        }
    }
    original = {
        "claim_checks": [
            {
                "claim": "Literal task:generic-case is in prose",
                "support_refs": [
                    "task:generic-case",
                    "task:another-case",
                    "skill:missing",
                ],
            }
        ],
        "status": "unknown",
        "reason": "Keep this uncertainty",
    }
    result, notes = normalize_reply(request, original)
    assert result["claim_checks"][0]["support_refs"] == [
        "task",
        "task:another-case",
        "skill:missing",
    ]
    assert result["claim_checks"][0]["claim"] == original["claim_checks"][0]["claim"]
    assert result["status"] == "unknown" and result["reason"] == original["reason"]
    assert original["claim_checks"][0]["support_refs"][0] == "task:generic-case"
    assert notes == [
        {"operation": "current_task_ref", "from": "task:generic-case", "to": "task"}
    ]
    request["context"]["available_evidence_refs"].append("task:generic-case")
    assert normalize_reply(request, original) == (
        original,
        [],
    )  # Declared handles have precedence.
    assert normalize_reply({}, original) == (original, [])


def test_declared_response_wrapper_is_lossless_and_not_semantic_repair():
    shape = {
        "decision": "try | keep",
        "proposal": {},
        "source_proposal_ids": [],
        "reason": "text",
    }
    inner = {
        "decision": "keep",
        "proposal": None,
        "source_proposal_ids": [],
        "reason": "Keep my decision.",
    }
    original = {"response": deepcopy(inner)}
    result, notes = normalize_reply({"response": shape}, original)
    assert result == inner and original == {"response": inner}
    assert notes == [{"operation": "unwrap_declared_response_envelope"}]
    for request in (
        {},
        {"response": {"response": {}}},
        {"response": {"decision": "text"}},
    ):
        assert normalize_reply(request, original) == (original, [])
    malformed = {"response": {"decision": "try"}}
    assert normalize_reply({"response": shape}, malformed) == (malformed, [])


@pytest.mark.parametrize("include_recovery", [False, True])
def test_measurement_envelope_preserves_optional_fields_without_loosening_other_contracts(
    include_recovery,
):
    from reward_engine.core.outcome.context import ResponseContractError

    shape = {
        "status": "pass/fail/unknown",
        "reason": "why",
        "gap": "defect",
        "evidence_refs": [],
        "claims": [],
        "measurement_recovery": [],
    }
    payload = {
        "status": "unknown",
        "reason": "Keep the uncertainty",
        "gap": "",
        "evidence_refs": [],
        "claims": [],
        "inquiry_result": {"status": "unknown", "reason": "No inquiry was supplied"},
    }
    if include_recovery:
        payload["measurement_recovery"] = []
    request = {"mode": "outcome_measure", "response": shape}
    wrapped = {"type": "json_object", "response": deepcopy(payload)}
    result, notes = normalize_reply(request, wrapped)
    assert result == payload and wrapped["response"] == payload
    assert notes == [
        {"operation": "unwrap_declared_typed_response_envelope", "type": "json_object"}
    ]
    for changed in [
        {**wrapped, "answer": "conflicting outer answer"},
        {**wrapped, "response": {**payload, "score": 100}},
        {**wrapped, "response": {k: v for k, v in payload.items() if k != "reason"}},
    ]:
        with pytest.raises(ResponseContractError):
            normalize_reply(request, changed)
    with pytest.raises(ResponseContractError):
        normalize_reply({**request, "mode": "different_operation"}, wrapped)


def test_completed_reply_one_semicolon_separator_preserves_values():
    import json

    import pytest

    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    value, notes = decode_completed_reply(
        '{"gap":"text; is evidence";"evidence_refs":["source"]}'
    )
    assert value == {"gap": "text; is evidence", "evidence_refs": ["source"]}
    assert notes[0]["operation"] == "replace_value_separator_semicolon"
    assert decode_completed_reply('{"text":"a;b"}')[1] == []
    for bad in (
        '{"a":1;"b":2;"c":3}',
        '{"gap":"unfinished;',
        '{"a":1;"b":NaN;}',
        '{"a";1}',
    ):
        with pytest.raises(json.JSONDecodeError):
            decode_completed_reply(bad)


def test_completed_reply_only_closes_a_missing_root_brace():
    import json

    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    value = {"rankings": [{"reason": "Keep exact disagreement", "score": 0}]}
    text = json.dumps(value)
    assert decode_completed_reply(text) == (value, [])
    assert decode_completed_reply(text[:-1]) == (
        value,
        [{"operation": "close_missing_root_object"}],
    )
    for broken in (
        '{"x": [1',
        '{"x": "unfinished',
        '{"x":',
        '{"x":1,',
        '{"x": 1 "y":2}',
        "[1,2",
    ):
        with pytest.raises(json.JSONDecodeError):
            decode_completed_reply(broken)


@pytest.mark.parametrize(
    "raw",
    ['{"text":"keep ,} exactly","items":[],}', '{"items":["a",],"status":"unknown"}'],
)
def test_one_trailing_comma_preserves_every_value(raw):
    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    result, notes = decode_completed_reply(raw)
    assert result in (
        {"text": "keep ,} exactly", "items": []},
        {"items": ["a"], "status": "unknown"},
    )
    assert notes[0]["operation"] == "remove_one_trailing_comma"
    assert raw[notes[0]["offset"]] == ","


@pytest.mark.parametrize(
    "raw", ['{"a":1,,}', '{"a":[1,],}', '{"a":"unfinished,}', '{"a":1,} {"a":2}']
)
def test_trailing_comma_recovery_cannot_invent_an_answer(raw):
    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    with pytest.raises(ValueError):
        decode_completed_reply(raw)


def test_one_typographic_closing_quote_preserves_content_not_arbitrary_json_repair():
    import json

    from reward_engine.adapters.claw_eval.reply_format import decode_completed_reply

    text = '{"question": "Why this ‘difference’？”,\n"value": 2}'
    result, notes = decode_completed_reply(text)
    assert result == {"question": "Why this ‘difference’？", "value": 2}
    assert notes[0]["operation"] == "normalize_one_closing_quote"
    for bad in ('{"q":"unfinished\n"value":2}', '{"q":"a”,\n"v":"b”,\n"x":1}'):
        with pytest.raises(json.JSONDecodeError):
            decode_completed_reply(bad)


def test_formatting_keeps_all_votes_and_reasons_and_resolves_only_observed_names():
    request = {
        "mode": "independent_final_wording_confirmation",
        "criteria": [{"criterion_id": "complaint_specific_comprehension"}],
        "experience_pool": {
            "traces": [
                {
                    "source_ref": "trace:trace_1",
                    "evidence_items": [
                        {
                            "content": {
                                "source_documents": [
                                    {
                                        "source": "observed_tool",
                                        "content": {"record": "real result"},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ]
        },
    }
    response = {
        "response": {
            "confirmations": [
                {
                    "criterion_id": "complaint_specific_complaint_specific_comprehension",
                    "decision": "reject",
                    "rationale": "Keep my exact dissent; never change it into approval.",
                    "source_refs": ["observed_tool", "unobserved_tool", "task:task1"],
                }
            ]
        }
    }
    saved = deepcopy(response)
    result, notes = normalize_confirmation(request, response)
    row = result["confirmations"][0]
    assert row["criterion_id"] == "complaint_specific_comprehension"
    assert row["source_refs"] == ["trace:trace_1", "unobserved_tool", "task:task1"]
    assert (
        row["decision"] == "reject"
        and row["rationale"] == saved["response"]["confirmations"][0]["rationale"]
    )
    assert response == saved and len(notes) == 3


def test_other_identifier_errors_are_not_guessed_and_no_missing_vote_is_created():
    request = {
        "mode": "independent_final_wording_confirmation",
        "criteria": [{"criterion_id": "quality"}],
        "experience_pool": {"traces": []},
    }
    response = {
        "confirmations": [
            {"criterion_id": "qulaity", "decision": "approve", "source_refs": []}
        ]
    }
    result, notes = normalize_confirmation(request, response)
    assert result == response and not notes
    assert normalize_confirmation(request, {"confirmations": []}) == (
        {"confirmations": []},
        [],
    )


def test_evidence_ref_list_format_does_not_change_vote_or_invent_evidence():
    response = {
        "criterion_findings": [
            {
                "criterion_id": "quality",
                "preference": "second",
                "evidence_refs": "evidence_1, evidence_2; missing_3",
                "reason": "Original comparison.",
            }
        ]
    }
    saved = deepcopy(response)
    result, notes = normalize_reply({}, response)
    assert result["criterion_findings"][0] == {
        **saved["criterion_findings"][0],
        "evidence_refs": ["evidence_1", "evidence_2", "missing_3"],
    }
    assert response == saved and len(notes) == 1
    assert normalize_reply({}, {"evidence_refs": ""})[0] == {"evidence_refs": []}
    assert normalize_reply({}, result) == (result, [])


def test_only_declared_single_list_envelope_is_recovered_without_changing_content():
    response = [
        {
            "criterion_id": "quality",
            "preference": "second",
            "reason": "Keep exact dissent.",
            "evidence_refs": ["evidence_1", "evidence_2"],
        }
    ]
    saved = deepcopy(response)
    result, notes = normalize_reply(
        {"response": {"criterion_findings": [{}]}}, response
    )
    assert result == {"criterion_findings": saved} and response == saved
    assert notes == [
        {"operation": "wrap_declared_list_envelope", "field": "criterion_findings"}
    ]
    assert normalize_reply({"response": {"hypotheses": [{}]}}, [])[0] == {
        "hypotheses": []
    }
    for shape in (
        {},
        {"reason": "text"},
        {"a": [], "b": []},
        {"a": [], "reason": "text"},
    ):
        with pytest.raises(ValueError, match="ambiguous"):
            normalize_reply({"response": shape}, response)


def test_array_envelope_does_not_supply_missing_fields_or_reinterpret_unknown_references():
    original = [
        {
            "criterion_id": "unknown-criterion",
            "preference": "second",
            "evidence_refs": ["candidate_1.artifact.final_response", "unknown-handle"],
        }
    ]
    response, notes = normalize_reply(
        {"response": {"criterion_findings": [{}]}}, original
    )
    assert response == {"criterion_findings": original}
    assert "reason" not in response["criterion_findings"][0]
    assert notes == [
        {"operation": "wrap_declared_list_envelope", "field": "criterion_findings"}
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "outcome_hypothesis_map",
        "outcome_hypothesis_reduce",
        "outcome_hypothesis_refine",
    ],
)
def test_outcome_catalog_id_stems_are_unique_recorded_and_nonmutating(mode):
    from copy import deepcopy

    from reward_engine.adapters.claw_eval.reply_format import normalize_reply

    request = {
        "mode": mode,
        "jury": {
            "gaps": [
                {"gap_id": "g01_source_support"},
                {"gap_id": "g02_a"},
                {"gap_id": "g02_b"},
            ]
        },
    }
    original = {
        "target_gap_ids": ["g01", "g02", "g0", "unknown", "g01_source_support"],
        "method": "Do not change my meaning",
    }
    saved = deepcopy(original)
    result, notes = normalize_reply(request, original)
    assert original == saved and result["method"] == original["method"]
    assert result["target_gap_ids"] == [
        "g01_source_support",
        "g02",
        "g0",
        "unknown",
        "g01_source_support",
    ]
    assert notes == [
        {
            "operation": "unique_catalog_id_prefix",
            "from": "g01",
            "to": "g01_source_support",
        }
    ]
    assert normalize_reply(request, result)[1] == []
    request = {
        "mode": "outcome_delivery_review",
        "available_tests": [{"test_id": "t02_bounds"}],
    }
    result, notes = normalize_reply(
        request, {"verify": [{"test_id": "t02", "reason": "Verify bounds"}]}
    )
    assert (
        result["verify"] == [{"test_id": "t02_bounds", "reason": "Verify bounds"}]
        and len(notes) == 1
    )
