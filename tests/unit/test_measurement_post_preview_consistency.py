"""Prospective request clarity, exact WB v9 answers and scripted offline controls.

Captured answers: WB v9 round_2/construction_checks.json batch index 3, check
language_format_usability_check. Ledger SHA256:
92884b6c1d1dd69f82bfc5d4273ded0a7973f347bb2faef3941daf1ce3764cf9.
Actual Role uses mocked HTTP and synthetic evidence, never a provider or native
quality measurement. Changed replies below are explicitly scripted simulations.
"""

import hashlib
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import build_context
from tests.unit.test_claim_contract_repair import replay
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

# Exact parsed native answers, portable without a dependency on ignored runs.
REPLIES = json.loads(r"""
[
  {
    "claims": [
      {
        "claim": "The delivered final_answer.md is written in Chinese Markdown, preserving English CLI/API/code/path terms verbatim.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md; throughout, e.g. section headings '## 1. 功能与入口总览' and code blocks `citrinectl scan ...`, `ScanClient.from_profile`, `config/defaults.yaml`",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The observed file content is a Markdown document whose prose is Chinese while identifiers such as `citrinectl`, `--format`, `CITRINE_PROFILE_FILE`, `ScanClient`, and paths like `src/citrinekit/client.py` are retained verbatim. Headings and tables use Markdown.",
        "judgment": "supported"
      },
      {
        "claim": "The document is organized around the requested axes: functionality/entry points, command or API examples, key parameters, evidence files, and caveats.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md sections '## 1. 功能与入口总览', '## 2. CLI 用法'/'3. Python API 用法' with example prompts, parameter tables, '### 证据文件索引', and '## 6. unsupported / ambiguous caveats'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document contains an entry-point table, CLI and Python API usage sections with code blocks and parameter tables, a caveat section for unsupported/ambiguous items, and an explicit '证据文件索引'. This reasonably approximates the requested organizational axes.",
        "judgment": "supported"
      },
      {
        "claim": "The document provides an end-to-end trace sufficient to start using the tool: example config profile → CLI invocation and Python API invocation → observed output fields.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md subsections '2.4 典型调用示例', '2.5 实测输出（只读运行确认）', '3.3 调用示例', '4.6 示例配置落到实际行为', and '5.3 输出字段/格式确认'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document shows a `local-dev` profile with `output.format: jsonl` and `include_warnings: true`, gives CLI commands such as `citrinectl scan tests/fixtures/minimal_job.yaml --profile local-dev --include-warnings`, gives Python API code using `ScanClient.from_profile(...).scan_job(...)`, and displays observed summary/jsonl/table output examples and JSONL record fields. This is an end-to-end trace from config to invocation to output.",
        "judgment": "supported"
      },
      {
        "claim": "The document reports a CLI invocation `citrinectl scan tests/fixtures/minimal_job.yaml` as producing summary output `job=minimal_job; warnings=0; records=2`, and `--format jsonl` as producing specific JSONL records.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '2.5 实测输出（只读运行确认）'",
        "evidence_refs": [
          "product_1:source:2"
        ],
        "observation": "The observed artifact text contains these outputs as assertions. The supplied evidence shows only the written artifact, not an independent execution trace proving the exact output. However, the question concerns language/format/usability; these outputs demonstrate the format and are plausible from the artifact. No contradicting evidence is supplied.",
        "judgment": "unverified"
      },
      {
        "claim": "The document includes the caveat that `scan.retry_count` and `scan.timeout_seconds` are merged into settings but not read by `ScanClient.scan_job()` or `render_result()`, so they do not currently affect scanning behavior.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '4.7 重要 caveat：profile 字段 vs 实际生效字段' and '## 6. unsupported / ambiguous caveats'",
        "evidence_refs": [
          "product_1:source:2"
        ],
        "observation": "The artifact explicitly states this caveat, which is relevant to developer usability because it prevents users from assuming the config fields have runtime effect.",
        "judgment": "supported"
      },
      {
        "claim": "The document is sufficient for a developer to begin using the tool from the doc alone, at least for one CLI call and one Python API call.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '2.4 典型调用示例', '3.3 调用示例', and '## 7. 快速上手清单'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document gives installation/run hints, concrete CLI commands, concrete Python import and client usage, profile config context, output field descriptions, and evidence paths. This is sufficient for a developer to attempt both a CLI and Python API call.",
        "judgment": "supported"
      }
    ],
    "status": "pass",
    "reason": "The observed final_answer.md is Chinese Markdown with English technical terms preserved verbatim. It is organized around the requested axes: a functionality/entry-point overview, CLI and Python API sections with examples and key parameter tables, an evidence file index, and a caveat section. It also contains an end-to-end trace from example profile config to CLI/API invocation to observed output fields and field descriptions. The only unverified material claim is that the exact '实测输出' values were independently reproduced; the artifact asserts them and the supplied evidence does not independently confirm execution, but this does not prevent the document from meeting the language/format/usability standard, which asks whether the doc has sufficient end-to-end depth to start using the tool, not whether every observed-output value is externally proven.",
    "gap": "",
    "evidence_refs": [
      "product_1:source:2",
      "tool:language_format_usability_check:0"
    ]
  },
  {
    "field_corrections": {
      "status": "unknown",
      "reason": "The observed final_answer.md is Chinese Markdown with English CLI/API/code/path terms preserved verbatim, is organized around the requested axes (功能/入口, 命令或 API 示例, 关键参数, 证据文件, caveat), and contains an end-to-end trace from example profile config to CLI/API invocation to output fields. However, one material claim—that the '实测输出' values were actually reproduced by execution—remains unverified: the supplied evidence shows only the written artifact, not an independent execution trace. Because a material claim is unverified and no tool budget remains to perform the measurement, the standard cannot be confirmed as fully satisfied.",
      "gap": "No independent execution evidence confirms the exact '实测输出' values (e.g. `job=minimal_job; warnings=0; records=2` and the JSONL records) reported in final_answer.md; the artifact asserts them but the supplied evidence does not verify the run. This leaves a material claim unverified, so the verdict is unknown rather than pass."
    }
  },
  {
    "claims": [
      {
        "claim": "The delivered final_answer.md is written in Chinese Markdown, preserving English CLI/API/code/path terms verbatim.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md; throughout, e.g. headings '## 1. 功能与入口总览', code blocks `citrinectl scan ...`, `ScanClient.from_profile`, `config/defaults.yaml`",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The observed file content is a Markdown document whose prose is Chinese while identifiers such as `citrinectl`, `--format`, `CITRINE_PROFILE_FILE`, `ScanClient`, and paths like `src/citrinekit/client.py` are retained verbatim. Headings and tables use Markdown.",
        "judgment": "supported"
      },
      {
        "claim": "The document is organized around the requested axes: functionality/entry points, command or API examples, key parameters, evidence files, and caveats.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md sections '## 1. 功能与入口总览', '## 2. CLI 用法'/'3. Python API 用法' with example prompts, parameter tables, '### 证据文件索引', and '## 6. unsupported / ambiguous caveats'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document contains an entry-point table, CLI and Python API usage sections with code blocks and parameter tables, a caveat section for unsupported/ambiguous items, and an explicit '证据文件索引'. This reasonably approximates the requested organizational axes.",
        "judgment": "supported"
      },
      {
        "claim": "The document provides an end-to-end trace sufficient to start using the tool: example config profile → CLI invocation and Python API invocation → observed output fields.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md subsections '2.4 典型调用示例', '2.5 实测输出（只读运行确认）', '3.3 调用示例', '4.6 示例配置落到实际行为', and '5.3 输出字段/格式确认'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document shows a `local-dev` profile with `output.format: jsonl` and `include_warnings: true`, gives CLI commands such as `citrinectl scan tests/fixtures/minimal_job.yaml --profile local-dev --include-warnings`, gives Python API code using `ScanClient.from_profile(...).scan_job(...)`, and displays observed summary/jsonl/table output examples and JSONL record fields. This is an end-to-end trace from config to invocation to output.",
        "judgment": "supported"
      },
      {
        "claim": "The document reports a CLI invocation `citrinectl scan tests/fixtures/minimal_job.yaml` as producing summary output `job=minimal_job; warnings=0; records=2`, and `--format jsonl` as producing specific JSONL records.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '2.5 实测输出（只读运行确认）'",
        "evidence_refs": [
          "product_1:source:2"
        ],
        "observation": "The observed artifact text contains these outputs as assertions. The supplied evidence shows only the written artifact, not an independent execution trace proving the exact output. However, the question concerns language/format/usability; these outputs demonstrate the format and are plausible from the artifact. No contradicting evidence is supplied.",
        "judgment": "unverified"
      },
      {
        "claim": "The document includes the caveat that `scan.retry_count` and `scan.timeout_seconds` are merged into settings but not read by `ScanClient.scan_job()` or `render_result()`, so they do not currently affect scanning behavior.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '4.7 重要 caveat：profile 字段 vs 实际生效字段' and '## 6. unsupported / ambiguous caveats'",
        "evidence_refs": [
          "product_1:source:2"
        ],
        "observation": "The artifact explicitly states this caveat, which is relevant to developer usability because it prevents users from assuming the config fields have runtime effect.",
        "judgment": "supported"
      },
      {
        "claim": "The document is sufficient for a developer to begin using the tool from the doc alone, at least for one CLI call and one Python API call.",
        "claim_refs": [
          "product_1:source:2"
        ],
        "claim_location": "final_answer.md '2.4 典型调用示例', '3.3 调用示例', and '## 7. 快速上手清单'",
        "evidence_refs": [
          "product_1:source:2",
          "tool:language_format_usability_check:0"
        ],
        "observation": "The document gives installation/run hints, concrete CLI commands, concrete Python import and client usage, profile config context, output field descriptions, and evidence paths. This is sufficient for a developer to attempt both a CLI and Python API call.",
        "judgment": "supported"
      }
    ],
    "status": "pass",
    "reason": "The observed final_answer.md is Chinese Markdown with English CLI/API/code/path terms preserved verbatim, is organized around the requested axes (功能/入口, 命令或 API 示例, 关键参数, 证据文件, caveat), and contains an end-to-end trace from example profile config to CLI/API invocation to output fields. The one unverified claim—that the '实测输出' values were actually reproduced by execution—is not material to this language/format/usability standard: the standard asks whether the document is usable and contains an end-to-end trace, not whether every reported runtime value was independently reproduced. The document's structure, language, and trace satisfy the soft criterion.",
    "gap": "",
    "evidence_refs": [
      "product_1:source:2",
      "tool:language_format_usability_check:0"
    ]
  }
]
""")

REPLY_HASHES = [
    "d6b79d66eab92915d6bf80ef5c21c76c95d786e110384532cdabb910ba453c42",
    "b6ea728dc5001d7530eccfe868d870cf9d2f092d3ac02a8534967d8200771a2b",
    "8730428b8b148605a2697d8cd82799a1251e653dfab41e15b357dc7a2cdf8b81",
]
REMINDER = "After assessing the newly supplied evidence"


def evidence_case(*, inquiry=False):
    context, evidence = build_context(
        {"task_id": "offline-wbv9-status", "prompt": "Contract replay only"},
        [
            {
                "artifact": {"final_response": "Synthetic contract replay"},
                "source_documents": [
                    {
                        "text": "Synthetic preview data " * 180
                        if i == 2
                        else f"Offline source {i}"
                    }
                    for i in range(3)
                ],
            }
        ],
    )
    evidence["tool:language_format_usability_check:0"] = {
        "text": "Synthetic recorded observation; no tool executes"
    }
    if inquiry:
        context["measurement_inquiries"] = {
            "contract": {"questions": ["Separate local question"]}
        }
    return context, evidence


def corrected():
    return {**deepcopy(REPLIES[0]), **deepcopy(REPLIES[1]["field_corrections"])}


def assert_reminder(requests):
    assert all(REMINDER not in r.get("evidence_followup", "") for r in requests[:-1])
    final = requests[-1]
    text = final["evidence_followup"]
    for fragment in (
        REMINDER,
        "Omitted verdict_relevance means material",
        "calling a claim nonmaterial only in prose does not change",
        "model-authored relevance_reason tied to THIS criterion",
        "Do not change relevance just to obtain pass",
        "New evidence may legitimately change judgment or relevance when supported",
        "prior provisional verdict is not fixed",
        "global verdict and any inquiry_result independent",
    ):
        assert fragment in text
    assert set(final["requested_evidence"]) == {"product_1:source:2"}
    assert "response_recovery" not in final and "field_correction_reply" not in final
    assert final["measurement_budget"]["remaining_tool_calls"] == 4
    return final


def test_exact_native_answer_hashes():
    assert [
        hashlib.sha256(
            json.dumps(
                r, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        for r in REPLIES
    ] == REPLY_HASHES


@pytest.mark.asyncio
@pytest.mark.parametrize("conservative", [False, True])
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_exact_native_failure_and_conservative_control(
    transport, conservative, syntax_retry
):
    replies = deepcopy(REPLIES)
    if conservative:
        replies[-1] = (
            corrected()
        )  # Scripted reuse of model-authored unknown, not host choice.
    if syntax_retry:
        replies.insert(1, '{"field_corrections":')
    result, requests = await replay(transport, replies, *evidence_case())
    final = assert_reminder(requests)
    assert final["provisional_check"] == corrected()
    for request in requests[1:-1]:
        assert request["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert attempts[-2]["errors"] == {}
    assert attempts[-2]["composed_response"] == corrected()
    assert attempts[-2]["response"] == REPLIES[1]
    assert attempts[-2]["pending_evidence"] == {"global": ["product_1:source:2"]}
    assert attempts[-1]["response"] == replies[-1]
    assert result["status"] == "unknown"
    if conservative:
        assert not result.get("error") and attempts[-1]["errors"] == {}
    else:
        assert result["error"]
        assert "claims[3].judgment" in attempts[-1]["errors"]["global"]
    # replay verifies exact physical call count and raw receipts: no fourth
    # contract attempt (syntax control adds only its existing syntax allowance).


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["advisory_without_reason", "prose_only", "material_contradicted"]
)
async def test_reminder_does_not_accept_inconsistent_fresh_assessments(
    transport, fault
):
    replies = deepcopy(REPLIES)
    claim = replies[-1]["claims"][3]
    if fault == "advisory_without_reason":
        claim["verdict_relevance"] = "advisory"
    elif fault == "prose_only":
        claim["relevance_reason"] = (
            "Scripted prose says nonmaterial but field is omitted."
        )
    else:
        claim["judgment"] = "contradicted"
    result, requests = await replay(transport, replies, *evidence_case())
    assert_reminder(requests)
    assert result["error"] and result["response_validation"]["attempts"][-1]["errors"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fresh_change", ["advisory", "supported"])
@pytest.mark.parametrize("local_status", [None, "unknown", "fail"])
async def test_scripted_new_evidence_allows_fresh_assessment_and_independent_inquiry(
    transport, fresh_change, local_status
):
    first = corrected()
    final = deepcopy(REPLIES[2])
    claim = final["claims"][3]
    if fresh_change == "advisory":
        claim["verdict_relevance"] = "advisory"
        claim["relevance_reason"] = (
            "Scripted simulation: the newly read full document establishes the required "
            "format and trace; independent runtime reproduction is outside this criterion."
        )
    else:
        claim["judgment"] = "supported"
        claim["observation"] = (
            "Scripted simulation: newly expanded evidence supports this claim."
        )
        final["reason"] = "Scripted simulation: all material claims are now supported."
    if local_status:
        local = {
            "status": local_status,
            "reason": "Separate scripted inquiry assessment",
            "gap": "Separate local issue",
            "evidence_refs": ["product_1:artifact:final_response"],
        }
        first["inquiry_result"] = deepcopy(local)
        final["inquiry_result"] = deepcopy(local)
    result, requests = await replay(
        transport, [first, final], *evidence_case(inquiry=bool(local_status))
    )
    assert_reminder(requests)
    assert requests[-1]["provisional_check"]["status"] == "unknown"
    assert result["status"] == "pass" and not result.get("error")
    if local_status:
        assert result["inquiry_result"] == local
    assert result["claims"][3] == claim


@pytest.mark.asyncio
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_same_evidence_relabel_still_rejected_without_fresh_reminder(
    transport, syntax_retry
):
    repair = deepcopy(REPLIES[2])
    repair["claims"][3].update(
        verdict_relevance="advisory", relevance_reason="Force scripted pass"
    )
    replies = [REPLIES[0]] + (['{"claims":'] if syntax_retry else []) + [repair]
    result, requests = await replay(transport, replies, *evidence_case())
    assert all(REMINDER not in r.get("evidence_followup", "") for r in requests)
    assert result["error"]
    assert (
        "preserve"
        in result["response_validation"]["attempts"][-1]["errors"]["global"].lower()
    )
    assert all(
        r["measurement_budget"]["remaining_tool_calls"] == 0 for r in requests[1:]
    )
