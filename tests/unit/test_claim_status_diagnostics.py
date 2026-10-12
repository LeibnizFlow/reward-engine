"""WBv5 captured contradiction, replayed with offline Role and synthetic evidence.

A/outcome_estimate.json: caveats_unsupported_ambiguous_check, reply 2.
Also observed in B/outcome_estimate.json: manual_usability_and_organization_check.
Both under runs/full_workbuddy_20261012_v5/tasks/api-usage-explain-cli-l3-001/round_1/.
This checks contract consistency, not correctness of the native assessment.
Tests neither read nor rewrite the frozen runs.
"""

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.harness import defaults, validate_claims
from tests.unit.test_workbuddy_stream_terminal_replay import encode, event
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport


def evidence_case():
    return build_context(
        {"task_id": "offline-capture", "prompt": "Audit the supplied standard"},
        [
            {
                "artifact": {
                    "final_response": "Synthetic structural replay, not native task evidence",
                    "changed_files": "Offline placeholder",
                },
                "source_documents": [{"text": f"Offline source {i}"} for i in range(3)],
            }
        ],
    )


def test_captured_material_claim_status_conflict_is_indexed_and_not_mutated():
    _, evidence = evidence_case()
    raw = deepcopy(CAPTURE)
    assert (
        hashlib.sha256(
            json.dumps(
                raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == CAPTURE_SHA256
    )
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issue = caught.value.details
    assert issue["field"] == "status" and issue["actual"] == "pass"
    assert issue["claim_index"] == 2
    assert issue["depends_on"] == {
        "claims[2].judgment": "unverified",
        "claims[2].verdict_relevance": "material",
    }
    assert raw == CAPTURE


@pytest.mark.parametrize(
    "judgment,status,accepted",
    [
        ("unverified", "pass", False),
        ("contradicted", "pass", False),
        ("supported", "pass", True),
        ("unverified", "unknown", True),
        ("contradicted", "fail", True),
    ],
)
def test_existing_status_claim_acceptance_is_unchanged(judgment, status, accepted):
    _, evidence = evidence_case()
    raw = deepcopy(CAPTURE)
    raw["claims"][2]["judgment"] = judgment
    raw["status"] = status
    before = deepcopy(raw)
    if accepted:
        assert validate_claims(raw, evidence) == before
    else:
        with pytest.raises(ResponseContractError):
            validate_claims(raw, evidence)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["model_conservative", "identical"])
async def test_actual_role_repairs_only_inconsistent_summary_or_rejects_repeat(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first, second = deepcopy(CAPTURE), deepcopy(CAPTURE)
    if repair == "model_conservative":
        # Scripted MODEL output authors a conservative summary, preserving every
        # claim and citation. The host must not select/replace the verdict.
        second.update(
            status="unknown",
            reason="The material source/schema comparison remains unverified.",
            gap="Independent source contents are unavailable in this evidence.",
        )
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [event({"content": json.dumps(reply, ensure_ascii=False)}, "stop")]
            ),
        )
        for reply in (first, second)
    )
    context, evidence = evidence_case()
    result = await defaults(role).run(
        {"test_id": "caveats", "criterion_id": "caveats", "checker": "evidence.review"},
        context,
        evidence,
    )
    assert len(calls) == 2 and not responses
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global" and issue["claim_index"] == 2
    assert issue["depends_on"]["claims[2].judgment"] == "unverified"
    assert "claims[2].judgment" in recovery["details"]["invalid_fields"]["global"]
    instruction = recovery["instruction"]
    assert issue["repair_instruction"] in instruction
    assert "same evidence" in instruction
    assert "minimally correct status/reason/gap" in instruction
    assert "Preserve claims, judgments, relevance and evidence refs" in instruction
    for forbidden in (
        "Do not upgrade an unsupported claim to supported",
        "downgrade material to advisory",
        "invent proof",
    ):
        assert forbidden in instruction
        assert forbidden in requests[0]["instruction"]
    assert (
        "pass requires ALL material claims to be supported"
        in requests[0]["instruction"]
    )
    assert "nonempty evidence_refs" in requests[0]["instruction"]
    assert requests[1]["context"] == requests[0]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 0}
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert attempts[0]["issues"][0] == issue
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(
        json.dumps(json.loads(r["raw_output"]), sort_keys=True) for r in receipts
    ) == sorted(json.dumps(r, sort_keys=True) for r in (first, second))
    assert result["status"] == "unknown"
    if repair == "model_conservative":
        assert not result.get("error")
        assert result["claims"] == first["claims"]
        assert result["reason"] == second["reason"] and result["gap"] == second["gap"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["error"]
        assert attempts[1]["issues"][0] == issue
        assert result["response_validation"]["selected_attempts"] == {}
    assert first == CAPTURE


CAPTURE_SHA256 = "9132ff3a20621f04756c280d2a27c1c8c1afae128818329bfa81f466e3499392"
CAPTURE = json.loads(
    '{"claims": [{"claim": "交付物 final_answer.md 明确列出可疑 doc/feature 路径并给出 flagging 依据：`--json`/`run` 属 stale/legacy（依据 examples/legacy/old_json_flag.txt、docs/migration-v1.md），experimental.py 与 notes/roadmap_draft.txt 为未支持功能（注释声明 not wired/not supported）。", "claim_refs": ["product_1:source:2", "product_1:artifact:final_response"], "claim_location": "final_answer.md 第 6 节 caveats 第 1-2 条", "evidence_refs": ["product_1:source:2"], "observation": "完整正文显示每条 caveat 均标注来源并给出 work-space 路径，且引用了 legacy/roadmap/experimental 文件的自身说明作为 flagging 依据。满足“列出可疑路径+依据”的子要求。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "对未验证/不确定行为使用显式 caveat 标签，且不以支持功能呈现：`scan.timeout` 视为 stale 文档表述、以源码/schema 为准；scan 配置未接入扫描逻辑、explain schema 路径硬编码、warning_count 语义、table 记录数、API 无 format、环境变量范围有限，均归入 caveat 章。", "claim_refs": ["product_1:source:2", "product_1:artifact:final_response"], "claim_location": "final_answer.md 第 6 节 caveats 第 3-9 条", "evidence_refs": ["product_1:source:2"], "observation": "文本明确写“统一标注为 caveat，不作为当前支持功能”，并对 scan 配置未接入等以源码证据对比 doc/defaults 的方式标为不明确/未接入路径。满足显式 caveat 标签子要求。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "区分 stale doc claims 与 source-code facts：文档写 scan.timeout，而 defaults.yaml:4 与 profile.schema.json:21 实际为 scan.timeout_seconds，并声明“以源码/schema 为准”。", "claim_refs": ["product_1:source:2"], "claim_location": "final_answer.md 第 6.3 条", "evidence_refs": ["product_1:source:2"], "observation": "交付文本本身明确做了 doc vs source/schema 的对照并选择源码事实；但未见原始 defaults.yaml/profile.schema.json/docs/config-resolution.md 的独立证据，故该对照断言本身在本次证据下不能被独立复核为真实的仓库事实。", "judgment": "unverified", "verdict_relevance": "material"}, {"claim": "每条 caveat 附工作区路径证据（如 docs/migration-v1.md、notes/roadmap_draft.txt、src/citrinekit/experimental.py、client.py:31-56、cli.py:47-52 等）。", "claim_refs": ["product_1:source:2"], "claim_location": "final_answer.md 第 6 节", "evidence_refs": ["product_1:source:2"], "observation": "caveat 条目均带 work-space 路径引用。引用的仓库文件内容未作为独立证据提供，因此路径存在性/内容的准确性未被本次证据直接证实，但产品确实提供了路径证据。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "对比全文与交付摘要：摘要所述核心 caveats（--json/run stale、experimental/roadmap 未支持、scan.timeout 字段名不一致、scan 配置未接入）与 final_answer.md 第 6 节一致，无相互矛盾。", "claim_refs": ["product_1:source:2", "product_1:artifact:final_response"], "claim_location": "final_response 核心发现 vs final_answer.md 第 6 节", "evidence_refs": ["product_1:source:2", "product_1:artifact:final_response"], "observation": "两处描述一致，未发现摘要与正文相互矛盾。", "judgment": "supported", "verdict_relevance": "material"}], "status": "pass", "reason": "交付的 final_answer.md 完整正文已可查：第 6 节以 caveat 章集中列出可疑路径，并为每条给出 flagging 依据（legacy 文件、docs/migration-v1.md、experimental.py 注释、roadmap_draft.txt、源码行号），对 scan.timeout、scan 配置未接入、explain 硬编码、warning_count、table 计数、API 无 format、环境变量范围等未验证/不明确行为统一用 caveat 标签且明确“不作为当前支持功能”，并在 timeout 字段名上区分了 stale 文档与源码/schema 事实，且均带工作区路径证据。据此满足本 hard 标准 full 锚点。个别引用仓库文件的真实内容未被独立证据复核，属路径存在性与内容准确性之不确定（已以 unverified 记录），但不构成对本标准要求的缺失。", "gap": "", "evidence_refs": ["task", "product_1:artifact:final_response", "product_1:artifact:changed_files", "product_1:source:2"], "measurement_recovery": []}'
)
