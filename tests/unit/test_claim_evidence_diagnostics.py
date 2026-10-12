"""Exact captured claim shape, with offline Role transport and synthetic evidence.

This checks contract repair, not the truth of the original Chinese assessment.
The original run is never read by tests or rewritten/regraded.
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

# Captured from runs/full_claw_20261012_v6/tasks/T111zh_expense_email_check/round_1/construction_checks.json: candidate 1, check_evidence_traceability_and_unknowns.
CAPTURE_SHA256 = "a746286fddee7ab07eb98216abbdb4d179ee07a93713930919eba6929c1fe3c7"
CAPTURE = json.loads(
    '{"claims": [{"claim": "报告为每一行核对项都标注了数据来源（收件箱邮件与财务系统交易记录），并在表头声明数据来源为4封邮件（msg_701~704）加财务系统记录。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "一、匹配正常 / 二、金额不符 / 三、状态异常 各表及开头“数据来源”行", "evidence_refs": ["product_1:source:2", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "observation": "artifact 开头声明数据来源为 msg_701~704 与财务系统记录，但表中逐行未引用具体邮件ID或交易描述，仅用发票号 INV-00x 与交易ID TXN-70x 间接定位；这些标识符在来源邮件正文与交易记录描述中确实出现，可追踪。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "张伟 差旅费 3200元，发票号 INV-001，交易 TXN-701，状态 completed，金额一致。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "一、匹配正常 表格第1行", "evidence_refs": ["product_1:source:4", "product_1:source:3"], "observation": "source:4 邮件正文明确 3200元、INV-001；source:3 交易 TXN-701 amount 3200.0、type 差旅费报销、description 含 张伟、INV-001、status completed。金额与状态与报告一致，未发现杜撰。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "陈静 客户招待餐费 420元，发票号 INV-004，交易 TXN-704，状态 completed，金额一致。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "一、匹配正常 表格第2行", "evidence_refs": ["product_1:source:7"], "observation": "source:7 邮件正文明确 420元、INV-004、客户招待餐费。source:3 交易预览被截断，未显示 TXN-704 的完整字段，但报告表格未出现与已知证据冲突的数字；其 TXN-704 金额与状态在本证据集中未完全可见。", "judgment": "unverified", "verdict_relevance": "advisory", "relevance_reason": "该行为与标准相关的可追踪性问题：交易记录仅部分可见，但报告没有为该行编造与可见证据矛盾的数字/状态，尚不构成伪造证据；缺少的是 TXN-704 的完整核对支持，属未验证而非已证伪。"}, {"claim": "李华 办公用品 850元，发票号 INV-002，交易 TXN-702 系统金额 680元，差异 170元，状态 completed。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "二、金额不符 表格", "evidence_refs": ["product_1:source:5", "product_1:source:3"], "observation": "source:5 邮件正文明确 850元、INV-002；source:3 交易 TXN-702 amount 680.0、type 办公用品报销、description 含 李华、INV-002；差异 850-680=170，报告数字与来源一致，未伪造。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "王明 外部培训费 5000元，发票号 INV-003，交易 TXN-703，状态 pending（未完成）。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "三、状态异常 表格", "evidence_refs": ["product_1:source:6", "product_1:source:3"], "observation": "source:6 邮件正文明确 5000元、INV-003、外部培训费。source:3 交易预览在 TXN-702 处截断，未显示 TXN-703 的状态，因此报告中 TXN-703 = pending 这一关键状态值在本证据集内未获直接支持；但未见与已知来源的矛盾。", "judgment": "unverified", "verdict_relevance": "material"}, {"claim": "汇总称4封邮件全部在财务系统找到对应交易，正常2笔、金额不符1笔、状态异常1笔。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "汇总", "evidence_refs": ["product_1:source:2", "product_1:source:3"], "observation": "source:2 确认 4 封邮件存在；source:3 交易列表预览只显示到 TXN-702 附近，无法确认 TXN-703、TXN-704 是否全部存在且字段完整，故“全部找到”与分类计数属部分未验证。", "judgment": "unverified", "verdict_relevance": "material"}, {"claim": "备注称 TXN-705（水电费）、TXN-706（工资）为非报销类交易，未纳入核对。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "注", "evidence_refs": ["product_1:source:3"], "observation": "source:3 交易列表预览在 TXN-702 处截断，未显示 TXN-705、TXN-706，故该备注无法由已供证据核实；也未显示与已见记录矛盾。", "judgment": "unverified", "verdict_relevance": "advisory", "relevance_reason": "该备注不改变核心四项核对的引用与未杜撰判定，属额外声明的可追踪性未验证。"}, {"claim": "报告未对无法取得或未核实的数据标注“unknown”。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "全文", "evidence_refs": [], "observation": "报告中 TXN-703 状态、TXN-704 完整记录、汇总计数在证据集中未全部可见，报告均以确定值/确定结论呈现，没有将这些未验证项标记为 unknown。", "judgment": "contradicted", "verdict_relevance": "material"}], "status": "fail", "reason": "报告对主要项目使用了与来源邮件/交易记录一致的发票号与交易ID，且已核对到的金额（3200、680/850、420）未出现与来源矛盾的伪造数字，满足“不伪造已知数字”的多数要求。但标准还要求“每项引用来源并保留未知”，报告把在本证据集中不可见的交易状态与汇总结论（尤其 TXN-703 的 pending 状态、TXN-704 的完整记录、全部交易均已找到及分类计数）直接写成确定事实，且未将任何未核实值标注为 unknown，因此未满足该硬标准。", "gap": "缺少对 TXN-703 状态、TXN-704 完整记录及汇总“全部找到/分类计数”的逐项来源支撑；这些无法从可见交易列表预览确认的字段未标注 unknown。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "measurement_recovery": []}'
)


def evidence_case():
    context, evidence = build_context(
        {
            "task_id": "offline-capture",
            "prompt": "Inspect traceability of the delivered report",
        },
        [
            {
                "artifact": {
                    "final_response": "Offline structural replay, not native task evidence"
                },
                "source_documents": [
                    {"text": f"Offline source handle {i}"} for i in range(8)
                ],
            }
        ],
    )
    return context, evidence


def test_captured_eighth_claim_error_identifies_field_without_mutating_reply():
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
    assert len(raw["claims"]) == 8
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issue = caught.value.details
    assert issue["field"] == "claims[7].evidence_refs"
    assert issue["claim_index"] == 7
    assert issue["judgment"] == "contradicted" and issue["actual"] == []
    assert "supplied evidence handles" in issue["reason"]
    assert "flip the judgment" in issue["reason"]
    assert raw == CAPTURE


@pytest.mark.parametrize("judgment", ["supported", "contradicted", "unverified"])
def test_evidence_requirement_and_acceptance_remain_unchanged(judgment):
    _, evidence = evidence_case()
    raw = deepcopy(CAPTURE)
    raw["claims"][7]["judgment"] = judgment
    before = deepcopy(raw)
    if judgment == "unverified":
        assert validate_claims(raw, evidence) == before
    else:
        with pytest.raises(ResponseContractError, match=r"claims\[7\].evidence_refs"):
            validate_claims(raw, evidence)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["model_citation", "identical", "invented_citation"])
async def test_actual_role_receives_indexed_same_evidence_repair_and_keeps_raw_attempts(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first, second = deepcopy(CAPTURE), deepcopy(CAPTURE)
    if repair == "model_citation":
        # Only the scripted model response supplies the previously absent refs.
        second["claims"][7]["evidence_refs"] = ["product_1:artifact:final_response"]
    elif repair == "invented_citation":
        second["claims"][7]["evidence_refs"] = ["invented:approval"]
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
        {
            "test_id": "traceability",
            "criterion_id": "traceability",
            "checker": "evidence.review",
        },
        context,
        evidence,
    )
    assert len(calls) == 2
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global"
    assert issue["field"] == "claims[7].evidence_refs"
    assert issue["claim_index"] == 7 and issue["judgment"] == "contradicted"
    assert issue["actual"] == []
    assert "claims[7].evidence_refs" in recovery["details"]["invalid_fields"]["global"]
    assert "same evidence" in recovery["instruction"]
    assert requests[1]["context"] == requests[0]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    assert "nonempty evidence_refs" in requests[0]["instruction"]
    assert (
        "empty allowed only for unverified"
        in requests[0]["response"]["claims"][0]["evidence_refs"][0]
    )
    assert result["execution_usage"] == {"model_requests": 2, "tool_calls": 0}
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[1]["response"] == second
    assert attempts[0]["issues"][0] == issue
    # The actual Role persists both provider replies even when validation fails.
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(
        json.dumps(json.loads(r["raw_output"]), sort_keys=True) for r in receipts
    ) == sorted(json.dumps(r, sort_keys=True) for r in (first, second))
    if repair == "model_citation":
        assert result["status"] == "fail" and not result.get("error")
        assert result["claims"] == second["claims"]
        assert result["reason"] == first["reason"] and result["gap"] == first["gap"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["status"] == "unknown" and result["error"]
        assert attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}
    assert CAPTURE["claims"][7]["evidence_refs"] == []
    assert first == CAPTURE
