"""Offline contract repair of retained Claw omissions and WB CS-1 counterexamples.

Captured Claw: full_claw_20261012_v6/tasks/T111zh_expense_email_check/round_1/
actors/B/checks/delivery_1.json, check_coverage_of_common_failure_modes, reply 1;
also retained in actors/B/RESULT.json review_session.versions[0].measurement.
Evidence is synthetic; no benchmark correctness or live repair is established.
"""

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.context import ResponseContractError, build_context
from reward_engine.core.outcome.harness import defaults, validate_claims
from tests.unit.test_claim_status_diagnostics import CAPTURE as WB_CAPTURE
from tests.unit.test_claim_status_diagnostics import evidence_case as wb_evidence
from tests.unit.test_workbuddy_stream_terminal_replay import encode, event
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

CLAW_SHA256 = "440150b2f2ab463dc05c09706810da89a818edb1f9673dabed7774e2df25722f"
CLAW_CAPTURE = json.loads(
    '{"claims": [{"claim": "报告完成了任务要求的四项步骤：提取邮件金额与发票号、查找财务交易、交叉核对、分类输出。", "claim_refs": ["product_1:artifact:final_response", "task"], "observation": "报告含邮件提取表（4封报销申请）、逐笔财务交易、金额/状态核对、以及匹配正常/金额不符/状态异常三类汇总，与任务要求的输出结构一致。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "msg_701/INV-001 金额3200元，TXN-701金额3200，状态completed，匹配正常。", "claim_refs": ["product_1:source:4", "product_1:source:3", "product_1:artifact:final_response"], "observation": "邮件正文明确3200元/INV-001；TXN-701金额3200.0、状态completed、描述含INV-001，支持报告结论。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "msg_702/INV-002 邮件金额850元，TXN-702金额680元，状态completed，属金额不符（差170元）。", "claim_refs": ["product_1:source:5", "product_1:source:9", "product_1:artifact:final_response"], "observation": "邮件850元/INV-002；TXN-702金额680.0、状态completed；差额170元计算正确。报告将其列入金额不符，符合任务分类要求。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "msg_703/INV-003 金额5000元，TXN-703金额5000，状态pending，属状态异常。", "claim_refs": ["product_1:source:6", "product_1:source:10", "product_1:artifact:final_response"], "observation": "邮件5000元/INV-003；TXN-703金额5000.0、状态pending；报告列入状态异常并说明未完成结算，支持该结论。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "msg_704/INV-004 金额420元，TXN-704金额420，状态completed，匹配正常。", "claim_refs": ["product_1:source:7", "product_1:artifact:final_response"], "observation": "邮件正文明确420元/INV-004；报告称TXN-704金额420、completed。但已提供的财务交易来源预览未完整展示TXN-704记录，TXN-704的金额与状态在本批证据中未被直接观测到。", "judgment": "unverified", "verdict_relevance": "material"}, {"claim": "msg_705为行政通知（团建活动），与报销无关，已排除。", "claim_refs": ["product_1:source:8", "product_1:artifact:final_response"], "observation": "msg_705发件人为hr@company.com，主题为行政通知团建活动，非报销申请，排除正确。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "报告显式处理无重复发票号：INV-001~004各出现一次。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "observation": "四封报销邮件分别含INV-001至INV-004，无重复；报告设有“其他发现-无重复发票号”一节，主动检查并说明。", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "重复发票号是本测试项要求覆盖的常见失败模式之一；无重复时显式声明已满足“检测并说明”要求，不影响其他分类结论。"}, {"claim": "报告区分了 posted-but-not-claimed 与 claimed-but-not-posted，并说明未发现claimed-but-not-posted。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:3"], "observation": "报告列出TXN-705/706为非报销类财务记录、无对应报销邮件，并明确“未发现报销邮件在财务系统查无记录”。这已满足测试项对两类时间差区分的soft要求。", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "TXN-705（水电费12000）、TXN-706（员工工资350000）为非报销类财务记录，收件箱无对应报销邮件。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:3"], "observation": "source:3预览仅部分展示交易列表，未完整呈现TXN-705/706字段；“非报销类/无对应邮件”的断言在现有证据中未被完全直接观测。", "judgment": "unverified", "verdict_relevance": "advisory", "relevance_reason": "该断言属于报告“其他发现”的旁支观察；即使未验证，也不影响任务核心四类报销核对结论及本测试项主要覆盖要求。"}, {"claim": "报告在状态语义层面将财务编码pending/待定处理翻译为“未完成结算/状态异常”，未将邮件中的“已获批准/请处理”等措辞误读为已成功。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:6", "product_1:source:10"], "observation": "msg_703正文称“已获部门经理批准，请处理”，但TXN-703状态为pending；报告将其列入状态异常并说明“暂不能认定报销成功”，正确区分了邮件措辞与财务状态语义。", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "反向的“批准”措辞在本场景出现在INV-003；报告将其翻译为待定/未完成结算，满足状态语义覆盖要求，不影响其他分类。"}, {"claim": "报告对金额/格式化差异是否有统一规则处理——本批数据未出现1250 vs 1250.00或CNY/RMB等格式差异。", "claim_refs": ["product_1:artifact:final_response"], "observation": "当前证据中邮件金额均为整数元，财务金额为浮点表示，未观测到需额外规则化解的格式化冲突；报告未显式声明通用格式规则，但本数据未暴露该缺口。", "judgment": "unverified", "verdict_relevance": "advisory", "relevance_reason": "格式化差异规则属条件性要求；现有证据未显示该场景确实存在格式化差异，不构成对该测试项结论的实质性影响。"}, {"claim": "报告明确声明未做任何系统写入操作，后续建议未被表述为已执行动作。", "claim_refs": ["product_1:artifact:final_response"], "observation": "报告结尾注明“未做任何系统写入操作”，并说明后续建议非已执行动作，符合流程透明性要求。", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "该声明影响操作安全与可审计性，但不是本测试项核心覆盖要求。"}], "status": "pass", "reason": "在测试项标准下，产品报告已显式覆盖本场景适用的多数常见失败模式：主动检查并声明无重复发票号；以逐笔交易支持金额差异INV-002（850 vs 680，差170元）并归入金额不符；将pending状态翻译为状态异常/未完成结算，未将邮件中“已批准/请处理”误读为成功；明确区分posted-but-not-claimed（TXN-705/706）与claimed-but-not-posted（未发现），且在适用场景下对状态语义差异有显式处理。格式化差异在当前数据中未实际出现，报告未展示通用规则属条件性软缺口；TXN-704与TXN-705/706的部分字段在现有预览中未完全观测，属未验证而非已证伪。核心覆盖要求成立，按soft标准不构成fail。", "gap": "", "evidence_refs": ["task", "product_1:artifact:final_response", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7", "product_1:source:8", "product_1:source:9", "product_1:source:10"], "inquiry_result": {"status": "pass", "reason": "测试项关注报告是否显式处理重复、格式差异、状态语义及posted/claimed时间差。报告显式声明无重复发票号、列出posted-but-not-claimed（TXN-705/706）并说明未发现claimed-but-not-posted，且将pending翻译为状态异常而非成功。格式化差异在本数据中未观测到实际冲突，属条件性未展开；无实际存在的格式差异未处理，故不构成实质性Task缺陷。", "gap": "", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"]}}'
)


def claw_evidence():
    return build_context(
        {"task_id": "offline-claw", "prompt": "Inspect the recorded report"},
        [
            {
                "artifact": {"final_response": "Synthetic structural replay"},
                "source_documents": [
                    {"text": f"Offline source {i}"} for i in range(11)
                ],
            }
        ],
    )


async def replay(transport, replies, context, evidence):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    before = deepcopy(replies)
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [
                    event(
                        {
                            "content": r
                            if isinstance(r, str)
                            else json.dumps(r, ensure_ascii=False)
                        },
                        "stop",
                    )
                ]
            ),
        )
        for r in replies
    )
    result = await defaults(role).run(
        {
            "test_id": "contract",
            "criterion_id": "contract",
            "checker": "evidence.review",
        },
        context,
        evidence,
    )
    assert replies == before
    assert len(calls) == len(replies) and not responses
    requests = [json.loads(c["messages"][1]["content"]) for c in calls]
    assert result["execution_usage"] == {
        "model_requests": len(replies),
        "tool_calls": 0,
    }
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    failures = [json.loads(p.read_text()) for p in role.output.glob("*/failure.json")]
    assert len(receipts) + len(failures) == len(replies)
    assert all(f["error_type"] == "JSONDecodeError" for f in failures)
    assert sorted(
        [r["raw_output"] for r in receipts] + [f["partial_output"] for f in failures]
    ) == sorted(
        r if isinstance(r, str) else json.dumps(r, ensure_ascii=False) for r in replies
    )
    return result, requests


def test_exact_twelve_missing_refs_and_pass_dependency_are_all_reported():
    _, evidence = claw_evidence()
    raw = deepcopy(CLAW_CAPTURE)
    assert (
        hashlib.sha256(
            json.dumps(
                raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == CLAW_SHA256
    )
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issues = caught.value.details["issues"]
    missing = [i for i in issues if i.get("claim_field") == "evidence_refs"]
    assert len(issues) == 13 and len(missing) == 12
    assert {i["field"] for i in missing} == {
        f"claims[{n}].evidence_refs" for n in range(12)
    }
    assert all(i["missing"] and i["actual"] is None for i in missing)
    assert next(i for i in issues if i["field"] == "status")["depends_on"] == {
        "claims[4].judgment": "unverified",
        "claims[4].verdict_relevance": "material",
    }
    assert raw == CLAW_CAPTURE


@pytest.mark.parametrize("bad_refs", [None, "product_1:source:2", {}, ["absent"]])
def test_nonlist_or_unknown_refs_do_not_hide_other_claims_or_status(bad_refs):
    _, evidence = claw_evidence()
    raw = deepcopy(CLAW_CAPTURE)
    raw["claims"][0]["evidence_refs"] = bad_refs
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(raw, evidence)
    issues = caught.value.details["issues"]
    assert len(issues) == 13
    assert issues[0]["field"] == "claims[0].evidence_refs"
    assert issues[0]["actual"] == bad_refs and not issues[0]["missing"]
    assert any(i["field"] == "status" for i in issues)
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["all_fields", "refs_only", "identical", "upgrade"])
async def test_all_missing_fields_and_status_share_one_actual_role_repair(
    transport, repair
):
    first, second = deepcopy(CLAW_CAPTURE), deepcopy(CLAW_CAPTURE)
    if repair != "identical":
        # Authored by the scripted model, not inferred/filled by the harness.
        for claim in second["claims"]:
            claim["evidence_refs"] = (
                []
                if claim["judgment"] == "unverified"
                else ["product_1:artifact:final_response"]
            )
    if repair == "all_fields":
        second.update(
            status="unknown",
            reason="The material TXN-704 assessment remains unverified.",
            gap="The transaction record has not been observed.",
        )
    elif repair == "upgrade":
        second["claims"][4].update(
            judgment="supported", evidence_refs=["product_1:artifact:final_response"]
        )
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    assert recovery["rejected_response"] == first
    assert len(recovery["details"]["issues"]) == 13
    assert "claims[4].judgment" in json.dumps(recovery["details"]["issues"])
    assert "minimally correct status/reason/gap" in recovery["instruction"]
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "all_fields":
        assert not result.get("error")
        assert result["claims"] == second["claims"]
        assert result["reason"] == second["reason"]
        assert [
            {k: v for k, v in c.items() if k != "evidence_refs"}
            for c in result["claims"]
        ] == first["claims"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["error"] and attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "upgrade",
        "downgrade",
        "delete",
        "refs",
        "reorder",
        "observation",
        "conservative",
    ],
)
@pytest.mark.parametrize("syntax_retry", [False, True])
async def test_cs1_actual_role_preserves_assessment_across_same_evidence_and_syntax(
    transport, change, syntax_retry
):
    first, second = deepcopy(WB_CAPTURE), deepcopy(WB_CAPTURE)
    if change == "upgrade":
        second["claims"][2]["judgment"] = "supported"
    elif change == "downgrade":
        second["claims"][2].update(
            verdict_relevance="advisory",
            relevance_reason="Reclassified outside the standard",
        )
    elif change == "delete":
        del second["claims"][2]
    elif change == "reorder":
        second["claims"][0], second["claims"][1] = (
            second["claims"][1],
            second["claims"][0],
        )
    else:
        second.update(
            status="unknown", reason="The material comparison remains unverified."
        )
        if change == "refs":
            second["claims"][2]["evidence_refs"] = ["product_1:artifact:final_response"]
        elif change == "observation":
            second["claims"][2]["observation"] = "A different assessment"
    replies = [first, '{"status":', second] if syntax_retry else [first, second]
    result, requests = await replay(transport, replies, *wb_evidence())
    assert all(r["context"] == requests[0]["context"] for r in requests)
    assert all(
        r["measurement_budget"]["remaining_tool_calls"] == 0 for r in requests[1:]
    )
    attempts = result["response_validation"]["attempts"]
    assert attempts[0]["response"] == first and attempts[-1]["response"] == second
    if syntax_retry:
        assert attempts[1]["raw_response"] == replies[1]
        assert requests[-1]["response_recovery"]["error_type"] == "JSONDecodeError"
    assert result["status"] == "unknown"
    if change == "conservative":
        assert not result.get("error") and result["claims"] == first["claims"]
    else:
        assert result["error"]
        assert "preserve" in attempts[-1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}


def preview_case():
    context, evidence = build_context(
        {"task_id": "offline-preview", "prompt": "Check the delivered fact"},
        [
            {
                "artifact": {"final_response": "Claim needing verification"},
                "source_documents": [
                    {"text": "x" * 1000 + " Newly visible confirming observation."}
                ],
            }
        ],
    )
    ref = "product_1:source:0"
    raw = {
        "claims": [
            {
                "claim": "The delivered fact holds",
                "claim_refs": ["product_1:artifact:final_response"],
                "evidence_refs": [ref],
                "observation": "The preview does not establish this fact",
                "judgment": "unverified",
                "verdict_relevance": "material",
            }
        ],
        "status": "pass",
        "reason": "Internally inconsistent summary",
        "gap": "",
        "evidence_refs": [ref],
    }
    return context, evidence, ref, raw


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_refs", [False, True])
async def test_new_preview_evidence_after_consistent_repair_permits_fresh_assessment(
    transport,
    missing_refs,
):
    context, evidence, ref, first = preview_case()
    second = {
        **deepcopy(first),
        "status": "unknown",
        "reason": "The material fact is unverified.",
    }
    third = deepcopy(first)
    third.update(reason="The newly supplied record establishes the fact.")
    third["claims"][0].update(
        judgment="supported", observation="The complete supplied record confirms it."
    )
    if missing_refs:
        del first["claims"][0]["evidence_refs"]
    result, requests = await replay(
        transport, [first, second, third], context, evidence
    )
    assert requests[1]["response_recovery"]["rejected_response"] == first
    assert ref not in requests[1].get("requested_evidence", {})
    assert requests[2]["requested_evidence"][ref] == evidence[ref]
    assert requests[2]["evidence_followup"] and not requests[2].get("response_recovery")
    assert result["status"] == "pass" and not result.get("error")
    assert result["claims"] == third["claims"]
    assert result["response_validation"]["selected_attempts"] == {"global": 3}


@pytest.mark.asyncio
async def test_all_advisory_pass_can_repair_summary_without_changing_claims(transport):
    first = deepcopy(WB_CAPTURE)
    for claim in first["claims"]:
        claim.update(
            verdict_relevance="advisory", relevance_reason="An ancillary observation"
        )
    second = {
        **deepcopy(first),
        "status": "unknown",
        "reason": "No material outcome was established.",
    }
    result, requests = await replay(transport, [first, second], *wb_evidence())
    assert any(
        i["field"] == "status"
        for i in requests[1]["response_recovery"]["details"]["issues"]
    )
    assert result["status"] == "unknown" and not result.get("error")
    assert result["claims"] == first["claims"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_kind", ["already_inline", "duplicate", "new_during_repair"]
)
async def test_read_request_cannot_unlock_same_evidence_repair(transport, read_kind):
    context, evidence, ref, first = preview_case()
    if read_kind == "already_inline":
        context, evidence = wb_evidence()
        first = deepcopy(WB_CAPTURE)
        ref = "product_1:source:2"
    replies = ([{"need_evidence": [ref]}] if read_kind == "duplicate" else []) + [
        first,
        {"need_evidence": [ref]},
    ]
    result, requests = await replay(transport, replies, context, evidence)
    assert result["status"] == "unknown" and result["error"]
    assert requests[-1]["measurement_budget"]["remaining_tool_calls"] == 0
    if read_kind == "duplicate":
        assert requests[-1]["requested_evidence"][ref] == evidence[ref]
    attempts = result["response_validation"]["attempts"]
    assert attempts[-1]["response"] == replies[-1]
    assert "existing evidence" in attempts[-1]["errors"]["repair"]
    assert result["response_validation"]["selected_attempts"] == {}
