"""Exact Clawv8 B review shape replayed through actual offline Role and validators.

Captured checks/model/f5ead940cc8549f0b3f8a5f08af9e7c6/response.json under
runs/full_claw_20261012_v8/tasks/T111zh_expense_email_check/round_1/actors/B.
The d4bd65da479e405f96d4ea4ab2e48942 repair answer was identical.
Synthetic evidence checks reference contracts, not native factual correctness.
Tests never read or rewrite runs and never call a live provider.
"""

import hashlib
import json
from copy import deepcopy

import httpx
import pytest

from reward_engine.adapters.workbuddy.models import Role
from reward_engine.core.outcome.context import ResponseContractError, ask, build_context
from reward_engine.core.outcome.delivery_triage import validate_review
from reward_engine.core.outcome.improvement import INSTRUCTION, schema, validate
from tests.unit.test_workbuddy_stream_terminal_replay import encode, event
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

CAPTURE_SHA256 = "4a5096768d96474f9eddaf98799ff1a645a29e1a447d315cd893533607f4e4a9"
CAPTURE = json.loads(
    '{"outcome_goal": "一份可直接交付财务复核的报销核对报告：完整覆盖4封报销邮件并正确排除非报销邮件，每笔按发票号定位到对应交易，逐笔给出金额比对（含差额）与状态判断，按匹配正常/金额不符/状态异常三类无重叠归类，且每条结论可追溯到邮件与交易ID，并对异常项给出可执行下一步。", "full_score_gaps": [{"id": "G1_status_field_omission", "gap": "邮件清单表(c1)只列了金额与发票号，未列申请人邮件中的其他可核对字段（如日期、事由），且未说明邮件日期(2026-10)与交易日期(2026-03)存在时间差——这一差异对“核对”场景可能提示报销与入账时间不一致，报告未提及。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:4", "product_1:source:9"]}, {"id": "G2_category_rule_edge_case", "gap": "报告声明了归类规则（金额一致且completed→正常；金额不一致→金额不符；金额一致但非completed→状态异常），但未说明当金额不符且状态异常同时出现时如何归类（任务open_question亦标记此边界未解决），仅口头说\'每笔仅归一类\'。当前数据未触发该分支，但缺少边界规则声明。", "evidence_refs": ["product_1:artifact:final_response", "reference:outcome_question:5"]}, {"id": "G3_recommendation_specificity", "gap": "异常项建议较笼统（\'核实差异\'\'跟进审批人\'），未给出具体可执行动作，如：核对TXN-702审批日志/核减凭据、确认INV-002是否部分票据缺失、对TXN-703查询pending阻塞原因并给出预计审批路径。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:10", "product_1:source:11"]}, {"id": "G4_currency_notation", "gap": "系统金额以CNY计价（finance记录currency=CNY），邮件金额表述为\'元\'，报告未显式声明两者币种一致，也未标明\'元=CNY\'，在跨境/多币种核对场景下可能产生歧义。", "evidence_refs": ["product_1:source:9", "product_1:source:3"]}], "improvements": [{"id": "I1_add_cross_check_dimensions", "gap_ids": ["G1"], "action": "在邮件清单与交易匹配表中增列\'邮件日期\'与\'交易日期\'两列，并在结论处加一行说明：本次4笔邮件时间(2026-10)晚于系统交易日期(2026-03)，可能为历史单据补报或系统日期记录方式，不影响本次金额/状态核对，但若为时序核对场景需注意。", "expected_gain": "使报告在\'核对\'任务上不仅比对金额与状态，还覆盖时间维度的合理性判断，提升报告深度与可复核性。", "verification": "报告中出现邮件日期列与交易日期列，且有一句明确的时间差说明；日期值可分别从source:4-7与source:9-12核对。", "evidence_refs": ["product_1:source:4", "product_1:source:9"]}, {"id": "I2_declare_edge_category_rule", "gap_ids": ["G2"], "action": "在\'说明\'部分将归类规则扩展为边界规则：若一笔同时存在金额不符与状态异常，优先归入\'金额不符\'（金额差异更需人工介入），并在报告中同时标注状态异常；若仅状态异常则归\'状态异常\'。当前4笔数据未触发该分支，需声明规则以保证报告在扩展数据下可用。", "expected_gain": "消除分类规则的隐含歧义，使报告在同类财务核对任务上具备可迁移的判定逻辑，符合open_question:5标记的边界未决问题。", "verification": "报告\'说明\'段出现明确的同时命中归类优先级规则，且与三类标签一致。", "evidence_refs": ["product_1:artifact:final_response", "reference:outcome_question:5"]}, {"id": "I3_concretize_recommendations", "gap_ids": ["G3"], "action": "将两处异常建议具体化：(a)李华/INV-002：建议调取TXN-702审批与核减记录、核对INV-002所附明细是否含未入账票据、确认170元是否需补付并走补充审批；(b)王明/INV-003：建议联系张总监确认pending阻塞原因（如缺预算科目或需上级会签），并给出预计放款路径。", "expected_gain": "建议从\'跟进/核实\'升级为可执行动作，使财务人员能据此直接推进处理，符合核对报告的实务用途。", "verification": "报告中每条异常项建议含至少一个具体可执行步骤（查询对象+动作），且与TXN-702/TXN-703证据一致。", "evidence_refs": ["product_1:source:10", "product_1:source:11"]}], "claim_checks": [{"claim": "李华报销金额850元与系统680元不符，差额170元。", "claim_ref": "product_1:artifact:final_response", "support_refs": ["product_1:source:5", "product_1:source:10"], "assessment": "supported", "reason": "邮件body明确850元(INV-002)，TXN-702金额680.0，差额170元计算正确，金额保留原始值未被改写。", "test_id": null}, {"claim": "王明报销金额一致但状态为pending，属状态异常。", "claim_ref": "product_1:artifact:final_response", "support_refs": ["product_1:source:6", "product_1:source:11"], "assessment": "supported", "reason": "INV-003邮件5000元与TXN-703金额5000.0一致，TXN-703 status=\'pending\'，未等同\'找到记录\'与\'完成\'，判断正确。", "test_id": null}, {"claim": "msg_705(行政通知：团建)非报销邮件，已排除。", "claim_ref": "product_1:artifact:final_response", "support_refs": ["product_1:source:8"], "assessment": "supported", "reason": "msg_705为hr@company.com发给all@company.com的行政通知，正文无报销金额/发票号，排除合理且保留了被排除项说明。", "test_id": null}], "reason": "决定性事实层已充分支撑：4封报销邮件金额/发票号、4条交易记录金额/状态、两类异常(INV-002金额差170元、INV-003 pending)均与source:4-12逐一对应，排除项正确，可追溯性(c5)与缺失/冲突处理(c6)在本次数据下均满足。剩余差距为全质量维度的深化：时间维度未纳入核对、边界归类规则未声明、异常建议较笼统、币种未显式声明——这些不构成事实错误，但影响报告在扩展与实务场景下的完整性与可用性，故整体判为needs_improvement。", "status": "needs_improvement", "gaps": [{"gap": "邮件清单未提供除金额/发票号外的可核对字段及邮件与交易时间差说明，削弱核对报告深度。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:4", "product_1:source:9"]}, {"gap": "分类规则未覆盖\'金额不符且状态异常\'同时命中的边界情况。", "evidence_refs": ["product_1:artifact:final_response", "reference:outcome_question:5"]}], "verify": [{"test_id": "c5_traceable_to_evidence_check", "reason": "确认报告每行结论均可反向定位到邮件/交易ID，且异常项给出了可定位证据来源。"}, {"test_id": "c6_explicit_missing_and_conflict_handling_check", "reason": "确认金额不符与状态异常均被显式标注且给出后续建议，未静默归入正常。"}]}'
)


EXACT_IDS = [
    "G1_status_field_omission",
    "G2_category_rule_edge_case",
    "G3_recommendation_specificity",
    "G4_currency_notation",
]


def evidence_case():
    context, evidence = build_context(
        {"task_id": "offline-review", "prompt": "Inspect the delivered report"},
        [
            {
                "artifact": {"final_response": "Offline structural replay"},
                "source_documents": [
                    {"text": f"Synthetic source {i}"} for i in range(13)
                ],
            }
        ],
    )
    evidence["reference:outcome_question:5"] = (
        "Synthetic question for reference validation"
    )
    context["available_evidence_refs"] = list(evidence)
    return context, evidence


def corrected_capture():
    row = deepcopy(CAPTURE)
    # Explicit SCRIPTED MODEL replacements, not host inference or alias mapping.
    row["improvements"][0]["gap_ids"] = ["G1_status_field_omission"]
    row["improvements"][1]["gap_ids"] = ["G2_category_rule_edge_case"]
    row["improvements"][2]["gap_ids"] = ["G3_recommendation_specificity"]
    return row


def test_exact_capture_reports_all_invalid_actions_and_allowed_own_ledger_ids():
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
        validate(raw, evidence)
    details = caught.value.details
    assert details["allowed_ids"] == EXACT_IDS
    assert details["field"] == "improvements[0].gap_ids"
    assert [i["field"] for i in details["issues"]] == [
        f"improvements[{n}].gap_ids" for n in range(3)
    ]
    assert [i["actual"] for i in details["issues"]] == [["G1"], ["G2"], ["G3"]]
    assert all(
        i["invalid_indices"] == [0] and not i["missing"] for i in details["issues"]
    )
    assert all(g in details["reason"] for g in EXACT_IDS)
    assert "Correct every listed" in caught.value.recovery_instruction
    assert raw == CAPTURE


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        "G1_status_field_omission",
        ["G1"],
        ["g1_status_field_omission"],
        ["G1_status_field_omission "],
        ["G1_status"],
        ["I1_add_cross_check_dimensions"],
        ["product_1:source:4"],
        ["G1_status_field_omission", "G2", 7, {}],
    ],
)
def test_exact_reference_rule_rejects_aliases_and_malformed_values_without_mutation(
    bad,
):
    _, evidence = evidence_case()
    raw = corrected_capture()
    raw["improvements"][0]["gap_ids"] = deepcopy(bad)
    before = deepcopy(raw)
    with pytest.raises(ResponseContractError) as caught:
        validate(raw, evidence)
    issue = caught.value.details["issues"][0]
    assert issue["actual"] == bad
    assert issue["invalid_indices"] == (
        [1, 2, 3]
        if isinstance(bad, list) and len(bad) == 4
        else [0]
        if isinstance(bad, list) and bad
        else []
    )
    assert raw == before


def test_missing_references_are_distinguished_and_exact_multiple_ids_still_pass():
    _, evidence = evidence_case()
    row = corrected_capture()
    del row["improvements"][1]["gap_ids"]
    with pytest.raises(ResponseContractError) as caught:
        validate(row, evidence)
    assert caught.value.details["issues"] == [
        {
            "field": "improvements[1].gap_ids",
            "actual": None,
            "missing": True,
            "invalid_indices": [],
        }
    ]
    row = corrected_capture()
    row["improvements"][0]["gap_ids"] = [
        "G1_status_field_omission",
        "G4_currency_notation",
    ]
    before = deepcopy(row)
    assert validate(row, evidence) == before and row == before


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["exact_ids", "identical", "only_first"])
async def test_actual_role_single_repair_receives_all_indices_and_preserves_raw_replies(
    transport, repair
):
    original, responses, calls = transport
    role = Role(
        {**original.config, "role_bounded_stage": True}, "judge", original.output
    )
    first = deepcopy(CAPTURE)
    second = corrected_capture() if repair == "exact_ids" else deepcopy(CAPTURE)
    if repair == "only_first":
        second["improvements"][0]["gap_ids"] = ["G1_status_field_omission"]
    responses.extend(
        httpx.Response(
            200,
            text=encode(
                [event({"content": json.dumps(row, ensure_ascii=False)}, "stop")]
            ),
        )
        for row in (first, second)
    )
    context, evidence = evidence_case()
    tests = {item["test_id"] for item in first["verify"]}

    async def run():
        return await ask(
            role,
            "outcome_delivery_review",
            context,
            evidence,
            INSTRUCTION,
            {
                **schema(),
                "status": "review status",
                "reason": "review conclusion",
                "gaps": [],
                "verify": [],
                "claim_checks": [],
            },
            verify_used_previews=True,
            max_evidence_rounds=2,
            close_evidence=True,
            max_contract_retries=1,
            validate=lambda value: validate(
                validate_review(value, evidence, tests), evidence
            ),
        )

    if repair == "exact_ids":
        result = await run()
        assert result["full_score_gaps"] == first["full_score_gaps"]
        assert result["improvements"] == second["improvements"]
        assert result["status"] == first["status"] == "needs_improvement"
        assert result["reason"] == first["reason"]
        assert result["claim_checks"] == first["claim_checks"]
        for old, new in zip(first["improvements"], result["improvements"], strict=True):
            assert {k: v for k, v in old.items() if k != "gap_ids"} == {
                k: v for k, v in new.items() if k != "gap_ids"
            }
    else:
        with pytest.raises(ResponseContractError) as caught:
            await run()
        expected = range(1, 3) if repair == "only_first" else range(3)
        assert [i["field"] for i in caught.value.details["issues"]] == [
            f"improvements[{i}].gap_ids" for i in expected
        ]
    assert len(calls) == 2 and not responses
    requests = [json.loads(call["messages"][1]["content"]) for call in calls]
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["allowed_ids"] == EXACT_IDS
    assert len(recovery["details"]["issues"]) == 3
    assert recovery["rejected_response"] == first
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[0].get("requested_evidence") == requests[1].get(
        "requested_evidence"
    )
    assert "copy each complete ID string" in requests[0]["instruction"]
    assert (
        "exact complete full_score_gaps[].id"
        in requests[0]["response"]["improvements"][0]["gap_ids"][0]
    )
    assert "Correct every listed improvements[i].gap_ids" in recovery["instruction"]
    assert "Preserve full_score_gaps IDs" in recovery["instruction"]
    assert "host does not infer or map aliases" in recovery["instruction"]
    receipts = [json.loads(p.read_text()) for p in role.output.glob("*/response.json")]
    assert len(receipts) == 2
    assert sorted(r["raw_output"] for r in receipts) == sorted(
        json.dumps(row, ensure_ascii=False) for row in (first, second)
    )
    assert first == CAPTURE
