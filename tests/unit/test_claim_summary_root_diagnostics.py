"""Clawv7 construction failures replayed offline; no semantic/native regrading.

Exact replies from runs/full_claw_20261012_v7/tasks/T111zh_expense_email_check/
round_1/construction_checks.json: row 0 report_clarity_and_language_check and
row 2 reconciliation_report_structure_check. Tests never read or change runs.
"""

import hashlib
import json
from copy import deepcopy

import pytest

from reward_engine.core.outcome.context import ResponseContractError
from reward_engine.core.outcome.harness import validate_claims, validate_result
from tests.unit.test_claim_contract_repair import claw_evidence, replay
from tests.unit.test_workbuddy_stream_terminal_replay import transport as transport

CAPTURE_SHA256 = "a838afaf0e4451146ab6c08f620b525a92e633f69a7498b38f8eaaeb6fa0b319"
CAPTURES = json.loads(
    '{"report_clarity_and_language_check": [{"claims": [{"claim": "The report is written in Chinese, matching the user\'s prompt language.", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "Whole report; headings and body text", "evidence_refs": ["task", "product_1:artifact:final_response"], "observation": "The final_response is written in Chinese; the Task prompt is also in Chinese. This satisfies the language-match subrequirement.", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "This supports the rubric standard but is not the sole determinant of pass/fail."}, {"claim": "The report uses the exact category labels 匹配正常 / 金额不符 / 状态异常.", "claim_refs": ["product_1:artifact:final_response", "task"], "claim_location": "Section 三 headings: ✅ 匹配正常（2笔）, ⚠️ 金额不符（1笔）, 🔴 状态异常（1笔）", "evidence_refs": ["product_1:artifact:final_response", "task"], "observation": "The three required labels appear verbatim as subsection headings. This satisfies the exact-label subrequirement.", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "Supports the rubric but does not alone determine overall correctness."}, {"claim": "Entries are scannable and include key facts (amount, invoice, status, verdict).", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "Sections 一、二、三 tables and bullet lists", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The report uses tables for email extraction and finance matching, and bullet lines under each category stating applicant, invoice, amounts, status and verdict. Each matched/flagged entry includes amount, invoice and status. However, the status-anomaly entry (INV-003) states amount consistency and status but does not explicitly repeat the invoice number in that bullet beyond the header reference; it is still identifiable. Entries are generally one-line/scannable.", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "Confirms scannability; minor omissions do not violate the soft standard."}, {"claim": "The report language and label usage make it concise enough to scan.", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "Overall structure with numbered sections and short bullets", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The report is structured into four brief sections with tables and short action bullets; length is moderate and easily scannable.", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "Supports usability but is not a hard requirement."}, {"claim": "The report contains no unsupported assertions regarding language or labels.", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "N/A", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The language and labels are directly observable in the artifact; no external verification needed.", "judgment": "supported", "verdict_relevance": "advisory", "relevance_reason": "No additional uncertainty affecting this standard."}], "status": "pass", "reason": "The report is written in Chinese, uses the exact labels 匹配正常 / 金额不符 / 状态异常, and presents entries in scannable tables/bullets with the key facts (amount, invoice, status, verdict). All material claims for this language/clarity standard are supported by the supplied artifact.", "gap": "", "evidence_refs": ["task", "product_1:artifact:final_response"], "measurement_recovery": []}, {"claims": [{"claim": "The report is written in Chinese, matching the user\'s prompt language.", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "Whole report; headings and body text", "evidence_refs": ["task", "product_1:artifact:final_response"], "observation": "The final_response is written in Chinese; the Task prompt is also in Chinese. This satisfies the language-match subrequirement.", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "The report uses the exact category labels 匹配正常 / 金额不符 / 状态异常.", "claim_refs": ["product_1:artifact:final_response", "task"], "claim_location": "Section 三 headings: ✅ 匹配正常（2笔）, ⚠️ 金额不符（1笔）, 🔴 状态异常（1笔）", "evidence_refs": ["product_1:artifact:final_response", "task"], "observation": "The three required labels appear verbatim as subsection headings. This satisfies the exact-label subrequirement.", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "Entries are scannable and include key facts (amount, invoice, status, verdict).", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "Sections 一、二、三 tables and bullet lists", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The report uses tables for email extraction and finance matching, and bullet lines under each category stating applicant, invoice, amounts, status and verdict. Each matched/flagged entry includes amount, invoice and status. The status-anomaly entry (INV-003) states amount consistency and status and remains identifiable via the header reference. Entries are generally one-line/scannable.", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "The report language and label usage make it concise enough to scan.", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "Overall structure with numbered sections and short bullets", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The report is structured into four brief sections with tables and short action bullets; length is moderate and easily scannable.", "judgment": "supported", "verdict_relevance": "material"}, {"claim": "The report contains no unsupported assertions regarding language or labels.", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "N/A", "evidence_refs": ["product_1:artifact:final_response"], "observation": "The language and labels are directly observable in the artifact; no external verification needed.", "judgment": "supported", "verdict_relevance": "material"}], "status": "pass", "reason": "The report is written in Chinese, uses the exact labels 匹配正常 / 金额不符 / 状态异常, and presents entries in scannable tables/bullets with the key facts (amount, invoice, status, verdict). All material claims for this language/clarity standard are supported by the supplied artifact.", "gap": "", "evidence_refs": ["task", "product_1:artifact:final_response"], "measurement_recovery": []}], "reconciliation_report_structure_check": [{"claims": [{"claim": "报销报告未按用户第4步要求输出三个必需类别（匹配正常 / 金额不符 / 状态异常）的规范分区。", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "报告仅在“分类汇总”下用“✅ 匹配正常（2 笔）”“⚠️ 金额不符（1 笔）”“⚠️ 状态异常（1 笔）”列出，未使用与用户措辞完全一致且独立的“匹配正常 / 金额不符 / 状态异常”类别标题。", "evidence_refs": ["task", "product_1:artifact:final_response"], "observation": "任务明确要求“列出：匹配正常的、金额不符的、状态异常的”。artifact 的“分类汇总”确实按这三类分组，但类别标题带有 emoji 与“（n 笔）”标注，属于可读的结构等价形式，未见类别缺失。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "这是本 check 的核心结构要求之一。"}, {"claim": "每一笔提取到的报销均被放入报告，并进入三个类别之一。", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "表格列出 4 笔：张伟 3200 / INV-001 / TXN-701 / completed；陈静 420 / INV-004 / TXN-704 / completed；李华 850 / INV-002 / TXN-702 / 680 / completed；王明 5000 / INV-003 / TXN-703 / pending。分类汇总分别计入正常 2 笔、金额不符 1 笔、状态异常 1 笔。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "observation": "邮件侧有 msg_701–msg_704 四封报销申请，财务侧有 TXN-701–TXN-704 四条交易；artifact 覆盖全部 4 笔，并在分类中每笔仅出现一次。没有双重归类未解释的情况。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本 check 要求每笔费用进入三类之一或解释双重归类。"}, {"claim": "每个分类项都引用了所比较的邮件与财务记录。", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "表格列有“发票号”“交易ID”“邮件金额”“系统金额”“状态”；分类汇总中亦给出发票号/交易ID对应关系，例如李华 INV-002 / TXN-702，王明 INV-003 / TXN-703。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7", "product_1:source:3"], "observation": "报告引用了四条财务交易 ID（TXN-701–TXN-704）和四张发票号/邮件金额，能对应到具体邮件与系统记录；但未直接显示邮件 message_id（如 msg_701），仅以申请人、发票号、金额作为邮件侧定位。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本 check 要求每个分类项能说明比较了哪封邮件和哪条财务记录。"}, {"claim": "报告对金额不符项与状态异常项分别给出理由，未将同一笔重复计入两个类别。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "李华：邮件 850 元，系统 680 元，差额 170 元，归为金额不符；王明：金额一致但 pending，归为状态异常。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:5", "product_1:source:3", "product_1:source:6"], "observation": "李华交易状态为 completed，因此仅金额不符；王明金额 5000 与邮件一致，仅状态为 pending，因此仅状态异常。每笔只归一类，无未解释的双重归类。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本 check 允许双重归类但要求解释；此处不存在双重归类问题。"}], "status": "pass", "reason": "报告包含三类结果分组（匹配正常、金额不符、状态异常），4 笔报销全部列入且各归一类；金额不符与状态异常均说明了比较依据，并引用邮件侧金额/发票号和财务侧交易 ID/系统金额。", "gap": "", "measurement_recovery": []}, {"claims": [{"claim": "报销报告包含用户要求的三个类别。", "claim_refs": ["task", "product_1:artifact:final_response"], "claim_location": "分类汇总：\'✅ 匹配正常（2 笔）\'、\'⚠️ 金额不符（1 笔）\'、\'⚠️ 状态异常（1 笔）\'", "evidence_refs": ["task", "product_1:artifact:final_response"], "observation": "任务第4步要求列出匹配正常的、金额不符的、状态异常的；artifact报告以匹配正常/金额不符/状态异常三个分组呈现，措辞一致，带有计数与符号标注。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "三个类别是用户明确要求的报告结构核心。"}, {"claim": "每笔提取到的报销均被放入三类之一，且无未解释的双重归类。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "表格列出张伟3200/INV-001/TXN-701/completed、李华850/INV-002/TXN-702/680/completed、王明5000/INV-003/TXN-703/pending、陈静420/INV-004/TXN-704/completed；分类为正常2笔、金额不符1笔、状态异常1笔。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "observation": "邮件列表msg_701–msg_704对应四封报销申请；财务交易TXN-701–TXN-704对应四条记录；报告覆盖全部四笔，每笔仅归一类：李华因金额不符（状态completed）归金额不符，王明因pending（金额一致）归状态异常，无未解释双重归类。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本check要求每笔费用进入三类之一或解释双重归类。"}, {"claim": "每个分类项引用了所比较的邮件和财务记录。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "表格含\'发票号\'\'交易ID\'\'邮件金额\'\'系统金额\'\'状态\'列；分类汇总中李华INV-002/TXN-702、王明INV-003/TXN-703等关联。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "observation": "报告为每笔记明发票号、交易ID、邮件金额与系统金额，足以定位所比较的邮件（通过申请人/发票号/金额）和财务记录（TXNID）；未直接显示邮件message_id，但邮件侧定位信息充分。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本check要求每个分类项说明比较了哪封邮件和哪条财务记录。"}, {"claim": "金额不符项与状态异常项分别说明原因，未重复归类。", "claim_refs": ["product_1:artifact:final_response"], "claim_location": "李华：邮件850 vs 系统680，差额170，归金额不符；王明：金额一致但pending，归状态异常。", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:3", "product_1:source:5", "product_1:source:6"], "observation": "李华交易状态completed，仅金额不符；王明金额5000与邮件一致，仅状态pending。每笔仅归一类，无未解释双重归类。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "本check允许双重归类但需解释，此处不存在双重归类问题。"}], "status": "pass", "reason": "报告按用户措辞包含匹配正常、金额不符、状态异常三类分组；四笔报销全部列入且各归一类；每笔引用发票号、交易ID、邮件金额与系统金额，足以对应所比较的邮件和财务记录。", "gap": "", "evidence_refs": ["task", "product_1:artifact:final_response", "product_1:source:2", "product_1:source:3", "product_1:source:4", "product_1:source:5", "product_1:source:6", "product_1:source:7"], "measurement_recovery": []}]}'
)


CLARITY = "report_clarity_and_language_check"
STRUCTURE = "reconciliation_report_structure_check"


def test_exact_captures_identify_all_advisory_status_and_missing_root_refs():
    assert (
        hashlib.sha256(
            json.dumps(
                CAPTURES, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == CAPTURE_SHA256
    )
    _, evidence = claw_evidence()
    first = deepcopy(CAPTURES[CLARITY][0])
    assert len(first["claims"]) == 5
    assert all(c["verdict_relevance"] == "advisory" for c in first["claims"])
    with pytest.raises(ResponseContractError) as caught:
        validate_claims(first, evidence)
    issue = caught.value.details
    assert issue["field"] == "status" and issue["actual"] == "pass"
    assert issue["material_claim_count"] == 0
    assert issue["depends_on"] == {
        f"claims[{i}].verdict_relevance": "advisory" for i in range(5)
    }
    assert "minimally correct status/reason/gap" in issue["repair_instruction"]
    assert first == CAPTURES[CLARITY][0]
    first = deepcopy(CAPTURES[STRUCTURE][0])
    assert "evidence_refs" not in first and all(
        c["evidence_refs"] for c in first["claims"]
    )
    with pytest.raises(ResponseContractError) as caught:
        validate_result(first, evidence)
    issue = caught.value.details
    assert (
        issue["field"] == "evidence_refs"
        and issue["actual"] is None
        and issue["missing"]
    )
    assert "Top-level evidence_refs" in issue["reason"]
    assert (
        "Claim-level evidence_refs do not replace this root field"
        in issue["repair_instruction"]
    )
    assert first == CAPTURES[STRUCTURE][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["summary_only", "recorded_relabeling", "identical"])
async def test_all_advisory_repair_corrects_summary_without_reclassifying_claims(
    transport, repair
):
    first = deepcopy(CAPTURES[CLARITY][0])
    second = deepcopy(
        CAPTURES[CLARITY][1] if repair == "recorded_relabeling" else first
    )
    if repair == "summary_only":
        # Authored by the scripted model; the host never substitutes a verdict.
        second.update(
            status="unknown",
            reason="Only advisory findings were assessed; no material basis establishes this criterion.",
            gap="",
        )
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["issues"][0]["material_claim_count"] == 0
    instruction = recovery["instruction"]
    assert "minimally correct status/reason/gap" in instruction
    assert "do not promote advisory claims to material" in instruction
    assert "same evidence" in instruction
    assert (
        "Material means needed to judge THIS criterion, including a soft criterion"
        in requests[0]["instruction"]
    )
    assert (
        "A soft criterion does not make its claims advisory"
        in requests[0]["instruction"]
    )
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "summary_only":
        assert not result.get("error") and result["claims"] == first["claims"]
        assert result["reason"] == second["reason"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert (
            result["error"] and result["response_validation"]["selected_attempts"] == {}
        )
        assert attempts[1]["errors"]["global"]
        if repair == "recorded_relabeling":
            assert "preserve already valid fields" in attempts[1]["errors"]["global"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["root_only", "recorded_rewrite", "identical"])
async def test_missing_root_refs_repair_keeps_exact_assessment(transport, repair):
    first = deepcopy(CAPTURES[STRUCTURE][0])
    second = deepcopy(CAPTURES[STRUCTURE][1] if repair == "recorded_rewrite" else first)
    if repair == "root_only":
        second["evidence_refs"] = ["task", "product_1:artifact:final_response"]
    result, requests = await replay(transport, [first, second], *claw_evidence())
    recovery = requests[1]["response_recovery"]
    issue = recovery["details"]["issues"][0]
    assert issue["scope"] == "global" and issue["field"] == "evidence_refs"
    assert issue["missing"] and issue["actual"] is None
    assert issue["repair_instruction"] in recovery["instruction"]
    assert (
        "Repair only the invalid top-level evidence_refs field"
        in recovery["instruction"]
    )
    assert "do not regenerate claims" in recovery["instruction"]
    assert "host does not fill them" in recovery["instruction"]
    assert requests[0]["context"] == requests[1]["context"]
    assert requests[1]["measurement_budget"]["remaining_tool_calls"] == 0
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    if repair == "root_only":
        assert result["status"] == "pass" and not result.get("error")
        for field in ("claims", "status", "reason", "gap"):
            assert result[field] == first[field]
        assert result["evidence_refs"] == second["evidence_refs"]
        assert result["response_validation"]["selected_attempts"] == {"global": 2}
    else:
        assert result["status"] == "unknown" and result["error"]
        assert result["response_validation"]["selected_attempts"] == {}
        assert attempts[1]["errors"]["global"]
        if repair == "recorded_rewrite":
            assert "preserve already valid fields" in attempts[1]["errors"]["global"]


@pytest.mark.parametrize(
    "refs", [None, [], "product_1:artifact:final_response", ["invented:proof"]]
)
def test_bad_root_refs_keep_exact_actual_without_mutation(refs):
    _, evidence = claw_evidence()
    first = deepcopy(CAPTURES[STRUCTURE][0])
    first["evidence_refs"] = refs
    before = deepcopy(first)
    with pytest.raises(ResponseContractError) as caught:
        validate_result(first, evidence)
    issue = caught.value.details
    assert issue["field"] == "evidence_refs" and issue["actual"] == refs
    assert issue["missing"] is False
    assert first == before


def test_existing_root_refs_acceptance_unchanged():
    _, evidence = claw_evidence()
    first = deepcopy(CAPTURES[STRUCTURE][0])
    first["evidence_refs"] = ["product_1:artifact:final_response"]
    assert validate_result(first, evidence) == first
    first["status"] = "unknown"
    del first["evidence_refs"]
    assert validate_result(first, evidence) == first


# WBv6 construction row 2: cross_validation_evidence_citation_check.
WB_SHA256 = "1208beef64c9965102f567f2a7e047475e1db349710fe0e5e0f94888b044ded2"
WB_CAPTURE = json.loads(
    '[{"claims": [{"claim": "最终交付物为中文 Markdown 并保存到 final_answer.md，且工作区证据显示该文件被创建（sha256/bytes/changed 字段），未改动示例仓库文件。", "claim_refs": ["product_1:artifact:final_response", "product_1:artifact:changed_files"], "claim_location": "final_response 开头与 changed_files 列表", "evidence_refs": ["product_1:artifact:changed_files", "product_1:source:2", "product_1:environment"], "observation": "changed_files 显示 final_answer.md changed=true、有 sha256/bytes；环境 deleted_files 为空；source:2 是 final_answer.md 正文片段，含中文 Markdown 标题；未显示 input/ 下文件被改。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "任务要求中文 Markdown、保存位置及只读边界。"}, {"claim": "关键断言普遍附带工作区内具体路径证据（如 src/citrinekit/cli.py、client.py、config/defaults.yaml、schemas/*.json、examples/output/scan_summary.jsonl）。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "最终回复要点概览及 final_answer.md 正文引用", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2", "product_1:source:0", "product_1:source:1"], "observation": "final_response 列出带行号/文件路径的 CLI/API/config/schema/output 证据；source:2 正文包含多个 input/sample_workspace/citrinectl/... 路径；source:0/1 是工作区文件证据。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "该标准核心是路径证据充分性。"}, {"claim": "主要结论由至少两类材料交叉支持（source + docs/config/schema/examples/output），README 不是唯一依据。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "final_response 概览与 final_answer.md 证据链", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "概览同时引用 src 源码、config、schemas、examples/output；正文按 功能/入口、命令或 API 示例、关键参数、证据文件、caveat 组织，并称 schema 与 output sample 交叉验证。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "标准明确要求多类材料交叉验证，不得仅靠 README。"}, {"claim": "caveat 类断言（--json/run 为 stale、experimental.py 未接入、paths 不运行校验、job YAML 未被解析、warning_count 逻辑、scan.timeout 应为 timeout_seconds）被列为重要限制。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "最终回复 caveat 段落与 final_answer.md 第 6 节类内容", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "final_response 明确列出这些 caveat，并称每条附工作区路径证据；source:2 正文片段重复了 schema/paths/job YAML 限制，但完整逐条证据未在预览中展开。", "judgment": "unverified", "verdict_relevance": "material", "relevance_reason": "若这些 caveat 缺乏可定位证据，会削弱本标准的证据链要求；当前仅见总结引用，未逐条核验底层文件。"}, {"claim": "交付物覆盖了任务要求的五大重点：CLI、Python API、config/default/schema、examples/output format、unsupported/ambiguous caveats。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "final_response 要点概览与 final_answer.md 目录式内容", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "概览逐项覆盖 CLI、API、config、schema/output、caveat；建议章节结构对应任务输出建议。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "任务要求重点覆盖这些方面。"}, {"claim": "交付物没有把 README 当作功能断言的唯一依据。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "证据文件引用来源", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:0", "product_1:source:2"], "observation": "README 在 source:0 中存在，但 final_response 的主要引用指向 src、config、schemas、examples；未发现以 README 单独支撑的断言。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "标准明确排除 README 唯一依据。"}], "status": "pass", "reason": "已交付的中文 Markdown final_answer.md 在 changed_files 中可确认存在，且最终回复与可见正文片段显示关键断言附有工作区内具体文件路径，来源涵盖源码、配置、schema、示例输出等多类材料并相互引用；未发现仅依赖 README 或单一文件支撑主要结论的证据。caveat 逐条底层证据因源预览不完整而未完全核验，但现有可见证据已足以支持本标准的路径证据与多源交叉要求。", "gap": "", "evidence_refs": ["product_1:artifact:changed_files", "product_1:artifact:final_response", "product_1:source:2", "product_1:source:0", "product_1:source:1", "product_1:environment"], "inquiry_result": {"question": "关键断言是否附具体工作区文件路径？主要结论是否由 docs/source/config/schema/examples/output 中至少两类材料交叉支持？README 是否被当作唯一依据？", "verdict": "pass", "reason": "最终回复和 final_answer.md 片段均显示引用具体路径（如 src/citrinekit/cli.py:11-35、client.py、config/defaults.yaml、schemas/*.json、examples/output/scan_summary.jsonl），覆盖源码、配置、schema、示例输出等多类材料；README 未被单独用作功能依据。未发现材料级缺陷。"}}, {"claims": [{"claim": "最终交付物为中文 Markdown 并保存到 final_answer.md，且工作区证据显示该文件被创建（sha256/bytes/changed 字段），未改动示例仓库文件。", "claim_refs": ["product_1:artifact:final_response", "product_1:artifact:changed_files"], "claim_location": "final_response 开头与 changed_files 列表", "evidence_refs": ["product_1:artifact:changed_files", "product_1:source:2", "product_1:environment"], "observation": "changed_files 显示 final_answer.md changed=true、有 sha256/bytes；环境 deleted_files 为空；source:2 是 final_answer.md 正文片段，含中文 Markdown 标题；未显示 input/ 下文件被改。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "任务要求中文 Markdown、保存位置及只读边界。"}, {"claim": "关键断言普遍附带工作区内具体路径证据（如 src/citrinekit/cli.py、client.py、config/defaults.yaml、schemas/*.json、examples/output/scan_summary.jsonl）。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "最终回复要点概览及 final_answer.md 正文引用", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2", "product_1:source:0", "product_1:source:1"], "observation": "final_response 列出带行号/文件路径的 CLI/API/config/schema/output 证据；source:2 正文包含多个 input/sample_workspace/citrinectl/... 路径；source:0/1 是工作区文件证据。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "该标准核心是路径证据充分性。"}, {"claim": "主要结论由至少两类材料交叉支持（source + docs/config/schema/examples/output），README 不是唯一依据。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "final_response 概览与 final_answer.md 证据链", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "概览同时引用 src 源码、config、schemas、examples/output；正文按 功能/入口、命令或 API 示例、关键参数、证据文件、caveat 组织，并称 schema 与 output sample 交叉验证。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "标准明确要求多类材料交叉验证，不得仅靠 README。"}, {"claim": "caveat 类断言（--json/run 为 stale、experimental.py 未接入、paths 不运行校验、job YAML 未被解析、warning_count 逻辑、scan.timeout 应为 timeout_seconds）被列为重要限制且各自附底层路径证据。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "最终回复 caveat 段落与 final_answer.md 第 6 节类内容", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "final_response 明确列出这些 caveat，并称每条附工作区路径证据；source:2 正文片段重复了 schema/paths/job YAML 限制，但完整逐条证据未在预览中展开，无法核验每条 caveat 是否都有可定位的底层文件路径。", "judgment": "unverified", "verdict_relevance": "material", "relevance_reason": "若这些 caveat 缺乏可定位证据，会削弱本标准的证据链要求；当前仅见总结引用，未逐条核验底层文件。"}, {"claim": "交付物覆盖了任务要求的五大重点：CLI、Python API、config/default/schema、examples/output format、unsupported/ambiguous caveats。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "final_response 要点概览与 final_answer.md 目录式内容", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:2"], "observation": "概览逐项覆盖 CLI、API、config、schema/output、caveat；建议章节结构对应任务输出建议。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "任务要求重点覆盖这些方面。"}, {"claim": "交付物没有把 README 当作功能断言的唯一依据。", "claim_refs": ["product_1:artifact:final_response", "product_1:source:2"], "claim_location": "证据文件引用来源", "evidence_refs": ["product_1:artifact:final_response", "product_1:source:0", "product_1:source:2"], "observation": "README 在 source:0 中存在，但 final_response 的主要引用指向 src、config、schemas、examples；未发现以 README 单独支撑的断言。", "judgment": "supported", "verdict_relevance": "material", "relevance_reason": "标准明确排除 README 唯一依据。"}], "status": "unknown", "reason": "已确认交付物为中文 Markdown final_answer.md，且多数关键断言附有具体工作区路径并涵盖源码、配置、schema、示例输出等多类材料，README 未被单独用作依据。但一条重要断言——caveat 类结论均逐条附底层路径证据——在现有证据中仅为交付方自述与摘要级引用，未被独立核验，属未决材料事实；因此本标准无法判定为 pass。", "gap": "需核验 final_answer.md 中每条 caveat（--json/run stale、experimental.py 未接入、paths 不做运行时校验、job YAML 未被解析、warning_count 逻辑、scan.timeout 命名）是否各自给出可定位的工作区内具体文件路径证据；当前仅有概述与片段，缺少逐条底层文件映射。", "evidence_refs": ["product_1:artifact:changed_files", "product_1:artifact:final_response", "product_1:source:2", "product_1:source:0", "product_1:source:1", "product_1:environment"], "inquiry_result": {"question": "关键断言是否附具体工作区文件路径？主要结论是否由 docs/source/config/schema/examples/output 中至少两类材料交叉支持？README 是否被当作唯一依据？", "verdict": "unknown", "reason": "已确认多数关键断言引用具体路径（如 src/citrinekit/cli.py:11-35、client.py、config/defaults.yaml、schemas/*.json、examples/output/scan_summary.jsonl），覆盖源码、配置、schema、示例输出等多类材料，README 未被单独用作功能依据。但 caveat 类断言的逐条底层路径证据未在现有片段中核验，故本问题无法判定为 pass；需展开 final_answer.md 相应章节以确认每条 caveat 附有可定位的工作区路径。"}}]'
)


@pytest.mark.asyncio
@pytest.mark.parametrize("repair", ["summary_only", "recorded_rewrite"])
async def test_wbv6_valid_claim_text_and_observation_are_immutable(transport, repair):
    from tests.unit.test_claim_status_diagnostics import evidence_case

    assert (
        hashlib.sha256(
            json.dumps(
                WB_CAPTURE, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        == WB_SHA256
    )
    first, second = deepcopy(WB_CAPTURE)
    changed = {
        k for k in first["claims"][3] if first["claims"][3][k] != second["claims"][3][k]
    }
    assert changed == {"claim", "observation"}
    assert (
        first["claims"][3]["judgment"]
        == second["claims"][3]["judgment"]
        == "unverified"
    )
    if repair == "summary_only":
        # The scripted model authors the summary without revising its assessment.
        second = {
            **deepcopy(first),
            **{k: second[k] for k in ("status", "reason", "gap")},
        }
    result, requests = await replay(transport, [first, second], *evidence_case())
    recovery = requests[1]["response_recovery"]
    assert recovery["details"]["issues"][0]["claim_index"] == 3
    assert (
        "Copy already valid claim fields exactly from rejected_response"
        in recovery["instruction"]
    )
    assert (
        "depends_on fields are constraints to preserve, not repair targets"
        in recovery["instruction"]
    )
    assert (
        "Do not paraphrase or clarify valid claim text or observations"
        in recovery["instruction"]
    )
    assert requests[0]["context"] == requests[1]["context"]
    attempts = result["response_validation"]["attempts"]
    assert [a["response"] for a in attempts] == [first, second]
    assert result["status"] == "unknown"
    if repair == "summary_only":
        assert not result.get("error") and result["claims"] == first["claims"]
        assert result["reason"] == second["reason"] and result["gap"] == second["gap"]
    else:
        assert result["error"]
        assert "claims[3]" in attempts[1]["errors"]["global"]
        assert result["response_validation"]["selected_attempts"] == {}
