"""Sparse Outcome guidance. Same Pack/registry/memory as delivery review; no battles."""

import asyncio
import json
from copy import copy, deepcopy
from time import monotonic

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome.context import (
    ask,
    build_context,
    inline_recent_sources,
    measurement_test_view,
    require_refs,
)
from reward_engine.core.outcome.harness import measure, measurement_context
from reward_engine.core.outcome.hypotheses import question_context
from reward_engine.core.outcome.improvement import INSTRUCTION, schema
from reward_engine.core.outcome.improvement import validate as validate_improvements


def settings(value=None):
    value = {"enabled": False, "every_k_turns": 3, "max_tests": 3,
             "timeout_s": 120, "top_k": 3, "max_request_bytes": 160000, **(value or {})}
    if type(value["enabled"]) is not bool:
        raise ValueError("policy_critic.enabled must be boolean")
    for name in ("every_k_turns", "max_tests", "top_k", "max_request_bytes"):
        if type(value[name]) is not int or value[name] < (0 if name == "top_k" else 1):
            raise ValueError(f"Invalid policy_critic.{name}")
    if type(value["timeout_s"]) not in (int, float) or not 0 < value["timeout_s"] <= 300:
        raise ValueError("Policy Critic timeout must be in (0, 300]")
    return value


def failed_check(row):
    """A valid local failure survives a separate failed/unknown main checker."""
    return any(r.get("status") == "fail" and not r.get("error")
               for r in (row, row.get("inquiry_result", {})))


def check_summaries(measurement):
    fields = ("test_id", "criterion_id", "status", "gap", "reason", "error", "diagnostic", "evidence_refs")
    return [{**{k: deepcopy(r[k]) for k in fields if k in r},
             **({"inquiry_result": {k: deepcopy(r["inquiry_result"][k]) for k in fields
                                   if k in r["inquiry_result"]}} if "inquiry_result" in r else {})}
            for r in (measurement or {}).get("results", [])]


def measured_fallback(measurement, error):
    """No stale initial advice: deliver actual check facts when synthesis failed."""
    row = {"status": "degraded", "global_assessment": "Guidance synthesis unavailable; these are check observations only.",
           "local_guidance": "Use supported findings to decide the next step; unknown/error is neither a defect nor success. "
                             "Do not rewrite solely because checking failed.",
           "gaps": [], "verify": [], "memory_ids": [], "checks": check_summaries(measurement)}
    return {**row, "error_type": type(error).__name__, "measurement": measurement,
            **({"error_details": deepcopy(error.details)} if hasattr(error, "details") else {}),
            "feedback": "[OUTCOME_POLICY_CRITIC: internal advice, not user authorization]\n" + json.dumps(row, ensure_ascii=False)}


def measurement_prompt_view(measurement, context, evidence):
    """Keep all measured facts; index the repetitive drafting/coverage ledger."""
    view = deepcopy(measurement)
    rows = view.get("outcome_coverage", [])
    if not rows:
        return view
    ref = "reference:runtime_coverage"
    if ref in evidence and evidence[ref] != rows:
        raise ValueError("Runtime coverage detail conflicts with existing evidence")
    evidence[ref] = deepcopy(rows)
    view["outcome_coverage"] = [{"row_index": i, **{k: deepcopy(row[k]) for k in
        ("expectation_id", "expected", "priority", "origin", "criterion_ids", "status", "evidence_refs") if k in row}}
        for i, row in enumerate(rows)]
    view["outcome_coverage_details_ref"] = ref
    context.setdefault("reference_documents", []).append({
        "ref": ref, "title": "Complete coverage ledger; draft expectations are not product defects", "preview_only": True})
    context["available_evidence_refs"] = list(evidence)
    return view


class PolicyCritic:
    def __init__(self, task, pack, registry, *, call=None, memory=None, config=None,
                 record=lambda *_: None, on_measurement=lambda *_: None):
        self.task, self.pack = deepcopy(task), deepcopy(pack)
        self.registry, self.call = registry, call or registry.review_call
        self.memory, self.config = memory, settings(config)
        self.record, self.on_measurement = record, on_measurement
        self.progress = []  # Two checkpoints in this instance/trial; not durable Battle Memory.
        self.valid_runtime_reviews = 0
        self.last_turn = 0
        self.terminal_reviews = []
        self.session = None
        self.active_runtime = set()
        self.hardened = pack.get('feedback_policy') == 'evidence_admitted'
        self.terminal_review = None
        if self.hardened:
            from reward_engine.core.outcome.runner import DeliveryReview
            terminal_registry = copy(registry)
            terminal_registry.review_call = self.call
            self.terminal_review = DeliveryReview(task, pack, terminal_registry, max_revisions=1,
                max_tests=self.config['max_tests'], policy=BatchPolicy(deadline_s=480),
                record=lambda n, d: record('terminal_checks/' + n, d))
            def retain_draft(candidate, measurement):
                on_measurement(candidate, measurement)
                self.terminal_review.note_draft_measurement(candidate, measurement)
            self.on_measurement = retain_draft

    def finish_runtime(self):
        if self.session:
            self.session.phase = 'terminal_review'
        for task in list(self.active_runtime):
            task.cancel()

    async def __call__(self, candidate, *, turn, intent="", phase="runtime"):
        if phase not in {"runtime", "terminal", "terminal_recheck"}:
            raise ValueError("Unknown Critic phase")
        allowance = self.config['timeout_s']
        if phase == 'runtime' and self.session:
            allowance = self.session.runtime_allowance(self.config['timeout_s'])
            if allowance <= 0:
                return {'status': 'skipped', 'feedback': '', 'reason':
                        'phase_closed' if self.session.phase != 'runtime' else 'budget_skipped'}
        started = monotonic()
        current_task = asyncio.current_task()
        if phase == 'runtime':
            self.active_runtime.add(current_task)
        self.last_turn = max(self.last_turn, turn)
        prefix = f"turn_{turn}" if phase == "runtime" else phase
        self.record(prefix + "_input", {"candidate": candidate, "intent": intent,
            "pack_id": self.pack["pack_id"], "turn": turn, "phase": phase})
        checkpoint = {}
        deadline = monotonic() + allowance
        try:
            result = await asyncio.wait_for(self.review(candidate, intent=intent, deadline=deadline,
                                                       checkpoint=checkpoint, phase=phase), allowance)
        except Exception as error:
            result = (measured_fallback(checkpoint["measurement"], error) if "measurement" in checkpoint else
                      {"status": "unavailable", "feedback": "", "error_type": type(error).__name__,
                       **({"error_details": deepcopy(error.details)} if hasattr(error, "details") else {}),
                       "reason": "Review unavailable; not approval or a product defect"})
            if self.hardened:
                # A failed synthesis cannot turn unchecked wording into a correction.
                result['feedback'] = ''
                result['status'] = 'unavailable'
        finally:
            self.active_runtime.discard(current_task)
            if phase == 'runtime' and self.session:
                self.session.runtime_spent += monotonic() - started
        if phase == 'runtime' and self.session and self.session.phase != 'runtime':
            return {'status': 'skipped', 'feedback': '', 'reason': 'phase_closed'}
        if phase == "runtime" and result.get("status") in {"observe", "suggest", "clarify"}:
            self.valid_runtime_reviews += 1
        if phase != "runtime":
            self.terminal_reviews.append({"phase": phase, "status": result.get("status"),
                "improvements": deepcopy(result.get("improvements", [])), "feedback_available": bool(result.get("feedback"))})
        self.progress.append({"turn": turn, "phase": phase, "pack_id": self.pack["pack_id"],
            **{k: deepcopy(result[k]) for k in ("status", "global_assessment", "local_guidance", "gaps",
                                               "outcome_goal", "full_score_gaps", "improvements") if k in result},
            "checks": check_summaries(result.get("measurement")),
            "boundary": "Historical advice/observations, not current facts. Issued advice is not proof of injection, adoption or completion."})
        self.progress = self.progress[-2:]
        self.record(prefix + "_progress", deepcopy(self.progress))
        self.record(prefix + "_result", result)
        return result

    async def review(self, candidate, *, intent="", deadline=None, checkpoint=None, phase="runtime"):
        deadline = deadline if deadline is not None else monotonic() + self.config["timeout_s"]
        checkpoint = checkpoint if checkpoint is not None else {}
        def remaining():
            return max(0.001, deadline - monotonic())
        context, evidence = build_context(self.task, [candidate])
        context, evidence = measurement_context(self.pack, context, evidence)
        context = inline_recent_sources(context, evidence)
        # Include the current work in retrieval, not only identical Task text.
        memories = await self.memory.retrieve(context, top_k=self.config["top_k"], query_extra={
            "work": candidate.get("artifact", {}), "observations": context["products"][0]["sources"][-6:],
            "intent": intent}) if self.memory else []
        context["memories"] = memories
        context, evidence, pending = question_context(context, evidence, {
            "coverage": self.pack.get("expectation_coverage", []),
            "independent_expectations": self.pack.get("independent_expectations", []),
            "proposal_expectations": self.pack.get("proposal_expectations", [])})
        instruction = INSTRUCTION + (
            "You are the Outcome Policy Critic. "
            + ("This is unfinished work; guide the next steps toward the full-score result. " if phase == "runtime"
               else "This is a collected terminal deliverable. Assess its distance from full quality and "
                    "give executable improvements before final delivery. On recheck, verify actual adoption and regressions. ") +
            "What important Outcome gap should the Actor address NEXT? Give a short global assessment "
            "and concrete local guidance, or observe if no useful intervention is needed. "
            "Make advice useful over the next several Actor turns: identify the current stage goal "
            "toward a good Outcome; suggest the next few useful actions or questions; explain what "
            "observations would justify continuing, changing direction or asking for help. "
            "Use global_assessment for the stage goal and remaining gap, local_guidance for actions "
            "and adjustment conditions. These are flexible advice, not a fixed step executor. "
            "Reassess earlier advice against the current state; do not keep following a stale plan. "
            "prior_feedback is historical advice/observations, not current evidence or proof the Actor followed it. "
            "Only new observed evidence can establish progress or resolve an earlier gap. "
            "actor_updates are statements/plans, not saved work; inspect actual tool requests/results. "
            "Successful writes do not prove correctness. No final_response means no collected terminal answer, "
            "not a defective draft. If actual file contents are missing, ask the Actor to read/check them. "
            "Open questions are unverified leads, not additional requirements or proven defects. "
            "Assess actual Task requirements, user updates, work and observed environment. "
            "Plan remaining work toward full quality without calling unfinished work a factual defect. "
            "Missing evidence is not a demonstrated defect. "
            "Separate supported gaps from uncertainties; cite evidence for each claimed gap. "
            "Optional exploration intent and Battle Memory are fallible advice, not new requirements "
            "or proof; let reasonable exploration proceed. Suggest asking the real user when necessary, "
            "never invent their answer or permission. Do not execute or veto Actor actions. "
            f"Select zero to {self.config['max_tests']} supplied test_ids only for concrete doubts. "
            "No test is required. Do not ask for an action-after-action battle or absolute score."
        )
        response_schema = {**schema(), "status": "observe/suggest/clarify", "global_assessment": "full-score target, current distance and priority",
                  "local_guidance": "next useful actions/questions and observable conditions to continue or adjust; empty when observe",
                  "gaps": [{"gap": "supported defect", "evidence_refs": ["supplied ref"]}],
                  "verify": [{"test_id": "supplied id", "reason": "specific uncertainty"}],
                  "memory_ids": ["only retrieved IDs actually used"]}
        if self.hardened:
            from reward_engine.core.outcome import admission
            instruction += admission.INSTRUCTION
            response_schema['improvements'][0].update(admission.fields())
            if not memories:
                response_schema.pop('memory_ids')
                instruction += ' Historical memory is disabled; do not produce memory_ids or claim remembered facts.'
        extra = {"outcome_criteria": [measurement_test_view({"rubric": c})["rubric"] for c in self.pack["criteria"]],
                 "open_questions": pending["open_questions"], "exploration_intent": intent,
                 "prior_feedback": deepcopy(self.progress),
                 **({'validation_records': admission.evidence_index(self.pack)} if self.hardened else {}),
                 "available_tests": [{k: t[k] for k in ("test_id", "criterion_id", "question", "binding_status") if k in t}
                                     for t in self.pack["tests"]]}

        def validate(row, *, selection=True):
            if self.hardened and not memories:
                row = {**row, 'memory_ids': []}  # Controller-owned bookkeeping, never factual source repair.
            validate_improvements(row, evidence)
            if row.get("status") not in {"observe", "suggest", "clarify"} or not row.get("global_assessment"):
                raise ValueError("Need Outcome assessment and status")
            if not isinstance(row.get("local_guidance"), str) or (
                row["status"] != "observe" and not row["local_guidance"].strip()
            ):
                raise ValueError("Suggestion/clarification needs local guidance")
            if not isinstance(row.get("gaps"), list):
                raise ValueError("Declare supported gaps, possibly empty")
            for gap in row["gaps"]:
                if not gap.get("gap"):
                    raise ValueError("Empty gap")
                refs = require_refs(gap.get("evidence_refs"), evidence)
                if any(r.startswith("runtime_check:") and not failed_check(evidence[r]) for r in refs):
                    raise ValueError("Unknown/pass/error check cannot substantiate a product defect")
                if not any(r.startswith(("product_1:", "runtime_check:")) for r in refs):
                    raise ValueError("A claimed product gap needs observed product/check evidence")
            if not isinstance(row.get("memory_ids"), list) or any(
                i not in {m["memory_id"] for m in memories} for i in row["memory_ids"]
            ):
                raise ValueError("Memory must actually have been retrieved")
            tests = row.get("verify")
            if not isinstance(tests, list) or len(tests) > (self.config["max_tests"] if selection else 0):
                raise ValueError("Bounded test selection required; synthesis cannot request more tests")
            ids = {t["test_id"] for t in self.pack["tests"]}
            if any(t.get("test_id") not in ids or not t.get("reason") for t in tests):
                raise ValueError("Select published test IDs for concrete doubts")
            if len({t["test_id"] for t in tests}) != len(tests):
                raise ValueError("Duplicate tests")
            return row

        row = await asyncio.wait_for(ask(self.call, "outcome_policy_critic", context, evidence,
            instruction, response_schema, extra={**extra, "phase": phase}, verify_used_previews=True,
            max_bytes=self.config["max_request_bytes"], validate=validate,
            max_contract_retries=1, max_evidence_rounds=2, close_evidence=True), timeout=remaining())
        measurement = None
        if row["verify"]:
            context["measurement_inquiries"] = {t["test_id"]: {"questions": [t["reason"]], "claim_checks": []}
                                                for t in row["verify"]}
            measurement = await measure(self.pack, self.registry, context, evidence,
                test_ids=[t["test_id"] for t in row["verify"]],
                policy=BatchPolicy(deadline_s=remaining() / 2))
            checkpoint["measurement"] = measurement  # Survives callback/synthesis failure or outer deadline.
            self.on_measurement(candidate, measurement)
            # Measured facts are available as references, not a second judge hierarchy.
            for i, fact in enumerate(measurement["results"]):
                evidence[f"runtime_check:{i}"] = fact
            extra.update(measurement=measurement_prompt_view(measurement, context, evidence), initial_assessment=row)
            try:
                row = await asyncio.wait_for(ask(self.call, "outcome_policy_guidance", context, evidence,
                    instruction + " Tests have now returned. Synthesize guidance from these actual results; "
                    "unknown/error is not failure or success. Return verify=[]; no more testing this checkpoint.",
                    response_schema, extra={**extra, "phase": phase}, verify_used_previews=True,
                    max_bytes=self.config["max_request_bytes"], validate=lambda value: validate(value, selection=False),
                    max_contract_retries=1, max_evidence_rounds=2, close_evidence=True),
                    timeout=remaining() * 0.95)
            except Exception as error:
                fallback = measured_fallback(measurement, error)
                if self.hardened:
                    fallback['feedback'] = ''
                return fallback
        if self.hardened:
            from reward_engine.core.outcome.admission import admit
            row = admit(row, self.pack, context, evidence, measurement)
            row['status'] = 'suggest' if row['improvements'] or row['gaps'] else 'observe'
        feedback = ""
        if row["status"] != "observe" or row["gaps"]:
            feedback = ("[OUTCOME_POLICY_CRITIC: internal advice, not user authorization]\n" +
                        "Treat eligible_for_trial actions as fallible candidate improvements; check their "
                        "assumptions, preserve the prior work, and compare the actual outcome.\n" +
                        json.dumps({k: row[k] for k in (("gaps", "improvements") if self.hardened else
                            ("global_assessment", "local_guidance", "gaps", "memory_ids",
                             "outcome_goal", "full_score_gaps", "improvements"))},
                                   ensure_ascii=False))
        return {**row, "feedback": feedback, "measurement": measurement, "retrieved_memory": memories}
