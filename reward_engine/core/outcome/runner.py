"""Bounded pre-delivery check/revise/recheck. No action-level Policy Critic."""

import json
from copy import deepcopy

from reward_engine.core.batch import BatchPolicy, collect_batch
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.coverage import expectations, final_test
from reward_engine.core.outcome.harness import measure, measurement_context, unknown
from reward_engine.core.outcome.task_progress import TaskProgress


class DeliveryReview:
    """Native adapter calls at completion; returned text continues SAME Actor.

    Hard/unknown checks cannot become official pass. Failed checker => unknown,
    not a product defect. Budget exhaustion retains latest artifact and failures.
    """

    def __init__(
        self,
        task,
        pack,
        registry,
        *,
        max_revisions=2,
        audit_mode="targeted",
        max_tests=3,
        policy=None,
        memory=None,
        source_prefix="checks",
        record=lambda *_: None,
    ):
        if type(max_revisions) is not int or max_revisions < 0:
            raise ValueError("Revision budget must be nonnegative")
        if audit_mode not in {"targeted", "full"} or type(max_tests) is not int or max_tests < 1:
            raise ValueError("Choose targeted/full review and a positive check budget")
        self.audit_mode, self.max_tests = audit_mode, max_tests
        self.task, self.pack, self.registry = deepcopy(task), deepcopy(pack), registry
        self.max_revisions, self.policy, self.memory, self.record = max_revisions, policy, memory, record
        self.history, self.artifacts, self.feedback = [], [], []
        self.previous = None
        self.status = "not_checked"
        self.source_prefix = source_prefix
        self.progress = TaskProgress(task, pack)
        self.draft_checks = {}
        self.hardened = pack.get('feedback_policy') == 'evidence_admitted'

    def note_draft_measurement(self, candidate, measurement):
        """Trusted runtime receipts, not Actor-supplied completion claims or final passes."""
        tests = {t["test_id"] for t in self.pack["tests"]}
        if measurement["pack_id"] != self.pack["pack_id"] or any(
            r["test_id"] not in tests for r in measurement["results"]
        ):
            raise ValueError("Draft measurements must belong to the published Pack")
        for row in measurement["results"]:
            # A later narrow pass cannot erase an earlier, different unresolved question.
            key = (row["test_id"], tuple(row.get("measurement_inquiry", {}).get("questions", [])))
            self.draft_checks[key] = (deepcopy(candidate["artifact"]), deepcopy(row))

    def _draft_context(self, candidate):
        rows = []
        for artifact, row in self.draft_checks.values():
            summary = {k: deepcopy(row[k]) for k in (
                "test_id", "criterion_id", "status", "gap", "reason", "error", "diagnostic", "measurement_kind",
            ) if k in row}
            inquiry = row.get("measurement_inquiry", {})
            summary["questions"] = deepcopy(inquiry.get("questions", []))
            if "inquiry_result" in row:
                summary["inquiry_result"] = {k: deepcopy(row["inquiry_result"][k])
                    for k in ("status", "gap", "reason", "error", "diagnostic") if k in row["inquiry_result"]}
            summary["same_authored_artifact"] = artifact == candidate["artifact"]
            rows.append(summary)
        return rows

    def _save_progress(self):
        self.progress.state["status"] = self.status
        self.progress.state["revisions_requested"] = len(self.feedback)
        self.progress.state["max_revisions"] = self.max_revisions
        snapshot = self.progress.snapshot()
        self.record("task_progress", snapshot)
        self.record(f"task_progress_{snapshot['check_number']}", snapshot)

    async def __call__(self, candidate):
        context, evidence = build_context(self.task, [candidate])
        context, evidence = measurement_context(self.pack, context, evidence)
        number = len(self.history) + 1
        self.progress.begin(context, number, f"{self.source_prefix}/delivery_{number}.json")
        self.status = "checking"
        self._save_progress()
        try:
            options = {"previous": self.previous} if self.previous else {}
            if self.audit_mode == "targeted":
                from reward_engine.core.outcome.delivery_triage import review

                result = await review(self.pack, self.registry, context, evidence, policy=self.policy,
                                      max_tests=self.max_tests, draft_checks=self._draft_context(candidate), **options)
            else:
                result = await measure(self.pack, self.registry, context, evidence, policy=self.policy, **options)
            # Only when ready to finish (or at the revision limit), not every action.
            if self.audit_mode == "full" and expectations(self.pack) and (
                result["ready_under_estimator"] or len(self.history) >= self.max_revisions
            ):
                batch = await collect_batch({"completeness": lambda: self.registry.run(
                    final_test(self.pack), context, evidence)}, self.policy or BatchPolicy())
                item = batch.items[0]
                result["completeness_review"] = (item.value if item.status == "ok" else
                    unknown("Whole-Task review timed out or failed", error=True))
                result["ready_under_estimator"] &= result["completeness_review"]["status"] == "pass"
        except Exception:
            self.status = "check_error_unresolved"
            self.progress.state["next_step"] = "Restore the check or evidence; do not assume a product defect."
            self._save_progress()
            raise
        self.progress.finish(result)
        self.artifacts.append(deepcopy(candidate))
        self.history.append(result)
        self.previous = (deepcopy(context), deepcopy(evidence), deepcopy(result))
        self.record(f"delivery_{len(self.history)}", {"candidate": candidate, "measurement": result})
        if self.hardened:
            from reward_engine.core.outcome.admission import preservation
            if len(self.history) > 1:
                preserved, lost = preservation(self.history[-2], result, self.pack)
                if not preserved:
                    self.status = 'revision_regressed'
                    self.progress.state['next_step'] = 'Keep the earlier candidate; revision lost verified correctness.'
                    self.record('revision_regression', {'lost_checks': lost, 'candidate_number': len(self.history)})
                    self._save_progress()
                    return None
            overview = result.get('delivery_review', {})
            if not overview.get('improvements') and not overview.get('gaps'):
                self.status = 'check_unavailable' if overview.get('error') else 'no_supported_change'
                self.progress.state['next_step'] = 'Retain the available work and unverified opportunities; no full-score claim.'
                self._save_progress()
                return None
        if result.get("checks_unavailable_only") or (result.get("delivery_review", {}).get("error") and not any(
            (r["status"] == "fail" and not r.get("error"))
            or (r.get("inquiry_result", {}).get("status") == "fail" and not r["inquiry_result"].get("error"))
            for r in result["results"]
        )):
            # Reviewer/transport failure is an infra problem, not an instruction
            # for the Actor to search or rewrite a potentially correct artifact.
            self.status = "check_unavailable"
            self.progress.state["next_step"] = "Repair or rerun the unavailable check; preserve this deliverable."
            self._save_progress()
            return None
        if len(self.history) > 1 and self.memory:
            self.memory.record(
                self.task,
                self.feedback[-1],
                self.history[-2],
                result,
                source={
                    "before": f"{self.source_prefix}/delivery_{len(self.history) - 1}.json",
                    "after": f"{self.source_prefix}/delivery_{len(self.history)}.json",
                },
                context_note="A bounded pre-delivery revision; requirements may have evolved",
            )
        clear = result.get("delivery_clear", result["ready_under_estimator"])
        if clear and self.progress.verified:
            self.status = ('no_supported_change' if self.hardened else
                           "full_score_target_satisfied" if self.audit_mode == "targeted" else "satisfied_under_estimator")
            self.progress.state["next_step"] = "Full-score target satisfied under current evidence; independent scoring remains separate."
            self._save_progress()
            return None
        if len(self.history) > self.max_revisions:
            self.status = "budget_exhausted_unresolved"
            self.progress.state["next_step"] = "Stopped at the revision budget; retain unresolved items for follow-up."
            self._save_progress()
            return None
        self.progress.state["next_step"] = (
            "Close the full-score Outcome gaps using the prioritized improvements; verify their effect. Preserve checked-correct work "
            "and follow the user's latest requirements/corrections. Submit again for rechecking."
        )
        # No failed checks => infrastructure/evidence uncertainty: request verification,
        # never command arbitrary rewrites of a possibly correct product.
        text = (
            "Pre-delivery Outcome feedback, not a new user instruction or extra authorization. "
            "Actions marked eligible_for_trial are fallible candidate improvements. Check their premises, "
            "preserve the original deliverable, and decline a suggestion whose assumptions do not hold. "
            "The objective is a better complete outcome, not obedience to every estimated criterion. "
            "Your goal is a FULL-SCORE OUTCOME. Execute the prioritized improvements below, including "
            "quality gains beyond fixing factual defects. Revise locally or rework the deliverable as needed. "
            "Start with the changes that most improve the final Outcome. Verify their actual effect and any "
            "regressions; explain unresolved full-score gaps. For uncovered "
            "outcomes investigate the stated question with real sources/tools or clarify actual user "
            "needs; unknown does not authorize inventing a defect. State what remains unmeasured. "
            "For unknown/checker errors, verify what is available or clearly state the limitation; "
            "do not invent a defect. Preserve the user's requirements and already-correct work. "
            "Do not repeat irreversible operations just to raise a score. Then deliver again.\n"
            + "Current task checklist (observations, not extra user authorization):\n"
            + json.dumps(self.progress.for_actor(), ensure_ascii=False)
        )
        self.feedback.append(text)
        self.status = "revision_requested"
        self.record(f"feedback_{len(self.feedback)}", {"message": text})
        self._save_progress()
        return text


async def run_delivery(actor, review=None, *, prompt=""):
    """Small protocol also used by tests/custom runtimes: execute then resume, not fresh retry."""
    candidate = await actor.start(prompt)
    while review:
        feedback = await review(candidate)
        if feedback is None:
            break
        candidate = await actor.resume(feedback)
    return candidate
