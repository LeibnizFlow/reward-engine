"""Current unfinished work, derived from observations; no additional judge or score."""

from copy import deepcopy

from reward_engine.core.outcome.coverage import report


class TaskProgress:
    """One delivery-review session only. Fresh actors never inherit completion claims."""

    def __init__(self, task, pack):
        self.pack = deepcopy(pack)
        self.state = {
            "task_id": task["task_id"],
            "pack_id": pack["pack_id"],
            "task_prompt": task["prompt"],
            "user_updates": [],
            "check_number": 0,
            "status": "not_checked",
            "items": [],
            "next_step": "Submit a real deliverable for checking.",
            "boundary": "Current Pack checks only; not proof all Task requirements or official grading pass.",
        }

    def begin(self, context, check_number, source):
        product = context["products"][0]
        # Keep actual user statements (including corrections), never assistant suggestions.
        # Previously received requirements survive a later collector omitting history.
        for update in product["user_updates"]:
            if update["text"] not in self.state["user_updates"]:
                self.state["user_updates"].append(update["text"])
        criteria = {c["id"]: c for c in self.pack["criteria"]}
        items = [
            {
                "test_id": t["test_id"],
                "criterion_id": t["criterion_id"],
                "standard": criteria[t["criterion_id"]].get("standard", ""),
                "question": t.get("question", ""),
                "priority": criteria[t["criterion_id"]].get("priority", "important"),
                "status": "unknown",
                "reason": "Awaiting measurement of the current deliverable; old passes are not reused.",
                "gap": "",
                "evidence_refs": [],
                "source": source,
            }
            for t in self.pack["tests"]
        ]
        covered = {t["criterion_id"] for t in items}
        for key, criterion in criteria.items():
            if key not in covered:
                items.append({"criterion_id": key, "standard": criterion.get("standard", ""),
                              "status": "unknown", "reason": "No Harness test bound to this criterion."})
        if product.get("reply_pending"):
            items.append({"requirement": "Respond to the actual user's latest message",
                          "status": "unknown", "reason": "Collected user reply is still pending."})
        current_updates = {u["text"] for u in product["user_updates"]}
        if any(text not in current_updates for text in self.state["user_updates"]):
            items.append({"requirement": "Check against previously received user requirements too",
                          "status": "unknown", "reason": "Current evidence omitted earlier user messages; "
                          "the saved requirements remain applicable until corrected by the user."})
        self.state.update(check_number=check_number, status="checking", items=items,
                          outcome_coverage=report(self.pack, []),
                          next_step="Wait for current evidence; do not reuse earlier completion claims.")

    def finish(self, measurement):
        targeted = measurement.get("review_mode") == "targeted"
        self.state["outcome_coverage"] = deepcopy(measurement.get("outcome_coverage", []))
        for row in self.state["outcome_coverage"]:
            if row["status"] == "unknown":
                self.state["items"].append({**deepcopy(row),
                    "reason": "Outcome not covered by a usable check: " + row["reason"], "gap": ""})
        if "completeness_review" in measurement:
            self.state["items"].append({"requirement": "Whole-Task outcome completeness",
                                        "priority": "important", **measurement["completeness_review"]})
        results = {row["test_id"]: row for row in measurement["results"]}
        for item in self.state["items"]:
            if targeted:
                item["blocking"] = "requirement" in item or item.get("test_id") in results
            if "test_id" not in item:
                continue
            row = results.get(item["test_id"])
            if row is None:
                item["reason"] = "No current result returned; this item remains unverified."
                continue
            for key in ("status", "gap", "reason", "evidence_refs", "error", "diagnostic", "measurement_kind", "claims"):
                if key in row:
                    item[key] = deepcopy(row[key])
        for row in measurement["results"]:
            if "inquiry_result" in row:
                self.state["items"].append({"requirement": "Resolve the specific measurement question",
                    "related_test_id": row["test_id"], "blocking": True,
                    "measurement_inquiry": deepcopy(row.get("measurement_inquiry", {})),
                    **deepcopy(row["inquiry_result"])})
        if targeted:
            overview = measurement["delivery_review"]
            for row in measurement.get("unresolved_checks", []):
                self.state["items"].append({"requirement": "Unresolved earlier check; not current product proof",
                    "related_test_id": row["test_id"], "blocking": True, "status": "unknown",
                    "reason": row.get("reason", "Check still unavailable"),
                    "previous_check": deepcopy(row)})
            self.state["items"].append({"requirement": "达成 Outcome Reward 还差什么？哪些主张值得核实？",
                "blocking": True, "status": "pass" if measurement["delivery_clear"] else "unknown",
                "outcome_status": overview["status"],
                "factual_assessment": overview.get("factual_assessment", "unknown"),
                "reason": overview["reason"], "gaps": deepcopy(overview["gaps"]),
                "verification_requests": deepcopy(overview["verify"]),
                "claim_checks": deepcopy(overview.get("claim_checks", [])),
                "outcome_goal": overview.get("outcome_goal", ""),
                "full_score_gaps": deepcopy(overview.get("full_score_gaps", [])),
                "improvements": deepcopy(overview.get("improvements", [])),
                "deferred_tests": measurement["deferred_tests"]})
        if self.pack.get('feedback_policy') == 'evidence_admitted':
            from reward_engine.core.outcome.admission import critical_tests
            critical = critical_tests(self.pack)
            for item in self.state['items']:
                item['blocking'] = (item.get('test_id') in critical or bool(item.get('improvements'))
                                    or 'requirement' in item and 'outcome_status' not in item
                                    and 'related_test_id' not in item)
                if 'outcome_status' in item:
                    # Audit the full proposal, but only admitted actions reach the Actor.
                    item['full_score_gaps'] = []
                    item['verification_requests'] = []
                    item['claim_checks'] = []
            self.state['advisory_improvements'] = deepcopy(measurement.get('delivery_review', {}).get('advisory_improvements', []))

    @property
    def verified(self):
        return bool(self.state["items"]) and all(i["status"] == "pass" for i in self.state["items"]
                                                if i.get("blocking", True))

    def snapshot(self):
        return deepcopy(self.state)

    def for_actor(self):
        """Compact current checklist, not the growing log or a new source of authority."""
        return {
            "task_prompt": self.state["task_prompt"],
            "actual_user_updates": deepcopy(self.state["user_updates"]),
            "remaining": [
                {k: deepcopy(v) for k, v in i.items() if k != "claims"}
                for i in sorted(self.state["items"], key=lambda i:
                    {"critical": 0, "important": 1, "minor": 2}.get(i.get("priority"), 1))
                if i["status"] != "pass" and i.get("blocking", True)
            ],
            "unmeasured_count": sum(i["status"] == "unknown" and not i.get("blocking", True)
                                    for i in self.state["items"]),
            "verified_under_current_checks": [
                {k: deepcopy(i[k]) for k in ("test_id", "criterion_id", "standard", "source") if k in i}
                for i in self.state["items"] if i["status"] == "pass"
            ],
            "next_step": self.state["next_step"],
            "boundary": self.state["boundary"],
        }
