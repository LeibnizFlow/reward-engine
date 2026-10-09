"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from copy import deepcopy


OLD = "Assume an unobserved runtime setting."


CORRECT = "Keep unknown runtime settings conditional."


REFINED = "Inspect actual runtime evidence before an unconditional claim."


def reduce_reply(request, criterion, *, defer_round=None):
    clauses, decisions = [], []
    number = request.get("construction_round", 1)
    for expected in request["independent_expectations"] + request["proposal_expectations"]:
        for part in expected.get("subrequirements", []):
            old = part["text"]
            new = CORRECT if old == OLD else REFINED if number == 2 and old == CORRECT else old
            decision = {"parts": [part["selection_id"]], "action": "retain", "criterion_id": criterion["id"]}
            if new != old:
                decision.update(action="rewrite", text=new, reason="The actual Task requires grounded claims",
                                evidence_refs=["task"])
            if number == defer_round and old == CORRECT:
                decision = {"parts": [part["selection_id"]], "action": "defer", "reason": "Need actual runtime evidence"}
            elif new not in clauses:
                clauses.append(new)
            decisions.append(decision)
    return {"criteria": [{**deepcopy(criterion), "subrequirements": clauses}], "gaps": [],
            "selection_protocol": "exact-alias-v1", "part_decisions": decisions}


def controlled_models(criterion, *, audit_hook=None, reducer_hook=None):
    calls = []
    def member(key):
        async def call(request):
            calls.append((key, deepcopy(request)))
            number = request.get("construction_round", 1)
            if number > 1:
                if audit_hook:
                    result = await audit_hook(key, request)
                    if result is not None:
                        return result
                result = deepcopy(request["previous_synthesis"])
                result["criterion_audit"] = [{"criterion_id": c["id"], "assessment": "keep",
                    "reason": "Checked every clause against the original Task", "evidence_refs": ["task"]}
                    for c in result["criteria"]]
                return result
            return {"criteria": [deepcopy(criterion)], "gaps": []}
        return call
    async def reducer(request):
        calls.append(("reduce", deepcopy(request)))
        if reducer_hook:
            result = await reducer_hook(request)
            if result is not None:
                return result
        return reduce_reply(request, criterion)
    return {key: member(key) for key in ("a", "b", "c")}, reducer, calls
