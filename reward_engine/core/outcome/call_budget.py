"""One request budget shared across evidence reads, contract and transport recovery."""

from contextvars import ContextVar

ACTIVE = ContextVar("outcome_call_budget", default=None)


def consume():
    budget = ACTIVE.get()
    if budget is not None:
        if budget["requests"] >= budget.get("max_requests", 6):
            limit = "six" if budget.get("max_requests", 6) == 6 else str(budget["max_requests"])
            raise RuntimeError(f"Logical operation exhausted its {limit} physical requests")
        budget["requests"] += 1
