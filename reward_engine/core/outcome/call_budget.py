"""One request budget shared across evidence reads, contract and transport recovery."""

from contextvars import ContextVar

ACTIVE = ContextVar("outcome_call_budget", default=None)


def consume():
    budget = ACTIVE.get()
    if budget is not None:
        if budget["requests"] >= 6:
            raise RuntimeError("Logical operation exhausted its six physical requests")
        budget["requests"] += 1
