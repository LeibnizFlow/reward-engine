"""Bind downstream rejection to this call's receipt across protocol wrappers."""

from contextvars import ContextVar
from functools import wraps

ACTIVE_RESPONSE = ContextVar("outcome_response_receipt", default=None)
PENDING_REJECTIONS = ContextVar("outcome_pending_response_rejections", default=None)


def _reject(callback, error):
    pending = PENDING_REJECTIONS.get()
    if pending is None:
        callback(error)
    else:
        pending.append((callback, error))


class ResponseReceipt:
    def __init__(self):
        self.on_reject = None

    def reject(self, error):
        if self.on_reject is not None:
            _reject(self.on_reject, error)


def register_rejection(callback):
    receipt = ACTIVE_RESPONSE.get()
    if receipt is not None:
        receipt.on_reject = callback


def defer_rejections(call):
    """Let a draft consumer decide whether a diagnostic is terminal or retained.

    Successful draft admission does not certify its contents as a final standard;
    it only preserves its deterministic replay for the downstream validator.
    """
    @wraps(call)
    async def scoped(*args, **kwargs):
        pending = []
        token = PENDING_REJECTIONS.set(pending)
        failed = False
        try:
            return await call(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            PENDING_REJECTIONS.reset(token)
            if failed:
                for callback, error in pending:
                    _reject(callback, error)
    return scoped
