"""Revision-bound failure history with explicit pause or task isolation policies."""

from collections import Counter
from pathlib import Path

from reward_engine.adapters.workbuddy.io import read, write


class FailureWindow:
    def __init__(self, config):
        policy = config.get("failure_policy")
        if policy not in (None, "pause_repeated_v1", "isolate_tasks_v1"):
            raise ValueError("Unknown failure policy")
        self.enabled = policy is not None
        self.isolated = policy == "isolate_tasks_v1"
        self.revision = config.get("identity")
        if self.enabled and not config.get("implementation_revision"):
            raise ValueError("Failure policy requires explicit implementation revision")
        self.path = (
            Path(config.get("implementation_revision", config.get("output", ".")))
            / "failure_window.json"
        )
        self.state = {"revision": self.revision, "terminal": [], "paused": False}
        if self.enabled and self.path.exists():
            self.state = read(self.path)
            if self.state.get("revision") != self.revision:
                raise ValueError("Failure window revision identity changed")

    @property
    def paused(self):
        return self.enabled and self.state["paused"]

    def terminal(self, name, result):
        if not self.enabled:
            return
        failed = isinstance(result, BaseException) or (
            isinstance(result, dict) and result.get("status") in {"failed", "partial"}
        )
        value = result if isinstance(result, dict) else {}
        row = {
            "task": name,
            "failed": failed,
            "stage": value.get("stage", getattr(result, "stage", "task")),
            "error_code": value.get(
                "error_code",
                value.get(
                    "error_type",
                    getattr(result, "error_code", type(result).__name__)
                    if failed
                    else "",
                ),
            ),
        }
        self.state["terminal"] = [*self.state["terminal"], row][-10:]
        failures = [r for r in self.state["terminal"] if r["failed"]]
        repeated = Counter((r["stage"], r["error_code"]) for r in failures)
        if not self.isolated and (
            len(failures) >= 5 or any(count >= 3 for count in repeated.values())
        ):
            self.state["paused"] = True
            self.state["reason"] = {
                "failed_in_last_ten": len(failures),
                "repeated": [
                    {"stage": k[0], "error_code": k[1], "count": v}
                    for k, v in repeated.items()
                    if v >= 3
                ],
            }
        if failed:
            self.state["success_streak"] = 0
        else:
            self.state["success_streak"] = self.state.get("success_streak", 0) + 1
            if self.state["success_streak"] == 10:
                self.state["admission_limit"] = min(
                    128, self.state.get("admission_limit", 4) * 2
                )
                self.state["success_streak"] = 0
        write(self.path, self.state)
