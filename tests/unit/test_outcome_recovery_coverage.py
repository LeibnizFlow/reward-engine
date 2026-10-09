from copy import deepcopy

import pytest

from reward_engine.core.outcome.harness import unresolved_measurements


def fixture():
    def probe(refs, status="observed"):
        return {
            "tool": "code.run",
            "status": status,
            "arguments": {"input_refs": refs, "program": "result = 1"},
        }

    receipts = {
        "before": probe(["a"]),
        "failed": probe(["a", "b"], "unknown"),
        "after_b": probe(["b"]),
        "after_a": probe(["a"]),
    }
    result = {
        "evidence_refs": ["before", "after_b", "after_a"],
        "measurement_recovery": [
            {
                "failed_ref": "failed",
                "replacement_refs": ["before", "after_b", "after_a"],
                "reason": "Split the failed combined read into two subsequent successful probes.",
            }
        ],
    }
    return result, receipts


def test_earlier_supplement_does_not_erase_later_complete_coverage():
    result, receipts = fixture()
    original = deepcopy((result, receipts))
    assert unresolved_measurements(result, receipts) == set()
    assert (
        result,
        receipts,
    ) == original  # Failure history and model text remain intact.


@pytest.mark.parametrize(
    "kind",
    [
        "earlier_only",
        "partial_later",
        "uncited",
        "failed_replacement",
        "wrong_tool",
        "nonexistent",
        "unrelated",
    ],
)
def test_recovery_still_requires_cited_later_success_over_all_failed_inputs(kind):
    result, receipts = fixture()
    refs = result["measurement_recovery"][0]["replacement_refs"]
    if kind == "earlier_only":
        refs[:] = ["before"]
        receipts["before"]["arguments"]["input_refs"] = ["a", "b"]
    elif kind == "partial_later":
        refs[:] = ["before", "after_b"]  # Earlier a cannot cover the missing later a.
    elif kind == "uncited":
        result["evidence_refs"].remove("after_a")
    elif kind == "failed_replacement":
        receipts["after_a"]["status"] = "unknown"
    elif kind == "wrong_tool":
        receipts["after_a"]["tool"] = "document.read"
    elif kind == "nonexistent":
        refs.append("missing")
    else:
        receipts["after_a"]["arguments"]["input_refs"] = ["other"]
    assert "failed" in unresolved_measurements(result, receipts)
