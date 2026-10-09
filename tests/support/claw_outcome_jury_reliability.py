"""Shared fixtures and reusable scenarios extracted without changing assertions."""

from reward_engine.core.outcome.context import build_context


from reward_engine.core.outcome.jury_drafts import (
    apply_part_decisions,
    compact_selection_inputs,
    expectations,
    resolve_parts,
)


def fixture():
    context, evidence = build_context({"task_id": "x", "prompt": "Return valid JSON"})
    c = {"id": "json", "kind": "hard", "origin": "task", "standard": "Valid JSON",
         "measurement_question": "Does it parse?", "evidence_refs": ["task"],
         "subrequirements": ["Valid JSON"]}
    inputs = {"proposals": [{}, {}], "independent_expectations": expectations([c], "a", "before"),
              "proposal_expectations": []}
    return context, evidence, c, inputs
