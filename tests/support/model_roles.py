"""Synthetic role models shared by offline campaign fixtures."""
PANEL = ["test-panel-a", "test-model-a", "test-panel-c"]

def role_mapping(*, include_final=False):
    roles = {
        r: "test-reasoner"
        for r in (
            "actor",
            "jury_reducer",
            "hypothesis_chooser",
            "harness_designer",
            "reward_calibration_judge",
        )
    }
    for family in ("jury", "hypothesis"):
        roles.update({f"{family}_member_{i}": m for i, m in enumerate(PANEL, 1)})
    if include_final:
        roles["final_selector"] = "test-reasoner"
    return roles
