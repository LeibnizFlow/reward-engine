"""Deterministic Reduce response for integration fixtures, not a quality judge."""


def keep_groups(request):
    assert request["mode"] == "task_review_group_reduce"
    groups = request["groups"]
    coverage = []
    for need in request["task_needs"]:
        ids = [
            g["group_id"]
            for g in groups
            if g["map_disposition"] == "retained"
            and any(need["need_id"] in c["need_ids"] for c in g["cards"])
        ]
        # Explicit deterministic mock of a Reduce verdict, not a production
        # coverage rule. Do not inherit a single-source reader's absence claim.
        coverage.append(
            {
                "need_id": need["need_id"],
                "group_ids": ids,
                "status": "related_knowledge" if ids else "unknown",
            }
        )
    return {
        "group_decisions": [
            {
                "group_id": g["group_id"],
                "basis_card_ids": [
                    c["id"] for c in g["cards"] if not c.get("context_only")
                ][:1],
                "disposition": g["map_disposition"],
                "reason": "Keep the source reader's selection in this offline fixture.",
                "reason_kind": "out_of_scope"
                if g["map_disposition"] == "excluded"
                else "useful",
                "duplicate_of": None,
                **({"use": "procedure"} if g["map_disposition"] == "retained" else {}),
            }
            for g in groups
        ],
        "task_need_coverage": coverage,
    }
