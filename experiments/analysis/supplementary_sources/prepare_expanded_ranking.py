"""Frozen Jury conversion used by the completed Jury diagnostic."""


def jury_scores(phase, cids):
    if phase["ranking"].get("status") != "compared":
        return [None] * len(cids), "jury_unavailable"
    ranks = {}
    mapping = phase["product_to_candidate"]
    unknown = phase["ranking"].get("unknown", [])
    for i, group in enumerate(phase["ranking"].get("groups", [])):
        for product in group:
            assert (
                product in mapping
                and mapping[product] not in ranks
                and product not in unknown
            )
            ranks[mapping[product]] = -i
    return [ranks.get(candidate) for candidate in cids], None
