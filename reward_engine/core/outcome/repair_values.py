"""Explicit, content-bound restoration offers from exact retain diagnostics."""

from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint


def retain_field_locations(base, diagnostics):
    """Locate diagnosed parent fields; never apply or infer a semantic repair."""
    locations = {}

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for mismatch in value.get("retain_mismatches", []):
                target = mismatch.get("target", {})
                matches = [(i, row) for i, row in enumerate(base.get("criteria", []))
                           if isinstance(row, dict) and row.get("id") == target.get("criterion_id")]
                if len(matches) != 1:
                    continue
                index, criterion = matches[0]
                for difference in mismatch.get("field_differences", []):
                    field = difference.get("field")
                    # text/clause are semantic projections, not parent field paths.
                    if not isinstance(field, str) or field in {"text", "clause"}:
                        continue
                    present = field in criterion
                    if difference.get("target_present") is not present:
                        continue
                    source_present = difference.get("source_present")
                    if source_present is True and "source_value" in difference:
                        op = "replace" if present else "add"
                    elif source_present is False and present:
                        op = "remove"
                    else:
                        continue
                    row = {"path": ["criteria", index, field],
                           "source_present": source_present, "target_present": present,
                           "op_for_exact_restoration": op}
                    if source_present:
                        row["source_value"] = deepcopy(difference["source_value"])
                    # Several retained parts may diagnose the same parent field.
                    # Keep conflicting source values separate for model review.
                    locations[fingerprint(row)] = row
            for key, child in value.items():
                if key != "retain_mismatches":
                    visit(child)

    visit(diagnostics)
    return list(locations.values())


def restoration_values(base, diagnostics):
    offers = {}

    def offer(path, value):
        row = {"path": path, "value": deepcopy(value)}
        identity = fingerprint(row)
        offers[identity] = {"id": identity, **row}

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for mismatch in value.get("retain_mismatches", []):
                target = mismatch.get("target", {})
                matches = [(i, row) for i, row in enumerate(base.get("criteria", []))
                           if isinstance(row, dict) and row.get("id") == target.get("criterion_id")]
                if len(matches) != 1:
                    continue
                index, criterion = matches[0]
                for difference in mismatch.get("field_differences", []):
                    field = difference.get("field")
                    if not difference.get("source_present") or "source_value" not in difference:
                        continue
                    path = ["criteria", index]
                    if field == "clause":
                        part = target.get("part_index")
                        clauses = criterion.get("clauses", [])
                        if not isinstance(clauses, list) or type(part) is not int or not 0 <= part < len(clauses):
                            continue
                        path += ["clauses", part]
                        source, destination = difference["source_value"], clauses[part]
                        if isinstance(source, dict) and isinstance(destination, dict):
                            # Clause comparisons omit local identity fields. Copy
                            # diagnosed fields, never replace the whole clause.
                            for key, original in source.items():
                                if key == "source_quotes" and mismatch.get("invalid_source_quotes"):
                                    continue
                                if key in destination and destination[key] != original:
                                    offer([*path, key], original)
                        continue
                    elif isinstance(field, str) and field in criterion:
                        path.append(field)
                    else:
                        continue  # No guessed destination or missing-field insertion.
                    offer(path, difference["source_value"])
            for key, child in value.items():
                if key != "retain_mismatches":
                    visit(child)

    visit(diagnostics)
    return list(offers.values())


def resolve_restorations(edits, offers):
    """Resolve only model-selected values at their exact diagnosed destinations."""
    if not isinstance(edits, list):
        return edits  # The existing edit validator owns the envelope contract.
    values = {row["id"]: row for row in offers}
    result = deepcopy(edits)
    for edit in result:
        if not isinstance(edit, dict) or "value_from" not in edit:
            continue
        key = edit["value_from"]
        offer = values.get(key) if isinstance(key, str) else None
        if (offer is None or "value" in edit or edit.get("op") != "replace"
                or edit.get("path") != offer["path"]):
            raise ValueError("value_from must select an offered exact restoration at its bound replace path, without value")
        edit.pop("value_from")
        edit["value"] = deepcopy(offer["value"])
    return result
