"""Model-authored atomic edits for Jury drafts; the caller validates the result."""

import json
from copy import deepcopy

from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.jury_repair import apply_edits, diagnostics
from reward_engine.core.outcome.reducer_projection import resolve_output_texts
from reward_engine.core.outcome.repair_values import (
    resolve_restorations,
    restoration_values,
    retain_field_locations,
)

MODES = frozenset({"outcome_jury_map", "outcome_jury_reduce"})


class ProtocolRepairCall:
    def __init__(self, call, *, max_bytes=None):
        self.call = call
        self.pending = None
        self.pending_diagnostics = None
        self.max_bytes = max_bytes if max_bytes is not None else getattr(
            call, "max_request_bytes", getattr(call, "request_budget_bytes", 160000)
        )

    def __getattr__(self, name):
        return getattr(self.call, name)

    async def _call(self, request):
        from reward_engine.core.outcome.context import RequestBudgetExceeded
        from reward_engine.core.outcome.reducer_projection import expand_texts, share_texts
        from reward_engine.core.outcome.selection_view import project

        current = project(request)
        if current is not request:
            # The repair base and diagnostics were expanded to bind exact
            # semantic values. Re-intern repeated strings after projection,
            # using a fresh table rather than retaining stale request aliases.
            current = share_texts(expand_texts(current))
        if len(json.dumps(current, ensure_ascii=False).encode()) > self.max_bytes:
            raise RequestBudgetExceeded(current, self.max_bytes, stage="protocol_repair")
        return await self.call(current)

    async def __call__(self, request):
        from reward_engine.core.outcome.context import ResponseContractError

        recovery = request.get("response_recovery", {})
        base = recovery.get("rejected_response")
        syntax_retry = recovery.get("error_type") == "JSONDecodeError" and self.pending is not None
        repair_retry = recovery.get("details", {}).get("field") == "protocol_repair" and self.pending is not None
        if not recovery:
            self.pending = None
            self.pending_diagnostics = None
        elif syntax_retry or repair_retry:
            base = self.pending
        if (request.get("mode") not in MODES or not isinstance(base, dict)
                or not isinstance(base.get("criteria"), list) or not base["criteria"]
                or recovery.get("details", {}).get("criterion_field_repairs")):
            return await self._call(request)
        # A projected reducer request may intern repeated strings. Bind the
        # semantic draft, not request-local aliases that can change on a retry.
        base, _ = resolve_output_texts(request, base)
        self.pending = deepcopy(base)
        current = deepcopy(request)
        current["response_recovery"]["rejected_response"] = deepcopy(base)
        if syntax_retry or repair_retry:
            # Fixing the edit envelope must not erase the original reason that
            # the draft was rejected. Keep syntax and validation errors distinct.
            current["response_recovery"]["syntax_error" if syntax_retry else "repair_error"] = {
                "error_type": recovery["error_type"], "details": deepcopy(recovery.get("details", {})),
            }
        else:
            # Projection aliases are local to a request; preserve semantic
            # diagnostics before a syntax retry can introduce a different table.
            self.pending_diagnostics, _ = resolve_output_texts(request, recovery.get("details", {}))
        current["response_recovery"]["details"] = deepcopy(self.pending_diagnostics)
        identity = fingerprint(base)
        current["protocol_repair"] = {
            "schema": "jury-draft-edits/v1", "base_sha256": identity,
            "draft_response_schema": deepcopy(request.get("response", {})),
            "draft_instruction": request.get("instruction", ""),
            "instruction": "The repair base is response_recovery.rejected_response. "
            "All paths address that draft, with edits applied sequentially and atomically. "
            "Diagnostics in response_recovery.details are read-only and are not edit paths. "
            "Untouched fields retain their exact bytes/values. This is not permission to change a verdict or weaken a requirement.",
        }
        ledger = "revision_decisions" if request["mode"] == "outcome_jury_map" else "part_decisions"
        fields = ("source_parts", "action", "targets", "reason", "evidence_refs") if ledger == "revision_decisions" else (
            "parts", "action", "criterion_id", "clause_id", "text", "reason", "evidence_refs")
        decisions = base.get(ledger, [])
        current["protocol_repair"]["decision_fields"] = [
            {"path": [ledger, index], "existing_fields": sorted(row),
             "absent_fields": [field for field in fields if field not in row]}
            for index, row in enumerate(decisions if isinstance(decisions, list) else [])
            if isinstance(row, dict)
        ]
        current["protocol_repair"]["decision_instruction"] = (
            "Member revision_decisions use source_parts, action and targets. A changed decision "
            "uses one of these actions: retain, rewrite, merge, delete, defer, add. "
            "inherit_from is never a decision action: it is an entire criteria[i] object "
            "containing only inherit_from with the expectation_id. Its decision remains retain "
            "and targets the original criterion id. Restoring missing criterion fields requires "
            "editing that criterion or using this criteria[i] inheritance representation; changing "
            "only revision_decisions does not restore a missing criterion field. A changed decision "
            "needs reason and evidence_refs; its requirement text lives in the target criterion's "
            "clauses/subrequirements, not in a decision text field. Retain needs the original "
            "wording AND evidence/scope/priority/verification exactly; adding evidence is a change. "
            "Preserve the original via explicit restore_values edits, or explicitly justify a rewrite."
            if ledger == "revision_decisions" else
            "Reducer part_decisions use parts and action. An explicit rewrite requires text, "
            "reason and evidence_refs in the decision itself; text must match its final clause exactly."
        )
        current["protocol_repair"]["draft_preflight"] = diagnostics(base, [])
        reference_issues = []
        available = request.get("context", {}).get("available_evidence_refs", [])
        available = {ref for ref in available if isinstance(ref, str)}
        def check_refs(row, path):
            refs = row.get("evidence_refs")
            if (not isinstance(refs, list) or not refs
                    or any(not isinstance(ref, str) or ref not in available for ref in refs)):
                reference_issues.append({
                    "path": [*path, "evidence_refs"], "field_present": "evidence_refs" in row,
                    "reason": "Supply a nonempty list of exact available evidence handles supporting "
                    "this specific requirement. Criterion and clause citations are checked separately. "
                    "Do not invent support or add citations merely to pass validation. Adding evidence "
                    "to a retained requirement changes its provenance and needs a grounded rewrite "
                    "or exact restoration of the original.",
                })
        if available:
            for i, criterion in enumerate(base["criteria"]):
                if not isinstance(criterion, dict) or "inherit_from" in criterion:
                    continue
                check_refs(criterion, ["criteria", i])
                clauses = criterion.get("clauses", [])
                for j, clause in enumerate(clauses if isinstance(clauses, list) else []):
                    if isinstance(clause, dict):
                        check_refs(clause, ["criteria", i, "clauses", j])
        current["protocol_repair"]["reference_preflight"] = reference_issues
        target_catalog = []
        for criterion in base["criteria"]:
            if not isinstance(criterion, dict) or not isinstance(criterion.get("id"), str):
                continue
            clauses = criterion.get("clauses")
            parts = ([clause.get("text") for clause in clauses if isinstance(clause, dict)]
                     if isinstance(clauses, list) and all(isinstance(c, dict) for c in clauses)
                     else criterion.get("subrequirements", []))
            for index, part in enumerate(parts if isinstance(parts, list) else []):
                target_catalog.append({"criterion_id": criterion["id"], "part_index": index,
                                       "text": deepcopy(part)})
        current["protocol_repair"]["target_catalog"] = target_catalog
        locations = retain_field_locations(base, self.pending_diagnostics)
        if locations:
            current["protocol_repair"]["retain_field_locations"] = locations
        if request["mode"] == "outcome_jury_reduce":
            current["protocol_repair"]["edit_budget"] = {"max_edits": 64, "instruction":
                "If many fields of an existing object or list need correction, replace that parent "
                "once with its complete corrected value, preserving unaffected fields and list rows. "
                "For example, a corrected part_decisions list may be one replace edit. Every edit "
                "still needs its own reason. Do not emit more than 64 leaf edits; nested values "
                "inside one replacement are not separate edit operations."}
        current["response"] = {"base_sha256": "exact repair base hash", "edits": [
            {"op": "replace/remove/add", "path": ["draft key", 0, "field"],
             "value": "model-authored JSON value for replace/add", "reason": "grounded correction"}]}
        current["instruction"] = (
            "Correct ONLY the diagnosed invalid Jury draft against the same supplied Task and evidence. "
            "Return an object with exactly two top-level keys: base_sha256 and edits. "
            "Use the exact base_sha256 and one to 64 explicit edits, not the complete draft. "
            "Do not add type, response, status or other wrapper/format metadata; "
            "the API response_format type=json_object is transport configuration, not an output field. "
            "Each edit needs op, path and reason; replace/add also need value. Replace/remove paths must exist. "
            "The edit's reason is mandatory even when its path ends in reason: value supplies the "
            "decision's reason, while the separate edit reason explains the correction. "
            "Locate the invalid draft field before choosing an edit; the validation error's field "
            "names the failed contract, not necessarily the draft path that needs correction. "
            "Add may only create an absent object key in an existing parent; it cannot overwrite or insert array entries. "
            "To insert or append criteria, clauses, gaps or decisions, use replace on the existing list "
            "itself and author its complete new value, preserving every unaffected row. For example, "
            "replace path=[\"criteria\"] with the complete criteria list; never use add with a numeric "
            "last path component such as [\"criteria\",4]. To change an existing list element, use replace. "
            "Paths start at criteria, gaps, uncertainties, revision_decisions, part_decisions or "
            "selection_protocol in the rejected draft. Never edit diagnostic objects or the request. "
            "Use zero-based array indices. Perform replacements and field additions BEFORE removing "
            "array entries; perform those removals last, in descending index order. Earlier removals "
            "shift all later array indices. To add a missing field, "
            "use add or replace its existing parent object, preserving its other fields. "
            "The original contract remains in protocol_repair.draft_response_schema and draft_instruction. "
            "A missing part_decisions ledger cannot be replaced by revision_decisions. "
            "Use the mode-specific decision_instruction and decision_fields inventory in protocol_repair. "
            "For member revision targets, target_catalog shows the existing zero-based targets and "
            "their text. A target is not an insertion position: do not invent another part_index. "
            "Do not map an unsupported addition to an unrelated existing clause just to pass. "
            "An absent_fields entry can only be introduced with add, not replace; an existing_fields "
            "entry can be replaced or removed, not added. This inventory describes presence, not "
            "permission to add every optional field. Choose only grounded corrections. An inherited criterion "
            "can be replaced with {inherit_from: exact expectation_id} alone, without an id override. "
            "Do not guess references, silently change retain to rewrite, remove Task requirements to "
            "evade validation, or copy a withdrawn clause into a final criterion. Explicit rewrites still "
            "need model-authored reasons and supplied supporting evidence. Keep all unaffected criteria, "
            "clauses and decisions unchanged. The entire reconstructed draft must pass all original validators. "
            "If syntax_error is present, fix the edit JSON syntax while still addressing the original "
            "validation diagnostics in response_recovery.details; do not invent a new repair objective. "
            "If repair_error is present, correct the rejected envelope or edit operations against "
            "this same base and original diagnostics. No edits from the rejected batch were applied. "
            "Do not request new tools or evidence during this correction."
        )
        current["response_recovery"]["instruction"] = current["instruction"]
        restorations = restoration_values(base, self.pending_diagnostics)
        if restorations:
            current["protocol_repair"]["restore_values"] = restorations
            current["instruction"] += (
                " For exact retain restoration, prefer an explicit replace edit with value_from set "
                "to the matching protocol_repair.restore_values id, its exact path and your reason, "
                "omitting value. This copies that supplied original value byte-for-byte instead of "
                "retyping it. It is optional and never automatically applied. Verify the source before "
                "selecting it; it does not authorize a rewrite or bypass any validation."
            )
            current["response"]["edits"][0]["value_from"] = (
                "optional exact restore_values id for replace; use instead of value"
            )
            current["response_recovery"]["instruction"] = current["instruction"]
        if locations:
            current["instruction"] += (
                " Address protocol_repair.retain_field_locations: these are actual criterion field "
                "paths in this rejected draft, even when the error field is revision_decisions. "
                "To keep retain, explicitly restore each diagnosed original field's presence and "
                "value using the listed operation and your own edit reason. Missing is distinct "
                "from null. Replacing only revision_decisions cannot fix these criterion fields. "
                "These locations are read-only diagnostics, not automatically applied edits. "
                "Review the original evidence and all other diagnostics; a semantic change still "
                "requires an explicit justified rewrite and full validation."
            )
            current["response_recovery"]["instruction"] = current["instruction"]
        reply = await self._call(current)
        if not isinstance(reply, dict) or set(reply) != {"base_sha256", "edits"} or reply["base_sha256"] != identity:
            fields = set(reply) if isinstance(reply, dict) else set()
            expected = {"base_sha256", "edits"}
            actual_hash = reply.get("base_sha256") if isinstance(reply, dict) else None
            details = {
                "expected_fields": sorted(expected),
                "extra_fields": sorted(fields - expected, key=str),
                "missing_fields": sorted(expected - fields),
                "actual_type": type(reply).__name__,
                "expected_base_sha256": identity,
                "actual_base_sha256": actual_hash,
                "base_hash_matches": actual_hash == identity,
            }
            error = ResponseContractError(
                "protocol_repair",
                f"Invalid repair envelope ({details['actual_type']}): "
                f"extra fields {details['extra_fields']}; missing fields {details['missing_fields']}; "
                f"base hash {'matches' if details['base_hash_matches'] else 'does not match'}. "
                "Return a JSON object with exactly base_sha256 and edits, bound to the current "
                "protocol_repair.base_sha256. No edits from this rejected reply were applied.",
            )
            error.details.update(details)
            raise error
        try:
            edits = resolve_restorations(reply["edits"], restorations)
            return apply_edits(base, {"edits": edits}, allow_add=True, allowed_roots={
                "criteria", "gaps", "uncertainties", "revision_decisions", "part_decisions", "selection_protocol"})
        except (ValueError, TypeError, KeyError) as error:
            failure = ResponseContractError("protocol_repair", str(error))
            if getattr(error, "details", None):
                failure.details["edit_failure"] = {
                    **error.details, "repair_base": "response_recovery.rejected_response",
                }
            raise failure from error
