"""Lossless vote-content normalization at the Claw transport boundary.

No new votes or reasons: unwrap an echoed response envelope, collapse a unique
duplicated ID prefix, and resolve actually observed source names to their trace
handles. Unknown names remain unknown and are rejected by the core parser.
"""
import json
import re
from copy import deepcopy

from reward_engine.core.outcome.context import ResponseContractError
from reward_engine.core.shared.provider_reply import json_answer_text


class EmptyModelReply(ResponseContractError):
    """The transport stopped without an answer; not a malformed verdict."""

    def __init__(self, reason):
        super().__init__('response', reason)


def decode_completed_reply(text, *, response_format=None):
    """Only for verified finish_reason=stop: normalize one clear syntax defect.

    Values are unchanged. An explicitly configured format echo may precede one
    complete object; other limited repairs are a root brace or delimiter typo.
    The caller records the untouched raw reply and this normalization. Callers
    omitting response_format keep the previous multi-object rejection behavior.
    """
    text = text.strip()
    if not text:
        raise EmptyModelReply("Completed model response has no answer")
    # Presentation-only recovery at the verified-stop boundary. Require ONE
    # already-complete JSON value: never close JSON, discard prose, or select
    # between answers when the surrounding Markdown is malformed.
    fence = re.fullmatch(r"(?::\s*)?```(?:json)?[ \t]*\n([\s\S]+)", text)
    if fence:
        body = fence.group(1).strip()
        if body.endswith("\n```"):
            body = body[:-4].rstrip()
        value = json.loads(body)
        return value, [{"operation": "unwrap_complete_json_presentation"}]
    text = json_answer_text(text)
    try:
        return json.loads(text), []
    except json.JSONDecodeError as error:
        if error.msg == 'Extra data' and response_format == {'type': 'json_object'}:
            # Recover only a verbatim echo of the configured transport format,
            # never choose between answers. Duplicate keys make the echo/answer
            # ambiguous, and the remaining object must parse without any repair.
            def unique_object(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise json.JSONDecodeError('Duplicate key in format-echo reply', text, 0)
                    value[key] = item
                return value

            def invalid_constant(_):
                raise json.JSONDecodeError('Non-JSON constant in format-echo reply', text, 0)

            decoder = json.JSONDecoder(object_pairs_hook=unique_object, parse_constant=invalid_constant)
            prefix, end = decoder.raw_decode(text)
            if prefix == response_format:
                answer = decoder.decode(text[end:].strip())
                if not isinstance(answer, dict) or answer == response_format:
                    raise error
                return answer, [{'operation': 'remove_echoed_response_format', 'offset': end}]
        # One completed root object followed by only one redundant closing brace.
        # Keep the entire valid object; never choose between multiple answers or
        # discard prose, another value, or a second malformed delimiter.
        if (error.msg == 'Extra data' and text.startswith('{')
                and text[error.pos:].strip() == '}'):
            value = json.loads(text[:error.pos])
            if isinstance(value, dict):
                return value, [{'operation': 'remove_duplicate_root_closing_brace', 'offset': error.pos}]
        # A completed object/array may contain ONE trailing comma. The parser
        # locates the syntax error outside string values; remove only that
        # delimiter and require the entire remaining reply to parse unchanged.
        index = error.pos
        if text[index:index + 1] in ("}", "]"):
            index = len(text[:index].rstrip()) - 1
        if (index >= 0 and text[index:index + 1] == ","
                and text[index + 1:].lstrip()[:1] in ("}", "]")):
            try:
                value = json.loads(text[:index] + text[index + 1:])
            except json.JSONDecodeError:
                pass
            else:
                return value, [{"operation": "remove_one_trailing_comma", "offset": index}]
        # A single JSON value separator typo is unambiguous; do not rewrite strings
        # or accept another malformed token after repairing it.
        if error.msg == "Expecting ',' delimiter" and text[error.pos:error.pos + 1] == ";":
            try:
                value = json.loads(text[:error.pos] + "," + text[error.pos + 1:])
            except json.JSONDecodeError:
                pass
            else:
                return value, [{"operation": "replace_value_separator_semicolon", "offset": error.pos}]
        # A single typographic closing quote at the reported line end. Preserve
        # its value verbatim; accept only if this ONE delimiter change parses all.
        if error.msg.startswith("Invalid control character") and text[error.pos:error.pos + 1] == "\n":
            start = text.rfind("\n", 0, error.pos) + 1
            index = text.rfind("”", start, error.pos)
            if index >= start and text[index + 1:error.pos].strip() == ",":
                try:
                    value = json.loads(text[:index] + '"' + text[index + 1:])
                except json.JSONDecodeError:
                    pass
                else:
                    return value, [{"operation": "normalize_one_closing_quote", "offset": index}]
        if error.pos != len(text) or not text.startswith("{"):
            raise
        return json.loads(text + "}"), [{"operation": "close_missing_root_object"}]


def normalize_reply(request, response):
    envelope_notes = []
    shape = request.get("response", {})
    evidence_protocol = bool(request.get('evidence_access'))

    if (request.get('mode') == 'outcome_jury_map'
            and isinstance(shape, dict) and 'criteria' in shape
            and isinstance(response, dict) and set(response) == {'outcome_jury'}
            and isinstance(response['outcome_jury'], dict)
            and isinstance(response['outcome_jury'].get('criteria'), list)
            and response['outcome_jury']['criteria']):
        # A single named draft envelope, not a competing answer. Preserve ALL
        # inner fields (including inquiries and objections); normal validation
        # and Reduce still decide whether the draft can become a standard.
        response = response['outcome_jury']
        envelope_notes.append({'operation': 'unwrap_jury_draft_envelope', 'field': 'outcome_jury'})

    def envelope_fields_match(payload):
        if not isinstance(payload, dict) or not isinstance(shape, dict):
            return False
        # ask() permits a preparatory evidence request independently of the
        # final answer schema. Preserve its handles for the core to validate;
        # a transport format echo must not turn it into a malformed verdict.
        if evidence_protocol and 'need_evidence' in payload:
            remaining = {key: value for key, value in payload.items() if key != 'need_evidence'}
            return not remaining or envelope_fields_match(remaining)
        if set(payload) == set(shape):
            return True
        # Measurement has optional recovery/inquiry records. A provider's
        # format wrapper must not make these stricter than the checker contract.
        # Keep every supplied field; no status, evidence or verdict is repaired.
        optional = {'measurement_recovery', 'inquiry_result'}
        return (request.get('mode') == 'outcome_measure'
                and {'status', 'reason', 'gap', 'evidence_refs', 'claims'} <= set(shape)
                and set(shape) - optional <= set(payload) <= set(shape) | optional)

    if (evidence_protocol and isinstance(shape, dict) and shape and 'response' not in shape
            and isinstance(response, dict)
            and set(response) in ({'response', 'need_evidence'}, {'type', 'response', 'need_evidence'})
            and ('type' not in response or response['type'] == 'json_object')
            and isinstance(response['response'], dict)
            and 'need_evidence' not in response['response']
            and envelope_fields_match(response['response'])):
        # There is one answer and one inquiry, not competing answers. Move the
        # inquiry inside the wrapper without interpreting or dropping anything.
        response = {key: value for key, value in response.items() if key != 'need_evidence'} | {
            'response': {**response['response'], 'need_evidence': response['need_evidence']},
        }
        envelope_notes.append({'operation': 'move_envelope_evidence_request_into_payload'})

    if (request.get('mode') == 'outcome_expectations' and isinstance(shape, dict)
            and set(shape) == {'criteria'} and isinstance(response, dict) and len(response) == 1):
        alias = next(iter(response))
        rows = response[alias]
        if (alias in {'outcome_expectations', 'outcome'} and isinstance(rows, list) and rows
                and all(isinstance(row, dict) and {'id', 'standard', 'kind', 'origin',
                    'evidence_refs', 'measurement_question', 'clauses'} <= set(row) for row in rows)):
            # Observed providers rename the sole declared list. Preserve every
            # criterion and clause byte; ordinary core validation still checks
            # exact quotes, provenance, applicability and the full contract.
            response = {'criteria': rows}
            envelope_notes.append({'operation': 'rename_declared_criteria_list', 'from': alias, 'to': 'criteria'})
    if (isinstance(response, dict) and {"type", "response"} <= set(response)
            and isinstance(shape, dict) and shape and "response" not in shape
            and (set(response) != {"type", "response"} or response["type"] != "json_object"
                 or not isinstance(response["response"], dict)
                 or not envelope_fields_match(response["response"]))):
        raise ResponseContractError(
            "response", "Typed envelope requires only type=json_object and response with exact declared fields",
        )
    if (isinstance(response, dict)
            and (set(response) == {"response"}
                 or (set(response) == {"type", "response"} and response["type"] == "json_object"))
            and isinstance(response["response"], dict) and isinstance(shape, dict)
            and shape and "response" not in shape
            and envelope_fields_match(response["response"])):
        # The only accepted metadata is the exact JSON-object format echo.
        # Never drop extra answers/fields or guess which inner contract to use.
        note = {"operation": "unwrap_declared_response_envelope"}
        if "type" in response:
            note = {"operation": "unwrap_declared_typed_response_envelope", "type": "json_object"}
        response = response["response"]
        envelope_notes.append(note)
    if isinstance(response, list):
        shape = request.get("response", {})
        if not isinstance(shape, dict) or len(shape) != 1 or not isinstance(next(iter(shape.values())), list):
            raise ValueError("Cannot assign an array to an ambiguous response envelope")
        key = next(iter(shape))
        response = {key: response}
        envelope_notes.append({"operation": "wrap_declared_list_envelope", "field": key})
    if not isinstance(response, dict):
        raise ResponseContractError("response", "Expected JSON object")
    result, notes = normalize_confirmation(request, response)
    notes = envelope_notes + notes
    result = deepcopy(result)
    context = request.get('context', {})
    task_id = context.get('task', {}).get('task_id')
    available = context.get('available_evidence_refs', [])
    task_alias = 'task:' + task_id if isinstance(task_id, str) and task_id else None

    def canonical_ref(ref):
        # The current Task's qualified ID is an exact identity, not a fuzzy
        # citation guess. Other tasks and unknown source handles remain invalid.
        if task_alias and ref == task_alias and 'task' in available and ref not in available:
            notes.append({'operation': 'current_task_ref', 'from': ref, 'to': 'task'})
            return 'task'
        return ref
    if request.get("schema_version") in {"listwise-joint-v1", "listwise-dual-order-v1"} and isinstance(result.get("rankings"), dict):
        rows = []
        for key, value in result["rankings"].items():
            if not isinstance(value, dict) or value.get("criterion_id", key) != key:
                raise ValueError("Ambiguous keyed ranking; cannot invent its Criterion identity")
            rows.append({"criterion_id": key, **value})
        result["rankings"] = rows
        notes.append({"operation": "keyed_rankings_to_declared_array"})

    def visit(value):
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key in {'evidence_refs', 'support_refs', 'claim_refs', 'need_evidence'} and isinstance(child, list):
                    value[key] = [canonical_ref(ref) for ref in child]
                elif key == 'claim_ref':
                    value[key] = canonical_ref(child)
            refs = value.get("evidence_refs")
            if isinstance(refs, str):
                # Parse a list, not its meaning. Unknown handles still fail the
                # core evidence boundary; never add a missing candidate's ref.
                value["evidence_refs"] = [part.strip() for part in re.split(r"[,;]", refs) if part.strip()]
                notes.append({"operation": "split_evidence_ref_list", "from": refs, "to": value["evidence_refs"]})
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result)
    # Models sometimes return the displayed ID stem without its descriptive
    # suffix. Resolve only an unambiguous delimiter-bounded catalog prefix;
    # never fuzzy-match a name or invent a tool/criterion not supplied.
    def resolve(original, ids):
        if not isinstance(original, str) or not original or original in ids:
            return original
        matches = [known for known in ids if known.startswith(original + "_")]
        if len(matches) == 1:
            notes.append({"operation": "unique_catalog_id_prefix", "from": original, "to": matches[0]})
            return matches[0]
        return original

    if request.get("mode") in {"outcome_hypothesis_map", "outcome_hypothesis_reduce", "outcome_hypothesis_refine"}:
        ids = {r["gap_id"] for r in request.get("jury", {}).get("gaps", [])}
        if isinstance(result.get("target_gap_ids"), list):
            result["target_gap_ids"] = [resolve(ref, ids) for ref in result["target_gap_ids"]]
    if request.get("mode") == "outcome_delivery_review":
        ids = {r["test_id"] for r in request.get("available_tests", [])}
        for row in result.get("verify", []):
            if isinstance(row, dict):
                row["test_id"] = resolve(row.get("test_id"), ids)
    if "outcome_comparison_protocol" in request:
        from reward_engine.core.shared.outcome_matrix import (
            normalize_matrix_reply,
        )

        result, matrix_notes = normalize_matrix_reply(request, result)
        notes.extend(matrix_notes)
    return result, notes


def normalize_confirmation(request, response):
    if request.get("mode") != "independent_final_wording_confirmation":
        return response, []
    result, notes = deepcopy(response), []
    if set(result) == {"response"} and isinstance(result["response"], dict):
        result = result["response"]
        notes.append({"operation": "unwrap_response_envelope"})
    ids = {row["criterion_id"] for row in request["criteria"]}
    sources = {}
    for trace in request["experience_pool"]["traces"]:
        for item in trace.get("evidence_items", []):
            content = item.get("content", {})
            if not isinstance(content, dict):
                continue
            for document in content.get("source_documents", []):
                name = document.get("source")
                if isinstance(name, str):
                    sources.setdefault(name, set()).add(trace["source_ref"])
    for row in result.get("confirmations", []):
        if not isinstance(row, dict):
            continue
        original = row.get("criterion_id")
        if isinstance(original, str) and original not in ids:
            # Only duplicated leading text: no edit distance, no semantic guess.
            matches = [known for known in ids if len(original) > len(known) and original.endswith(known)
                       and known.startswith(original[:-len(known)])]
            if len(matches) == 1:
                row["criterion_id"] = matches[0]
                notes.append({"operation": "collapse_duplicated_prefix", "from": original, "to": matches[0]})
        refs = row.get("source_refs")
        if not isinstance(refs, list):
            continue
        resolved = []
        for ref in refs:
            if isinstance(ref, str) and ref in sources:
                parents = sorted(sources[ref])
                resolved.extend(parents)
                notes.append({"operation": "observed_source_name_to_parent_traces", "from": ref, "to": parents})
            else:
                resolved.append(ref)
        row["source_refs"] = list(dict.fromkeys(resolved))
    return result, notes
