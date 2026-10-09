"""Real measurements for model-assisted checks; receipts are not quality scores."""

import asyncio
import json
from copy import deepcopy
from uuid import uuid4

from reward_engine.core.outcome.context import validate_current_product_refs
from reward_engine.core.outcome.numeric_program import (
    DEFAULT_STEPS,
    MAX_INPUT_BYTES,
    run_numeric,
)


class MeasurementRequestRejected(ValueError):
    """The probe never executed: its arguments or cited evidence were invalid.

    This is a correctable request error, not an observed measurement failure.
    It must not be recorded as an unavailable measurement that forces the whole
    check to unknown; the model may retry with valid refs/paths within the cap.
    """


def validate_probe_arguments(probe):
    """Reject a non-executable code placeholder before installing a recipe."""
    if probe.get('tool') == 'code.run':
        program = probe.get('arguments', {}).get('program')
        if not isinstance(program, str) or not program.strip():
            raise ValueError('code.run probe requires a non-empty program. Supply executable code; '
                             'if evidence must be inspected first, omit the optional probe and let '
                             'evidence.review request a concrete measurement. Do not emit code.run({}).')


class MeasurementTools:
    def __init__(self, *, document_reader=None, cuda_runner=None, record=lambda *_: None):
        self.document_reader, self.cuda_runner, self.record = document_reader, cuda_runner, record

    def catalog(self):
        return {
            "calculate": {"available": True, "arguments": {"expression": "numeric expression"},
                          "meaning": "Computed number, not proof the expression matches an artifact claim"},
            "code.run": {"available": True, "arguments": {"program": "assign result using numerical Python subset",
                            "input_refs": ["exact evidence refs"],
                            "input_paths": {"optional exact ref": ["nested_key", 0, "values"]}},
                         "input_contract": "inputs is a dictionary keyed by the exact input_refs strings. "
                         "Read inputs['exact:ref']; input_refs, data, refs, d and ref0 are NOT program variables. "
                         "Inspect the source schema first; no implicit table/row conversion occurs. "
                         "Optional input_paths selects one nested JSON value per cited ref before the 256000-byte input cap; "
                         "the program then sees that selected value at inputs[ref]. Omitted refs keep the whole source. "
                         "A request citing refs that are not supplied evidence, a mismatched path, or an oversized "
                         "selection never executes and does not consume the executed-measurement budget; correct the "
                         "refs/paths and retry within the cap.",
                         "example": {"program": "value = inputs['source:example']\nresult = len(value)",
                                     "input_refs": ["source:example"]},
                         "example_boundary": "Illustrative syntax only; replace source:example with an actual supplied ref.",
                         "functions": ["range", "abs", "min", "max", "len", "round", "int", "float", "str", "sqrt", "floor", "sorted", "find", "rfind"],
                         "text_search": "find(text, substring[, start[, end]]) and rfind(...) return a position or -1 with Python string semantics. Search existing long input text directly, then take a small slice; do not build range(len(text)) or concatenate all parts. Search work counts against the step budget.",
                         "scalar_conversion": "str(value) accepts one JSON scalar only; containers are not stringified.",
                         "indexed_assignment": "list[index] = JSON scalar updates an existing slot only; no slice, dictionary or container-valued assignment. Input evidence is copied, never written back.",
                         "limits": "No import/IO/user functions or arbitrary attributes/methods; assignments, for over range/list/string with break/continue, if, assert, arithmetic, JSON indexing, bounded slices/concatenation, single-loop list comprehensions, and/or, in/not in, is None/is not None. The only supported method is list.append(value) as a statement, with list length and cycle guards; "
                                   f"{DEFAULT_STEPS} steps. Not general Python or CUDA."},
            "document.read": {"available": True, "arguments": {"ref": "exact source ref", "offset": 0},
                              "external_fetch": self.document_reader is not None,
                              "alternative": "url already present in evidence; no invented URLs or hidden files"},
            "cuda.run": {"available": self.cuda_runner is not None,
                         "arguments": {"source": "CUDA source", "input_refs": ["exact evidence refs"]},
                         "meaning": "Only configured isolated GPU backend can execute; otherwise unavailable"},
        }

    async def run(self, request, evidence):
        tool, args = request.get("tool"), request.get("arguments", {})
        receipt = {"tool": tool, "arguments": deepcopy(args), "status": "unknown",
                   "request_rejected": False}
        try:
            if not isinstance(args, dict):
                raise MeasurementRequestRejected("Tool arguments must be an object")
            if tool == "calculate":
                result = run_numeric("result = (" + args["expression"] + ")")
            elif tool in {"code.run", "cuda.run"}:
                refs = args.get("input_refs", [])
                if not isinstance(refs, list) or any(not isinstance(r, str) or r not in evidence for r in refs):
                    raise MeasurementRequestRejected("Program inputs must cite actual evidence")
                inputs = {r: evidence[r] for r in refs}
                if tool == "code.run":
                    paths = args.get("input_paths", {})
                    if not isinstance(paths, dict) or set(paths) - set(refs):
                        raise MeasurementRequestRejected("input_paths must select only cited input_refs")
                    for ref, path in paths.items():
                        if (not isinstance(path, list) or len(path) > 16
                                or any(type(part) not in {str, int} for part in path)):
                            raise MeasurementRequestRejected(
                                "Use at most 16 exact JSON keys or array indices per input path")
                        selected = inputs[ref]
                        for part in path:
                            if (isinstance(selected, dict) and isinstance(part, str)
                                    or isinstance(selected, list) and type(part) is int):
                                selected = selected[part]
                            else:
                                raise MeasurementRequestRejected(
                                    "Input path does not match the observed JSON structure")
                        inputs[ref] = selected
                    # An oversized selection never reaches the program; it is a
                    # correctable request error, not an unavailable measurement.
                    if len(json.dumps(inputs, ensure_ascii=False).encode()) > MAX_INPUT_BYTES:
                        raise MeasurementRequestRejected(
                            "Program inputs too large; select relevant evidence")
                    result = run_numeric(args["program"], inputs)
                elif self.cuda_runner is None:
                    raise RuntimeError("No isolated CUDA execution backend/GPU configured")
                else:
                    result = await asyncio.wait_for(self.cuda_runner(args["source"], inputs), 30)
                    if (not isinstance(result, dict) or result.get("executed") is not True
                            or not result.get("device") or type(result.get("exit_code")) is not int):
                        raise ValueError("CUDA backend must return executed, actual device and exit_code")
            elif tool == "document.read":
                if "ref" in args:
                    if args["ref"] not in evidence:
                        raise MeasurementRequestRejected("Document ref is not supplied evidence")
                    source = evidence[args["ref"]]
                else:
                    url = args["url"]
                    def urls(value):
                        if isinstance(value, dict):
                            for key, item in value.items():
                                if key in {"url", "fetched_url"} and isinstance(item, str):
                                    yield item
                                else:
                                    yield from urls(item)
                        elif isinstance(value, list):
                            for item in value:
                                yield from urls(item)
                    if url not in set(urls(evidence)):
                        raise MeasurementRequestRejected("External source was not observed")
                    if self.document_reader is None:
                        raise RuntimeError("External document reader is unavailable")
                    source = await asyncio.wait_for(self.document_reader(url), 45)
                text = source if isinstance(source, str) else json.dumps(source, ensure_ascii=False)
                offset = args.get("offset", 0)
                if type(offset) is not int or offset < 0 or offset > len(text):
                    raise MeasurementRequestRejected("Invalid document offset")
                result = {"text": text[offset:offset + 12000], "offset": offset,
                          "total_chars": len(text), "excerpt_only": offset > 0 or len(text) > 12000}
            else:
                raise MeasurementRequestRejected("Unknown measurement tool")
            if len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode()) > 50000:
                raise ValueError("Tool output exceeds bounded receipt size")
            receipt.update(status="observed", result=result)
        except Exception as error:
            receipt.update(error_type=type(error).__name__, reason=str(error)[:300],
                           request_rejected=isinstance(error, MeasurementRequestRejected))
            if tool == 'code.run' and isinstance(args, dict):
                refs = args.get('input_refs', [])
                sources = {}
                for ref in refs[:6] if isinstance(refs, list) else []:
                    if not isinstance(ref, str) or ref not in evidence:
                        continue
                    selectors = []
                    def describe(value, path=(), selectors=selectors):
                        if len(selectors) >= 16:
                            return
                        if isinstance(value, (dict, list)):
                            selectors.append({'path': list(path), 'kind': type(value).__name__, 'length': len(value)})
                        if len(path) < 4 and isinstance(value, dict):
                            for key, child in list(value.items())[:8]:
                                describe(child, (*path, key))
                        elif len(path) < 4 and isinstance(value, list) and value:
                            describe(value[0], (*path, 0))
                    describe(evidence[ref])
                    sources[ref] = selectors
                receipt['recovery_hint'] = {
                    'input_sources': sources,
                    'instruction': "Each code.run starts a fresh program. Read inputs using the exact input_refs "
                        "strings, never invented aliases such as 'src'/'out'. Use input_paths with the observed "
                        "selectors to keep only needed tables/rows before the byte limit. After selection, "
                        "inputs[ref] is that selected value. list.append(value) is supported; arbitrary methods, "
                        "set, sum, imports and host IO are not. sorted(list) compares scalar multisets while "
                        "preserving duplicates; do not assume fixed row order or identifier ranges. Correct this computation locally and cite its "
                        "successful receipt; a prose estimate does not replace a failed calculation."}
        self.record("measurement_tool_" + uuid4().hex, receipt)
        return receipt


async def author_checker(test, registry, context, evidence, call, *, execution_recovery=None):
    """Current-product recipe, real dry run, at most one execution repair."""
    from reward_engine.core.outcome.context import ask, compact_learning_context

    # A reusable checker receives one current product in production. Showing the
    # historical pool here generated product_2 bindings which correctly failed
    # the portability guard before execution. Do not expose that pool to authoring.
    context, evidence = deepcopy(context), deepcopy(evidence)
    context['products'] = context['products'][:1]
    context['checks'] = []
    evidence = {ref: value for ref, value in evidence.items()
                if not ref.startswith('check:') and (not ref.startswith('product_') or ref.startswith('product_1:'))}
    context['available_evidence_refs'] = list(evidence)
    context = compact_learning_context(context)
    def validate_recipe(recipe):
        if not isinstance(recipe, dict):
            raise ValueError('Checker recipe must be an object')
        if recipe.get('unavailable_reason'):
            return recipe
        probes = recipe.get('probes')
        if not isinstance(probes, list) or not 1 <= len(probes) <= 3:
            raise ValueError('Return one to three execution probes. Use need_evidence to inspect sources before '
                             'authoring; do not include every preparatory read in the final execution recipe.')
        validate_current_product_refs(probes)
        if any(not isinstance(p, dict) or not isinstance(p.get('tool'), str)
               or not isinstance(p.get('arguments', {}), dict) for p in probes):
            raise ValueError('Malformed measurement recipe')
        for probe in probes:
            validate_probe_arguments(probe)
        return recipe
    instruction = (
        "Implement the missing checker as a short sequence of actual measurement tools. "
        "Return ONE TO THREE execution probes, never more. Use need_evidence with exact handles to "
        "inspect source contents while authoring; those preparatory reads are not execution probes. "
        "Use calculate for arithmetic, code.run for bounded numerical Python programs, "
        "document.read for source checks. Return probes, not a verdict or an expected answer. "
        "Use workspace.inspect/read when available to inspect parsed Office files by relative path. "
        "Each probe executes independently and cannot refer to an earlier probe's result. "
        "Keep incomplete excerpts explicit; a read is not deterministic proof of correctness. "
        "Programs must use current input_refs, not historical candidate answers. "
        "Only product_1 is the current product; never compare historical product_2 or hardcode its answer. "
        "code.run exposes inputs as a dictionary keyed by exact evidence handles: inputs['exact:ref']. "
        "The argument input_refs is NOT a variable inside the program. Inspect actual source schemas. "
        "Use only catalogued syntax/functions; bounded list.append(value) is supported as a statement. "
        "No string methods, sum, set, imports or invented aliases. "
        "A numerical program must assign JSON-compatible result; no imports or filesystem. "
        "If unavailable CUDA/general execution is essential, set unavailable_reason instead. "
        "A dry-run technical failure permits one repair of this same measurement, without changing "
        "the standard or replacing the computation with a constant answer. Do not pretend a missing capability exists.")
    recovery = deepcopy(execution_recovery)
    for attempt in range(2):
        recipe = await ask(call, "outcome_checker_author", context, evidence, instruction,
            {"probes": [{"tool": "catalog key", "arguments": {}}], "unavailable_reason": "or empty"},
            extra={"test": test, "measurement_tools": registry.tools.catalog(),
                   **({'execution_recovery': recovery} if recovery else {})},
            validate=validate_recipe, max_contract_retries=1, max_evidence_rounds=2, close_evidence=True)
        probes, receipts = recipe.get('probes'), []
        # A declared missing capability is not a syntax failure to redraw.
        if recipe.get('unavailable_reason'):
            raise ValueError('Checker capability unavailable: ' + str(recipe['unavailable_reason']))
        try:
            receipts = [await registry.tools.run(probe, evidence) for probe in probes]
            failures = [f"probe[{i}] {r['tool']}: {r.get('error_type', r['status'])}: {r.get('reason', '')}"
                        for i, r in enumerate(receipts) if r['status'] != 'observed']
            if failures:
                raise ValueError('Checker recipe could not execute; ' + '; '.join(failures)[:900])
        except (ValueError, TypeError, KeyError) as error:
            recovery = {'recipe': recipe, 'receipts': receipts, 'error': str(error),
                        'instruction': 'Fix only the concrete execution/binding error for the same current-product check.'}
            registry.tools.record('checker_author_trial_' + uuid4().hex,
                                  {'checker': test['checker'], 'attempt': attempt, **recovery})
            if attempt:
                raise
            continue
        registry.tools.record('checker_author_trial_' + uuid4().hex,
            {'checker': test['checker'], 'attempt': attempt, 'recipe': recipe, 'receipts': receipts})
        registry.bind_recipe(test['checker'], {'probes': deepcopy(probes)})
        return
