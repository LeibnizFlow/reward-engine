"""Real measurements for model-assisted checks; receipts are not quality scores."""

import asyncio
import json
from copy import deepcopy
from uuid import uuid4

from reward_engine.core.outcome.context import validate_current_product_refs
from reward_engine.core.outcome.measurement_requests import (
    MeasurementRequestRejected,
    preflight,
)
from reward_engine.core.outcome.measurement_requests import (
    enabled as recovery_enabled,
)
from reward_engine.core.outcome.numeric_program import (
    DEFAULT_STEPS,
    MAX_INPUT_BYTES,
    run_numeric,
)


def validate_probe_arguments(probe, *, policy=None):
    """Reject a non-executable code placeholder before installing a recipe."""
    if policy is not None:
        from reward_engine.core.outcome.measurement_requests import POLICY

        if policy == POLICY:
            preflight(probe)
            return
    if probe.get("tool") == "code.run":
        program = probe.get("arguments", {}).get("program")
        if not isinstance(program, str) or not program.strip():
            raise ValueError(
                "code.run probe requires a non-empty program. Supply executable code; "
                "if evidence must be inspected first, omit the optional probe and let "
                "evidence.review request a concrete measurement. Do not emit code.run({})."
            )


class MeasurementTools:
    def __init__(
        self, *, document_reader=None, cuda_runner=None, record=lambda *_: None,
        role_contract_policy=None,
    ):
        self.document_reader, self.cuda_runner, self.record = (
            document_reader,
            cuda_runner,
            record,
        )
        self.role_contract_policy = role_contract_policy

    def catalog_for(self, evidence):
        """Capabilities for these inputs; subclasses may require collected data."""
        return self.catalog()

    def validate_request(self, request, evidence, *, require_available=True, check_program=True):
        """Shared admission proof for execution, budgeting and retained receipts.

        Historical successful observations do not need a currently available
        external service. A claimed pre-execution rejection does need proof.
        """
        bindings = preflight(request, evidence, check_program=check_program)
        if require_available:
            tool = request.get("tool")
            if (tool == "document.read" and "ref" not in request.get("arguments", {})
                    and self.document_reader is None):
                raise MeasurementRequestRejected("External document reader is unavailable",
                                                 field="arguments.url", code="TOOL_UNAVAILABLE")
            capability = self.catalog_for(evidence).get(tool, {})
            if capability.get("available") is False:
                raise MeasurementRequestRejected(
                    capability.get("unavailable_reason", "Measurement capability is unavailable for these inputs"),
                    field="tool", actual=tool, code="TOOL_UNAVAILABLE")
        return bindings

    async def run_checked(self, request, evidence):
        """Reject unavailable capabilities before invoking any measurement."""
        if recovery_enabled(self):
            try:
                self.validate_request(request, evidence)
            except MeasurementRequestRejected as error:
                receipt = {"tool": request.get("tool"), "arguments": deepcopy(request.get("arguments", {})),
                           "status": "unknown", "request_rejected": True,
                           "execution_phase": "preflight", "contract_policy": self.role_contract_policy,
                           "error_type": type(error).__name__, "reason": str(error),
                           "diagnostics": error.diagnostics}
                self.record("measurement_tool_" + uuid4().hex, receipt)
                return receipt
        tool = request.get("tool")
        capability = self.catalog_for(evidence).get(tool, {})
        if isinstance(capability, dict) and capability.get("available") is False:
            receipt = {"tool": tool, "arguments": deepcopy(request.get("arguments", {})),
                       "status": "unknown", "request_rejected": True,
                       "error_type": "MeasurementRequestRejected",
                       "reason": capability.get("unavailable_reason", "Measurement capability is unavailable for these inputs")}
            if recovery_enabled(self):
                receipt.update(execution_phase="preflight", contract_policy=self.role_contract_policy,
                               diagnostics=[{"code": "TOOL_UNAVAILABLE", "field": "tool", "actual": tool,
                                             "rule": receipt["reason"]}])
            self.record("measurement_tool_" + uuid4().hex, receipt)
            return receipt
        return await self.run(request, evidence)

    def execution_metadata(self, request, evidence):
        if not recovery_enabled(self):
            return {}
        return {"execution_phase": "execution", "contract_policy": self.role_contract_policy,
                "input_bindings": preflight(request, evidence)}

    def validate_historical(self, receipt, evidence):
        """Adapters may prove equivalence when a deterministic tool contract changes."""

    def text_evidence(self, arguments, evidence):
        ref = arguments.get("ref")
        if not isinstance(ref, str) or not isinstance(evidence.get(ref), str):
            raise ValueError("Text checker requires exact text at supplied ref")
        return {"text": evidence[ref], "evidence_refs": [ref]}

    def catalog(self):
        return {
            "calculate": {
                "available": True,
                "arguments": {"expression": "numeric expression" if not recovery_enabled(self) else
                              "one expression, e.g. (2+3)/5; no assignment or semicolon-separated statements"},
                "meaning": "Computed number, not proof the expression matches an artifact claim",
            },
            "code.run": {
                "available": True,
                "arguments": {
                    "program": "assign result using numerical Python subset",
                    "input_refs": ["exact evidence refs"],
                    "input_paths": {"optional exact ref": ["nested_key", 0, "values"]},
                },
                "input_contract": "inputs is a dictionary keyed by the exact input_refs strings. "
                "Read inputs['exact:ref']; input_refs, data, refs, d and ref0 are NOT program variables. "
                "Bound inputs contain the complete supplied evidence values (or the explicit input_paths selections), "
                "subject to the input byte limit, even when the model context showed only a preview. "
                "Inspect the source schema first; no implicit table/row conversion occurs. "
                "Optional input_paths selects one nested JSON value per cited ref before the 256000-byte input cap; "
                "the program then sees that selected value at inputs[ref]. Omitted refs keep the whole source. "
                "A request citing refs that are not supplied evidence, a mismatched path, or an oversized "
                "selection never executes and does not consume the executed-measurement budget; correct the "
                "refs/paths and retry within the cap.",
                "example": {
                    "program": "value = inputs['source:example']\nresult = len(value)",
                    "input_refs": ["source:example"],
                },
                "example_boundary": "Illustrative syntax only; replace source:example with an actual supplied ref.",
                "functions": [
                    "range",
                    "abs",
                    "min",
                    "max",
                    "len",
                    "round",
                    "int",
                    "float",
                    "str",
                    "sqrt",
                    "floor",
                    "sorted",
                    "find",
                    "rfind",
                ],
                "text_search": "find(text, substring[, start[, end]]) and rfind(...) return a position or -1 with Python string semantics. Search existing long input text directly, then take a small slice; do not build range(len(text)) or concatenate all parts. Search work counts against the step budget.",
                "scalar_conversion": "str(value) accepts one JSON scalar or bounded acyclic JSON container. "
                "The full conversion must fit 10000 characters and the step/depth bounds; slicing the string "
                "after conversion does not avoid these bounds. Inspect small selected fields or slices instead.",
                "schema_inspection": "Use the supplied source structure or recovery_hint.input_sources to locate data, "
                "then inspect small selected values with print/str. type, hasattr, isinstance and arbitrary attributes "
                "are unavailable; use only the listed functions and supported syntax.",
                "indexed_assignment": "list[index] = JSON scalar updates an existing slot only; no slice, dictionary or container-valued assignment. Input evidence is copied, never written back.",
                "limits": "No import/IO/user functions or arbitrary attributes/methods; assignments, for over range/list/string with break/continue, if, assert, arithmetic, JSON indexing, bounded slices/concatenation, single-loop list comprehensions, and/or, in/not in, is None/is not None. Supported methods are dict.get, str.find/rfind/count/split and list.append(value) as a statement; print captures at most 10000 characters in the receipt, with list length and cycle guards; "
                f"{DEFAULT_STEPS} steps. Not general Python or CUDA.",
            },
            "document.read": {
                "available": True,
                "arguments": {"ref": "exact source ref", "offset": 0},
                "external_fetch": self.document_reader is not None,
                "alternative": "url already present in evidence; no invented URLs or hidden files",
            },
            "cuda.run": {
                "available": self.cuda_runner is not None,
                "arguments": {
                    "source": "CUDA source",
                    "input_refs": ["exact evidence refs"],
                },
                "meaning": "Only configured isolated GPU backend can execute; otherwise unavailable",
            },
        }

    async def run(self, request, evidence):
        tool, args = request.get("tool"), request.get("arguments", {})
        receipt = {
            "tool": tool,
            "arguments": deepcopy(args),
            "status": "unknown",
            "request_rejected": False,
        }
        try:
            receipt.update(self.execution_metadata(request, evidence))
            if not isinstance(args, dict):
                raise MeasurementRequestRejected("Tool arguments must be an object")
            if tool == "calculate":
                result = run_numeric("result = (" + args["expression"] + ")")
            elif tool in {"code.run", "cuda.run"}:
                refs = args.get("input_refs", [])
                if not isinstance(refs, list) or any(
                    not isinstance(r, str) or r not in evidence for r in refs
                ):
                    raise MeasurementRequestRejected(
                        "Program inputs must cite actual evidence"
                    )
                inputs = {r: evidence[r] for r in refs}
                if tool == "code.run":
                    paths = args.get("input_paths", {})
                    if not isinstance(paths, dict) or set(paths) - set(refs):
                        raise MeasurementRequestRejected(
                            "input_paths must select only cited input_refs"
                        )
                    for ref, path in paths.items():
                        if (
                            not isinstance(path, list)
                            or len(path) > 16
                            or any(type(part) not in {str, int} for part in path)
                        ):
                            raise MeasurementRequestRejected(
                                "Use at most 16 exact JSON keys or array indices per input path"
                            )
                        selected = inputs[ref]
                        for part in path:
                            if (
                                isinstance(selected, dict)
                                and isinstance(part, str)
                                or isinstance(selected, list)
                                and type(part) is int
                            ):
                                selected = selected[part]
                            else:
                                raise MeasurementRequestRejected(
                                    "Input path does not match the observed JSON structure"
                                )
                        inputs[ref] = selected
                    # An oversized selection never reaches the program; it is a
                    # correctable request error, not an unavailable measurement.
                    if (
                        len(json.dumps(inputs, ensure_ascii=False).encode())
                        > MAX_INPUT_BYTES
                    ):
                        raise MeasurementRequestRejected(
                            "Program inputs too large; select relevant evidence"
                        )
                    captured = []
                    try:
                        result = run_numeric(args["program"], inputs, stdout=captured)
                    finally:
                        if captured:
                            receipt["stdout"] = "".join(captured)
                elif self.cuda_runner is None:
                    raise RuntimeError(
                        "No isolated CUDA execution backend/GPU configured"
                    )
                else:
                    result = await asyncio.wait_for(
                        self.cuda_runner(args["source"], inputs), 30
                    )
                    if (
                        not isinstance(result, dict)
                        or result.get("executed") is not True
                        or not result.get("device")
                        or type(result.get("exit_code")) is not int
                    ):
                        raise ValueError(
                            "CUDA backend must return executed, actual device and exit_code"
                        )
            elif tool == "document.read":
                if "ref" in args:
                    if args["ref"] not in evidence:
                        raise MeasurementRequestRejected(
                            "Document ref is not supplied evidence"
                        )
                    source = evidence[args["ref"]]
                else:
                    url = args["url"]

                    def urls(value):
                        if isinstance(value, dict):
                            for key, item in value.items():
                                if key in {"url", "fetched_url"} and isinstance(
                                    item, str
                                ):
                                    yield item
                                else:
                                    yield from urls(item)
                        elif isinstance(value, list):
                            for item in value:
                                yield from urls(item)

                    if url not in set(urls(evidence)):
                        raise MeasurementRequestRejected(
                            "External source was not observed"
                        )
                    if self.document_reader is None:
                        raise RuntimeError("External document reader is unavailable")
                    source = await asyncio.wait_for(self.document_reader(url), 45)
                text = (
                    source
                    if isinstance(source, str)
                    else json.dumps(source, ensure_ascii=False)
                )
                offset = args.get("offset", 0)
                if type(offset) is not int or offset < 0 or offset > len(text):
                    if recovery_enabled(self) and "ref" not in args:
                        raise ValueError("Invalid offset after external document fetch")
                    raise MeasurementRequestRejected("Invalid document offset")
                result = {
                    "text": text[offset : offset + 12000],
                    "offset": offset,
                    "total_chars": len(text),
                    "excerpt_only": offset > 0 or len(text) > 12000,
                }
            else:
                raise MeasurementRequestRejected("Unknown measurement tool")
            if (
                len(json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
                > 50000
            ):
                raise ValueError("Tool output exceeds bounded receipt size")
            receipt.update(status="observed", result=result)
        except Exception as error:
            receipt.update(
                error_type=type(error).__name__,
                reason=str(error)[:300],
                request_rejected=isinstance(error, MeasurementRequestRejected),
            )
            if recovery_enabled(self) and isinstance(error, MeasurementRequestRejected):
                receipt.update(execution_phase="preflight", diagnostics=error.diagnostics,
                               contract_policy=self.role_contract_policy)
            if tool == "code.run" and isinstance(args, dict):
                refs = args.get("input_refs", [])
                sources = {}
                for ref in refs[:6] if isinstance(refs, list) else []:
                    if not isinstance(ref, str) or ref not in evidence:
                        continue
                    selectors = []

                    def describe(value, path=(), selectors=selectors):
                        if len(selectors) >= 16:
                            return
                        if isinstance(value, (dict, list)):
                            selectors.append(
                                {
                                    "path": list(path),
                                    "kind": type(value).__name__,
                                    "length": len(value),
                                }
                            )
                        if len(path) < 4 and isinstance(value, dict):
                            for key, child in list(value.items())[:8]:
                                describe(child, (*path, key))
                        elif len(path) < 4 and isinstance(value, list) and value:
                            describe(value[0], (*path, 0))

                    describe(evidence[ref])
                    sources[ref] = selectors
                receipt["recovery_hint"] = {
                    "input_sources": sources,
                    "instruction": "Each code.run starts a fresh program. Read inputs using the exact input_refs "
                    "strings, never invented aliases such as 'src'/'out'. Use input_paths with the observed "
                    "selectors to keep only needed tables/rows before the byte limit. After selection, "
                    "inputs[ref] is that selected value. list.append(value) is supported; arbitrary methods, "
                    "set, sum, imports and host IO are not. sorted(list) compares scalar multisets while "
                    "preserving duplicates; do not assume fixed row order or identifier ranges. Correct this computation locally and cite its "
                    "successful receipt; a prose estimate does not replace a failed calculation.",
                }
                if recovery_enabled(self) and not receipt["request_rejected"]:
                    receipt["recovery_hint"].update(
                        selection_policy="preserve_failed_selection_select_inside_program",
                        required_input_bindings=deepcopy(receipt.get("input_bindings", [])),
                        instruction="This request executed and failed. Keep its exact input_refs and "
                        "input_paths selections when correcting the program; changing a selection "
                        "does not cover the failed input. Select needed fields or slices INSIDE "
                        "the program before str(), concatenation or printing. For example, index "
                        "the observed content field directly instead of converting an entire "
                        "source object to text. Each code.run starts fresh: use the exact cited "
                        "reference strings in inputs[ref], with the original selected value. "
                        "Use the source structure below to choose real keys, never invented "
                        "aliases. list.append, dict.get, str.find/rfind/count/split are supported; "
                        "set, sum, imports and host IO are not. Cite the successful corrected "
                        "receipt in measurement_recovery; prose cannot replace a failed calculation.",
                    )
        self.record("measurement_tool_" + uuid4().hex, receipt)
        return receipt


async def author_checker(
    test, registry, context, evidence, call, *, execution_recovery=None
):
    """Current-product recipe, real dry run, at most one execution repair."""
    from reward_engine.core.outcome.context import ask, compact_learning_context

    # A reusable checker receives one current product in production. Showing the
    # historical pool here generated product_2 bindings which correctly failed
    # the portability guard before execution. Do not expose that pool to authoring.
    context, evidence = deepcopy(context), deepcopy(evidence)
    context["products"] = context["products"][:1]
    context["checks"] = []
    evidence = {
        ref: value
        for ref, value in evidence.items()
        if not ref.startswith("check:")
        and (not ref.startswith("product_") or ref.startswith("product_1:"))
    }
    context["available_evidence_refs"] = list(evidence)
    context = compact_learning_context(context)

    def validate_recipe(recipe):
        if not isinstance(recipe, dict):
            raise ValueError("Checker recipe must be an object")
        if recipe.get("unavailable_reason"):
            return recipe
        probes = recipe.get("probes")
        if not isinstance(probes, list) or not 1 <= len(probes) <= 3:
            raise ValueError(
                "Return one to three execution probes. Use need_evidence to inspect sources before "
                "authoring; do not include every preparatory read in the final execution recipe."
            )
        validate_current_product_refs(probes)
        if any(
            not isinstance(p, dict)
            or not isinstance(p.get("tool"), str)
            or not isinstance(p.get("arguments", {}), dict)
            for p in probes
        ):
            raise ValueError("Malformed measurement recipe")
        for probe in probes:
            validate_probe_arguments(probe, policy=getattr(registry.tools, "role_contract_policy", None))
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
        "the standard or replacing the computation with a constant answer. Do not pretend a missing capability exists."
    )
    recovery = deepcopy(execution_recovery)
    for attempt in range(2):
        recipe = await ask(
            call,
            "outcome_checker_author",
            context,
            evidence,
            instruction,
            {
                "probes": [{"tool": "catalog key", "arguments": {}}],
                "unavailable_reason": "or empty",
            },
            extra={
                "test": test,
                "measurement_tools": registry.tools.catalog(),
                **({"execution_recovery": recovery} if recovery else {}),
            },
            validate=validate_recipe,
            max_contract_retries=1,
            max_evidence_rounds=2,
            close_evidence=True,
        )
        probes, receipts = recipe.get("probes"), []
        # A declared missing capability is not a syntax failure to redraw.
        if recipe.get("unavailable_reason"):
            raise ValueError(
                "Checker capability unavailable: " + str(recipe["unavailable_reason"])
            )
        try:
            run = registry.tools.run_checked if recovery_enabled(registry.tools) else registry.tools.run
            receipts = [await run(probe, evidence) for probe in probes]
            failures = [
                f"probe[{i}] {r['tool']}: {r.get('error_type', r['status'])}: {r.get('reason', '')}"
                for i, r in enumerate(receipts)
                if r["status"] != "observed"
            ]
            if failures:
                raise ValueError(
                    "Checker recipe could not execute; " + "; ".join(failures)[:900]
                )
        except (ValueError, TypeError, KeyError) as error:
            recovery = {
                "recipe": recipe,
                "receipts": receipts,
                "error": str(error),
                "instruction": "Fix only the concrete execution/binding error for the same current-product check.",
            }
            registry.tools.record(
                "checker_author_trial_" + uuid4().hex,
                {"checker": test["checker"], "attempt": attempt, **recovery},
            )
            if attempt:
                raise
            continue
        registry.tools.record(
            "checker_author_trial_" + uuid4().hex,
            {
                "checker": test["checker"],
                "attempt": attempt,
                "recipe": recipe,
                "receipts": receipts,
            },
        )
        registry.bind_recipe(test["checker"], {"probes": deepcopy(probes)})
        return
