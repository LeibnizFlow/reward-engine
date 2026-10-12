"""Separate versioned strict PUBLIC path gate; never replace pipeline_audit.

Run against terminal opted-in runs only. Exit zero means the additional public
path gate passes. The unchanged original strict audit must independently pass.
Offline parsing and deterministic reconstruction use retained provider receipts.
No quality/grade interpretation, external provider/tool calls, or receipt writes occur here.
"""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from reward_engine.adapters.workbuddy.evidence import EXCLUDED
from reward_engine.adapters.workbuddy.io import digest
from reward_engine.adapters.workbuddy.public_delivery import (
    read_bound_contract,
    scan_workspace,
)
from reward_engine.core.outcome.public_delivery import (
    TEST,
    bind_definition,
    bind_pack,
    candidate_gate,
    fingerprint,
    safe_path,
    validate,
)

SCHEMA = "public_delivery.strict_audit.v3"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def path(self, name):
        relative = safe_path(str(name))
        path = self.root
        for part in relative.split("/"):
            path = path / part
            require(not path.is_symlink(), "Linked audit source: " + relative)
        require(path.resolve().is_relative_to(self.root), "Audit source escapes run")
        return path

    def bytes(self, name):
        data = self.path(name).read_bytes()
        self.hashes[str(name)] = hashlib.sha256(data).hexdigest()
        return data

    def read(self, name):
        try:
            return json.loads(self.bytes(name))
        except json.JSONDecodeError as error:
            # Corrupt audit metadata is never a model syntax-recovery signal.
            raise ValueError("Malformed retained JSON record: " + str(name)) from error

    def local(self, absolute):
        return Path(absolute).relative_to(self.root).as_posix()


class RetainedRole:
    """Read-only Role replay: exact requests, parsed retained provider bytes, no IO calls.

    This establishes origin within the retained evidence, not authenticity against
    an adversary able to replace provider receipts as well as all manifests.
    """

    def __init__(self, reader, folder, config, *, targeted=True):
        from reward_engine.adapters.workbuddy.models import Role

        self.reader, self.folder, self.config = reader, folder, config
        self.bounded_stage = config.get("role_bounded_stage", False)
        self.stage_timeout_s = config.get("role_stage_timeout_s", 600)
        self.request_budget_bytes = Role.request_budget_bytes
        self.evidence_expansion_limit = Role.evidence_expansion_limit
        self.jury_repair_mode = "targeted" if targeted else "full"
        self.entries, self.used, self.errors = {}, [], []
        directory = reader.path(folder)
        for path in sorted(directory.glob("**/request.json")):
            name = reader.local(path)
            request = reader.read(name)
            self.entries.setdefault(fingerprint(request), []).append((name, request))

    def transport(self, directory, body, summary, *, failed):
        """Bind a parsed summary to completed physical payload, before decoding JSON.

        A transport/receipt inconsistency must raise ValueError, never the
        JSONDecodeError consumed by the production bounded syntax-repair loop.
        """
        from reward_engine.adapters.workbuddy.io import model_matches
        from reward_engine.adapters.workbuddy.reply import is_usage_terminal_repeat

        def physical_json(text):
            try:
                return json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Malformed retained provider transport JSON"
                ) from error

        nonstream = body.get("stream") is False
        if nonstream:
            link = self.reader.read(directory + "/transport.json")
            require(
                link.get("mode") == "nonstream_recovery",
                "Missing retained nonstream recovery provenance",
            )
            original_name = self.reader.local(Path(link["original_failure"]))
            require(
                str(Path(original_name).parent.parent) == self.folder,
                "Nonstream recovery points outside its Role recorder",
            )
            original = self.reader.read(original_name)
            original_dir = str(Path(original_name).parent)
            require(
                original.get("error_type") == "ProviderStreamError",
                "Nonstream recovery has no actual provider transport failure",
            )
            first_body = self.reader.read(original_dir + "/request_body.json")
            expected_body = {
                k: v for k, v in first_body.items() if k != "stream_options"
            }
            expected_body["stream"] = False
            require(
                body == expected_body, "Nonstream recovery changed the original request"
            )
            stream_name = original_dir + "/stream_lines.json"
            events = []
            for line in self.reader.read(stream_name):
                if line.startswith("data:") and line[5:].strip() != "[DONE]":
                    event = physical_json(line[5:].strip())
                    if "error" in event:
                        events.append(event)
            require(
                events == [original.get("provider_error_event")],
                "Nonstream recovery failure differs from physical provider stream",
            )
            self.used.append(
                {
                    "request": original_dir + "/request.json",
                    "failure": original_name,
                    "failure_sha256": self.reader.hashes[original_name],
                    "transport": stream_name,
                    "transport_sha256": self.reader.hashes[stream_name],
                    "recovery": directory + "/transport.json",
                }
            )
            payload = directory + "/raw_response.txt"
            raw = physical_json(self.reader.bytes(payload))
            require(
                raw == self.reader.read(directory + "/raw_response.json"),
                "Retained nonstream response copies disagree",
            )
            choices = raw.get("choices", [])
            require(
                "error" not in raw and len(choices) == 1,
                "Incomplete retained nonstream response",
            )
            choice = choices[0]
            require(
                choice.get("finish_reason") == "stop"
                and choice.get("native_finish_reason") in (None, "stop")
                and type(choice.get("index", 0)) is int
                and choice.get("index", 0) == 0
                and raw.get("model")
                and model_matches(self.config, body["model"], raw["model"]),
                "Incomplete or mismatched retained nonstream response",
            )
            message = choice.get("message", {})
            require(
                message.get("tool_calls") is None
                and message.get("function_call")
                in (None, {"name": "", "arguments": ""}),
                "Unsupported retained nonstream calls",
            )
            content, tools, finished = message.get("content"), [], "stop"
            require(isinstance(content, str), "Missing retained nonstream content")
        else:
            payload = directory + "/stream_lines.json"
            parts, tools, functions, finished, terminals = [], [], [], None, []
            models, completion_ids = set(), set()
            for line in self.reader.read(payload):
                require(isinstance(line, str), "Malformed retained stream line")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = physical_json(data)
                require(
                    "error" not in event,
                    "Retained semantic stream contains provider error",
                )
                if event.get("model"):
                    models.add(event["model"])
                if event.get("id"):
                    completion_ids.add(event["id"])
                choices = event.get("choices", [])
                require(len(choices) <= 1, "Ambiguous retained semantic choices")
                for choice in choices:
                    require(
                        type(choice.get("index", 0)) is int
                        and choice.get("index", 0) == 0,
                        "Ambiguous retained semantic choice index",
                    )
                    delta = choice.get("delta", {})
                    tool_fragment = (
                        delta.get("tool_calls") is not None
                        and delta["tool_calls"] != []
                    )
                    function_fragment = delta.get(
                        "function_call"
                    ) is not None and delta["function_call"] != {
                        "name": "",
                        "arguments": "",
                    }
                    require(
                        not finished
                        or not (
                            delta.get("content")
                            or delta.get("reasoning_content")
                            or tool_fragment
                            or function_fragment
                        ),
                        "Retained semantic content after terminal",
                    )
                    if delta.get("content"):
                        parts.append(delta["content"])
                    if delta.get("tool_calls") is not None:
                        tools.append(delta["tool_calls"])
                    if delta.get("function_call") is not None:
                        functions.append(delta["function_call"])
                    if choice.get("finish_reason"):
                        require(
                            not terminals
                            or is_usage_terminal_repeat(terminals[0], event),
                            "Conflicting retained semantic terminals",
                        )
                        require(
                            choice.get("native_finish_reason")
                            in (None, choice["finish_reason"]),
                            "Conflicting retained native terminal",
                        )
                        terminals.append(event)
                        finished = choice["finish_reason"]
            require(
                finished in {"stop", "tool_calls"}
                and models
                and len(completion_ids) <= 1
                and all(
                    model_matches(self.config, body["model"], actual)
                    for actual in models
                ),
                "Incomplete or mismatched retained semantic transport",
            )
            require(
                all(chunk == {"name": "", "arguments": ""} for chunk in functions),
                "Unsupported retained legacy function call",
            )
            require(
                tools == summary.get("native_tool_call_chunks", [])
                and functions == summary.get("legacy_function_call_chunks", [])
                and finished == summary.get("finish_reason"),
                "Semantic summary differs from retained provider transport metadata",
            )
            content = "".join(parts)
        output_key = "partial_output" if failed else "raw_output"
        # Nonstream failure.json records only error_type/reason. Its raw HTTP
        # body is authoritative; never invent a partial_output for that schema.
        if not (nonstream and failed and output_key not in summary):
            require(
                summary.get(output_key) == content,
                "Semantic output differs from retained provider transport bytes",
            )
        return content, [chunk for chunk in tools if chunk != []], finished, payload

    def answer(self, name, request):
        from reward_engine.adapters.claw_eval.reply_format import normalize_reply
        from reward_engine.adapters.workbuddy.reply import (
            decode_native_tool_reply,
            decode_role_reply,
        )
        from reward_engine.core.outcome.reducer_projection import resolve_output_texts

        directory = str(Path(name).parent)
        body = self.reader.read(directory + "/request_body.json")
        try:
            bound_request = json.loads(body["messages"][-1]["content"])
        except json.JSONDecodeError as error:
            raise ValueError("Malformed retained provider request body") from error
        require(
            bound_request == request, "Retained provider body differs from Role request"
        )
        response_name = directory + "/response.json"
        failed = not self.reader.path(response_name).exists()
        record_name = directory + "/failure.json" if failed else response_name
        response = self.reader.read(record_name)
        if failed:
            require(
                response.get("error_type") == "JSONDecodeError",
                "No replayable completed semantic provider response",
            )
        content, chunks, finished, payload = self.transport(
            directory, body, response, failed=failed
        )
        source = {
            "request": name,
            "failure" if failed else "response": record_name,
            "transport": payload,
            "transport_sha256": self.reader.hashes[payload],
            "request_sha256": fingerprint(request),
        }
        if failed:
            require(
                finished == "stop" and not chunks,
                "Syntax recovery requires a completed text response",
            )
            try:
                decode_role_reply(content, response_format=body.get("response_format"))
            except json.JSONDecodeError:
                self.used.append(
                    {**source, "failure_sha256": self.reader.hashes[record_name]}
                )
                raise
            raise ValueError("Retained syntax failure unexpectedly parses")
        try:
            if chunks or finished == "tool_calls":
                value, _ = decode_native_tool_reply(
                    request, chunks, content=content, finish_reason=finished
                )
            else:
                value, _ = decode_role_reply(
                    content, response_format=body.get("response_format")
                )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Retained successful response does not decode successfully"
            ) from error
        value, _ = normalize_reply(request, value)
        value, _ = resolve_output_texts(request, value)
        require(
            value == response.get("answer"),
            "Retained semantic answer differs from provider output bytes",
        )
        self.used.append({**source, "answer_sha256": fingerprint(value)})
        self.last_request, self.last_answer = deepcopy(request), deepcopy(value)
        return deepcopy(value)

    async def __call__(self, request):
        try:
            return await self.replay(request)
        except json.JSONDecodeError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.errors.append(str(error))
            raise

    async def replay(self, request):
        matches = self.entries.get(fingerprint(request), [])
        require(
            bool(matches),
            "Semantic replay has no exact retained provider request: "
            + request["mode"],
        )
        # A completed nonstream recovery may share the failed stream's request.
        successful = [
            row
            for row in matches
            if self.reader.path(str(Path(row[0]).parent) + "/response.json").exists()
        ]
        selected = successful or matches
        if not successful:
            recovery = [
                row
                for row in matches
                if self.reader.path(
                    str(Path(row[0]).parent) + "/transport.json"
                ).exists()
            ]
            if recovery:
                selected = recovery  # Its original failure is checked against the physical error stream.
        answers = [self.answer(name, original) for name, original in selected]
        require(
            all(value == answers[0] for value in answers),
            "Conflicting retained responses for one semantic request",
        )
        return deepcopy(answers[0])


class RetainedCatalog:
    def __init__(self, catalog):
        self.value = deepcopy(catalog)

    def catalog(self):
        return deepcopy(self.value)

    async def run(self, *_):
        raise ValueError("Origin reconstruction must not execute measurement tools")


class RetainedAuthorTools(RetainedCatalog):
    """Replay authored probes from retained execution records, never run a tool.

    UUID filenames do not establish chronology. The production author protocol
    supplies attempt/probe order and exact recovery requests. A trial's embedded
    receipt must match a distinct physical record in that same recorder scope.
    Conflicting trial histories cannot be resolved by preferring a success.
    """

    def __init__(self, reader, folder, catalog, designer, test, consumed):
        super().__init__(catalog)
        self.reader, self.designer, self.test = reader, designer, test
        self.consumed = consumed
        self.used, self.problems = [], []
        self.attempt, self.index, self.active = 0, 0, None
        locations = [folder, str(Path(folder).parent / "measurement/measurements")]
        self.trials = []
        for location in locations:
            for path in sorted(
                reader.path(location).glob("checker_author_trial_*.json")
            ):
                name = reader.local(path)
                if name in consumed:
                    continue
                trial = reader.read(name)
                if trial.get("checker") == test["checker"]:
                    self.trials.append((name, trial))

    def checked(self, condition, message):
        if not condition:
            self.problems.append(message)
            raise ValueError(message)

    def select(self):
        from reward_engine.core.outcome.context import response_payload

        request, recipe = (
            self.designer.last_request,
            deepcopy(self.designer.last_answer),
        )
        if recipe.get("need_evidence") == [] and len(recipe) > 1:
            recipe.pop("need_evidence")
        recipe = response_payload(recipe, request["response"])
        self.checked(
            request.get("mode") == "outcome_checker_author"
            and request.get("test") == self.test,
            "Trial is not bound to the current author request",
        )
        matches = [
            (name, trial)
            for name, trial in self.trials
            if name not in self.consumed
            and trial.get("attempt") == self.attempt
            and trial.get("recipe") == recipe
        ]
        self.checked(bool(matches), "No retained trial for this ordered author attempt")
        self.checked(
            len({fingerprint(trial) for _, trial in matches}) == 1,
            "Conflicting retained trials for one author attempt",
        )
        name, trial = matches[0]
        self.checked(
            len(trial.get("receipts", [])) == len(recipe["probes"]),
            "Trial receipt count differs from authored probes",
        )
        # Same recorder directory only; a receipt elsewhere in the task or in a
        # later round is not interchangeable evidence of this execution.
        physical = []
        for path in sorted(
            self.reader.path(str(Path(name).parent)).glob("measurement_tool_*.json")
        ):
            source = self.reader.local(path)
            if source not in self.consumed:
                physical.append((source, self.reader.read(source)))
        bound = []
        for probe, receipt in zip(recipe["probes"], trial["receipts"], strict=True):
            self.checked(
                receipt.get("tool") == probe["tool"]
                and receipt.get("arguments") == probe.get("arguments", {}),
                "Trial receipt differs from actual authored probe request",
            )
            exact = [(source, value) for source, value in physical if value == receipt]
            self.checked(
                bool(exact),
                "Trial receipt has no matching retained measurement_tool execution record",
            )
            source, value = exact[0]
            bound.append((source, value))
            physical = [(p, v) for p, v in physical if p != source]
        self.active = (name, trial, bound)
        self.index = 0

    async def run(self, probe, evidence):
        if self.active is None:
            self.select()
        name, trial, bound = self.active
        self.checked(
            self.index < len(bound) and probe == trial["recipe"]["probes"][self.index],
            "Probe order differs from retained author trial",
        )
        source, value = bound[self.index]
        self.index += 1
        self.consumed.add(source)
        return deepcopy(value)

    def record(self, name, value):
        self.checked(
            name.startswith("checker_author_trial_") and self.active is not None,
            "Unexpected offline author record",
        )
        source, trial, bound = self.active
        self.checked(
            value == trial and self.index == len(bound),
            "Retained author trial differs from production reconstruction",
        )
        self.used.append(
            {
                "trial": source,
                "trial_sha256": self.reader.hashes[source],
                "attempt": self.attempt,
                "tools": [
                    {"source": p, "sha256": self.reader.hashes[p]} for p, _ in bound
                ],
            }
        )
        self.consumed.add(source)
        self.attempt += 1
        self.active = None


def _offline(coroutine):
    """Allow the synchronous auditor inside controlled async tests, without nested loops."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def audit_semantic_origin(
    reader, folder, config, raw_jury, raw_pack, *, previous=None, registry=None
):
    """Reconstruct genuine semantic definitions before applying any public projection.

    Uses shipped pure construction transforms with an offline Role. In particular,
    the first reducer response need not equal the final Jury: model-authored edits,
    invalid-batch rejection, normalization and coverage reconciliation are replayed.
    Prior semantic Packs are accepted only after their own origin audit succeeded.
    """
    from reward_engine.core.outcome import builder, jury
    from reward_engine.core.outcome.harness import defaults

    checkpoint = reader.read(folder + "/jury_reduce_checkpoint.json")
    require(
        not raw_pack.get("construction"),
        "Refined construction needs its own semantic-origin replay; cannot assume legacy provenance",
    )
    targeted = not reader.path(folder + "/jury_reduce_attempt_1").exists()
    reducer = RetainedRole(
        reader, folder + "/models/jury_reducer", config, targeted=targeted
    )
    emitted = {}
    expected_jury = _offline(
        jury.reduce_proposals(
            checkpoint["context"],
            checkpoint["evidence"],
            checkpoint["inputs"],
            reducer,
            reduce_max_bytes=checkpoint["reduce_max_bytes"],
            protocol_version=checkpoint["prompt_version"],
            record=lambda name, value: emitted.update({name: deepcopy(value)}),
        )
    )
    require(
        expected_jury == raw_jury,
        "Semantic Jury differs from retained model-origin reconstruction",
    )
    for name, value in emitted.items():
        if name != "jury":
            require(
                reader.read(folder + "/" + name + ".json") == value,
                "Retained reducer/repair record differs from model-origin reconstruction: "
                + name,
            )

    context = reader.read(folder + "/learning_context.json")
    require(
        context["task"] == checkpoint["context"]["task"],
        "Builder/Jury task provenance mismatch",
    )
    designer = RetainedRole(reader, folder + "/models/harness_designer", config)
    registry = registry or defaults(None)
    catalog = raw_pack["build_scope"]["checkers"]
    expected_catalog = {
        k: v
        for k, v in registry.catalog().items()
        if k != "public_delivery.paths.v1" and k not in registry.recipes
    }
    require(
        catalog == expected_catalog,
        "Custom checker catalog has no supported frozen registry provenance",
    )
    registry.tools = RetainedCatalog(raw_pack["build_scope"]["tools"])

    authors = []
    # A shared task recorder can contain several rounds. Carry consumption only
    # from already verified earlier Packs; never reuse one physical tool call to
    # substantiate two fresh author executions.
    consumed = getattr(registry, "_origin_consumed", set())
    registry._origin_consumed = consumed

    async def author(test, registry):
        from reward_engine.core.outcome.measurement_tools import author_checker

        original_tools = registry.tools
        tools = RetainedAuthorTools(
            reader, folder, original_tools.catalog(), designer, test, consumed
        )
        authors.append(tools)
        registry.tools = tools
        try:
            await author_checker(
                test, registry, context, checkpoint["evidence"], designer
            )
        finally:
            registry.tools = original_tools

    built = {}
    expected_pack = _offline(
        builder._build_semantic(
            context,
            checkpoint["evidence"],
            expected_jury,
            registry,
            designer,
            previous=previous,
            author=author,
            record=lambda name, value: built.update({name: deepcopy(value)}),
        )
    )
    require(
        not designer.errors,
        "Retained Builder provider origin is inconsistent: "
        + "; ".join(designer.errors),
    )
    problems = [problem for tools in authors for problem in tools.problems]
    require(
        not problems,
        "Retained generated-checker execution origin is inconsistent: "
        + "; ".join(problems),
    )
    # A fresh Pack UUID is deliberately nondeterministic; its semantic content
    # and every other field must reconstruct exactly. The effective projection
    # independently binds the retained UUID and complete semantic Pack hash.
    expected_pack["pack_id"] = raw_pack["pack_id"]
    require(
        expected_pack == raw_pack,
        "Semantic Pack differs from retained model-origin compilation/reuse",
    )
    for name, value in built.items():
        if name == "pack":
            continue
        saved = reader.read(folder + "/" + name + ".json")
        if name.endswith("pack"):
            value["pack_id"] = saved["pack_id"]
        require(
            saved == value,
            "Builder construction record differs from model-origin reconstruction: "
            + name,
        )
    return {
        "schema": "public_delivery.semantic_origin.v2",
        "jury_provider_sources": reducer.used,
        "builder_provider_sources": designer.used,
        "builder_reuse": built.get("builder_reuse"),
        "checker_author_trials": [trial for tools in authors for trial in tools.used],
        "jury_sha256": fingerprint(expected_jury),
        "semantic_pack_sha256": fingerprint(raw_pack),
        "boundary": "Offline deterministic reconstruction from retained provider records; no provider/tool calls or authenticity claim",
    }, registry


def audit_projection(raw_jury, raw_pack, jury, pack, contract):
    validate(
        contract, {"task_id": raw_pack["task_id"], "prompt": raw_pack["task_prompt"]}
    )
    require(
        jury == bind_definition(raw_jury, contract),
        "Effective Jury is not the exact declared public projection",
    )
    require(
        pack == bind_pack(raw_pack, contract),
        "Effective Pack is not the exact declared public projection",
    )
    require(
        {c["id"] for c in raw_jury["criteria"]}
        == {c["id"] for c in raw_pack["criteria"]},
        "Raw Jury/Builder criteria differ",
    )
    return {
        "model_criteria": len(raw_jury["criteria"]),
        "model_checks": len(raw_pack["tests"]),
        "bound_prerequisites": 1,
        "bound_checks": 1,
        "raw_jury_sha256": fingerprint(raw_jury),
        "raw_pack_sha256": fingerprint(raw_pack),
        "effective_pack_sha256": fingerprint(pack),
    }


def audit_snapshot(reader, directory, candidate=None, *, initial_hashes=None):
    directory = str(directory)
    saved = reader.read(directory + "/candidate.json")
    require(
        candidate is None or saved == candidate,
        "Candidate differs from native snapshot",
    )
    inventory = reader.read(directory + "/inventory.json")
    envelope = reader.read(directory + "/public_delivery_inventory.json")
    require(
        saved["environment"].get("public_delivery_inventory") == envelope,
        "Candidate inventory envelope differs",
    )
    actual = scan_workspace(
        reader.path(directory + "/workspace"),
        excluded=EXCLUDED,
        initial_hashes=initial_hashes,
    )
    require(
        envelope == actual and inventory == actual["files"],
        "Inventory differs from actual snapshot bytes or completeness",
    )
    require(
        saved["environment"]["file_count"] == len(inventory),
        "Candidate file count differs",
    )
    require(
        saved["artifact"]["changed_files"] == [r for r in inventory if r["changed"]],
        "Candidate changed files differ",
    )
    for row in inventory:
        data = reader.bytes(directory + "/workspace/" + row["path"])
        require(
            len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"],
            "Snapshot file hash/size mismatch",
        )
    return saved


def audit_measurement(contract, candidate, measured, pack):
    from reward_engine.core.outcome.context import build_context
    from reward_engine.core.outcome.harness import (
        MEASUREMENT_INPUT_SCHEMA,
        measurement_context,
        measurement_input_fingerprint,
    )

    task = {
        "task_id": pack["task_id"],
        "prompt": pack["task_prompt"],
        "public_delivery_contract": contract,
    }
    context, evidence = build_context(task, [candidate])
    context, evidence = measurement_context(pack, context, evidence)
    if measured.get("review_mode") == "targeted":
        from reward_engine.core.outcome.delivery_triage import prepare_review_inputs

        context, evidence, _ = prepare_review_inputs(pack, context, evidence)
        context["measurement_inquiries"] = measured["review_input"]["measurement_plan"][
            "measurement_inquiries"
        ]
    require(
        measured.get("measurement_input_schema") == MEASUREMENT_INPUT_SCHEMA
        and measured.get("measurement_input_sha256")
        == measurement_input_fingerprint(pack, context, evidence),
        "Gate measurement input fingerprint mismatch",
    )
    expected = candidate_gate(contract, candidate)
    rows = [r for r in measured.get("results", []) if r.get("test_id") == TEST]
    require(len(rows) == 1, "Every measurement needs the mandatory public check")
    row = rows[0]
    require(
        row.get("public_delivery_receipt") == expected
        and measured.get("public_delivery_gate") == expected,
        "Gate does not match exact contract/candidate/inventory",
    )
    require(
        row["status"] == expected["status"]
        and not row.get("error")
        and not row.get("reused_unchanged_input"),
        "Public check status/execution differs from its receipt",
    )
    require(measured["pack_id"] == pack["pack_id"], "Gate measured a different Pack")
    if expected["status"] != "pass":
        require(
            not measured.get("delivery_clear")
            and not measured.get("ready_under_estimator"),
            "Nonconforming prerequisite marked acceptable",
        )
    return expected


def audit_run(root, *, required_rounds=3):
    reader = Reader(root)
    report = {
        "schema": SCHEMA,
        "status": "fail",
        "scope": "Additional opt-in public path gate only; unchanged original pipeline strict audit remains required",
        "rounds": [],
        "source_hashes": reader.hashes,
        "boundary": "Supplied executor translation and quoted provenance; no automatic natural-language coverage, semantic truth, authenticity or native-quality claim",
    }
    try:
        config, manifest = reader.read("config.json"), reader.read("manifest.json")
        binding = config["public_delivery_contract"]
        require(
            manifest.get("public_delivery_contract") == binding,
            "Public contract manifest mismatch",
        )
        require(
            config["identity"]
            == digest({k: v for k, v in config.items() if k != "identity"})
            == manifest["config_identity"],
            "Config identity mismatch",
        )
        require(reader.local(config["output"]) == ".", "Config belongs to another run")
        reader.bytes(reader.local(binding["path"]))
        tid = safe_path(binding["task_id"])
        require(
            "/" not in tid and tid in config["task_ids"], "Invalid bound task identity"
        )
        prompt_name = reader.local(Path(config["tasks_root"]) / tid / "instruction.md")
        task = {"task_id": tid, "prompt": reader.bytes(prompt_name).decode("utf-8")}
        contract = read_bound_contract(binding, task)
        quoted_name = reader.local(
            Path(config["tasks_root"]) / tid / contract["source"]["path"]
        )
        require(
            reader.bytes(quoted_name) == task["prompt"].encode(),
            "Quoted source is not actual public Task",
        )
        task["public_delivery_contract"] = contract
        for name, sha in manifest["code_hashes"].items():
            require(
                hashlib.sha256(
                    reader.bytes(reader.local(Path(config["code"]) / safe_path(name)))
                ).hexdigest()
                == sha,
                "Frozen source changed: " + name,
            )
        # Reconstruct with the frozen package, including transitive pure helpers.
        package = Path(__file__).resolve().parents[2]
        frozen_package = reader.path(
            reader.local(Path(config["code"]) / "reward_engine")
        )
        declared = {
            name
            for name in manifest["code_hashes"]
            if name.startswith("reward_engine/")
        }
        actual = {
            "reward_engine/" + p.relative_to(frozen_package).as_posix()
            for p in frozen_package.rglob("*.py")
        }
        require(actual == declared, "Frozen package manifest omitted or added modules")
        for name in declared:
            installed = package / name.removeprefix("reward_engine/")
            require(
                hashlib.sha256(installed.read_bytes()).hexdigest()
                == manifest["code_hashes"][name],
                "Run frozen auditor: installed package differs from frozen source",
            )
        for module in (
            "core/outcome/public_delivery.py",
            "core/outcome/context.py",
            "core/outcome/harness.py",
            "core/outcome/delivery_triage.py",
            "core/outcome/builder.py",
            "core/outcome/loop.py",
            "adapters/workbuddy/evidence.py",
            "adapters/workbuddy/agent.py",
            "adapters/workbuddy/public_delivery.py",
            "adapters/workbuddy/public_delivery_audit.py",
        ):
            require(
                "reward_engine/" + module in declared,
                "Required frozen public-delivery module missing",
            )
        for name, sha in manifest["prepared_dataset_hashes"].items():
            data = reader.bytes(
                reader.local(Path(config["tasks_root"]).parent / safe_path(name))
            )
            require(
                hashlib.sha256(data).hexdigest() == sha,
                "Prepared public task dataset changed",
            )
        status = reader.read("validation_status.json")
        require(
            status.get("status") == "execution_completed"
            and status.get("completed_rounds") == required_rounds,
            "Opt-in strict gate requires all terminal rounds",
        )
        task_dir = "tasks/" + tid
        report["contract_sha256"] = fingerprint(contract)
        report["public_prompt_sha256"] = contract["public_prompt_sha256"]
        # Check seed snapshots too; they remain observations, not mandatory successes.
        incoming = []
        for index in range(1, 4):
            slot = f"{task_dir}/seeds/{index}"
            initial = audit_snapshot(reader, slot + "/snapshots/initial")
            hashes = {
                r["path"]: r["sha256"]
                for r in initial["environment"]["public_delivery_inventory"]["files"]
            }
            state, result = (
                reader.read(slot + "/actor_state.json"),
                reader.read(slot + "/RESULT.json"),
            )
            incoming.append(
                audit_snapshot(
                    reader,
                    f"{slot}/snapshots/delivery-{state['turn']}",
                    result["candidate"],
                    initial_hashes=hashes,
                )
            )
        previous_semantic, origin_registry = None, None
        for number in range(1, required_rounds + 1):
            folder = f"{task_dir}/round_{number}"
            pack = reader.read(folder + "/pack.json")
            facts = audit_projection(
                reader.read(folder + "/jury_semantic.json"),
                reader.read(folder + "/pack_semantic.json"),
                reader.read(folder + "/jury.json"),
                pack,
                contract,
            )
            origin, origin_registry = audit_semantic_origin(
                reader,
                folder,
                config,
                reader.read(folder + "/jury_semantic.json"),
                reader.read(folder + "/pack_semantic.json"),
                previous=previous_semantic,
                registry=origin_registry,
            )
            previous_semantic = reader.read(folder + "/pack_semantic.json")
            facts.update(round=number, arms={}, semantic_origin=origin)
            construction = reader.read(folder + "/construction_checks.json")
            require(
                len(construction) == len(incoming),
                "Construction public-gate coverage differs from actual incoming pool",
            )
            facts["construction_gates"] = [
                audit_measurement(contract, candidate, measured, pack)
                for candidate, measured in zip(incoming, construction, strict=True)
            ]
            checkpoint = reader.read(folder + "/checkpoint.json")
            require(
                checkpoint["task"] == task,
                "Checkpoint dropped or changed the public binding",
            )
            for arm in "ABC":
                slot = folder + "/actors/" + arm
                state, result = (
                    reader.read(slot + "/actor_state.json"),
                    reader.read(slot + "/RESULT.json"),
                )
                require(result.get("status") == "completed", "Actor not completed")
                initial = audit_snapshot(reader, slot + "/snapshots/initial")
                hashes = {
                    r["path"]: r["sha256"]
                    for r in initial["environment"]["public_delivery_inventory"][
                        "files"
                    ]
                }
                require(
                    type(state["turn"]) is int
                    and 1 <= state["turn"] <= (3 if arm == "B" else 1),
                    "Unexpected Actor turns",
                )
                versions = []
                for turn in range(1, state["turn"] + 1):
                    candidate = audit_snapshot(
                        reader,
                        f"{slot}/snapshots/delivery-{turn}",
                        initial_hashes=hashes,
                    )
                    if arm == "B":
                        checked = reader.read(f"{slot}/checks/delivery_{turn}.json")
                        require(
                            checked["candidate"] == candidate,
                            "B review does not bind its actual snapshot",
                        )
                        versions.append(
                            audit_measurement(
                                contract, candidate, checked["measurement"], pack
                            )
                        )
                require(
                    result["candidate"] == candidate,
                    "Actor RESULT is not final native snapshot",
                )
                delivery = reader.read(folder + f"/{arm}/delivery.json")
                require(
                    delivery["candidate"] == candidate,
                    "Final delivery differs from native snapshot",
                )
                final = audit_measurement(
                    contract,
                    candidate,
                    reader.read(folder + f"/{arm}/outcome_estimate.json"),
                    pack,
                )
                facts["arms"][arm] = {"versions": versions, "final": final}
                incoming.append(candidate)
            selection = reader.read(folder + "/selected_delivery.json")
            arm = selection.get("arm")
            require(
                arm in facts["arms"]
                and facts["arms"][arm]["final"]["status"] == "pass",
                "Selected final delivery does not conform",
            )
            require(
                selection.get("public_delivery_acceptable") is True
                and selection.get("public_delivery_gates")
                == {a: facts["arms"][a]["final"] for a in "ABC"},
                "Selection path-gate receipts mismatch",
            )
            require(
                selection.get("source") == f"round_{number}/{arm}/delivery.json",
                "Selected delivery source mismatch",
            )
            facts["selected_arm"] = arm
            report["rounds"].append(facts)
        report["status"] = "pass"
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["reason"] = f"{type(error).__name__}: {error}"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args(argv)
    report = audit_run(args.run_directory)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
