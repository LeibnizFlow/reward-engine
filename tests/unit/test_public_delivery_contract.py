"""Real workspace collection and scripted semantic false positives; no paid API."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.io import write
from reward_engine.adapters.workbuddy.public_delivery import (
    freeze_contract,
    read_bound_contract,
)
from reward_engine.adapters.workbuddy.public_delivery_audit import (
    Reader,
    audit_measurement,
    audit_projection,
    audit_snapshot,
)
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.delivery_triage import measurement_plan
from reward_engine.core.outcome.harness import (
    defaults,
    measure,
    measurement_context,
    measurement_input_fingerprint,
)
from reward_engine.core.outcome.loop import arm_prompts, select_delivery
from reward_engine.core.outcome.public_delivery import (
    CHECKER,
    CRITERION,
    TEST,
    actor_instruction,
    bind_definition,
    bind_pack,
    candidate_gate,
    mandatory_tests,
    semantic_pack,
    validate,
)
from reward_engine.core.outcome.runner import DeliveryReview


def task_and_contract(path="manual.md"):
    task = {
        "task_id": "public-manual",
        "prompt": f"Work in the current task directory.\nSave the final manual to `{path}`.\n",
    }
    contract = {
        "schema": "public_delivery.contract.v1",
        "task_id": task["task_id"],
        "public_prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
        "source": {
            "kind": "public_task_instruction",
            "path": "instruction.md",
            "quotes": [
                {"start_line": n, "end_line": n, "text": line}
                for n, line in enumerate(task["prompt"].splitlines(), 1)
            ],
        },
        "translation": {
            "method": "explicit_executor_translation",
            "authority": "Controlled fixture explicitly supplied public binding",
            "scope": "Exact file presence only",
        },
        "requirements": [
            {
                "id": "manual",
                "path": path,
                "kind": "regular_file",
                "source_quote_indices": [0, 1],
            }
        ],
    }
    task["public_delivery_contract"] = contract
    return task, contract


def raw_pack(task):
    criterion = {
        "id": "content",
        "kind": "soft",
        "origin": "task",
        "standard": "Useful manual",
        "measurement_question": "Is the manual useful?",
        "evidence_refs": ["task"],
        "anchors": {"fail": "Unusable", "partial": "Incomplete", "full": "Useful"},
    }
    return {
        "pack_id": "semantic-pack",
        "task_id": task["task_id"],
        "task_prompt": task["prompt"],
        "status": "callable",
        "criteria": [criterion],
        "tests": [
            {
                "test_id": "semantic",
                "criterion_id": "content",
                "checker": "scripted.semantic",
                "binding_status": "callable",
                "rubric": criterion,
                "question": "Useful?",
            }
        ],
    }


def snapshot(
    root,
    paths,
    *,
    initial=None,
    candidate_id="candidate",
    final_response="A complete useful manual",
):
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    for name, text in paths.items():
        file = workspace / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text)
    return collect(
        workspace,
        candidate_id=candidate_id,
        final_response=final_response,
        initial_hashes=initial,
        output=root,
        public_delivery=True,
    )[0]


def registry():
    async def model(request):
        assert request["mode"] == "outcome_delivery_review"
        return {
            "status": "target_satisfied",
            "reason": "Scripted favorable semantic judgment",
            "gaps": [],
            "verify": [],
            "claim_checks": [],
            "outcome_goal": "Deliver a useful manual",
            "full_score_gaps": [],
            "improvements": [],
        }

    result = defaults(model)

    async def favorable(test, context, evidence):
        return {
            "status": "pass",
            "reason": "Scripted semantic false positive retained",
            "gap": "",
            "evidence_refs": ["product_1:artifact:final_response"],
        }

    result.register(
        "scripted.semantic",
        favorable,
        "Controlled semantic judgment",
        kind="model_assisted",
    )
    return result


@pytest.mark.asyncio
async def test_wrong_path_semantic_pass_favorable_rank_cannot_accept(tmp_path):
    task, contract = task_and_contract()
    pack = bind_pack(raw_pack(task), contract)
    wrong = snapshot(
        tmp_path / "wrong", {"output/manual.md": "Authored complete manual"}
    )
    context, evidence = build_context(task, [wrong])
    measured = await measure(pack, registry(), context, evidence)
    assert measured["results"][0]["status"] == "pass"
    assert (
        measured["results"][0]["reason"] == "Scripted semantic false positive retained"
    )
    assert measured["public_delivery_gate"]["status"] == "fail"
    assert not measured["ready_under_estimator"]
    selection = select_delivery(
        {"status": "compared", "groups": [["B"], ["A"], ["C"]]},
        {a: a for a in "ABC"},
        {a: measured for a in "ABC"},
        public_delivery_contract=contract,
    )
    assert selection["arm"] is None and selection["public_delivery_acceptable"] is False
    assert audit_measurement(contract, wrong, measured, pack)["status"] == "fail"


@pytest.mark.asyncio
async def test_review_wrong_then_correct_retains_raw_judgments_and_original_audit_binding(
    tmp_path,
):
    from experiments.pipeline_audit import review_binding

    task, contract = task_and_contract()
    pack = bind_pack(raw_pack(task), contract)
    wrong = snapshot(tmp_path / "one", {"output/manual.md": "Good content"})
    right = snapshot(
        tmp_path / "two",
        {"manual.md": "Same good content", "output/manual.md": "Good content"},
    )
    records = {}
    controller = DeliveryReview(
        task,
        pack,
        registry(),
        max_tests=3,
        record=lambda k, v: records.update({k: deepcopy(v)}),
    )
    feedback = await controller(wrong)
    assert feedback and "manual.md" in feedback and "absent" in feedback
    assert (
        controller.history[0]["review_input"]["overview_before_measurement"]["status"]
        == "target_satisfied"
    )
    assert controller.history[0]["delivery_review"]["status"] == "needs_improvement"
    assert controller.history[0]["public_delivery_gate"]["status"] == "fail"
    assert not controller.history[0]["delivery_clear"]
    assert await controller(right) is None
    assert controller.status == "full_score_target_satisfied"
    assert [row["public_delivery_gate"]["status"] for row in controller.history] == [
        "fail",
        "pass",
    ]
    assert (
        len(controller.feedback) == 1
        and len(controller.export_session()["versions"]) == 2
    )
    previous = None
    bare_task = {k: v for k, v in task.items() if k != "public_delivery_contract"}
    for candidate, measured in zip((wrong, right), controller.history, strict=True):
        # Immutable original auditor rebuilds basic Task from task_progress: Pack restores the binding.
        _, previous = review_binding(measured, pack, bare_task, candidate, previous)
        audit_measurement(contract, candidate, measured, pack)
    assert records["delivery_1"]["candidate"] == wrong


@pytest.mark.parametrize(
    "required,actual,expected",
    [
        ("manual.md", "output/manual.md", "fail"),
        ("manual.md", "old-manual.md", "fail"),
        ("output/manual.md", "manual.md", "fail"),
        ("output/manual.md", "output/manual.md", "pass"),
        ("manual.md", "manual.md", "pass"),
    ],
)
def test_exact_paths_no_suffix_alias(tmp_path, required, actual, expected):
    _, contract = task_and_contract(required)
    candidate = snapshot(tmp_path / "s", {actual: "Authored text"})
    assert candidate_gate(contract, candidate)["status"] == expected


def test_unchanged_required_file_is_in_complete_inventory(tmp_path):
    _, contract = task_and_contract()
    digest = hashlib.sha256(b"Already authored").hexdigest()
    candidate = snapshot(
        tmp_path / "s", {"manual.md": "Already authored"}, initial={"manual.md": digest}
    )
    assert candidate["artifact"]["changed_files"] == []
    assert candidate_gate(contract, candidate)["status"] == "pass"
    assert (
        audit_snapshot(
            Reader(tmp_path), "s", candidate, initial_hashes={"manual.md": digest}
        )
        == candidate
    )


def test_incomplete_absence_unknown_and_excluded_is_not_missing(tmp_path):
    _, contract = task_and_contract()
    candidate = snapshot(tmp_path / "s", {"other.md": "text"})
    candidate["environment"]["public_delivery_inventory"]["complete"] = False
    assert candidate_gate(contract, candidate)["status"] == "unknown"
    _, excluded = task_and_contract(".cache/manual.md")
    hidden = snapshot(tmp_path / "hidden", {".cache/manual.md": "Actual but excluded"})
    assert candidate_gate(excluded, hidden)["status"] == "unknown"
    assert hidden["environment"]["public_delivery_inventory"]["files"] == []


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_leaf_or_ancestor_never_counts_as_regular_delivery(tmp_path, directory):
    target = tmp_path / "outside"
    target.mkdir()
    (target / "manual.md").write_text("Real text outside workspace")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ("output" if directory else "manual.md")).symlink_to(
        target if directory else target / "manual.md"
    )
    candidate, _ = collect(
        workspace, candidate_id="linked", final_response="Done", public_delivery=True
    )
    _, contract = task_and_contract("output/manual.md" if directory else "manual.md")
    assert candidate_gate(contract, candidate)["status"] == "fail"
    assert not candidate["environment"]["public_delivery_inventory"]["files"]


def test_directory_is_not_a_regular_file(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "manual.md").mkdir(parents=True)
    candidate, _ = collect(
        workspace, candidate_id="dir", final_response="Done", public_delivery=True
    )
    assert candidate_gate(task_and_contract()[1], candidate)["status"] == "fail"


@pytest.mark.parametrize(
    "path",
    [
        "../manual.md",
        "/manual.md",
        "a/../manual.md",
        "./manual.md",
        "a//manual.md",
        "a\\manual.md",
        "manual.md/",
        "x:\\manual.md",
        "x\n.md",
    ],
)
def test_unsafe_paths_rejected(path):
    task, contract = task_and_contract(path)
    with pytest.raises(ValueError):
        validate(contract, task)


@pytest.mark.parametrize(
    "change", ["prompt", "quote", "task", "source", "method", "refs"]
)
def test_stale_or_unproven_binding_rejected(change):
    task, contract = task_and_contract()
    contract = deepcopy(contract)
    if change == "prompt":
        task["prompt"] += "Changed public task."
    elif change == "quote":
        contract["source"]["quotes"][0]["text"] = "Invented"
    elif change == "task":
        contract["task_id"] = "different"
    elif change == "source":
        contract["source"]["kind"] = "private_grader"
    elif change == "method":
        contract["translation"]["method"] = "automatic_regex"
    else:
        contract["requirements"][0]["source_quote_indices"] = [999]
    with pytest.raises(ValueError):
        validate(contract, task)


def test_mandatory_check_outside_semantic_quota(tmp_path):
    task, contract = task_and_contract()
    pack = bind_pack(raw_pack(task), contract)
    raw = raw_pack(task)
    raw["tests"].append({**raw["tests"][0], "test_id": "other"})
    pack = bind_pack(raw, contract)
    overview = {
        "verify": [
            {"test_id": "semantic", "reason": "Content"},
            {"test_id": "other", "reason": "More content"},
        ]
    }
    plan = measurement_plan(pack, overview, max_tests=1)
    assert plan["selected_tests"] == [TEST, "semantic"]
    assert plan["deferred_tests"] == ["other"]


@pytest.mark.asyncio
async def test_empty_selected_tests_still_executes_gate_no_reuse(tmp_path):
    task, contract = task_and_contract()
    pack = bind_pack(raw_pack(task), contract)
    candidate = snapshot(tmp_path / "s", {"manual.md": "text"})
    c, e = build_context(task, [candidate])
    reg = registry()
    first = await measure(pack, reg, c, e, test_ids=[])
    second = await measure(pack, reg, c, e, test_ids=[], previous=(c, e, first))
    assert [r["test_id"] for r in first["results"]] == [TEST]
    assert second["execution_summary"]["submitted_checks"] == 1
    assert second["execution_summary"]["reused_checks"] == 0


def test_projection_rejects_unrelated_effective_changes():
    task, contract = task_and_contract()
    raw = raw_pack(task)
    jury = {"criteria": deepcopy(raw["criteria"])}
    effective, pack = bind_definition(jury, contract), bind_pack(raw, contract)
    assert semantic_pack(pack) == raw
    facts = audit_projection(jury, raw, effective, pack, contract)
    assert facts["model_criteria"] == 1 and facts["bound_prerequisites"] == 1
    for target, field in (
        (effective["criteria"][0], "standard"),
        (pack["tests"][0], "question"),
    ):
        old = target[field]
        target[field] = "Unrelated executor rewrite"
        with pytest.raises(ValueError, match="exact declared public projection"):
            audit_projection(jury, raw, effective, pack, contract)
        target[field] = old


@pytest.mark.asyncio
async def test_builder_retains_semantic_pack_and_model_output(tmp_path):
    from reward_engine.core.outcome.builder import build
    from tests.support.claw_outcome_builder_harness import setup

    _, _, jury, reply = setup()
    task, contract = task_and_contract()
    candidate = snapshot(tmp_path / "s", {"manual.md": "Content"}, final_response="{}")
    context, evidence = build_context(task, [candidate])
    rawreply = deepcopy(reply)
    records = {}
    requests = []

    async def call(request):
        requests.append(deepcopy(request))
        return reply

    pack = await build(
        context,
        evidence,
        jury,
        defaults(call),
        call,
        record=lambda k, v: records.update({k: deepcopy(v)}),
    )
    assert reply == rawreply
    assert records["builder_attempt_1"]["output"] == rawreply
    assert records["pack"] == bind_pack(records["pack_semantic"], contract) == pack
    assert len(records["pack_semantic"]["tests"]) == 1 and len(pack["tests"]) == 2
    assert len(requests) == 1


def test_contract_is_in_task_context_actor_prompt_and_identity(tmp_path):
    task, contract = task_and_contract()
    candidate = snapshot(tmp_path / "s", {"manual.md": "text"})
    pack = bind_pack(raw_pack(task), contract)
    c, e = build_context(task, [candidate])
    c, e = measurement_context(pack, c, e)
    assert c["task"]["public_delivery_contract"] == contract
    assert all(
        "manual.md" in text
        for text in arm_prompts(pack, {"status": "keep", "method": ""}).values()
    )
    assert "source_quote_indices" not in actor_instruction(
        contract
    )  # prompt carries exact quotes and paths, not private inputs
    changed = deepcopy(contract)
    changed["translation"]["scope"] += " Version 2."
    pack2 = bind_pack(raw_pack(task), changed)
    assert pack2["pack_id"] != pack["pack_id"]
    c2, e2 = build_context({**task, "public_delivery_contract": changed}, [candidate])
    assert measurement_input_fingerprint(pack, c, e) != measurement_input_fingerprint(
        pack2, c2, e2
    )
    with pytest.raises(ValueError, match="missing from published"):
        measurement_context(raw_pack(task), c, e)


def test_freeze_binds_public_source_and_exact_data_bytes(tmp_path):
    task, contract = task_and_contract()
    taskroot = tmp_path / "tasks" / task["task_id"]
    taskroot.mkdir(parents=True)
    (taskroot / "instruction.md").write_text(task["prompt"])
    source = tmp_path / "contract.json"
    write(source, contract)
    output = tmp_path / "run"
    output.mkdir()
    binding = freeze_contract(source, output, tmp_path / "tasks", [task["task_id"]])
    assert Path(binding["path"]).read_bytes() == source.read_bytes()
    assert read_bound_contract(binding, task) == contract
    Path(binding["path"]).write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        read_bound_contract(binding, task)


def test_actual_snapshot_byte_tampering_is_rejected(tmp_path):
    candidate = snapshot(tmp_path / "s", {"manual.md": "original"})
    audit_snapshot(Reader(tmp_path), "s", candidate)
    (tmp_path / "s/workspace/manual.md").write_text("tampered")
    with pytest.raises(ValueError, match="actual snapshot bytes"):
        audit_snapshot(Reader(tmp_path), "s", candidate)


def test_no_contract_collection_and_selection_unchanged(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "x.md").write_text("x")
    candidate, _ = collect(workspace, candidate_id="unbound", final_response="done")
    assert "public_delivery_inventory" not in candidate["environment"]
    context, _ = build_context({"task_id": "x", "prompt": "Task"}, [candidate])
    assert "candidate_sha256" not in context["products"][0]
    assert select_delivery({"status": "unavailable"}, {"A": "p"}, {})["arm"] == "A"


def test_tracked_binding_matches_exact_public_instruction():
    repo = Path(__file__).resolve().parents[2]
    contract = json.loads(
        (
            repo / "configs/public_delivery/api-usage-explain-cli-l3-001.v1.json"
        ).read_text()
    )
    prompt = (
        repo / "configs/public_delivery/api-usage-explain-cli-l3-001.instruction.md"
    )
    # Exact quoted source SHA is independently reviewable; the test does not need archived runs.
    assert (
        contract["public_prompt_sha256"]
        == "80c117555e628bc9915f5f4b42a1c56de71d93028a6031f9a5456bd85993f6ed"
    )
    assert [q["start_line"] for q in contract["source"]["quotes"]] == [1, 21]
    assert contract["translation"]["method"] == "explicit_executor_translation"
    validate(
        contract,
        {"task_id": contract["task_id"], "prompt": prompt.read_bytes().decode()},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [("checker", CHECKER), ("test_id", TEST), ("criterion_id", CRITERION)],
)
async def test_model_cannot_author_reserved_checker_even_if_registered(field, value):
    from reward_engine.core.outcome.builder import build
    from reward_engine.core.outcome.public_delivery import install
    from tests.support.claw_outcome_builder_harness import setup

    context, evidence, jury, reply = setup()
    reply["tests"][0][field] = value
    records = {}

    async def model(_):
        return reply

    reg = defaults(model)
    install(reg)
    with pytest.raises(ValueError, match="reserved public prerequisite"):
        await build(
            context,
            evidence,
            jury,
            reg,
            model,
            record=lambda k, v: records.update({k: deepcopy(v)}),
        )
    assert records["builder_attempt_1"]["output"] == reply


@pytest.mark.asyncio
async def test_model_and_artifact_metadata_cannot_supply_authority(tmp_path):
    from reward_engine.core.outcome.builder import build
    from tests.support.claw_outcome_builder_harness import setup

    _, _, jury, reply = setup()
    task, contract = task_and_contract()
    bare = {k: v for k, v in task.items() if k != "public_delivery_contract"}
    candidate = snapshot(
        tmp_path / "s", {"output/manual.md": "text"}, final_response="{}"
    )
    candidate["artifact"]["public_delivery_contract"] = contract
    jury["public_delivery_contract"] = contract
    reply["public_delivery_contract"] = contract
    context, evidence = build_context(bare, [candidate])

    async def model(_):
        return reply

    unbound = await build(context, evidence, jury, defaults(model), model)
    assert "public_delivery_contract" not in unbound
    assert "public_delivery_contract" not in context["task"]
    assert mandatory_tests(unbound) == []
    forged = deepcopy(contract)
    forged["requirements"][0]["path"] = "output/manual.md"
    jury["public_delivery_contract"] = forged
    effective = bind_definition(jury, contract)
    assert effective["public_delivery_contract"] == contract
    assert effective["public_delivery_projection"]["ignored_untrusted_metadata"] == [
        "public_delivery_contract"
    ]
    assert jury["public_delivery_contract"] == forged


def test_conflicting_or_stale_pack_rejected_before_measurement(tmp_path):
    task, contract = task_and_contract()
    candidate = snapshot(tmp_path / "s", {"manual.md": "text"})
    c, e = build_context(task, [candidate])
    pack = bind_pack(raw_pack(task), contract)
    conflict = deepcopy(contract)
    conflict["requirements"][0]["path"] = "different.md"
    with pytest.raises(ValueError, match="Conflicting public delivery"):
        measurement_context(bind_pack(raw_pack(task), conflict), c, e)
    pack["public_delivery_contract"]["public_prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Stale public instruction"):
        measurement_context(pack, c, e)


def test_collection_read_failure_is_unknown_without_invented_absence(
    tmp_path, monkeypatch
):
    import os

    from reward_engine.adapters.workbuddy import public_delivery

    workspace = tmp_path / "w"
    workspace.mkdir()
    (workspace / "manual.md").write_text("Unreadable")
    original = os.open

    def denied(path, *args, **kwargs):
        if Path(path).name == "manual.md":
            raise PermissionError("controlled unreadable file")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(public_delivery.os, "open", denied)
    candidate, _ = collect(
        workspace,
        candidate_id="unreadable",
        final_response="done",
        public_delivery=True,
    )
    inventory = candidate["environment"]["public_delivery_inventory"]
    assert inventory["complete"] is False and inventory["blocked"] == [
        {"path": "manual.md", "reason": "read_error"}
    ]
    assert candidate_gate(task_and_contract()[1], candidate)["status"] == "unknown"


@pytest.mark.asyncio
async def test_selection_filters_wrong_b_but_keeps_right_a_or_c(tmp_path):
    task, contract = task_and_contract()
    pack = bind_pack(raw_pack(task), contract)
    candidates = {
        a: snapshot(
            tmp_path / a, {("manual.md" if a in "AC" else "output/manual.md"): a}
        )
        for a in "ABC"
    }
    measurements = {}
    for a, candidate in candidates.items():
        c, e = build_context(task, [candidate])
        measurements[a] = await measure(pack, registry(), c, e)
    ranking = {"status": "compared", "groups": [["B"], ["C"], ["A"]]}
    result = select_delivery(
        ranking, {a: a for a in "ABC"}, measurements, public_delivery_contract=contract
    )
    assert result["arm"] == "C" and result["public_delivery_acceptable"]
    result = select_delivery(
        {"status": "unavailable"},
        {a: a for a in "ABC"},
        measurements,
        public_delivery_contract=contract,
    )
    assert result["arm"] == "A"
    measurements["A"] = measurements["B"]
    result = select_delivery(
        ranking, {a: a for a in "ABC"}, measurements, public_delivery_contract=contract
    )
    assert result["arm"] == "C"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["direct", "targeted", "atomic", "reuse", "partial_reuse", "syntax"]
)
async def test_full_scripted_pipeline_and_additional_auditor(
    tmp_path, monkeypatch, scenario
):
    """Real Jury/Builder/controller/loop, actual collected snapshots, no parser mock."""
    import httpx

    from reward_engine.adapters.workbuddy import models
    from reward_engine.adapters.workbuddy.public_delivery_audit import audit_run
    from reward_engine.core.batch import BatchPolicy
    from reward_engine.core.outcome.loop import run_task
    from tests.unit.test_public_delivery_manifest import prepare_public_run

    rounds = 3 if scenario in {"reuse", "partial_reuse"} else 1
    task, contract = task_and_contract()
    config = prepare_public_run(tmp_path, monkeypatch, task, contract)
    root = Path(config["output"])
    taskdir = root / "tasks" / task["task_id"]
    modes = []

    def record(name, value):
        write(taskdir / (name + ".json"), value)

    def native_snapshot(slot, turn, paths, initial):
        return snapshot(
            slot / f"snapshots/delivery-{turn}",
            paths,
            initial=initial,
            final_response="{}",
            candidate_id=slot.name,
        )

    def initial_snapshot(slot):
        snapshot(
            slot / "snapshots/initial", {}, candidate_id=slot.name, final_response=""
        )
        return {}

    seeds = []
    for n in range(1, 4):
        slot = taskdir / f"seeds/{n}"
        initial = initial_snapshot(slot)
        candidate = native_snapshot(slot, 1, {"manual.md": "Seed manual"}, initial)
        seeds.append(candidate)
        write(slot / "actor_state.json", {"turn": 1})
        write(slot / "RESULT.json", {"status": "completed", "candidate": candidate})

    responses = {}
    original_client = httpx.AsyncClient
    syntax_sent = False

    async def upstream(request):
        nonlocal syntax_sent
        body = json.loads(request.content)
        answer = await responses[body["model"]](
            json.loads(body["messages"][-1]["content"])
        )
        output = json.dumps(answer)
        if (
            scenario == "syntax"
            and body["model"].endswith("/jury_reducer")
            and not syntax_sent
        ):
            syntax_sent = True
            output = "{invalid syntax"
        event = {
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": output},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    monkeypatch.setattr(models, "auth", lambda *_: "offline-only")
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(upstream), **kwargs
        ),
    )

    def roles(role, prefix):
        async def call(request):
            mode = request["mode"]
            modes.append((prefix, role, mode))
            if "ranking" in mode:
                products = request["context"]["products"]
                groups = [[p["product_id"]] for p in reversed(products)]
                # Prefer current B over current A, including the scripted semantic false positive.
                ids = [p["product_id"] for p in products]
                groups = [[p] for p in [ids[-2], ids[-3], ids[-1], *ids[:-3]]]
                return {
                    "groups": groups,
                    "unknown": [],
                    "reason": "Favorable semantic ranking",
                    "observations": [
                        {
                            "product_id": p["product_id"],
                            "reason": "Observed authored output",
                            "evidence_refs": [
                                p["product_id"] + ":artifact:final_response"
                            ],
                        }
                        for p in products
                    ],
                }
            if mode == "outcome_jury_repair":
                edits = [
                    {
                        "op": "replace",
                        "path": ["criteria", 0, "evidence_refs"],
                        "value": ["task"],
                        "reason": "Use actual public task evidence",
                    }
                ]
                if scenario == "atomic" and not request["validation_repair"].get(
                    "invalid_edits"
                ):
                    edits.append(
                        {
                            "op": "add",
                            "path": ["criteria", 0, "bogus"],
                            "value": "not allowed",
                            "reason": "Invalid batch control",
                        }
                    )
                if request["validation_repair"].get("invalid_edits"):
                    assert (
                        request["validation_repair"]["invalid_draft"]["criteria"][0][
                            "evidence_refs"
                        ]
                        == []
                    )
                return {"edits": edits}
            if "jury" in mode or mode == "outcome_expectations":
                reply = {
                    "criteria": [
                        {
                            "id": "valid",
                            "standard": "Valid JSON final reply",
                            "kind": "hard",
                            "origin": "task",
                            "measurement_question": "Does the reply parse?",
                            "evidence_refs": []
                            if mode == "outcome_jury_reduce"
                            and scenario in {"targeted", "atomic"}
                            else ["task"],
                        }
                    ],
                    "gaps": [],
                    "coverage": [
                        {
                            "expectation_id": e["expectation_id"],
                            "criterion_ids": ["valid"],
                            "status": "mapped",
                            "reason": "Fixture expectation",
                            "subrequirements": [
                                {
                                    "part_id": p["part_id"],
                                    "status": "mapped",
                                    "criterion_ids": ["valid"],
                                    "reason": "Fixture mapping",
                                    "retained_text": "Valid JSON final reply",
                                }
                                for p in e["subrequirements"]
                            ],
                        }
                        for e in request.get("independent_expectations", [])
                        + request.get("proposal_expectations", [])
                    ],
                }
                if (
                    scenario == "partial_reuse"
                    and prefix != "round_1"
                    and mode == "outcome_jury_reduce"
                ):
                    reply["criteria"].append(
                        {
                            "id": "syntax",
                            "standard": "Final reply has JSON syntax",
                            "kind": "hard",
                            "origin": "task",
                            "measurement_question": "Is JSON syntax valid?",
                            "evidence_refs": ["task"],
                        }
                    )
                return reply
            if "hypothesis" in mode:
                return {"status": "keep", "method": ""}
            if mode == "outcome_builder":
                return {
                    "rubrics": [
                        {
                            "criterion_id": cid,
                            "anchors": {
                                "fail": "invalid",
                                "partial": "fragment",
                                "full": "valid",
                            },
                        }
                        for cid in request["required_criterion_ids"]
                    ],
                    "tests": [
                        {
                            "test_id": "parse_" + cid,
                            "criterion_id": cid,
                            "checker": "json.parse",
                            "question": "Parse?",
                            "arguments": {"ref": "product_1:artifact:final_response"},
                        }
                        for cid in request["required_criterion_ids"]
                    ],
                }
            raise AssertionError(mode)

        model = prefix + "/" + role
        responses[model] = call
        return models.Role(
            {
                **config,
                "roles": {**config["roles"], role: model},
                "gateway_local": "http://offline.invalid",
            },
            role,
            taskdir / prefix / "models" / role,
        )

    def pipeline_registry():
        result = registry()
        # This end-to-end fixture uses the shipped json.parse checker. Custom
        # registries need separate frozen implementation provenance in the audit.
        for mapping in (result.checkers, result.kinds, result.dependencies):
            mapping.pop("scripted.semantic")
        return result

    async def sample(arm, number, prompt, pack):
        assert "manual.md" in prompt
        slot = taskdir / f"round_{number}/actors/{arm}"
        initial = initial_snapshot(slot)
        path = "output/manual.md" if arm == "B" else "manual.md"
        candidate = native_snapshot(slot, 1, {path: "Authored manual"}, initial)
        controller = None
        turns = 1
        if pack:
            controller = DeliveryReview(
                task,
                pack,
                pipeline_registry(),
                record=lambda name, row: write(slot / "checks" / (name + ".json"), row),
            )
            feedback = await controller(candidate)
            assert feedback and "manual.md" in feedback
            candidate = native_snapshot(
                slot,
                2,
                {"manual.md": "Authored corrected manual", path: "Original retained"},
                initial,
            )
            assert await controller(candidate) is None
            turns = 2
        row = {
            "status": "completed",
            "candidate": candidate,
            "cost": {},
            "review": deepcopy(controller.history) if controller else [],
        }
        write(slot / "actor_state.json", {"turn": turns})
        write(slot / "RESULT.json", row)
        return row

    result = await run_task(
        task,
        seeds,
        rounds=rounds,
        roles=roles,
        registry=pipeline_registry(),
        sample=sample,
        full_audit=True,
        continuation_policy="fixed_rounds",
        record=record,
        policy=BatchPolicy(deadline_s=10),
    )
    assert result["completed_rounds"] == rounds
    assert result["selected_delivery"]["arm"] == "B"
    assert len([m for m in modes if m[2] == "outcome_jury_map"]) == 3 * rounds
    assert len([m for m in modes if m[2] == "outcome_builder"]) == (
        2 if scenario == "partial_reuse" else 1
    )
    write(
        root / "validation_status.json",
        {"status": "execution_completed", "completed_rounds": rounds},
    )
    report = audit_run(root, required_rounds=rounds)
    assert report["status"] == "pass", report
    assert report["code_binding"]["complete_file_set"] is True
    assert (
        "code/reward_engine/legacy/adapters/opencode_plugin.mjs"
        in report["source_hashes"]
    )
    assert [r["status"] for r in report["rounds"][0]["arms"]["B"]["versions"]] == [
        "fail",
        "pass",
    ]
    assert (
        report["rounds"][0]["model_criteria"]
        == report["rounds"][0]["bound_prerequisites"]
        == 1
    )
    # The default strict gate still requires the full three rounds.
    assert audit_run(root)["status"] == ("pass" if rounds == 3 else "fail")
    for row in report["rounds"]:
        assert row["semantic_origin"]["jury_provider_sources"]
    if scenario == "reuse":
        assert report["rounds"][0]["semantic_origin"]["builder_provider_sources"]
        assert all(
            not row["semantic_origin"]["builder_provider_sources"]
            for row in report["rounds"][1:]
        )
    if scenario == "partial_reuse":
        origin = report["rounds"][1]["semantic_origin"]
        assert origin["builder_reuse"]["reused_criteria"] == ["valid"]
        assert origin["builder_reuse"]["rebuilt_criteria"] == ["syntax"]
        assert origin["builder_provider_sources"]
        assert not report["rounds"][2]["semantic_origin"]["builder_provider_sources"]
    repairs = [m for m in modes if m[2] == "outcome_jury_repair"]
    assert (
        len(repairs)
        == {
            "direct": 0,
            "targeted": 1,
            "atomic": 2,
            "reuse": 0,
            "partial_reuse": 0,
            "syntax": 0,
        }[scenario]
    )

    if scenario == "syntax":
        failures = [
            r
            for r in report["rounds"][0]["semantic_origin"]["jury_provider_sources"]
            if "failure" in r
        ]
        assert len(failures) == 1
        assert failures[0]["failure"] in report["source_hashes"]
        assert failures[0]["transport"] in report["source_hashes"]
        stream = root / failures[0]["transport"]
        saved = stream.read_bytes()
        valid_reply = next(
            (taskdir / "round_1/models/jury_reducer").glob("*/response.json")
        )
        physical_success = json.loads(
            valid_reply.with_name("stream_lines.json").read_text()
        )
        truncated = deepcopy(physical_success)
        chunk = json.loads(truncated[0][5:])
        chunk["choices"][0]["finish_reason"] = "length"
        truncated[0] = "data: " + json.dumps(chunk)
        for forged in ([], physical_success, truncated, ["data: {invalid event"]):
            write(stream, forged)
            rejected = audit_run(root, required_rounds=rounds)
            assert rejected["status"] == "fail", forged
            assert rejected["reason"].startswith("ValueError:"), rejected
            assert failures[0]["transport"] in rejected["source_hashes"]
        stream.write_bytes(saved)
        assert audit_run(root, required_rounds=rounds)["status"] == "pass"

    # PD1: jointly forged raw/effective copies still satisfy public projection,
    # but must not acquire model authority from their mutual consistency.
    folder = taskdir / "round_1"
    rawjury = json.loads((folder / "jury_semantic.json").read_text())
    rawpack = json.loads((folder / "pack_semantic.json").read_text())
    for kind in ("jury", "pack"):
        raw = deepcopy(rawjury if kind == "jury" else rawpack)
        raw["criteria"][0]["standard"] = (
            "UNAUTHORED assertion absent from every provider reply"
        )
        projection = (
            bind_definition(raw, contract)
            if kind == "jury"
            else bind_pack(raw, contract)
        )
        rawpath, effectivepath = (
            folder / (kind + "_semantic.json"),
            folder / (kind + ".json"),
        )
        saved_raw, saved_effective = rawpath.read_bytes(), effectivepath.read_bytes()
        write(rawpath, raw)
        write(effectivepath, projection)
        assert "model-origin" in audit_run(root, required_rounds=rounds)["reason"]
        rawpath.write_bytes(saved_raw)
        effectivepath.write_bytes(saved_effective)

    # The normalized answer is not authority when original provider bytes disagree.
    responsepath = next((folder / "models/jury_reducer").glob("*/response.json"))
    saved_response = responsepath.read_bytes()
    response = json.loads(saved_response)
    response["answer"]["unauthored"] = "Do not accept a forged parsed copy"
    write(responsepath, response)
    assert "provider output bytes" in audit_run(root, required_rounds=rounds)["reason"]
    responsepath.write_bytes(saved_response)
    response = json.loads(saved_response)
    response["raw_output"] = json.dumps(
        {"criteria": [], "unauthored": "forged text and parsed answer"}
    )
    response["answer"] = json.loads(response["raw_output"])
    write(responsepath, response)
    assert (
        "provider transport bytes" in audit_run(root, required_rounds=rounds)["reason"]
    )
    responsepath.write_bytes(saved_response)
    # Repair records cannot be reordered or replaced independently of provider calls.
    if repairs:
        editpath = folder / "jury_reduce_repair_1_edits.json"
        saved_edits = editpath.read_bytes()
        write(editpath, {"edits": []})
        assert (
            "repair record differs" in audit_run(root, required_rounds=rounds)["reason"]
        )
        editpath.write_bytes(saved_edits)
    assert audit_run(root, required_rounds=rounds)["status"] == "pass"
    # Tamper only effective semantics: raw receipt and declared contract are unchanged.
    packfile = taskdir / "round_1/pack.json"
    saved = json.loads(packfile.read_text())
    forged = deepcopy(saved)
    forged["criteria"][0]["standard"] = "Host rewrite"
    write(packfile, forged)
    assert (
        "exact declared public projection"
        in audit_run(root, required_rounds=rounds)["reason"]
    )
    write(packfile, saved)
    snapshotfile = taskdir / "round_1/actors/B/snapshots/delivery-2/workspace/manual.md"
    snapshotfile.write_text("Rewritten after measurement")
    assert "actual snapshot bytes" in audit_run(root, required_rounds=rounds)["reason"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        "contract",
        "candidate",
        "inventory",
        "input",
        "row_status",
        "summary",
        "missing_check",
    ],
)
async def test_auditor_rejects_gate_receipt_tampering(tmp_path, tamper):
    task, contract = task_and_contract()
    candidate = snapshot(tmp_path / "s", {"manual.md": "Authored text"})
    pack = bind_pack(raw_pack(task), contract)
    context, evidence = build_context(task, [candidate])
    measured = await measure(pack, registry(), context, evidence)
    row = next(r for r in measured["results"] if r["test_id"] == TEST)
    if tamper in {"contract", "candidate", "inventory"}:
        row["public_delivery_receipt"][tamper + "_sha256"] = "0" * 64
    elif tamper == "input":
        measured["measurement_input_sha256"] = "0" * 64
    elif tamper == "row_status":
        row["status"] = "unknown"
    elif tamper == "summary":
        measured["public_delivery_gate"]["status"] = "fail"
    else:
        measured["results"].remove(row)
    with pytest.raises(ValueError):
        audit_measurement(contract, candidate, measured, pack)


def test_actor_authored_inventory_file_cannot_create_authority(tmp_path):
    task, contract = task_and_contract()
    candidate = snapshot(
        tmp_path / "s",
        {
            "public_delivery_contract.json": json.dumps(contract),
            "inventory.json": json.dumps(
                [{"path": "manual.md", "sha256": "0" * 64, "bytes": 10}]
            ),
        },
    )
    assert candidate_gate(contract, candidate)["status"] == "fail"
    assert {
        r["path"]
        for r in candidate["environment"]["public_delivery_inventory"]["files"]
    } == {"inventory.json", "public_delivery_contract.json"}


def test_validation_prepare_forwards_freezes_and_binds_cli_contract(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from experiments import validate_pipeline
    from experiments.workbuddy import experiment
    from reward_engine.adapters.workbuddy.io import read

    task, contract = task_and_contract()
    taskroot = tmp_path / "dataset/tasks" / task["task_id"]
    taskroot.mkdir(parents=True)
    (taskroot / "instruction.md").write_text(task["prompt"])
    source = tmp_path / "supplied.json"
    write(source, contract)
    output = tmp_path / "run"
    captured = []

    def prepare(run, *, upstream, task_ids, public_delivery_contract):
        captured.append(public_delivery_contract)
        run.mkdir()
        code = run / "code"
        (code / "experiments").mkdir(parents=True)
        binding = freeze_contract(
            public_delivery_contract, run, taskroot.parent, task_ids
        )
        config = {
            "output": str(run),
            "code": str(code),
            "roles": {"actor": "offline", "jury_reducer": "offline"},
            "public_delivery_contract": binding,
        }
        write(
            run / "manifest.json",
            {"code_hashes": {}, "public_delivery_contract": binding},
        )
        return config

    monkeypatch.setattr(experiment, "prepare", prepare)
    args = SimpleNamespace(
        output=output,
        benchmark="workbuddy",
        task_id=task["task_id"],
        public_delivery_contract=source,
    )
    config = validate_pipeline.prepare(args)
    assert captured == [source]
    assert (
        read(output / "manifest.json")["public_delivery_contract"]
        == config["public_delivery_contract"]
    )
    assert read_bound_contract(config["public_delivery_contract"], task) == contract
    args.benchmark = "claw_eval"
    with pytest.raises(ValueError, match="WorkBuddy-only"):
        validate_pipeline.prepare(args)


@pytest.mark.asyncio
async def test_workbuddy_task_wrapper_carries_only_validated_frozen_binding(
    tmp_path, monkeypatch
):
    from experiments.workbuddy import experiment, resume

    task, contract = task_and_contract()
    taskroot = tmp_path / "dataset/tasks" / task["task_id"]
    taskroot.mkdir(parents=True)
    (taskroot / "instruction.md").write_text(task["prompt"])
    source = tmp_path / "source.json"
    write(source, contract)
    output = tmp_path / "run"
    output.mkdir()
    binding = freeze_contract(source, output, taskroot.parent, [task["task_id"]])
    config = {
        "output": str(output),
        "tasks_root": str(taskroot.parent),
        "public_delivery_contract": binding,
        "batch": {},
    }
    captured = []

    async def launch(config, actual_task, slot, **kwargs):
        captured.append(deepcopy(actual_task))
        return {"status": "completed", "candidate": {"candidate_id": slot.name}}

    async def run(actual_task, seeds, **kwargs):
        assert actual_task == task
        return {"status": "partial", "completed_rounds": 0, "rounds": []}

    monkeypatch.setattr(experiment, "launch", launch)
    monkeypatch.setattr(experiment, "run_task", run)
    monkeypatch.setattr(experiment, "registry", lambda *_: object())
    monkeypatch.setattr(resume, "completed_rounds", lambda *_: ([], None))
    await experiment.one_task(config, task["task_id"])
    assert captured == [task, task, task]
    (taskroot / "instruction.md").write_text(task["prompt"] + "Different task version")
    with pytest.raises(ValueError, match="Stale public instruction"):
        await experiment.one_task(config, task["task_id"])
    assert len(captured) == 3  # rejected before any further Actor calls


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "repair", "failed"])
async def test_generated_author_origin_binds_actual_tools_in_order(
    tmp_path, monkeypatch, scenario
):
    """Physical local calculation receipts; offline Role, no tools during audit."""
    import httpx

    from reward_engine.adapters.workbuddy import models
    from reward_engine.adapters.workbuddy.public_delivery_audit import (
        RetainedRole,
        audit_semantic_origin,
    )
    from reward_engine.core.outcome import builder, jury
    from reward_engine.core.outcome.measurement_tools import MeasurementTools

    task, _ = task_and_contract()
    candidate = snapshot(tmp_path / "snapshot", {"manual.md": "Authored manual"})
    context, evidence = build_context(task, [candidate])
    folder = "tasks/public-manual/round_1"
    root = tmp_path / "run"
    output = root / folder
    # Native records share the task measurement recorder across rounds.
    tools_output = root / "tasks/public-manual/measurement/measurements"
    config = {
        "output": str(root),
        "role_bounded_stage": True,
        "roles": {"jury_reducer": "offline", "harness_designer": "offline"},
        "gateway_local": "http://offline.invalid",
    }
    definition = {
        "criteria": [
            {
                "id": "valid",
                "standard": "Return valid JSON",
                "kind": "hard",
                "origin": "task",
                "measurement_question": "Does it parse?",
                "evidence_refs": ["task"],
            }
        ],
        "gaps": [],
        "coverage": [],
    }
    provider_calls = []
    client = httpx.AsyncClient

    def upstream(request):
        body = json.loads(request.content)
        req = json.loads(body["messages"][-1]["content"])
        provider_calls.append(req)
        if req["mode"] == "outcome_jury_reduce":
            answer = definition
        elif req["mode"] == "outcome_builder":
            answer = {
                "rubrics": [
                    {
                        "criterion_id": "valid",
                        "anchors": {"fail": "bad", "partial": "some", "full": "good"},
                    }
                ],
                "tests": [
                    {
                        "test_id": "calc",
                        "criterion_id": "valid",
                        "checker": "authored.calculation",
                        "question": "Read actual calculation?",
                        "arguments": {},
                    }
                ],
            }
        elif req["mode"] == "outcome_checker_author":
            success = (
                scenario == "success"
                or scenario == "repair"
                and "execution_recovery" in req
            )
            answer = {
                "probes": [
                    {
                        "tool": "calculate",
                        "arguments": {"expression": "1+1" if success else "1/0"},
                    }
                ]
            }
        else:
            raise AssertionError(req["mode"])
        event = {
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": json.dumps(answer)},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    monkeypatch.setattr(models, "auth", lambda *_: "offline")
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )

    def record(name, value):
        write(output / (name + ".json"), value)

    tools = MeasurementTools(
        record=lambda name, value: write(tools_output / (name + ".json"), value)
    )
    write(output / "learning_context.json", context)
    raw_jury = await jury.reduce_proposals(
        context,
        evidence,
        {
            "proposals": [definition, definition],
            "independent_expectations": [],
            "proposal_expectations": [],
        },
        models.Role(config, "jury_reducer", output / "models/jury_reducer"),
        record=record,
    )
    raw_pack = await builder._build_semantic(
        context,
        evidence,
        raw_jury,
        defaults(None, tools=tools),
        models.Role(config, "harness_designer", output / "models/harness_designer"),
        record=record,
    )
    assert raw_pack["tests"][0]["binding_status"] == (
        "needs_implementation" if scenario == "failed" else "callable"
    )
    physical = list(tools_output.glob("measurement_tool_*.json"))
    assert len(physical) == (1 if scenario == "success" else 2)
    calls_before = len(provider_calls)

    async def forbidden(*_):
        raise AssertionError("Audit must not execute measurement tools")

    monkeypatch.setattr(MeasurementTools, "run", forbidden)

    def audit(pack=raw_pack):
        reader = Reader(root)
        origin, _ = audit_semantic_origin(reader, folder, config, raw_jury, pack)
        return origin, reader

    origin, reader = audit()
    assert len(origin["checker_author_trials"]) == len(physical)
    assert [r["attempt"] for r in origin["checker_author_trials"]] == list(
        range(len(physical))
    )
    assert all(p.relative_to(root).as_posix() in reader.hashes for p in physical)
    assert len(provider_calls) == calls_before

    if scenario == "failed":
        # Reproduce F1 without altering any provider or actual failed tool record.
        frozen = {p: p.read_bytes() for p in output.glob("models/*/*/*.json")}
        frozen.update({p: p.read_bytes() for p in physical})
        trials = sorted(tools_output.glob("checker_author_trial_*.json"))
        first = next(p for p in trials if json.loads(p.read_text())["attempt"] == 0)
        saved = first.read_bytes()
        forged = json.loads(saved)
        forged.pop("error")
        forged.pop("instruction")
        forged["receipts"] = [
            {
                "tool": p["tool"],
                "arguments": p["arguments"],
                "status": "observed",
                "result": 2,
                "request_rejected": False,
            }
            for p in forged["recipe"]["probes"]
        ]
        reg = defaults(None, tools=tools)

        async def false_author(test, registry):
            registry.bind_recipe(
                test["checker"], {"probes": forged["recipe"]["probes"]}
            )

        forged_pack = await builder._build_semantic(
            context,
            evidence,
            raw_jury,
            reg,
            RetainedRole(Reader(root), folder + "/models/harness_designer", config),
            author=false_author,
        )
        forged_pack["pack_id"] = raw_pack["pack_id"]
        assert forged_pack["tests"][0]["binding_status"] == "callable"
        # Adding a contradictory trial cannot choose success over the true failure.
        added = tools_output / "checker_author_trial_forged.json"
        write(added, forged)
        with pytest.raises(ValueError, match="Conflicting retained trials"):
            audit(forged_pack)
        added.unlink()
        # Replacing the summary also fails: no physical successful execution exists.
        write(first, forged)
        with pytest.raises(ValueError, match="matching retained measurement_tool"):
            audit(forged_pack)
        first.write_bytes(saved)
        assert all(p.read_bytes() == data for p, data in frozen.items())
        audit()
    else:
        # Every returned result field is bound, not merely tool/arguments/status.
        trial = next(
            p
            for p in tools_output.glob("checker_author_trial_*.json")
            if json.loads(p.read_text())["receipts"][0]["status"] == "observed"
        )
        saved = trial.read_bytes()
        forged = json.loads(saved)
        forged["receipts"][0]["result"] = 999
        write(trial, forged)
        with pytest.raises(ValueError, match="matching retained measurement_tool"):
            audit()
        trial.write_bytes(saved)
        # A matching receipt in another recorder/round does not establish this call.
        source = next(
            p for p in physical if json.loads(p.read_text())["status"] == "observed"
        )
        destination = root / "tasks/public-manual/round_2" / source.name
        destination.parent.mkdir(parents=True)
        source.rename(destination)
        with pytest.raises(ValueError, match="matching retained measurement_tool"):
            audit()
        destination.rename(source)
        audit()
    assert len(provider_calls) == calls_before


@pytest.mark.asyncio
async def test_nonstream_syntax_origin_requires_actual_failed_content(
    tmp_path, monkeypatch
):
    """Real Role transport recovery followed by genuine bounded JSON recovery."""
    import httpx

    from reward_engine.adapters.workbuddy import models
    from reward_engine.adapters.workbuddy.public_delivery_audit import RetainedRole
    from reward_engine.core.outcome.context import ask

    requests = []
    client = httpx.AsyncClient

    def upstream(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                text='data: {"error":{"message":"simulated provider stream failure"}}\n\n',
            )
        if not body["stream"]:
            return httpx.Response(
                200,
                json={
                    "model": "offline",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": "{invalid syntax"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        event = {
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": '{"ok":true}'},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200, text="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
        )

    monkeypatch.setattr(models, "auth", lambda *_: "offline")
    monkeypatch.setattr(
        models.httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(upstream), **kw),
    )
    config = {
        "output": str(tmp_path),
        "role_bounded_stage": True,
        "roles": {"jury_reducer": "offline"},
        "gateway_local": "http://offline.invalid",
    }
    role = models.Role(config, "jury_reducer", tmp_path / "model")
    role.nonstream_recovery_models = {"offline"}
    context, evidence = build_context(
        {"task_id": "syntax", "prompt": "Return an object"}, []
    )

    async def invoke(call):
        return await ask(
            call,
            "origin_syntax_fixture",
            context,
            evidence,
            "Return an object",
            {"ok": "boolean"},
        )

    assert await invoke(role) == {"ok": True}
    assert len(requests) == 3
    reader = Reader(tmp_path)
    replay = RetainedRole(reader, "model", config)
    assert await invoke(replay) == {"ok": True}
    assert len(requests) == 3  # no HTTP during replay
    failures = [row for row in replay.used if "failure" in row]
    assert len(failures) == 2
    assert all(
        row["failure"] in reader.hashes and row["transport"] in reader.hashes
        for row in failures
    )
    rawfile = next((tmp_path / "model").glob("*/raw_response.txt"))
    parsed = rawfile.with_suffix(".json")
    original, original_parsed = rawfile.read_bytes(), parsed.read_bytes()
    for content, stop in [('{"ok":false}', "stop"), ("{invalid syntax", "length")]:
        forged = json.loads(original)
        forged["choices"][0]["message"]["content"] = content
        forged["choices"][0]["finish_reason"] = stop
        write(rawfile, forged)
        write(parsed, forged)
        with pytest.raises(ValueError) as caught:
            await invoke(RetainedRole(Reader(tmp_path), "model", config))
        assert not isinstance(caught.value, json.JSONDecodeError)
    rawfile.write_bytes(original)
    parsed.write_bytes(original_parsed)
    assert await invoke(RetainedRole(Reader(tmp_path), "model", config)) == {"ok": True}
    assert len(requests) == 3
