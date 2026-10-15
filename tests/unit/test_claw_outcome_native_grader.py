"""Offline tests. Native grader source is loaded only in the private grading tests."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from reward_engine.adapters.claw_eval import native_grader as hook
from reward_engine.adapters.support import read_json, write_json


class Chunk:
    def __init__(self, content, model="qwen3.8-max", finish="stop"):
        self.model, self.usage = model, None
        self.choices = [NS(delta=NS(content=content), finish_reason=finish)]

    def model_dump_json(self):
        return json.dumps(
            {"model": self.model, "content": self.choices[0].delta.content}
        )


class Stream(list):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class Client:
    def __init__(self, chunk):
        self.chat = NS(completions=self)
        self.chunk, self.requests = chunk, []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return Stream([self.chunk])

    def close(self):
        pass


def test_native_request_preserved_streamed_and_low_score_not_retried(tmp_path):
    client = Client(Chunk('{"score":0,"reasoning":"No saved draft"}'))
    transport = hook.StreamingClient(client, {}, tmp_path)
    messages = [{"role": "user", "content": "native rubric and exact evidence"}]
    response = transport.create(model="unused", messages=messages)
    assert json.loads(response.choices[0].message.content)["score"] == 0
    assert len(client.requests) == 1
    assert client.requests[0]["messages"] == messages
    assert client.requests[0]["model"] == "qwen3.8-max"
    assert client.requests[0]["stream"] is True
    assert read_json(tmp_path / "judge_001/result.json")["returned_models"] == [
        "qwen3.8-max"
    ]


@pytest.mark.parametrize(
    "content,model,finish",
    [
        ('{"score":1,"reasoning":"cut"}', "qwen3.8-max", "length"),
        ('{"score":1,"reasoning":"ok"}', "wrong-model", "stop"),
        ('{"score":2,"reasoning":"bad"}', "qwen3.8-max", "stop"),
        ('{"score":NaN,"reasoning":"bad"}', "qwen3.8-max", "stop"),
        ('{"score":true,"reasoning":"bad"}', "qwen3.8-max", "stop"),
        ("not JSON", "qwen3.8-max", "stop"),
    ],
)
def test_judge_failure_cannot_be_caught_as_candidate_zero(
    tmp_path, content, model, finish
):
    client = Client(Chunk(content, model, finish))
    with pytest.raises(hook.JudgeUnavailable):
        try:
            hook.StreamingClient(client, {}, tmp_path).create(messages=[])
        except Exception:
            pytest.fail("Native grader would silently turn this into zero")
    assert len(client.requests) == 2


def test_current_local_runtime_needs_native_artifacts_without_calling_model(tmp_path):
    evaluate = hook.evaluator({"native_grader": {}}, tmp_path / "private")
    result = asyncio.run(
        evaluate({"task_id": "T110"}, {"runtime_directory": str(tmp_path)})
    )
    assert result["status"] == "needs_native_artifacts"
    assert "task_score" not in result
    assert not (tmp_path / "private").exists()


def test_plugin_only_passes_native_files_not_learning_context(tmp_path, monkeypatch):
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "trace.jsonl").touch()
    write_json(trial / "environment-snapshot.json", {})
    requests = []

    async def capture(request, output, **kwargs):
        requests.append(request)
        return {"status": "graded", "task_score": 0.4}

    monkeypatch.setattr(hook, "run_private", capture)
    evaluate = hook.evaluator(
        {"native_grader": {"benchmark_root": "/benchmark", "task_yaml": "/task.yaml"}},
        tmp_path,
    )
    result = asyncio.run(
        evaluate(
            {"task_id": "T110", "prompt": "not substituted"},
            {
                "native_grader_trial": str(trial),
                "candidate": {"ranking": 1},
                "memory": ["secret hint"],
            },
        )
    )
    assert result["task_score"] == 0.4
    assert set(requests[0]) == {
        "benchmark_root",
        "task_yaml",
        "task_id",
        "trace",
        "environment_snapshot",
        "judge",
    }


def test_remote_grader_interpreter_is_explicit_not_mac_default(tmp_path, monkeypatch):
    (tmp_path / "trace.jsonl").touch()
    write_json(tmp_path / "environment-snapshot.json", {})
    calls = []

    async def capture(request, output, **kwargs):
        calls.append(kwargs)
        return {"status": "graded"}

    monkeypatch.setattr(hook, "run_private", capture)
    remote = "/opt/native-grader/bin/python"
    evaluate = hook.evaluator(
        {
            "native_grader": {
                "benchmark_root": "/benchmark",
                "task_yaml": "/task.yaml",
                "python": remote,
            }
        },
        tmp_path / "private",
    )
    asyncio.run(evaluate({"task_id": "T068"}, {"native_grader_trial": str(tmp_path)}))
    assert calls == [{"timeout_s": 900, "python": remote}]


def test_grader_rejects_implicit_python_before_creating_run(tmp_path):
    with pytest.raises(ValueError, match="explicit existing interpreter"):
        asyncio.run(hook.run_private({}, tmp_path / "private", python="python"))
    assert not (tmp_path / "private").exists()


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    root = Path(
        os.environ.get(
            "CLAW_EVAL_ROOT",
            str(
                Path(__file__).resolve().parents[2] / "_local/data/claw_eval/benchmark"
            ),
        )
    )
    if not (root / "src/claw_eval").is_dir():
        pytest.skip("Native Claw-Eval checkout not available")
    monkeypatch.syspath_prepend(str(root / "src"))
    task = root / "tasks/T110_complaint_investigation/task.yaml"
    from claw_eval.models.task import TaskDefinition

    task_id = TaskDefinition.from_yaml(task).task_id
    trace = tmp_path / "trace.jsonl"
    events = [
        {
            "type": "trace_start",
            "trace_id": "fixture",
            "task_id": task_id,
            "model": "fixture",
        },
        {
            "type": "audit_snapshot",
            "trace_id": "fixture",
            "service_name": "gmail",
            "audit_url": "fixture",
            "audit_data": {"calls": []},
        },
        {"type": "trace_end", "trace_id": "fixture"},
    ]
    trace.write_text("\n".join(json.dumps(row) for row in events))
    snapshot = tmp_path / "environment-snapshot.json"
    write_json(snapshot, {})
    request = {
        "benchmark_root": str(root),
        "task_yaml": str(task),
        "task_id": task_id,
        "trace": str(trace),
        "environment_snapshot": str(snapshot),
    }
    monkeypatch.setenv("YIBU_API_KEY", "offline-test-only")
    client = Client(Chunk('{"score":0.5,"reasoning":"fixture"}'))
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    output = tmp_path / "private"
    output.mkdir()
    return request, output, client


def test_real_t110_native_grader_with_empty_actions_is_not_success(native_case):
    request, output, client = native_case
    result = hook.grade(request, output)
    assert result["status"] == "graded"
    # Native T110 still asks three content questions and applies a tool penalty.
    assert result["scores"]["completion"] == 0.0437
    assert result["task_score"] < 0.75
    assert result["official_leaderboard_equivalent"] is False
    assert len(client.requests) == 3  # No saved draft: no fourth draft-quality call.


def test_native_score_formula_and_native_judge_prompt(native_case, monkeypatch):
    request, output, client = native_case
    from claw_eval.models.trace import DimensionScores

    client.chunk = Chunk('{"score":0.5,"reasoning":"Missing {field} in artifact"}')

    class Grader:
        def grade(self, messages, dispatches, task, *, audit_data, judge, env_snapshot):
            result = judge.evaluate_actions(
                task.prompt.text, "actual fixture audit", "fixture rubric"
            )
            assert result.score == 0.5
            assert result.reasoning == "Missing {field} in artifact"
            return DimensionScores(completion=result.score, robustness=1, safety=1)

    monkeypatch.setattr(
        "claw_eval.graders.registry.get_grader", lambda *a, **kw: Grader()
    )
    result = hook.grade(request, output)
    assert result["task_score"] == 0.6
    assert "actual fixture audit" in client.requests[0]["messages"][1]["content"]
    assert len(result["judge_calls"]) == 1


def test_wrong_task_rejected_before_judge(native_case):
    request, output, client = native_case
    request["task_id"] = "other-task"
    with pytest.raises(ValueError, match="identities"):
        hook.grade(request, output)
    assert not client.requests


def test_snapshot_errors_reach_native_grader_unchanged(native_case, monkeypatch):
    request, output, client = native_case
    snapshot = {"file:/output.txt": {"error": "file not found"}}
    write_json(Path(request["environment_snapshot"]), snapshot)
    from claw_eval.models.trace import DimensionScores

    class Grader:
        def grade(self, messages, dispatches, task, *, audit_data, judge, env_snapshot):
            assert env_snapshot == snapshot
            return DimensionScores(completion=0)

    monkeypatch.setattr(
        "claw_eval.graders.registry.get_grader", lambda *a, **kw: Grader()
    )
    result = hook.grade(request, output)
    assert result["snapshot_error_keys"] == ["file:/output.txt"]
    assert result["task_score"] == 0
    assert not client.requests


def test_private_worker_runs_and_reports_invalid_task_without_model_calls(
    native_case, tmp_path
):
    request, _, _ = native_case
    request["task_id"] = "wrong-task"
    result = asyncio.run(hook.run_private(request, tmp_path / "subprocess"))
    assert result["status"] == "unavailable"
    assert result["error_type"] == "ValueError"
    assert "task_score" not in result
    assert (tmp_path / "subprocess/worker.log").is_file()


def test_deadline_kills_worker_without_awarding_zero(tmp_path, monkeypatch):
    class Process:
        returncode = None
        killed = False

        async def wait(self):
            if not self.killed:
                await asyncio.Event().wait()
            self.returncode = -9

        def kill(self):
            self.killed = True

    process = Process()

    async def launch(*args, **kwargs):
        assert args[0] == hook.PYTHON
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    result = asyncio.run(hook.run_private({}, tmp_path / "deadline", timeout_s=0.01))
    assert process.killed
    assert result["status"] == "unavailable"
    assert "task_score" not in result


def test_artifact_quality_uses_complete_body_without_arm_or_critic_history(
    native_case, tmp_path
):
    import hashlib

    request, output, client = native_case
    text = "A complete deliverable with a derivation and actionable correction."
    artifact = tmp_path / "opaque-artifact.json"
    write_json(
        artifact,
        {"final_response": text, "arm": "C", "critic_feedback": "SECRET_CRITIC_ADVICE"},
    )
    request.update(
        evaluation_kind="artifact_quality",
        artifact_file=str(artifact),
        artifact_sha256=hashlib.sha256(text.encode()).hexdigest(),
        judge={"model": "qwen3.8-max", "base_url": "https://yibuapi.com/v1"},
    )
    result = hook.grade_artifact(request, output)
    assert result["content_judge_score"] == 0.5 and "task_score" not in result
    assert result["artifact_sha256"] == request["artifact_sha256"]
    wire = json.dumps(client.requests[0]["messages"])
    assert text in wire and "SECRET_CRITIC_ADVICE" not in wire
    assert (
        len(client.requests) == 1 and result["protocol"] == "outcome-artifact-quality"
    )


def test_artifact_quality_rejects_changed_body_before_model(native_case, tmp_path):
    request, output, client = native_case
    artifact = tmp_path / "artifact.json"
    write_json(artifact, {"final_response": "Changed after manifest freeze"})
    request.update(artifact_file=str(artifact), artifact_sha256="wrong", judge={})
    with pytest.raises(ValueError, match="changed"):
        hook.grade_artifact(request, output)
    assert not client.requests
