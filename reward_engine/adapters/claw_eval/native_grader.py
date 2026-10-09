"""Private native Claw-Eval regrading hook. No grader imports in the Actor process."""

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

from reward_engine.adapters.support import PYTHON, read_json, write_json


class JudgeUnavailable(BaseException):
    """Escape native graders' catch-Exception-and-award-zero error handling."""


class StreamingClient:
    """Keep native judge prompts; replace only its HTTP transport and retry policy."""

    def __init__(self, client, settings, output):
        self.client, self.settings, self.output = client, settings, output
        self.chat = SimpleNamespace(completions=self)
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        request.update(
            model=self.settings.get("model", "qwen3.8-max"),
            temperature=self.settings.get("temperature", 0.6),
            max_tokens=self.settings.get("max_tokens", 32768),
            stream=True,
            stream_options={"include_usage": True},
        )
        directory = self.output / f"judge_{self.calls:03d}"
        directory.mkdir()
        write_json(directory / "request.json", request)
        # Two transport/format attempts, never retry because a score is low.
        for attempt in range(2):
            text, finish, models, usage = "", None, set(), None
            try:
                with (directory / f"stream_{attempt}.jsonl").open("w") as ledger:
                    with self.client.chat.completions.create(**request) as stream:
                        for chunk in stream:
                            ledger.write(chunk.model_dump_json() + "\n")
                            ledger.flush()
                            if chunk.model:
                                models.add(chunk.model)
                            if chunk.usage:
                                usage = chunk.usage.model_dump()
                            for choice in chunk.choices:
                                text += choice.delta.content or ""
                                finish = choice.finish_reason or finish
                allowed = self.settings.get("allowed_models", [request["model"]])
                if not models or not models.issubset(set(allowed)):
                    raise ValueError("Unexpected or missing returned model identity")
                if finish != "stop":
                    raise ValueError("Judge did not finish a complete answer")
                clean = text.strip()
                if clean.startswith("```json") and clean.endswith("```"):
                    clean = clean[7:-3].strip()
                parsed = json.loads(clean)
                score = parsed.get("score")
                if (
                    type(score) not in (int, float)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                    or not isinstance(parsed.get("reasoning"), str)
                ):
                    raise ValueError("Invalid score/reasoning response")
                write_json(
                    directory / "result.json",
                    {
                        "answer": parsed,
                        "returned_models": sorted(models),
                        "usage": usage,
                        "attempt": attempt,
                        "finish_reason": finish,
                    },
                )
                # Native parser extracts a flat JSON object by regex. Escape literal
                # braces inside reasoning without changing the decoded explanation.
                reason = (
                    json.dumps(parsed["reasoning"])
                    .replace("{", "\\u007b")
                    .replace("}", "\\u007d")
                )
                clean = '{"score":' + json.dumps(score) + ',"reasoning":' + reason + "}"
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=clean))]
                )
            except Exception as error:
                write_json(
                    directory / f"failure_{attempt}.json",
                    {
                        "error_type": type(error).__name__,
                        "partial_answer": text,
                        "finish_reason": finish,
                        "returned_models": sorted(models),
                        "usage": usage,
                    },
                )
        raise JudgeUnavailable(
            "Judge transport/format failed twice; not a candidate score"
        )


def grade(request, output):
    """Run in a separate process with authentic native trace + terminal snapshot."""
    root = Path(request["benchmark_root"]).resolve()
    sys.path.insert(0, str(root / "src"))
    from claw_eval.graders.llm_judge import LLMJudge
    from claw_eval.graders.registry import get_grader
    from claw_eval.models.scoring import compute_task_score, is_pass
    from claw_eval.models.task import TaskDefinition
    from claw_eval.trace.reader import load_trace
    from openai import OpenAI

    task_path = Path(request["task_yaml"]).resolve()
    task = TaskDefinition.from_yaml(task_path)
    start, messages, dispatches, media, end, audits = load_trace(request["trace"])
    if start.task_id != task.task_id or request["task_id"] != task.task_id:
        raise ValueError("Trace, experiment and task YAML identities differ")
    if end is None or end.trace_id != start.trace_id:
        raise ValueError("A completed native trace is required")
    if not audits and task.services:
        raise ValueError("Native service audit snapshots are missing")
    snapshot = read_json(Path(request["environment_snapshot"]))
    if not isinstance(snapshot, dict):
        raise ValueError("Terminal environment snapshot must be a mapping")
    required = (
        [f"cmd:{x}" for x in task.env_snapshot_commands]
        + [
            f"file:{x}"
            for x in task.env_snapshot_files
            if "*" not in x and "?" not in x
        ]
        + [f"local_file:{x}" for x in task.local_grader_files]
    )
    if any(key not in snapshot for key in required):
        raise ValueError("Required native terminal snapshot entries are missing")
    settings = request.get("judge", {})
    base_url = settings.get("base_url", "https://yibuapi.com/v1")
    model = settings.get("model", "qwen3.8-max")
    # Native grader imports remain private; native request formatting is unchanged.
    judge = LLMJudge(model_id=model, base_url=base_url, api_key="unused")
    judge.client.close()
    client = OpenAI(
        api_key=os.environ[settings.get("api_key_env", "YIBU_API_KEY")],
        base_url=base_url,
        timeout=settings.get("read_timeout_s", 120),
        max_retries=0,
    )
    transport = StreamingClient(client, settings, output)
    judge.client = transport
    try:
        grader = get_grader(
            task.task_id, tasks_dir=root / "tasks", task_dir=task_path.parent
        )
        kwargs = {"audit_data": audits, "judge": judge}
        params = inspect.signature(grader.grade).parameters
        if "media_events" in params:
            kwargs["media_events"] = media
        if "env_snapshot" in params:
            kwargs["env_snapshot"] = snapshot
        scores = grader.grade(messages, dispatches, task, **kwargs)
        score = compute_task_score(scores)
        return {
            "status": "graded",
            "protocol": "native-claw-eval-custom-judge",
            "task_id": task.task_id,
            "trace_id": start.trace_id,
            "judge_model": model,
            "judge_provider": base_url,
            "temperature": settings.get("temperature", 0.6),
            "scores": scores.model_dump(),
            "task_score": score,
            "score_threshold_met": is_pass(score),
            "official_leaderboard_equivalent": False,
            "snapshot_error_keys": [
                key
                for key, value in snapshot.items()
                if isinstance(value, dict) and value.get("error")
            ],
            "judge_calls": judge.get_call_log(),
        }
    finally:
        client.close()


def grade_artifact(request, output):
    """Private, artifact-only quality measurement; never impersonates a native trace."""
    root = Path(request["benchmark_root"]).resolve()
    sys.path.insert(0, str(root / "src"))
    from claw_eval.graders.llm_judge import LLMJudge
    from claw_eval.models.task import TaskDefinition
    from openai import OpenAI

    task = TaskDefinition.from_yaml(Path(request["task_yaml"]))
    if task.task_id != request["task_id"]:
        raise ValueError("Artifact grading Task mismatch")
    artifact = read_json(Path(request["artifact_file"]))
    text = artifact.get("final_response")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Artifact quality requires an actual nonempty deliverable")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != request["artifact_sha256"]:
        raise ValueError("Artifact changed after the blind grading manifest was frozen")
    prompt = task.prompt.text
    if request.get("public_task"):
        from reward_engine.core.outcome.construction import fingerprint

        public = request["public_task"]
        if (
            public.get("task_id") != task.task_id
            or fingerprint(public) != request.get("public_task_sha256")
            or not isinstance(public.get("prompt"), str)
            or not public["prompt"].startswith(prompt)
        ):
            raise ValueError("Blind grading public Task scope mismatch")
        prompt = public["prompt"]
    settings = request["judge"]
    base_url, model = settings["base_url"], settings["model"]
    judge = LLMJudge(model_id=model, base_url=base_url, api_key="unused")
    judge.client.close()
    client = OpenAI(
        api_key=os.environ[settings.get("api_key_env", "YIBU_API_KEY")],
        base_url=base_url,
        timeout=settings.get("read_timeout_s", 120),
        max_retries=0,
    )
    judge.client = StreamingClient(client, settings, output)
    try:
        result = judge.evaluate(
            prompt, "Assistant final deliverable:\n" + text, "", task.judge_rubric
        )
        return {
            "status": "graded",
            "protocol": "outcome-artifact-quality",
            "task_id": task.task_id,
            "artifact_sha256": digest,
            "artifact_chars": len(text),
            "content_judge_score": result.score,
            "judge_model": model,
            "judge_provider": base_url,
            "temperature": settings.get("temperature", 0.6),
            "judge_calls": judge.get_call_log(),
            "official_leaderboard_equivalent": False,
            "scope": "Only the complete deliverable; no arm label, revision history, Critic advice or tool effort.",
        }
    finally:
        client.close()


async def run_private(request, output, *, timeout_s=900, python=PYTHON):
    """Whole-grade deadline; no retry/relaunch of Actor and no trace mutation."""
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("Grader Python must be an explicit existing interpreter path")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.private.json", request)
    with (output / "worker.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            python,
            "-m",
            __name__,
            "--request",
            str(output / "request.private.json"),
            "--output",
            str(output),
            stdout=log,
            stderr=log,
        )
        try:
            await asyncio.wait_for(process.wait(), timeout_s)
        except (TimeoutError, asyncio.CancelledError) as error:
            if process.returncode is None:
                process.kill()
            await process.wait()
            write_json(
                output / "result.private.json",
                {"status": "unavailable", "reason": "deadline_or_cancel"},
            )
            if isinstance(error, asyncio.CancelledError):
                raise
    result_path = output / "result.private.json"
    result = (
        read_json(result_path)
        if result_path.exists()
        else {"status": "unavailable", "reason": "worker_failed"}
    )
    return {**result, "record_directory": str(output)}


def evaluator(config, output):
    """outcome_loop.official_evaluator plugin. Missing native artifacts are not zeros."""
    settings = config["native_grader"]

    async def evaluate(task, row):
        trial = row.get("native_grader_trial") or row.get("runtime_directory")
        if not trial:
            return {"status": "needs_native_artifacts"}
        trial = Path(trial).resolve()
        trace, snapshot = trial / "trace.jsonl", trial / "environment-snapshot.json"
        if not trace.is_file() or not snapshot.is_file():
            return {"status": "needs_native_artifacts", "trial": str(trial)}
        request = {
            "benchmark_root": settings["benchmark_root"],
            "task_yaml": settings["task_yaml"],
            "task_id": task["task_id"],
            "trace": str(trace),
            "environment_snapshot": str(snapshot),
            "judge": settings.get("judge", {}),
        }
        return await run_private(
            request,
            output / uuid.uuid4().hex,
            timeout_s=settings.get("timeout_s", 900),
            python=settings.get("python", PYTHON),
        )

    return evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "result.private.json").exists():
        parser.error("Do not overwrite an existing grade")
    try:
        request = read_json(args.request)
        result = (
            grade_artifact(request, args.output)
            if request.get("evaluation_kind") == "artifact_quality"
            else grade(request, args.output)
        )
    except (JudgeUnavailable, Exception) as error:
        result = {"status": "unavailable", "error_type": type(error).__name__}
    write_json(args.output / "result.private.json", result)


if __name__ == "__main__":
    main()
