"""Mock HTTP and real core timeout controls, never actual provider requests."""

import asyncio
import json

import httpx
import pytest

from reward_engine.examples import minimal
from reward_engine.examples.live import Live, LiveConfig


def completion():
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({"final_response": '{"total":6}'}),
                    },
                }
            ]
        },
    )


@pytest.mark.parametrize("concurrency", [1, 2])
def test_shared_admission_bounds_roles_and_actors_without_thinking_override(
    concurrency,
):
    active = peak = count = 0

    async def handler(request):
        nonlocal active, peak, count
        body = json.loads(request.content)
        assert "thinking" not in body and "enable_thinking" not in body
        active += 1
        peak = max(peak, active)
        count += 1
        try:
            await asyncio.sleep(0.01)
            return completion()
        finally:
            active -= 1

    async def exercise():
        backend = Live(
            LiveConfig(
                "one-model",
                "https://mock.invalid/v1",
                "test-only-key",
                role_concurrency=concurrency,
            ),
            transport=httpx.MockTransport(handler),
        )
        return await asyncio.gather(
            *(backend.role({"mode": "mock"}) for _ in range(3)),
            *(backend.actor([], arm="A", revision=0) for _ in range(3)),
        )

    results = asyncio.run(exercise())
    assert len(results) == count == 6 and peak == concurrency


def test_real_core_stage_timeout_records_cancellation_and_cannot_succeed(tmp_path):
    cancelled = 0

    async def handler(request):
        nonlocal cancelled
        messages = json.loads(request.content)["messages"]
        if not messages[0]["content"].startswith("You are an internal"):
            return completion()
        try:
            await (
                asyncio.Event().wait()
            )  # only the actual core stage deadline releases this
        except asyncio.CancelledError:
            cancelled += 1
            raise

    backend = Live(
        LiveConfig(
            "one-model", "https://mock.invalid/v1", "test-only-key", stage_timeout=0.02
        ),
        transport=httpx.MockTransport(handler),
    )
    root = tmp_path / "stage-timeout"
    with pytest.raises(RuntimeError):
        asyncio.run(minimal.run(root, backend=backend))
    assert cancelled > 0
    failures = [
        json.loads(p.read_text()) for p in (root / "calls").glob("*/failure.json")
    ]
    assert failures and all(f["error_type"] == "CancelledError" for f in failures)
    assert all("exact origin is not determined" in f["reason"] for f in failures)
    assert not (root / "summary.json").exists()
    scope = json.loads((root / "scope.json").read_text())
    assert scope["runtime_settings"]["stage_timeout_s"] == 0.02


@pytest.mark.parametrize(
    "error",
    [RuntimeError("Authorization: secret"), TimeoutError("Authorization: secret")],
)
def test_recorder_never_copies_unmarked_exception_text(tmp_path, error):
    async def fail(request):
        raise error

    recorder = minimal.Recorder(tmp_path / "unsafe-error")
    with pytest.raises(type(error)):
        asyncio.run(recorder.call(fail, {}, "mock"))
    raw = (recorder.root / "calls/0001-mock/failure.json").read_text()
    assert "Authorization" not in raw and "secret" not in raw
    assert json.loads(raw)["reason"]


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": 0},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"stage_timeout": -1},
        {"stage_timeout": True},
        {"role_concurrency": 0},
        {"role_concurrency": 1.5},
        {"role_concurrency": True},
        {"thinking": "enabled"},
        {"jury_repair_mode": "automatic"},
        {"jury_repair_mode": None},
        {"jury_repair_mode": []},
    ],
)
def test_invalid_runtime_settings_rejected(options):
    with pytest.raises(ValueError):
        LiveConfig("one-model", "https://mock.invalid/v1", "test-only-key", **options)


def test_cli_passes_explicit_live_runtime_settings(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    seen = []

    async def capture(output, *, backend):
        seen.append(backend.config)
        return {"test": "CLI configuration only; no provider call"}

    monkeypatch.setattr(minimal, "run", capture)
    assert (
        minimal.main(
            [
                "--output",
                str(tmp_path / "cli"),
                "--live",
                "--model",
                "one-model",
                "--base-url",
                "https://mock.invalid/v1",
                "--request-timeout",
                "600",
                "--stage-timeout",
                "1800",
                "--role-concurrency",
                "1",
                "--thinking",
                "disabled",
                "--jury-repair-mode",
                "targeted",
            ]
        )
        == 0
    )
    assert len(seen) == 1
    assert (
        seen[0].timeout,
        seen[0].stage_timeout,
        seen[0].role_concurrency,
        seen[0].thinking,
    ) == (600, 1800, 1, "disabled")
    assert seen[0].jury_repair_mode == "targeted"
    assert "test-only-key" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "option",
    [
        ["--request-timeout", "600"],
        ["--stage-timeout", "1800"],
        ["--role-concurrency", "1"],
        ["--thinking", "provider_default"],
        ["--jury-repair-mode", "targeted"],
    ],
)
def test_live_only_options_cannot_silently_run_offline(tmp_path, option, capsys):
    with pytest.raises(SystemExit) as error:
        minimal.main(["--output", str(tmp_path / "not-created"), *option])
    assert error.value.code == 2
    assert "require --live" in capsys.readouterr().err
    assert not (tmp_path / "not-created").exists()
