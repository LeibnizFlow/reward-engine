"""A single SIGINT must cancel pipeline work before embedded servers drain."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


def run_child(root, benchmark):
    """Use real Runner/Uvicorn/HTTP, replacing only benchmark work and provisioning."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import patch

    import httpx
    import uvicorn

    from experiments import validate_pipeline as driver
    from experiments.claw_eval.opencode import campaign, runtime
    from experiments.workbuddy import experiment
    from reward_engine.adapters.workbuddy.io import write

    servers, requests_started = [], {}
    release_requests = asyncio.Event()

    def record(kind, **data):
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"kind": kind, **data}) + "\n")

    def application(name):
        requests_started[name] = asyncio.Event()

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message["type"] == "lifespan.shutdown":
                        record("server_shutdown", server=name)
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            else:
                assert scope["type"] == "http" and scope["path"] == "/hold"
                requests_started[name].set()
                record("request_started", server=name)
                try:
                    # Uvicorn cannot finish draining until the pipeline cancels
                    # and releases this request. The old signal ownership stalls here.
                    await release_requests.wait()
                    await send(
                        {"type": "http.response.start", "status": 200, "headers": []}
                    )
                    await send({"type": "http.response.body", "body": b"released"})
                finally:
                    record("request_finished", server=name)

        return app

    class ObservedServer(uvicorn.Server):
        def __init__(self, config):
            super().__init__(config)
            servers.append(self)

    async def long_task(*_):
        async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
            requests = [
                asyncio.create_task(
                    client.get(
                        f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/hold"
                    )
                )
                for server in servers
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *(event.wait() for event in requests_started.values())
                    ),
                    timeout=3,
                )
                # Publish readiness only once every real server has an in-flight
                # request, so the signal cannot race startup or miss the drain bug.
                write(
                    root / "ready.json",
                    {
                        "servers": len(servers),
                        "graceful_timeouts": [
                            s.config.timeout_graceful_shutdown for s in servers
                        ],
                    },
                )
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                record("task_cancelled")
                raise
            finally:
                release_requests.set()
                try:
                    responses = await asyncio.wait_for(
                        asyncio.gather(*requests), timeout=3
                    )
                    assert all(response.status_code == 200 for response in responses)
                finally:
                    for request in requests:
                        if not request.done():
                            request.cancel()
                    await asyncio.gather(*requests, return_exceptions=True)
                await asyncio.sleep(0)
                record("task_cleanup_completed")

    write(
        root / "config.json",
        {
            "output": str(root),
            "validation_benchmark": benchmark,
            "validation_task_id": "offline-task",
            "benchmark_root": str(root),
            "gateway_port": 0,
            "tools_port": 0,
        },
    )
    with (
        patch.object(driver, "application", lambda _: application("gateway")),
        patch.object(driver.uvicorn, "Server", ObservedServer),
        patch.object(
            runtime, "Runtime", lambda _: SimpleNamespace(app=application("tools"))
        ),
        patch.object(campaign, "verify_freeze"),
        patch.object(experiment, "verify"),
        patch.object(campaign, "task_run", long_task),
        patch.object(experiment, "one_task", long_task),
        patch.object(
            sys, "argv", ["validate_pipeline", "--execute", "--output", str(root)]
        ),
    ):
        try:
            driver.main()
        except KeyboardInterrupt:
            record(
                "runner_interrupted",
                listeners_closed=all(
                    not listener.is_serving()
                    for server in servers
                    for listener in server.servers
                ),
            )
            raise SystemExit(130) from None
        raise AssertionError("Controlled task returned without interruption")


@pytest.mark.parametrize("benchmark,server_count", [("workbuddy", 1), ("claw_eval", 2)])
def test_one_sigint_cancels_pipeline_and_closes_real_servers(
    tmp_path, benchmark, server_count
):
    repo = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(tmp_path),
            benchmark,
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready.json").exists():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(
                    f"Local validation child failed before readiness: {stdout}\n{stderr}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("Local validation servers did not become ready")
            time.sleep(0.02)
        ready = json.loads((tmp_path / "ready.json").read_text())
        assert ready == {
            "servers": server_count,
            "graceful_timeouts": [10] * server_count,
        }
        process.send_signal(signal.SIGINT)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "One SIGINT did not cancel pipeline work before server draining; child still alive after 5 seconds"
            )
        assert process.returncode == 130, (stdout, stderr)
    finally:
        # Kill only our isolated test child on failure; never signal a campaign PID.
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)

    rows = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    kinds = [row["kind"] for row in rows]
    assert kinds.count("task_cancelled") == 1
    assert kinds.count("task_cleanup_completed") == 1
    assert kinds.count("request_started") == server_count
    assert kinds.count("request_finished") == server_count
    assert kinds.count("server_shutdown") == server_count
    assert kinds.index("task_cancelled") < kinds.index("request_finished")
    assert kinds.index("task_cleanup_completed") < kinds.index("server_shutdown")
    assert rows[-1] == {"kind": "runner_interrupted", "listeners_closed": True}
    status = json.loads((tmp_path / "validation_status.json").read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "CancelledError"


if __name__ == "__main__":
    assert len(sys.argv) == 4 and sys.argv[1] == "--child"
    run_child(Path(sys.argv[2]), sys.argv[3])
