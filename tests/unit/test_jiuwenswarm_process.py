import asyncio
import json
from pathlib import Path

import pytest

from reward_engine.adapters.jiuwenswarm.process import (
    JiuwenSwarmProcess,
    ProcessProtocolError,
    _validate_record_stream,
)


def result(**kwargs):
    return dict(
        schema_version="0.1",
        type="result",
        sequence=0,
        request_id="request",
        session_id="session",
        status="completed",
        exit_code=0,
        **kwargs,
    )


def validate(records, request=None, returncode=0):
    return _validate_record_stream(
        "\n".join(json.dumps(r) for r in records),
        request or {"request_id": "request"},
        returncode,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sequence", False),
        ("sequence", 1),
        ("usage", []),
        ("output", 1),
        ("exit_code", True),
        ("status", "ok"),
        ("type", "bogus"),
        ("request_id", "changed"),
        ("session_id", ""),
        ("extra", 1),
        ("usage", {"x": float("nan")}),
        ("usage", {"x": float("inf")}),
    ],
)
def test_invalid_result(field, value):
    record = result()
    record[field] = value
    with pytest.raises(ProcessProtocolError):
        validate([record])


def test_events_and_identity():
    event = dict(
        schema_version="0.1",
        type="event",
        sequence=0,
        request_id="request",
        session_id=None,
        event_type="future.event",
        payload={"data": [None, 1]},
    )
    terminal = result()
    terminal["sequence"] = 1
    assert validate([event, terminal])[0] == (event,)
    for changed in (
        {"event_type": ""},
        {"payload": []},
        {"payload": {"event_type": "different"}},
        {"extra": 1},
    ):
        with pytest.raises(ProcessProtocolError):
            validate([{**event, **changed}, terminal])
    with pytest.raises(ProcessProtocolError):
        validate([event, terminal], {"request_id": "request", "session_id": "other"})
    with pytest.raises(ProcessProtocolError):
        validate([result()], returncode=1)
    with pytest.raises(ProcessProtocolError):
        validate([event])
    with pytest.raises(ProcessProtocolError):
        validate([result(), terminal])


@pytest.mark.parametrize(
    "status,code", [("failed", 1), ("cancelled", 130), ("timed_out", 124)]
)
def test_valid_failure(status, code):
    record = {
        **result(),
        "status": status,
        "exit_code": code,
        "error": {"code": "FUTURE_CODE", "message": "Failure"},
    }
    assert validate([record], returncode=code)[1] == record
    record["error"]["retryable"] = 1
    with pytest.raises(ProcessProtocolError):
        validate([record], returncode=code)


def executable(tmp_path, body):
    import sys

    script = tmp_path / "fixture"
    script.write_text(f"#!{sys.executable}\n" + body)
    script.chmod(0o755)
    return str(script)


@pytest.mark.asyncio
async def test_durable_success_and_resume(tmp_path):
    binary = executable(
        tmp_path,
        """import json,sys,os
request=json.load(sys.stdin)
assert os.path.exists(os.environ['HOME'])
print('diagnostic',file=sys.stderr)
print(json.dumps(dict(schema_version='0.1',type='result',sequence=0,
 request_id=request['request_id'],session_id=request['session_id'] or 'session',
 status='completed',exit_code=0,output='answer')))
""",
    )
    client = JiuwenSwarmProcess(tmp_path / "instance", binary=binary)
    first = await client.run("task", workspace=tmp_path, label="first")
    second = await client.run(
        "revise", workspace=tmp_path, session_id=first.session_id, label="second"
    )
    assert first.output == "answer" and second.session_id == first.session_id
    assert (client.raw_dir / "first.request.json").exists()
    assert (client.raw_dir / "first.stderr").read_text() == "diagnostic\n"
    assert (
        json.loads((client.raw_dir / "first.execution.json").read_text())["status"]
        == "exited"
    )
    with pytest.raises(FileExistsError):
        await client.run("task", workspace=tmp_path, label="first")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_output_and_descendants(tmp_path, cancel):
    binary = executable(
        tmp_path,
        """import sys,time,subprocess,os
sys.stdin.read()
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
print(child.pid,flush=True)
print('retained',file=sys.stderr,flush=True)
time.sleep(60)
""",
    )
    client = JiuwenSwarmProcess(tmp_path / "instance", binary=binary)
    task = asyncio.create_task(
        client.run(
            "task",
            workspace=tmp_path,
            label="turn",
            host_timeout_s=10 if not cancel else 30,
        )
    )
    try:
        async with asyncio.timeout(35):
            while (
                not (client.raw_dir / "turn.jsonl").exists()
                or not (client.raw_dir / "turn.jsonl").stat().st_size
            ):
                if task.done():
                    await task
                    pytest.fail("Synthetic process exited without its startup output")
                await asyncio.sleep(0.01)
        if cancel:
            task.cancel()
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert (client.raw_dir / "turn.stderr").read_text() == "retained\n"
    pid = int((client.raw_dir / "turn.jsonl").read_text())
    stat = Path(f"/proc/{pid}/stat")
    assert not stat.exists() or stat.read_text().split()[2] == "Z"
    assert not (client.raw_dir / "turn.result.json").exists()
    assert not (client.raw_dir / "turn.terminal.json").exists()
    meta = json.loads((client.raw_dir / "turn.execution.json").read_text())
    assert meta["status"] == ("cancelled" if cancel else "timed_out")
    assert meta["returncode"] is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("execution_budget", [0.1, None])
async def test_host_timeout_reports_budget_and_preserved_evidence(tmp_path, execution_budget):
    binary = executable(
        tmp_path,
        "import sys,time\n"
        "sys.stdin.read()\n"
        "print('synthetic ongoing work',file=sys.stderr,flush=True)\n"
        "time.sleep(60)\n",
    )
    client = JiuwenSwarmProcess(tmp_path / "instance", binary=binary)
    with pytest.raises(TimeoutError) as captured:
        await client.run("task", workspace=tmp_path, timeout_s=execution_budget,
                         host_timeout_s=2, label="deadline")
    receipt = client.raw_dir / "deadline.execution.json"
    metadata = json.loads(receipt.read_text())
    assert "host deadline 2s" in str(captured.value)
    assert str(receipt) in str(captured.value)
    expected_budget = "0.1s" if execution_budget is not None else "unbounded"
    assert f"requested execution budget: {expected_budget}" in str(captured.value)
    assert metadata["status"] == "timed_out"
    assert metadata["timeout_source"] == "host"
    assert metadata["requested_timeout_s"] == execution_budget
    assert metadata["host_timeout_s"] == 2
    assert metadata["timeout_elapsed_s"] >= 2
    assert metadata["duration_s"] >= metadata["timeout_elapsed_s"]
    assert metadata["returncode"] is not None
    assert not (client.raw_dir / "deadline.result.json").exists()
    assert (client.raw_dir / "deadline.stderr").read_text() == "synthetic ongoing work\n"


@pytest.mark.asyncio
async def test_native_timeout_remains_a_native_failure(tmp_path):
    binary = executable(
        tmp_path,
        """import json,sys
request=json.load(sys.stdin)
print(json.dumps(dict(schema_version='0.1',type='result',sequence=0,
 request_id=request['request_id'],session_id='session',status='timed_out',
 exit_code=124,error=dict(code='TIMEOUT',message='Native deadline elapsed'))))
sys.exit(124)
""",
    )
    client = JiuwenSwarmProcess(tmp_path / "instance", binary=binary)
    result = await client.run("task", workspace=tmp_path, timeout_s=10, label="native")
    with pytest.raises(RuntimeError, match="Native deadline elapsed"):
        result.require_completed()
    metadata = json.loads((client.raw_dir / "native.execution.json").read_text())
    assert metadata["timeout_source"] == "native"
    assert metadata["requested_timeout_s"] == 10
    assert metadata["host_timeout_s"] == 25
    assert "timeout_elapsed_s" not in metadata
    assert result.returncode == 124


@pytest.mark.asyncio
async def test_launch_failure(tmp_path):
    client = JiuwenSwarmProcess(tmp_path / "new", binary="/no/such/process")
    with pytest.raises(FileNotFoundError):
        await client.run("task", workspace=tmp_path, label="failed")
    assert (client.raw_dir / "failed.request.json").exists()
    assert (
        json.loads((client.raw_dir / "failed.execution.json").read_text())["status"]
        == "launch_failed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code", [("failed", 1), ("cancelled", 130), ("timed_out", 124)]
)
async def test_subprocess_legitimate_failure_is_not_protocol_error(
    tmp_path, status, code
):
    binary = executable(
        tmp_path,
        f"""import sys,json
r=json.load(sys.stdin)
print(json.dumps(dict(schema_version='0.1',type='result',sequence=0,request_id=r['request_id'],
 session_id=None,status={status!r},exit_code={code},error={{'code':'FIXTURE','message':'legitimate failure'}})))
sys.exit({code})
""",
    )
    client = JiuwenSwarmProcess(tmp_path / "instance", binary=binary)
    receipt = await client.run("task", workspace=tmp_path, label="turn")
    assert receipt.result["status"] == status
    assert receipt.returncode == code
    assert (client.raw_dir / "turn.result.json").exists()
    assert not (client.raw_dir / "turn.invalid.json").exists()
    with pytest.raises(RuntimeError, match=status):
        receipt.require_completed()


def test_instance_config_isolation_and_native_tool_policy(tmp_path):
    import yaml

    from reward_engine.adapters.jiuwenswarm.process import make_instance

    template = tmp_path / "template.yaml"
    template.write_text("models: {}\npermissions: {enabled: false}\nmodes: {}\n")
    data = make_instance(
        tmp_path / "first",
        template=template,
        model={
            "api_base": "http://provider/v1",
            "model": "test",
            "api_key": "NEVER_PERSIST",
        },
        allowed_tools=["mcp_claw_Read"],
    )
    document = yaml.safe_load((data / "config/config.yaml").read_text())
    assert document["permissions"]["defaults"]["*"] == "deny"
    assert document["permissions"]["tools"]["mcp_claw_Read"] == "allow"
    assert document["permissions"]["tools"]["task_tool"] == "deny"
    assert document["react"]["subagent_runtime"]["enabled"] is False
    assert "NEVER_PERSIST" not in (data / "config/config.yaml").read_text()
    assert "NEVER_PERSIST" not in (data / "instance.json").read_text()
    assert not (tmp_path / "second").exists()


def test_bundle_rejects_unrecorded_code(tmp_path):
    from reward_engine.adapters.jiuwenswarm.bundle import verify
    from reward_engine.adapters.jiuwenswarm.process import file_hash

    (tmp_path / "code.py").write_text("pass\n")
    (tmp_path / "bin").mkdir()
    launcher = tmp_path / "bin/jiuwenswarm-process"
    launcher.write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "jiuwenswarm-runtime-bundle-v1",
                "schema": "0.1",
                "packages": {"workswarm": "0.2.7"},
                "files": {
                    "code.py": file_hash(tmp_path / "code.py"),
                    "bin/jiuwenswarm-process": file_hash(launcher),
                },
            }
        )
    )
    assert verify(tmp_path)["jiuwenswarm_version"] == "0.2.7"
    (tmp_path / "injected.pth").write_text("import injected\n")
    with pytest.raises(ValueError, match="inventory"):
        verify(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["agent.code.normal", "agent.work.normal"])
async def test_mode_wire_and_same_session_mode_guard(tmp_path, mode):
    binary = executable(
        tmp_path,
        """import json,sys
r=json.load(sys.stdin)
assert ('agent' in r) == (r['mode']=='agent.code.normal')
assert 'Execution instruction' in (r.get('agent',{}).get('instructions','')+r['input'])
print(json.dumps(dict(schema_version='0.1',type='result',sequence=0,request_id=r['request_id'],session_id='session',status='completed',exit_code=0,output='ok')))
""",
    )
    client = JiuwenSwarmProcess(tmp_path / "actor", binary=binary)
    result = await client.run(
        "task", workspace=tmp_path, mode=mode, instructions="Execution instruction"
    )
    assert result.output == "ok"
    opposite = (
        "agent.work.normal" if mode == "agent.code.normal" else "agent.code.normal"
    )
    with pytest.raises(ValueError, match="Cannot change mode"):
        await client.run(
            "next", workspace=tmp_path, session_id="session", mode=opposite
        )
    assert len(list(client.raw_dir.glob("*.request.json"))) == 1
