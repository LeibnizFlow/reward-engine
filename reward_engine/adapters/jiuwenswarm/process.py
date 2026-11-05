"""Durable, interpreter-independent client for Process CLI schema 0.1."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = "0.1"
PROTOCOL = "jiuwenswarm-process-v1"
DEFAULT_BINARY = "jiuwenswarm-process"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


class ProcessProtocolError(RuntimeError):
    """The child did not produce a valid JiuwenSwarm Process document."""


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProcessProtocolError(f"{name} must be a non-empty string")
    return value


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProcessProtocolError("Non-finite JSON number")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _object(value, allowed, required, name):
    if not isinstance(value, dict):
        raise ProcessProtocolError(f"{name} must be an object")
    if set(value) - set(allowed) or set(required) - set(value):
        raise ProcessProtocolError(f"{name} has missing or unknown fields")


def _validate_record_stream(
    text: str, request: Mapping[str, Any], returncode: int | None = None
):
    """Mirror the installed schema without importing the agent in the host."""
    records = []
    for index, line in enumerate(text.splitlines()):
        try:
            record = json.loads(line)
        except ValueError as error:
            raise ProcessProtocolError(f"record {index} is not valid JSON") from error
        _finite_json(record)
        common = {"schema_version", "type", "sequence", "request_id", "session_id"}
        required = common - {"session_id"}
        if not isinstance(record, dict):
            raise ProcessProtocolError("record must be an object")
        kind = record.get("type")
        if kind == "event":
            _object(
                record,
                common | {"event_type", "payload"},
                required | {"event_type"},
                "event",
            )
            _required_string(record["event_type"], "event_type")
            payload = record.get("payload")
            if payload is not None:
                if not isinstance(payload, dict):
                    raise ProcessProtocolError(
                        "event.payload must be an object or null"
                    )
                event_type = payload.get("event_type")
                if event_type is not None and (
                    not isinstance(event_type, str)
                    or (event_type and event_type != record["event_type"])
                ):
                    raise ProcessProtocolError("payload.event_type mismatch")
        elif kind == "result":
            _object(
                record,
                common | {"status", "exit_code", "output", "error", "usage"},
                required | {"status", "exit_code"},
                "result",
            )
            if record["status"] not in (
                "completed",
                "failed",
                "cancelled",
                "timed_out",
            ):
                raise ProcessProtocolError("Unsupported terminal status")
            code = record["exit_code"]
            if type(code) is not int or code < 0:
                raise ProcessProtocolError("exit_code must be a non-negative integer")
            if record.get("output") is not None and not isinstance(
                record["output"], str
            ):
                raise ProcessProtocolError("output must be a string or null")
            if not isinstance(record.get("usage", {}), dict):
                raise ProcessProtocolError("usage must be an object")
            error = record.get("error")
            if error is not None:
                _object(
                    error,
                    {"code", "message", "retryable", "details"},
                    {"code", "message"},
                    "error",
                )
                _required_string(error["code"], "error.code")
                _required_string(error["message"], "error.message")
                if type(error.get("retryable", False)) is not bool or not isinstance(
                    error.get("details", {}), dict
                ):
                    raise ProcessProtocolError("Invalid error.retryable/details")
            if record["status"] == "completed":
                if code != 0 or error is not None or record.get("session_id") is None:
                    raise ProcessProtocolError("Invalid completed result")
            elif code == 0 or error is None:
                raise ProcessProtocolError(
                    "Non-completed result requires nonzero exit_code and error"
                )
        else:
            raise ProcessProtocolError("Unsupported record type")
        if record["schema_version"] != SCHEMA_VERSION:
            raise ProcessProtocolError("Unsupported schema")
        if type(record["sequence"]) is not int or record["sequence"] != index:
            raise ProcessProtocolError("sequence must be contiguous integers from zero")
        _required_string(record["request_id"], "request_id")
        if record["request_id"] != request["request_id"]:
            raise ProcessProtocolError("Changed request identity")
        if record.get("session_id") is not None:
            _required_string(record["session_id"], "session_id")
        records.append(record)
    if not records or [i for i, r in enumerate(records) if r["type"] == "result"] != [
        len(records) - 1
    ]:
        raise ProcessProtocolError("JSONL must end with exactly one terminal result")
    result = records[-1]
    sessions = {r["session_id"] for r in records if r.get("session_id") is not None}
    if request.get("session_id") is not None:
        sessions.add(request["session_id"])
    # Early failures may have no resolved session; any emitted identity must match.
    if len(sessions) > 1 or (
        any(r.get("session_id") for r in records)
        and result.get("session_id") not in sessions
    ):
        raise ProcessProtocolError("Changed or lost session identity")
    if returncode is not None and result["exit_code"] != returncode:
        raise ProcessProtocolError(
            "Terminal exit_code differs from actual process returncode"
        )
    return tuple(records[:-1]), result


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hash(root: Path) -> str:
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if "__pycache__" in path.parts:
            continue
        relative = str(path.relative_to(root))
        if path.is_symlink():
            rows.append((relative, "link", os.readlink(path)))
        elif path.is_file():
            rows.append(
                (relative, "file", file_hash(path), path.stat().st_mode & 0o777)
            )
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def make_instance(
    root: Path,
    *,
    model: Mapping[str, Any],
    template: Path,
    mcp: Sequence[Mapping[str, Any]] = (),
    tools: Sequence[str] = (),
    allowed_tools: Sequence[str] | None = None,
    sandboxed_shell: bool = False,
    execution_timeout_s: float | None = None,
) -> Path:
    """Generate per-instance config from an explicitly pinned package template."""
    import yaml

    if execution_timeout_s is not None and (
        type(execution_timeout_s) not in (int, float)
        or not math.isfinite(execution_timeout_s)
        or execution_timeout_s <= 0
    ):
        raise ValueError("execution_timeout_s must be positive and finite")
    # The Process request and host watchdog still enforce the remaining Actor
    # budget. A shorter SDK first-output timer also counts gateway queue time
    # and can otherwise terminate a live session before that budget expires.
    first_chunk_timeout = 300 if execution_timeout_s is None else execution_timeout_s
    root = Path(root).resolve()
    data = root / "data"
    config = data / "config"
    config.mkdir(parents=True, exist_ok=True)
    document = yaml.safe_load(Path(template).read_text())
    document["models"] = {
        "defaults": [
            {
                "model_client_config": {
                    "api_base": model["api_base"],
                    "api_key": "${JIUWEN_API_KEY}",
                    "model_name": model["model"],
                    "client_provider": "OpenAI",
                    "endpoint_profile": "openai_compatible",
                    "timeout": max(360, first_chunk_timeout),
                    "stream_first_chunk_timeout": first_chunk_timeout,
                    "stream_idle_timeout": 120,
                    "verify_ssl": True,
                },
                "model_config_obj": {
                    "max_tokens": model.get("max_output_tokens", 32768)
                },
                "is_default": True,
            }
        ]
    }
    document["mcp"] = {"servers": [dict(entry) for entry in mcp]}
    document.setdefault("memory", {})["engine"] = "none"
    document["auto_memory_enabled"] = False
    document["progressive_tool_enabled"] = False
    document["tools"] = []
    document.setdefault("react", {})["subagent_runtime"] = {"enabled": False}
    document["task_memory"] = {"enabled": False}
    from .policy import permissions, policy

    contract = policy(allowed_tools)
    document["permissions"] = permissions(contract, sandboxed_shell=sandboxed_shell)
    for name in ("agent", "code"):
        mode = document.setdefault("modes", {}).setdefault(name, {})
        mode["memory"] = {"enabled": False}
        mode["rails"] = []
        mode["tools"] = list(tools)
    (config / "config.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    )
    _write_json(
        data / "instance.json",
        {
            "protocol": PROTOCOL,
            "template_sha256": file_hash(template),
            "model": {k: v for k, v in model.items() if k != "api_key"},
        },
    )
    return data


@dataclass(frozen=True)
class ProcessResult:
    request: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    result: dict[str, Any]
    stdout: str
    stderr: str
    returncode: int
    duration_s: float

    @property
    def session_id(self):
        return self.result.get("session_id")

    @property
    def output(self):
        return self.result.get("output") or ""

    def require_completed(self):
        if self.result["status"] != "completed":
            raise RuntimeError(
                f"JiuwenSwarm {self.result['status']}: {self.result.get('error')}"
            )
        return self


class JiuwenSwarmProcess:
    """One process per turn; optional container prefix with explicit abort hook."""

    def __init__(
        self,
        root: Path,
        *,
        binary: str = DEFAULT_BINARY,
        env: Mapping[str, str] | None = None,
        prefix: Sequence[str] = (),
        child_root: str | None = None,
        abort: Callable[[], Awaitable[None]] | None = None,
    ):
        self.root = Path(root).resolve()
        self.binary, self.env, self.prefix = str(binary), dict(env or {}), list(prefix)
        self.child_root = child_root or str(self.root)
        self.abort = abort
        self.mode = "agent.code.normal"
        self.raw_dir = self.root / "process"
        self._tasks: set[asyncio.Task] = set()

    async def run(
        self,
        prompt: str,
        *,
        workspace: Path,
        session_id: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        host_timeout_s: float | None = None,
        request_id: str | None = None,
        label: str | None = None,
        mode: str | None = None,
        instructions: str = "Complete the task in the supplied workspace and report the final result.",
        tools: Sequence[str] = ("*",),
    ) -> ProcessResult:
        from .policy import mode as validate_mode

        mode = validate_mode({"jiuwenswarm_mode": mode or self.mode})
        binding = self.root / "session-mode.json"
        if binding.exists() and json.loads(binding.read_text())["mode"] != mode:
            raise ValueError(
                "Cannot change mode in an existing session; create an independent run"
            )
        _required_string(prompt, "input")
        _required_string(instructions, "instructions")
        for value, name in (
            (session_id, "session_id"),
            (request_id, "request_id"),
            (model, "model"),
        ):
            if value is not None:
                _required_string(value, name)
        if isinstance(tools, (str, bytes)) or len(set(tools)) != len(tools):
            raise ValueError("tools must contain distinct tool names")
        for tool in tools:
            _required_string(tool, "tool")
        if host_timeout_s is not None and (
            type(host_timeout_s) not in (int, float)
            or not math.isfinite(host_timeout_s)
            or host_timeout_s <= 0
        ):
            raise ValueError("host_timeout_s must be positive and finite")
        if timeout_s is not None and (
            type(timeout_s) not in (int, float)
            or not math.isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be positive and finite")
        request = {
            "schema_version": SCHEMA_VERSION,
            "type": "run",
            "request_id": request_id or uuid4().hex,
            "session_id": session_id,
            "input": prompt,
            "agent": {
                "name": "benchmark_agent",
                "instructions": instructions,
                "model": model,
                "tools": list(tools),
                "skills": [],
            },
            "mode": mode,
            "workspace": {
                "cwd": str(workspace),
                "project_dir": str(workspace),
                "trusted_dirs": [str(workspace)],
            },
            "timeout_seconds": timeout_s,
        }
        if mode == "agent.work.normal":
            request.pop("agent")
            request["input"] = instructions + "\n\n" + prompt
        _write_json(binding, {"mode": mode})
        label = label or request["request_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label) or label in (".", ".."):
            raise ValueError("label must be a safe artifact basename")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        request_path = self.raw_dir / f"{label}.request.json"
        # Never overwrite evidence from an earlier attempt.
        with request_path.open("x") as stream:
            json.dump(request, stream, ensure_ascii=False, allow_nan=False)
        out, err = self.raw_dir / f"{label}.jsonl", self.raw_dir / f"{label}.stderr"
        child_env = {
            **self.env,
            "JIUWENSWARM_DATA_DIR": self.child_root + "/data",
            "JIUWENSWARM_CONFIG_DIR": self.child_root + "/data/config",
            "PYTHONUNBUFFERED": "1",
        }
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            in ("PATH", "LANG", "LC_ALL", "TMPDIR", "DOCKER_HOST", "DOCKER_CONTEXT")
        }
        # Isolate all HOME/XDG consumers, not just JiuwenSwarm's main config.
        home = self.root / "home"
        home.mkdir(exist_ok=True)
        env.update(
            HOME=str(home),
            XDG_CONFIG_HOME=str(home / ".config"),
            XDG_DATA_HOME=str(home / ".local/share"),
        )
        if self.prefix:
            command = [
                *self.prefix,
                "env",
                "-u",
                "LD_LIBRARY_PATH",
                *[f"{k}={v}" for k, v in child_env.items()],
                self.binary,
                "--run-json",
                "-",
            ]
        else:
            env.update(child_env)
            command = [self.binary, "--run-json", "-"]
        started = asyncio.get_running_loop().time()
        deadline = (
            host_timeout_s
            if host_timeout_s is not None
            else (timeout_s + 15 if timeout_s is not None else None)
        )
        execution_receipt = self.raw_dir / f"{label}.execution.json"
        metadata = {
            "status": "starting", "returncode": None, "signals": [],
            "requested_timeout_s": timeout_s, "host_timeout_s": deadline,
        }
        proc = None
        task = asyncio.current_task()
        self._tasks.add(task)

        async def cleanup():
            if self.abort is not None:
                try:
                    await self.abort()
                except Exception as error:
                    metadata["cleanup_error"] = str(error)
            if proc is not None:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                        metadata["signals"].append(sig.name)
                    except ProcessLookupError:
                        break
                    if sig == signal.SIGTERM:
                        # Also clean up descendants if their leader exited early.
                        await asyncio.sleep(0.05)
                        try:
                            await asyncio.wait_for(proc.wait(), 5)
                        except TimeoutError:
                            pass
                await proc.wait()

        try:
            with out.open("xb") as stdout, err.open("xb") as stderr:
                # Shield creation so cancellation cannot lose the child handle.
                spawn = asyncio.create_task(
                    asyncio.create_subprocess_exec(
                        *command,
                        cwd=str(self.root if self.prefix else workspace),
                        env=env,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                )
                try:
                    proc = await asyncio.shield(spawn)
                except asyncio.CancelledError:
                    proc = await spawn
                    raise
                metadata["status"] = "running"
                try:
                    await asyncio.wait_for(
                        proc.communicate(json.dumps(request, ensure_ascii=False).encode()),
                        deadline,
                    )
                except TimeoutError as error:
                    metadata["timeout_source"] = "host"
                    metadata["timeout_elapsed_s"] = (
                        asyncio.get_running_loop().time() - started
                    )
                    requested = (
                        f"{timeout_s:g}s" if timeout_s is not None else "unbounded"
                    )
                    raise TimeoutError(
                        f"JiuwenSwarm process {label!r} exceeded host deadline "
                        f"{deadline:g}s (requested execution budget: {requested}); "
                        f"execution receipt: {execution_receipt}"
                    ) from error
            try:
                stdout_text = out.read_text(encoding="utf-8")
            except UnicodeError as error:
                raise ProcessProtocolError("stdout is not UTF-8 JSONL") from error
            stderr_text = err.read_text(errors="replace")
            try:
                events, result = _validate_record_stream(
                    stdout_text, request, proc.returncode
                )
            except ProcessProtocolError as error:
                metadata["status"] = "protocol_error"
                _write_json(
                    self.raw_dir / f"{label}.invalid.json",
                    {"error": str(error), "returncode": proc.returncode},
                )
                raise
            _write_json(self.raw_dir / f"{label}.result.json", result)
            _write_json(self.raw_dir / f"{label}.events.json", list(events))
            if result["status"] == "timed_out":
                metadata["timeout_source"] = "native"
            if result["status"] in {"cancelled", "timed_out"}:
                await cleanup()
            metadata["status"] = "exited"
            return ProcessResult(
                request,
                events,
                result,
                stdout_text,
                stderr_text,
                proc.returncode,
                asyncio.get_running_loop().time() - started,
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                metadata["status"] = "cancelled"
            elif isinstance(error, TimeoutError):
                metadata["status"] = "timed_out"
            elif metadata["status"] != "protocol_error":
                metadata["status"] = (
                    "launch_failed" if proc is None else "execution_failed"
                )
            metadata["error"] = type(error).__name__
            cleanup_task = asyncio.create_task(cleanup())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            try:
                cleanup_task.result()
            except Exception as cleanup_error:
                metadata["cleanup_error"] = str(cleanup_error)
            raise
        finally:
            metadata.update(
                returncode=proc.returncode if proc else None,
                duration_s=asyncio.get_running_loop().time() - started,
            )
            _write_json(execution_receipt, metadata)
            self._tasks.discard(task)

    async def close(self):
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
