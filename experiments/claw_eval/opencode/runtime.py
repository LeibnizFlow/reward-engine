"""OpenCode sampler with host-owned Claw dispatch, audits, and native trace."""

import asyncio
import base64
import json
import os
import shutil
import socket
import time
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request

from reward_engine.adapters.workbuddy.evidence import collect
from reward_engine.adapters.workbuddy.io import auth, digest, read, write
from reward_engine.core.outcome.runner import review_result_fields, run_delivery


async def command(*args, timeout=120, stdout=None):
    process = await asyncio.create_subprocess_exec(
        *map(str, args),
        stdout=stdout or asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        data, _ = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(
            f"{args[0]} exited {process.returncode}: {(data or b'').decode(errors='replace')[-2000:]}"
        )
    return (data or b"").decode()


def free_port():
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", 0))
        return sock.getsockname()[1]


def opencode_config(route, base_url, proxy_token, context_window, max_output_tokens):
    """Build the pinned OpenCode provider without importing the legacy runner."""
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    provider_id = "campaign"
    provider_model = provider_id + "/" + route
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": provider_model,
        "small_model": provider_model,
        "autoupdate": False,
        "permission": "allow",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Claw campaign proxy",
                "options": {"baseURL": base_url, "apiKey": proxy_token},
                "models": {
                    route: {
                        "name": route,
                        "limit": {
                            "context": context_window,
                            "output": max_output_tokens,
                        },
                    }
                },
            }
        },
    }


def public_task(config, task_id):
    from claw_eval.models.task import TaskDefinition

    task = TaskDefinition.from_yaml(
        Path(config["benchmark_root"]) / "tasks" / task_id / "task.yaml"
    )
    if task.user_agent.enabled:
        raise ValueError("This campaign has no user-agent tasks")
    documents = [
        {"path": p, "text": (Path(task.task_file).parent / p).read_text()}
        for p in task.prompt.attachments
    ]
    return task, {
        "task_id": task_id,
        "prompt": task.prompt.text,
        "documents": documents,
    }


class Runtime:
    def __init__(self, config):
        self.config = config
        self.sessions = {}
        self.start_lock = asyncio.Lock()
        self.app = FastAPI()
        self.app.post("/tool/{trial_id}")(self.tool)
        self.app.get("/search")(self.search)

    async def search(self, request: Request):
        if request.query_params.get("api_key") != auth(self.config):
            raise HTTPException(401)
        from experiments.claw_eval.opencode.search import search

        query = request.query_params.get("q", "")
        maximum = max(1, min(10, int(request.query_params.get("num", "10"))))
        response = await search(query, maximum)
        write(
            Path(self.config["output"]) / "search_transport" / (uuid4().hex + ".json"),
            {
                "provider": "bing_rss",
                "query": query,
                "maximum": maximum,
                "response": response,
                "at": time.time(),
            },
        )
        return response

    async def tool(self, trial_id: str, request: Request):
        actor = self.sessions.get(trial_id)
        if (
            actor is None
            or request.headers.get("authorization") != "Bearer " + actor.token
        ):
            raise HTTPException(401)
        body = await request.json()
        if body["method"] == "tools/list":
            return {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.input_schema,
                    }
                    for t in actor.tools
                ]
            }
        if body["method"] != "tools/call" or not actor.accept_tools:
            raise HTTPException(400)
        from claw_eval.models.content import ToolUseBlock

        params = body["params"]
        if params["name"] not in {t.name for t in actor.tools}:
            raise HTTPException(400, "Undeclared tool")
        call = ToolUseBlock(
            id="call_" + uuid4().hex,
            name=params["name"],
            input=params.get("arguments", {}),
        )
        result, event, images = await asyncio.to_thread(
            actor.dispatcher.dispatch, call, actor.trial_id
        )
        row = {
            "event": event.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "discarded_visual_frames": bool(images),
        }
        with (actor.output / "tool_events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        actor.events.append((call, result, event))
        # Keep original public tool names and genuine service responses for graders.
        actor.trace_message("assistant", [call])
        actor.writer.write_event(event)
        actor.trace_message("user", [result])
        content = [{"type": "text", "text": p.text} for p in result.content]
        if images:
            content += [
                {"type": "image", "data": p.data, "mimeType": p.mime_type}
                for p in images
            ]
        return {"content": content, "isError": result.is_error}

    async def sample(self, task_id, output, *, guidance="", review=None):
        identity = digest(
            {
                "task": task_id,
                "guidance": guidance,
                "review": bool(review),
                "protocol": self.config.get(
                    "actor_protocol_identity", self.config["identity"]
                ),
            }
        )
        output = Path(output)
        if (output / "RESULT.json").exists():
            row = read(output / "RESULT.json")
            changed = row.get("identity") != identity
            invalid = changed or row.get("status") != "completed"
            if invalid and self.config.get("recover_changed_actor_identity"):
                root = Path(self.config["output"]).resolve()
                relative = output.resolve().relative_to(root)
                revision = (
                    root
                    / "implementation_revisions"
                    / (str(time.time_ns()) + "_actor_identity_mismatch")
                )
                destination = revision / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(output, destination)
                write(
                    revision / "revision.json",
                    {
                        "reason": "Actor guidance identity changed after construction recovery",
                        "slot": str(relative),
                        "previous_identity": row.get("identity"),
                        "expected_identity": identity,
                        "previous_status": row.get("status"),
                        "at": time.time(),
                    },
                )
            elif invalid:
                raise ValueError("Cannot reuse a changed or failed Actor slot")
            else:
                return row
        if output.exists():
            raise RuntimeError(
                "Interrupted Actor slot needs inspection; not resampling"
            )
        self.check_resources()
        output.mkdir(parents=True)
        write(output / "STARTED.json", {"identity": identity, "at": time.time()})
        actor = Actor(self, task_id, output)
        try:
            await actor.setup()
            candidate = await run_delivery(actor, review, prompt=guidance)
            await actor.finalize()
            row = {
                "status": "completed",
                "identity": identity,
                "candidate": candidate,
                "runtime_directory": str(output),
                "native_grader_trial": str(output),
                "session": actor.session,
                "review_status": review.status if review else "disabled",
                "actor_execution_s": actor.execution_s,
                "memory": review.memory.rows if review and review.memory else [],
                "task_progress": review.progress.snapshot() if review else None,
                "final_measurement": review.history[-1]
                if review and review.history
                else None,
                **review_result_fields(review, candidate),
            }
            write(output / "RESULT.json", row)
            return row
        except BaseException as error:
            write(
                output / "FAILURE.json",
                {
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "session": actor.session,
                    "identity": identity,
                },
            )
            raise
        finally:
            await actor.close()

    def check_resources(self):
        for path in (self.config["output"], self.config["docker_data_root"]):
            if shutil.disk_usage(path).free < 20 * 1024**3:
                raise RuntimeError("Storage admission floor reached")
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if available < 20 * 1024**3:
            raise RuntimeError("Memory admission floor reached")


class Actor:
    def __init__(self, runtime, task_id, output):
        self.runtime, self.config, self.output = runtime, runtime.config, output
        self.task, self.public = public_task(self.config, task_id)
        self.trial_id, self.token = uuid4().hex, uuid4().hex
        self.container = "claw-office-" + self.trial_id
        self.session, self.turn, self.execution_s = None, 0, 0
        self.events, self.raw_events = [], []
        self.services = self.dispatcher = self.writer = None
        self.accept_tools = False
        self.started = time.monotonic()

    def trace_message(self, role, content, *, timestamp=None):
        from claw_eval.models.message import Message
        from claw_eval.models.trace import TraceMessage

        self.writer.write_event(
            TraceMessage(
                trace_id=self.trial_id,
                message=Message(role=role, content=content),
                **({"timestamp": timestamp} if timestamp else {}),
            )
        )

    async def setup(self):
        from claw_eval.models.trace import TraceStart
        from claw_eval.runner.dispatcher import ToolDispatcher
        from claw_eval.runner.sandbox_dispatcher import SandboxToolDispatcher
        from claw_eval.runner.sandbox_tools import SANDBOX_TOOLS
        from claw_eval.runner.services import ServiceManager
        from claw_eval.trace.writer import TraceWriter

        self.workspace = self.output / "workspace"
        self.workspace.mkdir()
        await command(
            "docker",
            "run",
            "-d",
            "--name",
            self.container,
            "--label",
            "campaign=" + self.config["campaign_id"],
            "--memory",
            "4g",
            "--cpus",
            "2",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-p",
            "127.0.0.1::8080",
            "-v",
            str(self.workspace) + ":/workspace",
            "-v",
            self.config["opencode_binary"] + ":/usr/local/bin/opencode:ro",
            "-v",
            str(Path(__file__).with_name("mcp_stdio.py")) + ":/opt/claw_mcp.py:ro",
            self.config["image"],
        )
        port = (
            (await command("docker", "port", self.container, "8080/tcp"))
            .strip()
            .rsplit(":", 1)[1]
        )
        self.sandbox_url = "http://127.0.0.1:" + port
        async with httpx.AsyncClient(trust_env=False) as client:
            for _attempt in range(60):
                try:
                    response = await client.get(self.sandbox_url + "/health")
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise RuntimeError("Sandbox failed health check")
        files = list(self.task.sandbox_files or self.task.environment.fixtures)
        files += [p for p in self.task.prompt.attachments if p not in files]
        self.inject(files)
        async with self.runtime.start_lock:
            # Reserve distinct free ports, then start services before releasing the admission lock.
            reserved, remap = [], {}
            try:
                for service in self.task.services:
                    sock = socket.socket()
                    sock.bind(("0.0.0.0", 0))
                    reserved.append(sock)
                    remap[service.port] = sock.getsockname()[1]

                def url(value):
                    from urllib.parse import urlsplit, urlunsplit

                    parts = urlsplit(value)
                    return urlunsplit(
                        parts._replace(netloc="localhost:" + str(remap[parts.port]))
                    )

                for service in self.task.services:
                    service.port = remap[service.port]
                    service.health_check = url(service.health_check)
                    if service.reset_endpoint:
                        service.reset_endpoint = url(service.reset_endpoint)
                    service.env = {**service.env, "PORT": str(service.port)}
                    if service.name == "web_real":
                        service.env.update(
                            {
                                "SERP_API_URL": "http://127.0.0.1:"
                                + str(self.config["tools_port"])
                                + "/search",
                                "SERP_DEV_KEY": auth(self.config),
                                "WEB_REAL_CACHE_DIR": str(
                                    self.output / "service_cache"
                                ),
                            }
                        )
                for endpoint in self.task.tool_endpoints:
                    endpoint.url = url(endpoint.url)
                for sock in reserved:
                    sock.close()
                self.services = ServiceManager(
                    self.task.services,
                    cwd=Path(self.config["benchmark_root"]),
                    mock_today=self.task.environment.mock_today,
                )
                await asyncio.to_thread(self.services.__enter__)
            finally:
                for sock in reserved:
                    sock.close()
        existing = {t.name for t in self.task.tools}
        self.tools = list(self.task.tools) + [
            t for t in SANDBOX_TOOLS if t.name not in existing
        ]
        self.dispatcher = SandboxToolDispatcher(
            ToolDispatcher(self.task.get_endpoint_map()), sandbox_url=self.sandbox_url
        )
        self.writer = TraceWriter(self.output / "trace.jsonl")
        self.writer.write_event(
            TraceStart(
                trace_id=self.trial_id,
                task_id=self.task.task_id,
                model=self.config["roles"]["actor"],
            )
        )
        write(self.output / "public_task.json", self.public)
        prompt = self.task.prompt.text
        if self.task.prompt.attachments:
            prompt += "\n\nAttached public files in /workspace:\n" + "\n".join(
                self.task.prompt.attachments
            )
        if self.task.environment.mock_today:
            prompt += "\nCurrent task date: " + self.task.environment.mock_today
        self.instruction = prompt
        self.trace_message("user", prompt)
        oc = opencode_config(
            self.config["roles"]["actor"],
            self.config["gateway_container"] + "/wb-" + self.trial_id,
            auth(self.config),
            200000,
            32768,
        )
        oc["mcp"] = {
            "claw": {
                "type": "local",
                "command": [
                    "python",
                    "/opt/claw_mcp.py",
                    self.config["tools_container"] + "/tool/" + self.trial_id,
                    self.token,
                ],
                "enabled": True,
                "timeout": 180000,
            }
        }
        # All task execution passes through the auditable native tool dispatcher.
        builtins = (
            "bash",
            "edit",
            "glob",
            "grep",
            "read",
            "skill",
            "task",
            "todowrite",
            "todoread",
            "webfetch",
            "websearch",
            "codesearch",
            "write",
            "question",
            "apply_patch",
            "lsp",
            "plan_enter",
            "plan_exit",
            "batch",
        )
        oc["tools"] = {name: False for name in builtins}
        oc["permission"] = {"*": "deny", "claw_*": "allow"}
        oc["instructions"] = []
        encoded = base64.b64encode(json.dumps(oc).encode()).decode()
        await command(
            "docker",
            "exec",
            self.container,
            "sh",
            "-c",
            "mkdir -p /root/.config/opencode; echo "
            + encoded
            + " | base64 -d > /root/.config/opencode/opencode.json",
        )
        self.initial = collect(
            self.workspace, candidate_id=self.trial_id, final_response=""
        )[1]
        self.runtime.sessions[self.trial_id] = self
        self.accept_tools = True

    def inject(self, paths):
        for relative in paths:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Unsafe fixture path")
            source = Path(self.task.task_file).parent / relative
            if not source.exists():
                source = Path(self.config["benchmark_root"]) / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            target = self.workspace / relative
            if not target.resolve().is_relative_to(self.workspace.resolve()):
                raise ValueError(
                    "Fixture destination escapes the workspace through a symlink"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    async def start(self, prompt):
        return await self._turn(
            self.instruction + ("\n\nOptional guidance:\n" + prompt if prompt else "")
        )

    async def resume(self, feedback):
        if not self.session:
            raise ValueError("Revision requires the original session")
        return await self._turn(feedback)

    async def _turn(self, prompt):
        self.turn += 1
        for continuation in range(3):
            remaining = self.config["actor_timeout_s"] - self.execution_s
            if remaining <= 0:
                raise TimeoutError("Actor execution budget exhausted")
            args = [
                "docker",
                "exec",
                "-e",
                "OPENCODE_DISABLE_AUTO_UPDATE=true",
                "-e",
                "OPENCODE_FAKE_VCS=git",
                self.container,
                "opencode",
                "run",
                "--dir",
                "/workspace",
                "--pure",
                "--auto",
                "--format",
                "json",
                "--thinking",
                "--model",
                "campaign/" + self.config["roles"]["actor"],
            ]
            if self.session:
                args += ["--session", self.session]
            raw = self.output / f"actor-{self.turn}-{continuation}.jsonl"
            started = time.monotonic()
            try:
                with raw.open("wb") as stream:
                    await command(*args, "--", prompt, timeout=remaining, stdout=stream)
            finally:
                self.execution_s += time.monotonic() - started
            events = []
            for line in raw.read_text().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                events.append(event)
            self.raw_events += events
            errors = [e for e in events if e.get("type") == "error"]
            if errors:
                raise RuntimeError("OpenCode emitted an error; see actor log")
            sessions = {e.get("sessionID") for e in events if e.get("sessionID")}
            if len(sessions) != 1 or (self.session and self.session not in sessions):
                raise ValueError("Missing or changed OpenCode session")
            self.session = sessions.pop()
            terminal = [e for e in events if e.get("type") == "step_finish"]
            reason = terminal[-1]["part"].get("reason") if terminal else None
            if reason == "stop":
                break
            if reason != "length" or continuation == 2:
                raise ValueError("Incomplete OpenCode terminal reply")
            prompt = "Continue the same task and workspace after the output limit; finish the pending work."
        parts = [e["part"] for e in events if e.get("type") == "text"]
        last = parts[-1].get("messageID") if parts else None
        answer = "\n".join(
            p.get("text", "") for p in parts if p.get("messageID") == last
        )
        if not answer.strip():
            raise ValueError("No public final response")
        from datetime import datetime, timezone

        for event in events:
            if event.get("type") == "text" and event.get("part", {}).get("text"):
                timestamp = datetime.fromtimestamp(
                    event["timestamp"] / 1000, timezone.utc
                ).isoformat()
                self.trace_message(
                    "assistant", event["part"]["text"], timestamp=timestamp
                )
        snapshot = self.output / "snapshots" / str(self.turn)
        await asyncio.to_thread(
            shutil.copytree, self.workspace, snapshot / "workspace", symlinks=True
        )
        candidate = collect(
            snapshot / "workspace",
            candidate_id=self.trial_id,
            final_response=answer,
            initial_hashes=self.initial,
            output=snapshot,
        )[0]
        candidate["source_documents"] += [
            {
                "source_kind": "observed_public_tool_result",
                "tool": call.name,
                "request": call.input,
                "content": result.model_dump()["content"],
                "is_error": result.is_error,
            }
            for call, result, event in self.events
        ]
        write(snapshot / "candidate.json", candidate)
        write(
            self.output / "actor_state.json",
            {
                "session": self.session,
                "turn": self.turn,
                "actor_execution_s": self.execution_s,
            },
        )
        return candidate

    async def finalize(self):
        from claw_eval.cli import _collect_env_snapshot
        from claw_eval.models.trace import AuditSnapshot, TraceEnd

        self.accept_tools = False
        async with httpx.AsyncClient(trust_env=False, timeout=15) as client:
            for service in self.task.services:
                if service.reset_endpoint:
                    url = service.reset_endpoint.rsplit("/reset", 1)[0] + "/audit"
                    response = await client.get(url)
                    response.raise_for_status()
                    self.writer.write_event(
                        AuditSnapshot(
                            trace_id=self.trial_id,
                            service_name=service.name,
                            audit_url=url,
                            audit_data=response.json(),
                        )
                    )
        # Private grader files arrive only after OpenCode has fully exited.
        self.inject(self.task.sandbox_grader_files)
        snapshot = await asyncio.to_thread(
            _collect_env_snapshot, self.sandbox_url, self.task
        )
        for relative in self.task.local_grader_files:
            source = Path(self.task.task_file).parent / relative
            snapshot["local_file:" + relative] = {
                "content": base64.b64encode(source.read_bytes()).decode(),
                "encoding": "base64",
            }
        write(self.output / "environment-snapshot.json", snapshot)
        finishes = [
            e["part"] for e in self.raw_events if e.get("type") == "step_finish"
        ]
        self.writer.write_event(
            TraceEnd(
                trace_id=self.trial_id,
                total_turns=len(finishes),
                input_tokens=sum(p.get("tokens", {}).get("input", 0) for p in finishes),
                output_tokens=sum(
                    p.get("tokens", {}).get("output", 0) for p in finishes
                ),
                wall_time_s=time.monotonic() - self.started,
            )
        )
        self.writer.close()
        trace_path = self.output / "trace.jsonl"
        raw_trace = self.output / "trace.unsorted.jsonl"
        shutil.copyfile(trace_path, raw_trace)
        rows = [
            json.loads(line)
            for line in raw_trace.read_text().splitlines()
            if line.strip()
        ]
        rows.sort(
            key=lambda row: (row["type"] == "trace_end", row.get("timestamp", ""))
        )
        temporary = trace_path.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        temporary.replace(trace_path)

    async def close(self):
        self.accept_tools = False
        self.runtime.sessions.pop(self.trial_id, None)
        try:
            await command("docker", "rm", "-f", self.container)
        except Exception:
            pass
        if self.services:
            await asyncio.to_thread(self.services.__exit__, None, None, None)
        if self.dispatcher:
            self.dispatcher.close()
        if self.writer:
            self.writer.close()
