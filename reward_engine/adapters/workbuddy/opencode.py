"""OpenCode transport extracted from the published WorkBuddy runner.

Source: data/workbuddy_office/environment/workbuddy_swarm/agent.py.
Only installation/configuration/session transport is retained; the historical
reward implementation is not imported or executed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from pathlib import Path
from typing import Any, override

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment
from workbuddy_bench.agents._agent_user import ensure_agent_user
from workbuddy_bench.runner.model_endpoints import host_reachable_url
from workbuddy_swarm.endpoint import ProxyJsonEndpoint, api_base

OPENCODE_PROVIDER_ID = "swarm"


def opencode_config(
    *,
    route: str,
    base_url: str,
    proxy_token: str,
    context_window: int,
    max_output_tokens: int,
    provider_id: str = OPENCODE_PROVIDER_ID,
    provider_name: str = "WorkBuddy Swarm Proxy",
) -> dict[str, Any]:
    """Build an explicit Chat Completions provider for OpenCode."""
    provider_model = f"{provider_id}/{route}"
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": provider_model,
        "small_model": provider_model,
        "autoupdate": False,
        "permission": "allow",
        "provider": {
            provider_id: {
                "npm": "@ai-sdk/openai-compatible",
                "name": provider_name,
                "options": {
                    "baseURL": api_base(base_url),
                    "apiKey": proxy_token,
                },
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


class OpenCodeTransport(OpenCode):
    """Pinned OpenCode installation, configuration, and bounded session execution."""

    provider_id = OPENCODE_PROVIDER_ID
    provider_name = "WorkBuddy Swarm Proxy"

    def __init__(self, logs_dir: Path, *args: Any, **kwargs: Any) -> None:
        version = str(kwargs.pop("OPENCODE_VERSION", "1.18.16"))
        actor_timeout = kwargs.pop("SWARM_ACTOR_TIMEOUT_SEC", None)
        revision_timeout = kwargs.pop("SWARM_REVISION_TIMEOUT_SEC", None)
        self._actor_timeout_sec = float(actor_timeout) if actor_timeout else None
        self._revision_timeout_sec = (
            float(revision_timeout) if revision_timeout else None
        )
        self._mount_path = str(kwargs.pop("mount_path", "/opt/opencode-swarm"))
        connection = kwargs.pop("connection", None) or {}
        self._connection_mode = str(connection.get("mode") or "")
        self._proxy_url = str(connection.get("proxy_url") or "")
        self._route = str(
            connection.get("model_route") or kwargs.get("model_name") or ""
        )
        self._model_params = dict(kwargs.pop("model_params", None) or {})
        self._context_window = int(kwargs.pop("context_window", 200_000) or 200_000)
        kwargs.pop("context_compact_pct", None)
        kwargs.pop("instance_id", None)
        self._trial_id = self._trial_id_from_logs_dir(logs_dir)
        self._task_id = self._task_id_from_logs_dir(logs_dir)
        super().__init__(
            logs_dir,
            *args,
            version=version,
            opencode_config={},
            **kwargs,
        )

    @staticmethod
    def _trial_id_from_logs_dir(logs_dir: Path) -> str:
        path = Path(logs_dir)
        return path.parent.name if path.name == "agent" else path.name

    @classmethod
    def _task_id_from_logs_dir(cls, logs_dir: Path) -> str:
        path = Path(logs_dir)
        trial_dir = path.parent if path.name == "agent" else path
        lock_path = trial_dir / "lock.json"
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            task_name = (lock.get("task") or {}).get("name")
            if isinstance(task_name, str) and task_name.strip():
                return task_name.strip()
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        return (
            cls._trial_id_from_logs_dir(logs_dir).rsplit("__", 1)[0] or "unknown-task"
        )

    @staticmethod
    @override
    def name() -> str:
        return "opencode-swarm"

    @override
    def get_version_command(self) -> str | None:
        return 'export PATH="/usr/local/bin:$PATH"; opencode --version'

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        mount_binary = f"{self._mount_path.rstrip('/')}/bin/opencode"
        await self.exec_as_root(
            environment,
            command=(
                f"test -x {shlex.quote(mount_binary)} || "
                f'(echo "missing split-mount binary: {mount_binary}" >&2; exit 2); '
                f"ln -sf {shlex.quote(mount_binary)} /usr/local/bin/opencode; "
                'export PATH="/usr/local/bin:$PATH"; opencode --version'
            ),
        )
        await ensure_agent_user(self, environment)

    def _proxy_token(self) -> str:
        if not self._route:
            raise ValueError("OpenCode Swarm agent has no model route")
        return f"{self._trial_id}::{self._route}" if self._trial_id else self._route

    def _endpoint(self) -> ProxyJsonEndpoint:
        if self._connection_mode != "local_proxy" or not self._proxy_url:
            raise ValueError("OpenCode Swarm requires model_connection=local_proxy")
        return ProxyJsonEndpoint(
            model=self._route,
            base_url=host_reachable_url(self._proxy_url),
            api_key=self._proxy_token(),
        )

    async def _inventory(self, environment: BaseEnvironment) -> str:
        command = (
            "pwd; "
            "find . -maxdepth 5 -type f "
            "! -path './.git/*' ! -path './tests/*' ! -path '*/tests/*' "
            "! -path './.opencode/*' ! -path './node_modules/*' "
            "-print 2>/dev/null | LC_ALL=C sort | head -2000; "
            "if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree "
            ">/dev/null 2>&1; then git status --short --untracked-files=all; fi"
        )
        result = await self.exec_as_agent(environment, command=command)
        return str(result.stdout or "")[:24_000]

    async def _write_config(self, environment: BaseEnvironment) -> dict[str, str]:
        config = opencode_config(
            route=self._route,
            base_url=self._proxy_url,
            proxy_token=self._proxy_token(),
            context_window=self._context_window,
            max_output_tokens=int(
                self._model_params.get("max_output_tokens") or 32_768
            ),
            provider_id=self.provider_id,
            provider_name=self.provider_name,
        )
        encoded = base64.b64encode(json.dumps(config).encode()).decode()
        home = (
            'agent_home="$(awk -F: -v u="$(id -un)" \'$1==u {print $6}\' '
            '/etc/passwd)"; export HOME="${agent_home:-/tmp}"; '
        )
        await self.exec_as_agent(
            environment,
            command=(
                f'{home}mkdir -p "$HOME/.config/opencode"; '
                f"echo {shlex.quote(encoded)} | base64 -d > "
                '"$HOME/.config/opencode/opencode.json"'
            ),
            env={"OPENCODE_FAKE_VCS": "git"},
        )
        return {
            "OPENCODE_FAKE_VCS": "git",
            "OPENCODE_DISABLE_AUTO_UPDATE": "true",
        }

    async def _opencode_run(
        self,
        environment: BaseEnvironment,
        prompt: str,
        *,
        raw_name: str,
        session: str | None = None,
    ) -> None:
        raw_path = f"/logs/agent/{raw_name}"
        combined_path = "/logs/agent/opencode.txt"
        append = (
            f"cat {raw_path} >> {combined_path}"
            if session
            else f"cp {raw_path} {combined_path}"
        )
        session_flag = f"--session {shlex.quote(session)} " if session else ""
        home = (
            'agent_home="$(awk -F: -v u="$(id -un)" \'$1==u {print $6}\' '
            '/etc/passwd)"; export HOME="${agent_home:-/tmp}"; '
        )
        command = (
            f'{home}export PATH="/usr/local/bin:$PATH"; '
            f"opencode run --pure --auto --format json --thinking "
            f"--model {shlex.quote(f'{self.provider_id}/{self._route}')} "
            f"{session_flag}-- "
            f"{shlex.quote(prompt)} > {raw_path} 2>&1; "
            f"run_status=$?; {append}; cat {raw_path}; exit $run_status"
        )
        await self.exec_as_agent(
            environment,
            command=command,
            env=await self._write_config(environment),
        )

    async def _sync_raw_log(
        self,
        environment: BaseEnvironment,
        raw_name: str,
        *,
        append: bool,
    ) -> None:
        raw_path = f"/logs/agent/{raw_name}"
        combined_path = "/logs/agent/opencode.txt"
        operation = (
            f"cat {shlex.quote(raw_path)} >> {shlex.quote(combined_path)}"
            if append
            else f"cp {shlex.quote(raw_path)} {shlex.quote(combined_path)}"
        )
        try:
            await self.exec_as_agent(
                environment,
                command=f"test ! -f {shlex.quote(raw_path)} || {operation}",
            )
        except Exception:
            # Timeout finalization must not turn a usable partial workspace into
            # an agent failure merely because the raw trajectory could not sync.
            pass

    async def _stop_opencode(self, environment: BaseEnvironment) -> None:
        try:
            await self.exec_as_root(
                environment,
                command=(
                    "pkill -TERM -f '[o]pencode run --pure' 2>/dev/null || true; "
                    "sleep 1; "
                    "pkill -KILL -f '[o]pencode run --pure' 2>/dev/null || true"
                ),
            )
        except Exception:
            # The container may already have stopped by the time Harbor
            # cancellation reaches this cleanup path.
            pass

    async def _opencode_run_bounded(
        self,
        environment: BaseEnvironment,
        prompt: str,
        *,
        raw_name: str,
        session: str | None = None,
        timeout_sec: float | None,
    ) -> bool:
        if timeout_sec is None:
            await self._opencode_run(
                environment,
                prompt,
                raw_name=raw_name,
                session=session,
            )
            return True
        try:
            async with asyncio.timeout(timeout_sec):
                await self._opencode_run(
                    environment,
                    prompt,
                    raw_name=raw_name,
                    session=session,
                )
        except TimeoutError:
            await self._stop_opencode(environment)
            await self._sync_raw_log(
                environment,
                raw_name,
                append=session is not None,
            )
            return False
        return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def session_id(events: list[dict[str, Any]]) -> str | None:
    for event in events:
        value = event.get("sessionID") or event.get("session_id")
        if value:
            return str(value)
    return None
