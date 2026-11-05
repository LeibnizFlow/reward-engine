"""Owned task services with bounded readiness and durable startup diagnostics."""

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from reward_engine.adapters.workbuddy.io import write


class ServiceStartError(RuntimeError):
    pass


class ServiceManager:
    def __init__(self, services, cwd, mock_today=None, *, output):
        self.services, self.cwd, self.mock_today = services, cwd, mock_today
        self.output = Path(output)
        self.spawned = []
        self.lock = threading.RLock()
        self.closed = False

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        try:
            # Reuse the client; recreating its SSL context on every HTTP probe
            # can itself consume the startup deadline on a loaded filesystem.
            with httpx.Client(trust_env=False, timeout=2) as client:
                for index, service in enumerate(self.services):
                    self._spawn(service, index, client)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _spawn(self, service, index, client):
        command = shlex.split(service.command)
        if command and command[0] in ("python", "python3"):
            command[0] = sys.executable
        env = {key: value for key, value in os.environ.items()
               if key.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}
        env.update(service.env or {})
        env["PYTHONUNBUFFERED"] = "1"
        if self.mock_today:
            env["MOCK_TODAY"] = self.mock_today
        log_path = self.output / f"{index}.log"
        with self.lock, log_path.open("xb") as log:
            if self.closed:
                raise ServiceStartError("Service startup was cancelled")
            proc = subprocess.Popen(command, cwd=self.cwd, env=env,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            self.spawned.append(proc)
        timeout = max(60, service.ready_timeout)
        start = time.monotonic()
        metadata = {"service": service.name, "pid": proc.pid, "timeout_s": timeout,
                    "log": str(log_path), "status": "starting"}
        receipt = self.output / f"{index}.json"
        write(receipt, metadata)
        try:
            while time.monotonic() - start < timeout:
                if self.closed:
                    raise ServiceStartError("Service startup was cancelled")
                if proc.poll() is not None:
                    raise ServiceStartError(
                        f"Service {service.name!r} exited with {proc.returncode}; log: {log_path}")
                try:
                    response = client.request(service.health_check_method.upper(),
                                              service.health_check,
                                              headers={"X-Health-Check": "1"})
                    if response.status_code < 500:
                        metadata["status"] = "ready"
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.3)
            raise ServiceStartError(
                f"Service {service.name!r} did not become ready within {timeout}s; log: {log_path}")
        except BaseException as error:
            metadata.update(status="failed", error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            metadata.update(elapsed_s=time.monotonic() - start, returncode=proc.poll())
            write(receipt, metadata)

    def __exit__(self, *_):
        with self.lock:
            self.closed = True
            processes, self.spawned = self.spawned, []
        for proc in reversed(processes):
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                if sig == signal.SIGTERM:
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        pass
            proc.wait()
