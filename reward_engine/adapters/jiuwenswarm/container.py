"""Docker transport shared by Claw, Office checkpoint workers and Harbor."""

import asyncio
from pathlib import Path

from reward_engine.adapters.workbuddy.io import write

from .bundle import identity
from .policy import capabilities, mode, policy
from .process import JiuwenSwarmProcess, file_hash, make_instance

MOUNT = "/opt/jiuwenswarm"
TEMPLATE = "lib/python3.12/site-packages/jiuwenswarm/resources/config.yaml"


def backend(config):
    value = config.get("actor_backend", "opencode")
    if value not in ("opencode", "jiuwenswarm"):
        raise ValueError(f"Unknown actor_backend: {value}")
    return value


def bundle_path(config):
    path = Path(config["jiuwenswarm_bundle"]).resolve()
    if not (path / "manifest.json").is_file():
        raise ValueError("Build the pinned JiuwenSwarm runtime bundle first")
    return path


def fingerprints(config):
    return {
        **identity(bundle_path(config)),
        "jiuwenswarm_mode": mode(config),
        "jiuwenswarm_tool_policy": file_hash(Path(__file__).with_name("policy.py")),
    }


async def docker(*args):
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *map(str, args),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode:
        raise RuntimeError(
            f"docker {args[0]} failed: {stderr.decode(errors='replace')[-2000:]}"
        )
    return stdout.decode()


async def prepare(
    container,
    root,
    config,
    connection,
    *,
    user="dev",
    mcp=(),
    allowed_tools=None,
    config_in_place=False,
):
    """Replace config only on resume; checkpoint data/session bytes stay intact."""
    selected_mode = mode(config)
    write(Path(root) / "capabilities.json", capabilities(config, policy(allowed_tools)))
    child_root = (
        f"/home/{user}/.jiuwenswarm-benchmark"
        if user != "root"
        else "/root/.jiuwenswarm-benchmark"
    )
    generated = Path(root) / "jiuwenswarm-config"
    data = make_instance(
        generated,
        template=bundle_path(config) / TEMPLATE,
        model={
            "api_base": connection["base_url"],
            "model": connection["model"],
            "max_output_tokens": connection.get("max_output_tokens", 32768),
        },
        mcp=mcp,
        allowed_tools=allowed_tools,
        sandboxed_shell=user != "root" and allowed_tools is None,
        execution_timeout_s=config.get("actor_timeout_s"),
    )
    await docker(
        "exec",
        "--user",
        user,
        container,
        "mkdir",
        "-p",
        child_root + "/data/config",
        child_root + "/home",
    )
    if not config_in_place:
        await docker(
            "cp",
            str(data / "config/config.yaml"),
            container + ":" + child_root + "/data/config/config.yaml",
        )
        await docker(
            "exec", "--user", "root", container, "chown", "-R", user, child_root
        )

    async def abort():
        # Killing docker-exec's host process alone leaves the real actor alive.
        await docker("kill", container)
        running = (
            await docker("inspect", "--format", "{{.State.Running}}", container)
        ).strip()
        if running != "false":
            raise RuntimeError("Actor container did not stop")

    client = JiuwenSwarmProcess(
        Path(root) / "jiuwenswarm",
        binary=MOUNT + "/bin/jiuwenswarm-process",
        prefix=(
            "docker",
            "exec",
            "-i",
            "--user",
            user,
            "--workdir",
            "/workspace",
            container,
        ),
        child_root=child_root,
        abort=abort,
        env={
            "JIUWEN_API_KEY": connection["api_key"],
            "HOME": child_root + "/home",
            "XDG_CONFIG_HOME": child_root + "/home/.config",
            "XDG_DATA_HOME": child_root + "/home/.local/share",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )

    client.mode = selected_mode
    return client
