"""One native Harbor trial; the official result is stored outside its candidate."""

import argparse
import asyncio
import signal
from pathlib import Path
from uuid import uuid4

from experiments.workbuddy.scoring import native_contract, scoring_errors
from reward_engine.adapters.model_config import validate_native_grader
from reward_engine.adapters.workbuddy.io import auth, read, write


def official_result(spec, serialized, root):
    rewards = (serialized.get("verifier_result") or {}).get("rewards") or {}
    score = rewards.get("overall")
    trial_dir = root / "harbor" / spec["candidate_id"]
    score_path = trial_dir / "verifier/score.json"
    detail = {}
    try:
        detail = read(score_path) if score_path.exists() else {}
        task_path = Path(spec["task_path"])
        contract = native_contract(
            spec["task"]["task_id"],
            (task_path / "task.toml").read_bytes(),
            (task_path / "tests/judge.yaml").read_bytes(),
        )
        errors = scoring_errors(detail, contract=contract, rewards=rewards)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        errors = [{"error_type": "native_grade_contract", "reason": str(error)}]
    if not isinstance(detail, dict):
        detail = {}
    valid = not errors and not serialized.get("exception_info")
    return {
        "status": "not_run" if not spec["grade"] else "scored" if valid else "failed",
        "overall": score if valid and spec["grade"] else None,
        "rewards": rewards,
        "native_test_status": detail.get("test_status"),
        "native_score": str(score_path),
        "scoring_errors": errors,
        "raw_overall": score,
        "exception": serialized.get("exception_info"),
        "trial": str(trial_dir),
    }


def trial_config(spec, config):
    from harbor.models.trial.config import TrialConfig

    channel = spec["candidate_id"]
    gateway = config["gateway_container"]
    # Absence of the opt-in native route retains the historical actor-shared route.
    # Compatibility for old frozen configurations without role fields.
    # Fresh prepare always requires and freezes an explicit actor model.
    model = config.get("roles", {}).get("actor", "deepseek-v4-flash")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("The frozen actor model must be a nonempty name")
    grader = (
        validate_native_grader(config["native_grader"])
        if "native_grader" in config
        else {"model": model, "max_tokens": 32768}
    )
    limit = spec["actor_timeout_s"] + (6000 if spec.get("pack") else 300)
    value = {
            "task": {"path": spec["task_path"]},
            "trial_name": channel,
            "trials_dir": str(Path(spec["output"]) / "harbor"),
            "timeout_multiplier": 2.0,
            "environment_build_timeout_multiplier": 6.0,
            "agent": {
                "import_path": "reward_engine.adapters.workbuddy.agent:OutcomeOpenCodeAgent",
                "model_name": model,
                "override_timeout_sec": limit,
                "extra_allowed_hosts": ["host.docker.internal"],
                "kwargs": {
                    "OUTCOME_SPEC": spec["spec_path"],
                    "OPENCODE_VERSION": "1.18.16",
                    "SWARM_ACTOR_TIMEOUT_SEC": spec["actor_timeout_s"],
                    "mount_path": "/opt/opencode-baseline",
                    "context_window": 200000,
                    "model_params": {"max_output_tokens": 32768, "temperature": 0.95},
                    "connection": {
                        "mode": "local_proxy",
                        "proxy_url": gateway + "/" + channel,
                        "model_route": model,
                    },
                },
            },
            "environment": {
                "type": "docker",
                "import_path": "workbuddy_bench.runner.docker_environment:WorkBuddyDockerEnvironment",
                "force_build": False,
                "delete": True,
                "mounts": [
                    {
                        "type": "bind",
                        "source": config.get("harness_mount", config.get("jiuwenswarm_bundle")),
                        "target": "/opt/opencode-baseline",
                        "read_only": True,
                    }
                ],
            },
            "verifier": {
                "import_path": "workbuddy_bench.judge:CompositeVerifier",
                "disable": not spec["grade"],
                "env": {
                    "WORKBUDDY_VERIFIER_LLM_BASE_URL": gateway
                    + "/grade-"
                    + channel
                    + "/v1",
                    "WORKBUDDY_VERIFIER_LLM_API_KEY": auth(config),
                    "WORKBUDDY_VERIFIER_LLM_MODEL": grader["model"],
                    "WORKBUDDY_VERIFIER_LLM_MAX_OUTPUT_TOKENS": str(
                        grader["max_tokens"]
                    ),
                },
            },
        }
    if config.get("actor_backend", "opencode") == "jiuwenswarm":
        value["agent"]["import_path"] = "reward_engine.adapters.jiuwenswarm.agent:OutcomeJiuwenSwarmAgent"
        value["agent"]["kwargs"] = {"OUTCOME_SPEC": spec["spec_path"],
                                    "connection": {"proxy_url": gateway, "model_route": model}}
        value["environment"]["mounts"] = [{"type": "bind", "source": config["jiuwenswarm_bundle"],
            "target": "/opt/jiuwenswarm", "read_only": True}]
    return TrialConfig.model_validate(value)


async def run(spec_path, *, grade_only=False, campaign_config=None):
    from harbor.trial.trial import Trial

    spec = read(spec_path)
    spec["spec_path"] = str(spec_path.resolve())
    config = read(spec["campaign_config"])
    if campaign_config is not None:
        from experiments.revision import checkpoint_identity, verifier
        updated = read(campaign_config)
        if not grade_only or checkpoint_identity(updated) != checkpoint_identity(config):
            raise ValueError("Grading revision belongs to another experiment")
        verifier(updated)(updated)
        config = updated
    root = Path(spec["output"])
    if grade_only:
        from reward_engine.adapters.jiuwenswarm.checkpoints import verify_actor
        from reward_engine.adapters.workbuddy.io import digest

        actor = read(root / "RESULT.json")
        verify_actor(actor)
        binding = digest({"spec": spec["spec_identity"], "candidate": actor["candidate"],
                          "artifacts": actor["artifact_bindings"], "config": config.get("checkpoint_config_identity", config["identity"])})
        destination = root / "grading.private" / uuid4().hex
        destination.mkdir(parents=True)
        grading_spec = {**spec, "output": str(destination), "grade": True}
        # Harbor's serialized config redacts verifier credentials. Keep the
        # validated object in memory; serialization is an audit view, not a
        # lossless configuration round trip.
        value = trial_config(grading_spec, config)
        value.agent.import_path = "reward_engine.adapters.workbuddy.frozen_delivery:FrozenDeliveryAgent"
        value.agent.kwargs = {"actor_result": str(root / "RESULT.json")}
        value.environment.mounts = []
        trial = await Trial.create(value)
        result = await trial.run()
        serialized = result.model_dump(mode="json")
        write(destination / "TRIAL_RESULT.private.json", serialized)
        official = {**official_result(grading_spec, serialized, destination), "binding": binding,
                    "record_directory": str(destination)}
        write(destination / "OFFICIAL.private.json", official)
        write(root / "OFFICIAL.private.json", official)
        return 0 if official["status"] == "scored" else 2
    trial = await Trial.create(trial_config(spec, config))
    result = await trial.run()
    serialized = result.model_dump(mode="json")
    write(root / "TRIAL_RESULT.private.json", serialized)
    actor_path = root / "ACTOR_RESULT.json"
    if actor_path.exists():
        row = read(actor_path)
    else:
        row = {"status": "failed", "error": serialized.get("exception_info")}
    row["spec_identity"] = spec["spec_identity"]
    row["runtime_directory"] = str(root)
    if config.get("generation_protocol") == "three_b_fixed_baselines_v1":
        from reward_engine.adapters.jiuwenswarm.checkpoints import artifact_bindings
        row["artifact_bindings"] = artifact_bindings(root)
    write(root / "RESULT.json", row)
    write(root / "OFFICIAL.private.json", official_result(spec, serialized, root))
    return 0 if row["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument("--campaign-config", type=Path)
    args = parser.parse_args()
    async def entry():
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        return await run(args.spec, grade_only=args.grade_only, campaign_config=args.campaign_config)
    return asyncio.run(entry())


if __name__ == "__main__":
    raise SystemExit(main())
