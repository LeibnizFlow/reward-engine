"""One native Harbor trial; the official result is stored outside its candidate."""

import argparse
import asyncio
import math
from pathlib import Path

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from experiments.workbuddy.scoring import scoring_errors
from reward_engine.adapters.workbuddy.io import auth, read, write


def official_result(spec, serialized, root):
    rewards = (serialized.get("verifier_result") or {}).get("rewards") or {}
    score = rewards.get("overall")
    trial_dir = root / "harbor" / spec["candidate_id"]
    score_path = trial_dir / "verifier/score.json"
    detail = read(score_path) if score_path.exists() else {}
    fatal = detail.get("test_status") in {"build_error", "judge_error"}
    fatal = fatal or any(
        r.get("status") in {"build_error", "judge_error"}
        for r in detail.get("judges", [])
    )
    errors = scoring_errors(detail)
    valid = (
        type(score) in (int, float)
        and math.isfinite(score)
        and bool(detail)
        and not fatal
        and not errors
        and not serialized.get("exception_info")
    )
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
    channel = spec["candidate_id"]
    gateway = config["gateway_container"]
    limit = spec["actor_timeout_s"] + (6000 if spec.get("pack") else 300)
    return TrialConfig.model_validate(
        {
            "task": {"path": spec["task_path"]},
            "trial_name": channel,
            "trials_dir": str(Path(spec["output"]) / "harbor"),
            "timeout_multiplier": 2.0,
            "environment_build_timeout_multiplier": 6.0,
            "agent": {
                "import_path": "reward_engine.adapters.workbuddy.agent:OutcomeOpenCodeAgent",
                "model_name": "deepseek-v4-flash",
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
                        "model_route": "deepseek-v4-flash",
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
                        "source": config["harness_mount"],
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
                    "WORKBUDDY_VERIFIER_LLM_MODEL": "deepseek-v4-flash",
                    "WORKBUDDY_VERIFIER_LLM_MAX_OUTPUT_TOKENS": "32768",
                },
            },
        }
    )


async def run(spec_path):
    spec = read(spec_path)
    spec["spec_path"] = str(spec_path.resolve())
    config = read(spec["campaign_config"])
    root = Path(spec["output"])
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
    write(root / "RESULT.json", row)
    write(root / "OFFICIAL.private.json", official_result(spec, serialized, root))
    return 0 if row["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(run(args.spec))


if __name__ == "__main__":
    raise SystemExit(main())
