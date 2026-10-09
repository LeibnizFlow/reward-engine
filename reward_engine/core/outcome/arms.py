"""Versioned experimental treatments, shared by the loop and frozen-Pack entry."""

from reward_engine.core.outcome.policy_critic import settings as critic_settings

LEGACY = "legacy_abcd"
FEEDBACK = "hypothesis_feedback_v1"


def has_hypothesis(method):
    return (
        isinstance(method, dict)
        and method.get("status") == "propose"
        and isinstance(method.get("method"), str)
        and bool(method["method"].strip())
    )


def arm_spec(protocol, arm, *, legacy_critic=False):
    if protocol not in {LEGACY, FEEDBACK} or arm not in {"A", "B", "C", "D"}:
        raise ValueError("Unknown arm_protocol or invalid A/B/C/D arms")
    if protocol == FEEDBACK:
        return {
            "hypothesis": arm == "B",
            "static_reward": False,
            "delivery_review": arm == "C",
            "policy_critic": arm == "D",
        }
    review = arm in {"B", "D"}
    return {
        "hypothesis": arm in {"C", "D"},
        "static_reward": review,
        "delivery_review": review,
        "policy_critic": review and legacy_critic,
    }


def configure(config, default_arms):
    """Normalize the copied experiment config before any paid work starts."""
    protocol = config.setdefault("arm_protocol", LEGACY)
    arms = config.setdefault(
        "arms", list("ABCD") if protocol == FEEDBACK else default_arms
    )
    if (
        not isinstance(arms, list)
        or not arms
        or any(not isinstance(a, str) for a in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError("Choose distinct A/B/C/D arms")
    for arm in arms:
        arm_spec(protocol, arm)
    critic = dict(config.get("policy_critic") or {})
    if protocol == FEEDBACK:
        config.setdefault("rounds", 1)
        critic.setdefault("every_k_turns", 1)
        if "D" in arms and critic.get("enabled") is False:
            raise ValueError("D requires Runtime Policy Critic; omit D to disable it")
        critic["enabled"] = "D" in arms
        if "C" in arms and (
            type(config.get("max_revisions", 2)) is not int
            or config.get("max_revisions", 2) < 1
        ):
            raise ValueError(
                "C requires at least one same-session revision and recheck"
            )
        if config.get("actor_harness", {}).get("enabled"):
            raise ValueError(
                "This protocol routes Harness through C review/D Critic, not extra Actor tools"
            )
    config["policy_critic"] = critic_settings(critic)
    return arms
