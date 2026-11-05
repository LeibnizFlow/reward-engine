"""New experiment defaults remain separate from legacy saved configuration."""

from argparse import ArgumentParser
from copy import deepcopy

import pytest

from reward_engine.adapters.jiuwenswarm import config
from reward_engine.adapters.jiuwenswarm.container import backend
from reward_engine.core.outcome.three_b import PROTOCOL


@pytest.mark.parametrize("flag,value,expected", [
    ("--runtime", "jiuwen-swarm", "jiuwenswarm"),
    ("--runtime", "jiuwenswarm", "jiuwenswarm"),
    ("--runtime", "opencode", "opencode"),
    ("--actor-backend", "opencode", "opencode"),
])
def test_runtime_cli_preserves_canonical_config(flag, value, expected):
    parser = ArgumentParser()
    config.add_arguments(parser)
    assert config.arguments(parser.parse_args([flag, value])) == {"actor_backend": expected}
    assert config.arguments(parser.parse_args([])) == {}


def test_conflicting_runtime_flags_are_rejected():
    parser = ArgumentParser()
    config.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--runtime", "jiuwenswarm", "--actor-backend", "opencode"])


def test_new_defaults_and_absent_legacy_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(
        config,
        "fingerprints",
        lambda c: {"mode": c["jiuwenswarm_mode"], "bundle": c["jiuwenswarm_bundle"]},
    )
    value = config.options(jiuwenswarm_bundle=tmp_path)
    assert value["actor_backend"] == "jiuwenswarm"
    assert value["jiuwenswarm_mode"] == "agent.code.normal"
    assert value["generation_protocol"] == PROTOCOL and value["rounds"] == 3
    assert value["hypothesis_enabled"] is False and value["full_audit"] is True
    assert value["role_bounded_stage"] is True
    assert value["role_stream"] is False and value["role_protocol_edits"] is True
    assert value["role_stage_timeout_s"] == 1800
    legacy = {"rounds": 3}
    config.verify(legacy)
    assert legacy == {"rounds": 3} and backend(legacy) == "opencode"
    assert (
        config.options(actor_backend="opencode", generation_protocol="legacy_abcd")[
            "actor_backend"
        ]
        == "opencode"
    )


@pytest.mark.parametrize("rounds", [0, 4, True, 1.5])
def test_invalid_round_counts(rounds):
    with pytest.raises(ValueError):
        config.options(actor_backend="opencode", rounds=rounds)


@pytest.mark.parametrize(
    "constraints",
    [
        {"selection_policy": "independent_pool_v1"},
        {"public_delivery_contract": {"id": "x"}},
        {"recover_from": "old"},
        {"canary_from": "old"},
        {"reuse_first_round_from": "old"},
    ],
)
def test_constraints_fail_before_bundle_or_preparation(constraints):
    with pytest.raises(ValueError, match="incompatible"):
        config.options(**constraints)


def test_bundle_required_and_mode_change_rejected(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="explicit immutable"):
        config.options()
    monkeypatch.setattr(
        config, "fingerprints", lambda c: {"mode": c["jiuwenswarm_mode"]}
    )
    frozen = config.options(jiuwenswarm_bundle=tmp_path)
    config.verify(frozen)
    changed = deepcopy(frozen)
    changed["jiuwenswarm_mode"] = "agent.work.normal"
    with pytest.raises(ValueError, match="identity changed"):
        config.verify(changed)


def test_prepare_and_startup_do_not_scan_runtime_tree(tmp_path, monkeypatch):
    import json
    from reward_engine.adapters.jiuwenswarm import bundle

    root = tmp_path / 'runtime'
    root.mkdir()
    manifest = root / 'manifest.json'
    document = {'format': 'jiuwenswarm-runtime-bundle-v1', 'schema': '0.1',
                'packages': {'workswarm': '0.2.7'}, 'files': {'payload': 'declared-hash'}}
    manifest.write_text(json.dumps(document))
    (root / 'bin').mkdir()
    (root / 'bin/jiuwenswarm-process').write_text('#!/bin/sh\nexit 0\n')

    def forbidden(*args, **kwargs):
        pytest.fail('Experiment preparation/startup must not scan the runtime tree')

    monkeypatch.setattr(bundle, 'tree_hashes', forbidden)
    frozen = config.options(jiuwenswarm_bundle=root, rounds=1)
    assert frozen['runtime_integrity_check'] == 'manifest_only'
    config.verify(frozen)
    # Runtime payload changes are deliberately outside automatic startup checks.
    (root / 'payload').write_text('changed')
    config.verify(frozen)
    # Configuration still binds the declared bundle manifest.
    document['files']['payload'] = 'changed-declaration'
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='identity changed'):
        config.verify(frozen)
