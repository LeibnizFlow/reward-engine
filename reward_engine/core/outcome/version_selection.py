"""Select immutable delivered versions without rewriting the native Actor trace."""

import asyncio
from copy import deepcopy

from reward_engine.core.batch import BatchPolicy
from reward_engine.core.outcome import jury
from reward_engine.core.outcome.admission import critical_tests, preservation
from reward_engine.core.outcome.construction import fingerprint
from reward_engine.core.outcome.context import build_context
from reward_engine.core.outcome.harness import measure


def review_scope(task, candidate):
    """Public requirements binding, independent of the proposed artifact's quality."""
    context, _ = build_context(task, [candidate])
    return fingerprint(
        {
            "task": task,
            "user_updates": context["products"][0]["user_updates"],
            "task_scope_sha256": candidate.get("task_scope_sha256"),
            "artifact_task_scope_sha256": candidate["artifact"].get(
                "task_scope_sha256"
            ),
        }
    )


def validate_review_session(row, task, pack):
    """Reject incomplete retention evidence before any measurement or model call.

    Historical/custom version lists keep their existing contract. Versioned
    DeliveryReview exports and any regression claim require the complete binding.
    """
    session = row.get("review_session") or {}
    if not isinstance(session, dict):
        raise ValueError("Invalid review session")
    retention = session.get("retention")
    regressed = (
        row.get("review_status") == "revision_regressed"
        or session.get("status") == "revision_regressed"
    )
    retained_fields = {
        "retained_candidate",
        "retained_measurement",
        "retained_delivery",
    }
    if not (
        regressed
        or retention is not None
        or retained_fields & row.keys()
        or "schema_version" in session
    ):
        return

    def require(condition, reason):
        if not condition:
            raise ValueError("Invalid review session: " + reason)

    try:
        require(
            session.get("schema_version") == "outcome.delivery_review.v1",
            "missing or unsupported versioned history",
        )
        require(
            session["pack_id"] == pack["pack_id"]
            and session["pack_sha256"] == fingerprint(pack)
            and session["task_sha256"] == fingerprint(task),
            "Task or Pack binding mismatch",
        )
        require(
            row.get("review_status", session["status"]) == session["status"],
            "review status mismatch",
        )
        versions = session["versions"]
        require(isinstance(versions, list) and bool(versions), "missing versions")
        for number, version in enumerate(versions, 1):
            candidate, measurement = version["candidate"], version["measurement"]
            require(
                version["candidate_id"] == f"delivery_{number}",
                "missing, duplicate or reordered version",
            )
            require(
                version["artifact_sha256"] == fingerprint(candidate["artifact"])
                and version["candidate_sha256"] == fingerprint(candidate),
                "immutable candidate binding mismatch",
            )
            require(
                version["measurement_sha256"] == fingerprint(measurement)
                and measurement["pack_id"] == pack["pack_id"]
                and version["pack_sha256"] == session["pack_sha256"],
                "measurement or Pack binding mismatch",
            )
            require(
                isinstance(measurement["results"], list)
                and all(
                    isinstance(r, dict)
                    and isinstance(r.get("test_id"), str)
                    and r.get("status") in {"pass", "fail", "unknown"}
                    for r in measurement["results"]
                ),
                "malformed measurement results",
            )
            require(
                isinstance(version["measurement_source"], str)
                and bool(version["measurement_source"].strip()),
                "missing measurement source",
            )
            require(
                version["requirements_sha256"] == review_scope(task, candidate),
                "public requirements binding mismatch",
            )
        latest = versions[-1]
        require(
            session["latest_version_id"] == latest["candidate_id"]
            and latest["candidate_sha256"] == fingerprint(row["candidate"]),
            "native candidate differs from terminal version",
        )
        if "final_measurement" in row:
            require(
                fingerprint(row["final_measurement"]) == latest["measurement_sha256"],
                "native measurement differs from terminal version",
            )
        if not regressed:
            require(
                retention is None and not retained_fields & row.keys(),
                "retention without a regression decision",
            )
            return
        require(
            len(versions) >= 2 and isinstance(retention, dict),
            "missing retention history",
        )
        retained = versions[-2]
        require(
            retention["status"] == "artifact_retained"
            and retention["reason"] == "critical_check_regression"
            and retention["native_state_restored"] is False
            and retention["retained_version_id"] == retained["candidate_id"]
            and retention["rejected_version_id"] == latest["candidate_id"],
            "retention unavailable or references inconsistent versions",
        )
        require(
            retained["requirements_sha256"] == latest["requirements_sha256"],
            "cannot retain an earlier public Task scope",
        )
        preserved, lost = preservation(
            retained["measurement"], latest["measurement"], pack
        )
        require(
            not preserved and retention["lost_checks"] == lost,
            "regression evidence mismatch",
        )
        if retained_fields & row.keys():
            require(retained_fields <= row.keys(), "incomplete retained result")
            require(
                fingerprint(row["retained_candidate"]) == retained["candidate_sha256"]
                and fingerprint(row["retained_measurement"])
                == retained["measurement_sha256"],
                "retained candidate/measurement mismatch",
            )
            receipt = row["retained_delivery"]
            require(
                all(
                    receipt[k] == retained[k]
                    for k in (
                        "candidate_id",
                        "artifact_sha256",
                        "candidate_sha256",
                        "measurement_sha256",
                        "measurement_source",
                    )
                )
                and receipt["native_state_restored"] is False
                and receipt["official_evaluation"] == "not_run",
                "retained receipt binding mismatch",
            )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "Invalid review session: incomplete or malformed binding"
        ) from error


def choose(versions, ranking, pack):
    """Require strict preference and current critical evidence; otherwise keep prior."""
    selected = versions[0]
    decisions = []
    positions = {
        p: i for i, group in enumerate(ranking.get("groups", [])) for p in group
    }
    for version in versions[1:]:
        before, after = (
            selected.get("measurement") or {},
            version.get("measurement") or {},
        )
        preserved, lost = preservation(before, after, pack)
        known = {
            r["test_id"]
            for r in after.get("results", [])
            if r.get("status") in {"pass", "fail"} and not r.get("error")
        }
        complete = critical_tests(pack) <= known
        preferred = (
            ranking.get("status") == "compared"
            and selected["product_id"] in positions
            and version["product_id"] in positions
            and positions[version["product_id"]] < positions[selected["product_id"]]
        )
        accept = preferred and preserved and complete
        decisions.append(
            {
                "candidate_id": version["candidate_id"],
                "accepted": accept,
                "strictly_preferred": preferred,
                "critical_checks_complete": complete,
                "lost_critical_checks": lost,
            }
        )
        if accept:
            selected = version
    return selected, decisions


async def select_versions(
    task,
    pack,
    rows,
    registry,
    roles,
    *,
    record=lambda *_: None,
    reduce_max_bytes=393216,
):
    from reward_engine.core.outcome.task_sequence import (
        TaskSequence,
        require_version_scope,
    )

    # Validate every exported session before any arm can initiate model/tool work.
    for row in rows.values():
        current_task = row.get("effective_task") or task
        current_pack = row.get("effective_pack") or (
            TaskSequence(current_task).rebase(pack) if current_task != task else pack
        )
        validate_review_session(row, current_task, current_pack)

    if any(row.get("effective_task") for row in rows.values()):
        scoped = {}
        for arm, row in rows.items():
            if not row.get("candidate"):
                continue
            current_task = row.get("effective_task") or task
            current_pack = row.get("effective_pack") or (
                TaskSequence(current_task).rebase(pack)
                if current_task != task
                else pack
            )
            local = {
                k: deepcopy(v)
                for k, v in row.items()
                if k not in {"effective_task", "effective_pack"}
            }
            versions = local.get("review_session", {}).get("versions", [])
            require_version_scope(versions, local["candidate"])
            for version in versions:
                if version["artifact_sha256"] != fingerprint(
                    version["candidate"]["artifact"]
                ):
                    raise ValueError("Immutable delivery hash mismatch")
            if len(versions) <= 1:
                if versions and versions[0]["artifact_sha256"] != fingerprint(
                    local["candidate"]["artifact"]
                ):
                    raise ValueError(
                        "Native candidate differs from its recorded current-scope version"
                    )
                local["selected_candidate"] = deepcopy(local["candidate"])
                local["delivery_selection"] = {
                    "candidate_id": "delivery_1",
                    "artifact_sha256": fingerprint(local["candidate"]["artifact"]),
                    "decisions": [],
                    "native_trace_preserved": True,
                    "boundary": "Only one delivered version in the current public Task scope",
                }
                local["version_checks"] = versions or [
                    {
                        "candidate_id": "delivery_1",
                        "candidate": deepcopy(local["candidate"]),
                        "artifact_sha256": fingerprint(local["candidate"]["artifact"]),
                        "measurement": None,
                    }
                ]
                scoped[arm] = {"status": "single_version"}
            else:
                scoped[arm] = await select_versions(
                    current_task,
                    current_pack,
                    {arm: local},
                    registry,
                    roles,
                    record=lambda n, d, a=arm: record("task_scopes/" + a + "/" + n, d),
                    reduce_max_bytes=reduce_max_bytes,
                )
            for key in ("selected_candidate", "delivery_selection", "version_checks"):
                if key in local:
                    row[key] = local[key]
        selection = {
            "arm": None,
            "reason": "Native dialogues keep their own revealed user requirements and deliveries",
        }
        record("selected_delivery", selection)
        return {
            "ranking": {"status": "per_task_scope", "arms": scoped},
            "selection": selection,
        }

    groups, unique = {}, {}
    for arm, row in rows.items():
        if not row.get("candidate"):
            continue
        versions = deepcopy(row.get("review_session", {}).get("versions", []))
        if not versions:
            versions = [
                {
                    "candidate_id": "delivery_1",
                    "candidate": deepcopy(row["candidate"]),
                    "artifact_sha256": fingerprint(row["candidate"]["artifact"]),
                    "measurement": None,
                }
            ]
        require_version_scope(versions, row["candidate"])
        for version in versions:
            digest = fingerprint(version["candidate"]["artifact"])
            if version["artifact_sha256"] != digest:
                raise ValueError("Immutable delivery hash mismatch")
            unique.setdefault(digest, version["candidate"])
        groups[arm] = versions
    if not unique:
        return {
            "ranking": {"status": "unavailable"},
            "selection": {"arm": None, "reason": "no_artifacts"},
        }
    # Rank complete artifact bodies only: no arm labels, revision history, feedback or prior grades.
    ordered = sorted(unique)
    products = {digest: f"product_{i}" for i, digest in enumerate(ordered, 1)}
    anonymous = [
        {
            "artifact": deepcopy(unique[digest]["artifact"]),
            "source_documents": deepcopy(unique[digest].get("source_documents", [])),
        }
        for digest in ordered
    ]
    for candidate in anonymous:
        candidate["artifact"].pop("public_consultation", None)
        candidate["artifact"].pop("earlier_public_deliveries", None)
    critical = sorted(critical_tests(pack))

    async def check(digest):
        context, evidence = build_context(task, [unique[digest]])
        try:
            result = await measure(
                pack,
                registry,
                context,
                evidence,
                policy=BatchPolicy(deadline_s=480),
                test_ids=critical,
            )
        except Exception as error:
            result = {
                "pack_id": pack["pack_id"],
                "results": [],
                "error": type(error).__name__,
            }
        result["artifact_sha256"] = digest
        record("version_checks/" + digest, result)
        return digest, result

    measured = dict(await asyncio.gather(*(check(digest) for digest in ordered)))
    context, evidence = build_context(task, anonymous)
    # Rank complete outcomes against the original Task, independently of our
    # generated standards, attack rationale and suggested edits.
    try:
        ranking = await jury.rank(
            context,
            evidence,
            {str(i): roles(f"jury_member_{i}") for i in range(1, 4)},
            roles("jury_reducer"),
            policy=BatchPolicy(deadline_s=900),
            reduce_max_bytes=reduce_max_bytes,
            record=lambda n, d: record("version_ranking/" + n, d),
        )
    except Exception as error:
        ranking = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    record("version_ranking/result", ranking)
    finalists = {}
    for arm, versions in groups.items():
        for version in versions:
            digest = version["artifact_sha256"]
            version.update(
                product_id=products[digest],
                measurement=deepcopy(measured[digest]),
                measurement_sha256=fingerprint(measured[digest]),
                measurement_source="version_checks/" + digest + ".json",
            )
        selected, decisions = choose(versions, ranking, pack)
        selection = {
            k: deepcopy(selected[k])
            for k in ("candidate_id", "artifact_sha256", "product_id")
        }
        selection.update(
            check_source="version_checks/" + selected["artifact_sha256"] + ".json",
            decisions=decisions,
            native_trace_preserved=True,
            boundary="Fallible public ranking and scoped checks; no score guarantee",
        )
        rows[arm]["selected_candidate"] = deepcopy(selected["candidate"])
        rows[arm]["delivery_selection"] = selection
        rows[arm]["version_checks"] = versions
        finalists[arm] = selected
        record(arm + "/delivery_selection", selection)
    baseline = "A" if "A" in finalists else next(iter(finalists))
    across = [dict(finalists[baseline], candidate_id=baseline)] + [
        dict(v, candidate_id=a) for a, v in finalists.items() if a != baseline
    ]
    selected, decisions = choose(across, ranking, pack)
    selection = {
        "arm": selected["candidate_id"],
        "artifact_sha256": selected["artifact_sha256"],
        "product_id": selected["product_id"],
        "decisions": decisions,
        "source": selected["candidate_id"] + "/delivery_selection.json",
    }
    record("selected_delivery", selection)
    return {"ranking": ranking, "selection": selection}
