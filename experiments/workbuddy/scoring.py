"""Validate retained native Office scores; never normalize or rerun a judge.

Contracts: upstream judge/core/models.py and the Office shared/verifier/{items,
llm,scoring}.py. A rule FAIL may carry genuine fractional aggregate credit.
An unavailable/malformed verdict is an execution failure, never a valid zero.
"""

import math
import tomllib


class NativeScoreError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise NativeScoreError(reason)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value, *, unit=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
        and (not unit or value <= 1)
    )


def native_contract(task_id, task_toml, judge_yaml):
    """Read authored expected items/weights, with the worker's enabled LLM route."""
    import yaml

    task = tomllib.loads(task_toml.decode())
    try:
        config = yaml.safe_load(judge_yaml) or {}
    except yaml.YAMLError as error:
        raise NativeScoreError("Malformed authored native judge.yaml") from error
    require(isinstance(config, dict), "Malformed native judge.yaml")
    source = task.get("metadata", {}).get("source_case", task_id)
    require(text(source) and text(task_id), "Missing native task/source identity")
    rubrics = config.get("llm_judge", {}).get("rubrics", [])
    require(isinstance(rubrics, list), "Malformed authored native rubric list")
    rubric_ids = [r.get("id") for r in rubrics if isinstance(r, dict)]
    require(
        len(rubric_ids) == len(rubrics)
        and all(text(key) for key in rubric_ids)
        and len(set(rubric_ids)) == len(rubric_ids),
        "Missing/duplicate authored rubric identities",
    )
    judges = {"rule": ("rule_script", "rule", [f"{source}:rule"])}
    if rubric_ids:
        judges["llm_judge"] = (
            "llm_judge",
            "llm",
            [f"{source}:llm:{key}" for key in rubric_ids],
        )
    return {
        "task_id": task_id,
        "source_case": source,
        "judges": judges,
        "score_merge": config.get("score_merge") or {},
    }


def _objects(value, name, *, nonempty=True):
    require(
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(row, dict) for row in value),
        f"Malformed {name}: expected {'nonempty ' if nonempty else ''}object list",
    )
    return value


def _unique(rows, key, name):
    values = [row.get(key) for row in rows]
    require(all(text(v) for v in values), f"Missing {name} identities")
    require(len(values) == len(set(values)), f"Duplicate {name} identities")
    return dict(zip(values, rows, strict=True))


def _verdict(row, where):
    require(isinstance(row, dict), f"Malformed verdict at {where}")
    for key in ("item_id", "judge_name", "judge_type"):
        require(text(row.get(key)), f"Missing verdict {key} at {where}")
    status = row.get("status")
    require(
        status in ("pass", "fail", "review", "abstain", "build_error", "judge_error"),
        f"Invalid verdict status at {where}",
    )
    require(
        status in ("pass", "fail"), f"Unavailable native verdict {status} at {where}"
    )
    require(number(row.get("score"), unit=True), f"Invalid verdict score at {where}")
    require(
        row.get("confidence") is None or number(row["confidence"], unit=True),
        f"Invalid verdict confidence at {where}",
    )
    require(isinstance(row.get("reason"), str), f"Malformed verdict reason at {where}")
    require(
        isinstance(row.get("evidence_ids"), list)
        and all(text(ref) for ref in row["evidence_ids"]),
        f"Malformed verdict evidence at {where}",
    )
    require(
        isinstance(row.get("metadata"), dict), f"Malformed verdict metadata at {where}"
    )
    require(
        not row["metadata"].get("error_type"),
        f"Native verdict error {row['metadata'].get('error_type')} at {where}",
    )
    if row["judge_type"] == "rule_script":
        require(
            status == ("pass" if row["score"] >= 1.0 else "fail"),
            f"Rule verdict status/score disagreement at {where}",
        )
    if row["judge_type"] == "llm_judge":
        require(
            row["score"] == (1.0 if status == "pass" else 0.0),
            f"LLM verdict status/score disagreement at {where}",
        )


def _payload_int(value):
    """Pinned reward_payload.as_int semantics; retained values stay untouched."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _payload_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}
    return bool(value)


def _rule_payload_counts(raw, aggregate):
    """Check scorable source evidence and mirror pinned counts_from_payload."""
    passed = _payload_int(raw.get("tests_passed"))
    total = _payload_int(raw.get("tests_total"))
    tests = raw.get("tests")
    entries = (
        [entry for entry in tests if isinstance(entry, dict)]
        if isinstance(tests, list)
        else []
    )
    require(
        passed is not None
        or total is not None
        or bool(entries)
        or aggregate is not None,
        "Unscorable native rule source payload",
    )
    if (passed is None or total is None) and isinstance(tests, list):
        passed = sum(_payload_bool(entry.get("passed")) for entry in entries)
        total = len(entries)
    passed = 0 if passed is None or passed < 0 else passed
    total = 0 if total is None or total < 0 else total
    passed = min(passed, total)
    if total == 0 and aggregate is not None:
        return (1 if aggregate >= 1.0 else 0, 1)
    return passed, total


def _rule_payload_status(judge, verdict):
    """Reconcile the source fields used by the pinned rule runner, not comments.

    reward_payload.score_from_payload chooses the first usable aggregate field;
    status_from_payload prefers an explicit native status. Counts need not imply
    that aggregate: authored gates/weights may deliberately change it.
    """
    metadata = judge["metadata"]
    raw = metadata.get("raw")
    require(isinstance(raw, dict), "Missing/malformed native rule source payload")
    source_status = metadata.get("test_status")
    require(text(source_status), "Missing native rule source status")
    for status in (source_status, raw.get("test_status")):
        require(
            not isinstance(status, str)
            or status.strip().lower() not in {"build_error", "judge_error"},
            "Fatal native rule source status",
        )
    # Mirror the upstream field precedence and numeric conversion for checking
    # consistency only. Never replace or normalize the retained raw payload.
    aggregate = None
    for key in ("reward", "overall", "test_pass_rate", "score"):
        value = raw.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (ValueError, TypeError):
            continue
        require(math.isfinite(parsed), "Nonfinite native rule source aggregate")
        aggregate = max(0.0, min(parsed, 1.0))
        break
    passed, total = _rule_payload_counts(raw, aggregate)
    require(
        (metadata.get("tests_passed"), metadata.get("tests_total")) == (passed, total),
        "Native rule source/metadata count disagreement",
    )
    if aggregate is not None:
        require(
            verdict["score"] == aggregate,
            "Native rule source aggregate/verdict disagreement",
        )
    status = raw.get("test_status")
    if isinstance(status, str) and status:
        status = "full_pass" if status == "pass" else status
    else:
        score = aggregate
        if score is None:
            score = passed / total if total > 0 else 0.0
            if total <= 0:
                status = "build_error"
        if not status:
            status = (
                "full_pass"
                if score >= 1
                else "partial_pass"
                if score > 0
                else "no_pass"
            )
    require(source_status == status, "Native rule payload/metadata status disagreement")
    normalized = source_status.strip().lower()
    expected = (
        normalized
        if normalized in {"build_error", "judge_error"}
        else "pass"
        if normalized in {"pass", "full_pass"}
        else "fail"
    )
    require(
        judge["status"] == expected, "Native rule source/envelope status disagreement"
    )


def _validate(detail, contract):
    require(isinstance(detail, dict) and detail, "Missing/malformed native score")
    require(
        detail.get("test_status") in ("full_pass", "partial_pass", "no_pass"),
        "Native build/judge failed or test status invalid",
    )
    for key in ("reward", "overall", "test_pass_rate"):
        require(number(detail.get(key), unit=True), f"Invalid native {key}")
    verdicts = _objects(detail.get("verdicts"), "native verdicts")
    for index, row in enumerate(verdicts):
        _verdict(row, f"verdicts[{index}]")
    diagnostics = detail.get("diagnostics")
    require(isinstance(diagnostics, dict), "Missing native diagnostics/plan")
    plan = diagnostics.get("plan")
    require(
        isinstance(plan, dict) and plan.get("scoring_policy") == "pass_rate",
        "Missing/unsupported native scoring plan",
    )
    require(
        text(plan.get("task_id")) and text(plan.get("dataset_id")),
        "Missing native plan task/dataset",
    )
    items = _unique(_objects(plan.get("items"), "plan items"), "id", "plan item")
    specs = _unique(_objects(plan.get("judges"), "plan judges"), "name", "plan judge")
    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "Missing native plan metadata")
    require(
        all(number(item.get("weight")) for item in items.values()),
        "Invalid plan item weight",
    )
    if contract is not None:
        require(
            plan["task_id"] == contract["task_id"]
            and metadata.get("source_case") == contract["source_case"],
            "Native plan task/source mismatch",
        )
        require(
            metadata.get("score_merge") == contract["score_merge"],
            "Native merge differs from authored judge configuration",
        )
        require(
            set(specs) == set(contract["judges"]),
            "Native judge roster differs from authored task",
        )
        for name, (kind, family, expected) in contract["judges"].items():
            spec = specs[name]
            require(
                spec.get("type") == kind
                and spec.get("family") == family
                and spec.get("item_ids") == expected,
                "Native expected item coverage differs from authored task",
            )
        require(
            metadata.get("llm_judge_enabled") is True,
            "Configured native LLM judge route not enabled",
        )
        require(
            all(item["weight"] == 1.0 for item in items.values()),
            "Office authored item weights changed",
        )
    judges = _unique(
        _objects(detail.get("judges"), "native judges"), "judge_name", "judge"
    )
    require(set(judges) == set(specs), "Native judge coverage incomplete or unexpected")
    nested = []
    covered = set()
    for name, spec in specs.items():
        expected = spec.get("item_ids")
        require(
            isinstance(expected, list)
            and expected
            and all(text(k) for k in expected)
            and len(set(expected)) == len(expected),
            "Malformed expected judge item coverage",
        )
        require(set(expected) <= set(items), "Judge refers to undeclared plan item")
        judge = judges[name]
        require(
            judge.get("judge_type") == spec.get("type")
            and judge.get("judge_family") == spec.get("family"),
            "Native judge identity/family mismatch",
        )
        require(
            judge.get("status") in ("pass", "fail")
            and judge.get("ok") is True
            and judge.get("errors") == [],
            f"Native judge failed/unavailable: {name}",
        )
        rows = _objects(judge.get("verdicts"), f"judge {name} verdicts")
        for index, row in enumerate(rows):
            _verdict(row, f"judges[{name}].verdicts[{index}]")
            require(
                row["judge_name"] == name and row["judge_type"] == spec["type"],
                "Verdict belongs to another judge",
            )
        actual = _unique(rows, "item_id", f"judge {name} item")
        require(
            set(actual) == set(expected),
            f"Native item coverage incomplete or unexpected: {name}",
        )
        covered.update(actual)
        nested.extend(rows)
        scores = judge.get("scores")
        require(
            isinstance(scores, dict)
            and scores
            and all(text(k) and number(v, unit=True) for k, v in scores.items()),
            "Invalid native judge scores",
        )
        require(isinstance(judge.get("metadata"), dict), "Malformed judge metadata")
        if spec["type"] == "llm_judge":
            mean = round(sum(r["score"] for r in rows) / len(rows), 4)
            require(
                scores.get("llm_judge") == mean
                and scores.get("rubric_binary_mean") == mean,
                "Native LLM component/verdict disagreement",
            )
        elif spec["type"] == "rule_script":
            require(
                len(rows) == 1
                and scores.get("pass_rate") == round(rows[0]["score"], 4),
                "Native rule component/verdict disagreement",
            )
            if "overall" in scores:
                require(
                    scores["overall"] == round(rows[0]["score"], 4),
                    "Native rule overall/verdict disagreement",
                )
            _rule_payload_status(judge, rows[0])
        else:
            raise NativeScoreError(
                "Unsupported native judge type for aggregate validation"
            )
    require(covered == set(items), "Native plan contains unmeasured expected items")

    def key(row):
        return row["judge_name"], row["item_id"]

    require(
        len({key(row) for row in verdicts}) == len(verdicts),
        "Duplicate top-level native verdict",
    )
    require(
        sorted(verdicts, key=key) == sorted(nested, key=key),
        "Top-level/native judge verdict disagreement",
    )
    # Validate existing arithmetic only. Do not overwrite/round the retained values.
    merge = metadata.get("score_merge", {})
    require(isinstance(merge, dict), "Malformed native score merge configuration")
    rule = [j for j in judges.values() if j["judge_type"] == "rule_script"]
    llm = [j for j in judges.values() if j["judge_type"] == "llm_judge"]
    require(len(rule) == 1 and len(llm) <= 1, "Unsupported native component roster")
    rule_score = rule[0]["scores"]["pass_rate"]
    require(detail["test_pass_rate"] == rule_score, "Native rule aggregate mismatch")
    expected_score = rule_score
    if merge.get("method") == "weighted_sum" and llm:
        rw, lw = merge.get("rule_weight", 1.0), merge.get("llm_weight", 0.0)
        require(
            number(rw) and number(lw) and rw + lw > 0,
            "Invalid authored native merge weights",
        )
        ls = llm[0]["scores"]["llm_judge"]
        expected_score = round((rule_score * rw + ls * lw) / (rw + lw), 4)
        require(
            detail.get("llm_judge_component_score") == ls,
            "Native LLM aggregate mismatch",
        )
        expected_merge = {
            "method": "weighted_sum",
            "applied": True,
            "rule_weight": rw,
            "llm_weight": lw,
            "rule_score": rule_score,
            "llm_score": ls,
            "overall": expected_score,
        }
        for location in (
            detail.get("metadata", {}).get("score_merge"),
            diagnostics.get("score_merge"),
        ):
            require(
                isinstance(location, dict)
                and all(location.get(k) == v for k, v in expected_merge.items()),
                "Native merge receipt/component disagreement",
            )
    require(
        detail["overall"] == detail["reward"] == expected_score,
        "Native aggregate/verdict disagreement",
    )
    expected_status = (
        "full_pass"
        if expected_score >= 1
        else "partial_pass"
        if expected_score > 0
        else "no_pass"
    )
    require(
        detail["test_status"] == expected_status, "Native aggregate/status disagreement"
    )
    counts = rule[0]["metadata"]
    for key in ("tests_passed", "tests_total"):
        require(
            type(counts.get(key)) is int
            and counts[key] >= 0
            and type(detail.get(key)) is int
            and detail[key] == counts[key],
            "Native test count missing or inconsistent",
        )
    require(
        0 < counts["tests_total"] and counts["tests_passed"] <= counts["tests_total"],
        "Invalid native test counts",
    )


def scoring_errors(detail, *, contract=None, rewards=None):
    """Return explicit failures, including malformed inputs, without altering scores."""
    try:
        _validate(detail, contract)
        if rewards is not None:
            require(
                isinstance(rewards, dict) and number(rewards.get("overall"), unit=True),
                "Missing/invalid native trial reward",
            )
            for key in (
                "reward",
                "overall",
                "test_pass_rate",
                "llm_judge_component_score",
                "tests_passed",
                "tests_total",
            ):
                if key in rewards:
                    require(
                        number(rewards[key]) and rewards[key] == detail.get(key),
                        f"Native trial reward/component disagreement: {key}",
                    )
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
        return [{"error_type": "native_grade_contract", "reason": str(error)}]
    return []
