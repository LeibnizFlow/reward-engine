"""Joint Reward/Harness construction; no ranking reproduction prerequisite."""

import re
from copy import deepcopy
from uuid import uuid4

from reward_engine.core.outcome.builder_refinement import (
    CONSTRUCTION_VERSION,
    completed_construction,
    refine_draft,
    scope_generated_checkers,
)
from reward_engine.core.outcome.construction import VERSION as AUDIT_VERSION
from reward_engine.core.outcome.construction import fingerprint, validate_rounds
from reward_engine.core.outcome.context import (
    ask,
    requirement_context,
    requirement_scope,
    validate_current_product_refs,
)
from reward_engine.core.outcome.measurement_tools import validate_probe_arguments

METHOD_USES = {"verification_rule", "procedure", "unclassified"}
REFERENCE_USES = {"reference_implementation", "illustrative_example"}
CONDITION_USES = {"limitation_note"}


def knowledge_roles(context):
    """Group Task Review priors by their typed use; nothing is dropped, only labelled.

    Jury criteria remain the basis of every check. Method priors may shape how a
    check measures; reference priors illustrate and are never the standard;
    Same-source condition priors travel as context, not mandatory preconditions.
    `unclassified` (pre-`use` replies) counts as method for legacy replay only.
    """
    roles = {"method_refs": [], "reference_refs": [], "condition_refs": [], "unclassified_refs": [], "conditions_unstated_refs": [],
             "conditions_by_method": {}, "source_class": {}}
    sources = {}
    for skill in context.get("skills", []):
        content, ref = skill.get("content"), skill.get("ref")
        if not isinstance(content, dict) or content.get("kind") != "task_review_prior" or not isinstance(ref, str):
            continue
        use = content.get("use", "unclassified")
        bucket = ("condition_refs" if isinstance(use, str) and use in CONDITION_USES else
                  "reference_refs" if isinstance(use, str) and use in REFERENCE_USES else
                  "method_refs" if isinstance(use, str) and use in METHOD_USES else "unclassified_refs")
        roles[bucket].append(ref)
        if content.get("conditions_unstated"):
            roles["conditions_unstated_refs"].append(ref)
        if content.get("source_class"):
            roles["source_class"][ref] = content["source_class"]
        sources[ref] = {m.get("source_id") for m in content.get("mechanisms", []) if isinstance(m, dict)} | {
            c.get("source_id") for c in content.get("citations", []) if isinstance(c, dict)}
        sources[ref].discard(None)
    for method in roles["method_refs"]:
        related = [c for c in roles["condition_refs"] if sources[c] & sources[method]]
        if related:
            roles["conditions_by_method"][method] = related
    if not any(roles[k] for k in ("method_refs", "reference_refs", "condition_refs", "unclassified_refs")):
        return None
    roles["boundary"] = ("Typed labels from Task Review; Task/Jury criteria remain the standard. Method priors may "
                         "define how to measure; references illustrate, not prescribe. Same-source condition notes "
                         "are potentially relevant context, not proof they constrain this method. Unclassified refs "
                         "need interpretation, not automatic use as measurement rules.")
    return roles


def _prior_refs(test, roles):
    """Every skill:task_review ref mentioned anywhere in the test, plus its declared knowledge_refs."""
    known = {r for key in ("method_refs", "reference_refs", "condition_refs", "unclassified_refs")
             for r in roles.get(key, [])}
    found = set()

    def visit(value):
        if isinstance(value, str):
            if value in known:
                found.add(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit({k: v for k, v in test.items() if k != "knowledge"})
    return found


def bind_knowledge(test, roles):
    """Attach role-typed prior handles and same-source conditions to one test; code adds, never removes."""
    test.pop("knowledge", None)  # Derived state must never outlive the current rubric/recipe.
    if not roles:
        return test
    cited = _prior_refs(test, roles)
    methods = sorted(cited & set(roles["method_refs"]))
    references = sorted(cited & set(roles["reference_refs"]))
    conditions = set(cited & set(roles["condition_refs"]))
    for method in methods:
        conditions.update(roles["conditions_by_method"].get(method, []))
    test["knowledge"] = {
        "method_refs": methods, "reference_refs": references,
        "applicability_refs": sorted(conditions),
        "unclassified_refs": sorted(cited & set(roles.get("unclassified_refs", []))),
        "conditions_unstated": any(m in roles["conditions_unstated_refs"] for m in methods),
        "basis": "criterion+method" if methods else "criterion+reference_only" if references else "criterion",
        "boundary": "Task/Jury criterion is the standard. applicability_refs are related notes, NOT a conjunction "
                    "of mandatory conditions. First establish whether a note actually limits the chosen method; "
                    "irrelevant notes add no requirement. An essential unverified condition makes that method's "
                    "measurement unknown, not a product failure; another valid method may still measure it. "
                    "References illustrate, not prescribe. conditions_unstated and unclassified_refs are "
                    "uncertainties, not proof of applicability or automatic check failure.",
    }
    return test


def bind_current_artifacts(test, allowed):
    """Bind declared artifact handles in executable arguments to the future product.

    Source indices, user statements, prose and arbitrary code are not aliases.
    Leave those untouched so the historical-product guard still rejects them.
    """
    changes = []

    def bind(ref, path):
        if not isinstance(ref, str):
            return ref
        current = re.sub(r"^product_(?:[2-9]|[1-9][0-9]+):artifact:", "product_1:artifact:", ref)
        if current != ref and current in allowed:
            changes.append({"path": path, "before": ref, "after": current})
            return current
        return ref

    def arguments(values, path):
        if not isinstance(values, dict):
            return
        if "ref" in values:
            values["ref"] = bind(values["ref"], path + "/ref")
        refs = values.get("evidence_refs")
        if isinstance(refs, list) and all(isinstance(ref, str) for ref in refs):
            values["evidence_refs"] = list(dict.fromkeys(
                bind(ref, f"{path}/evidence_refs/{i}") for i, ref in enumerate(refs)))

    arguments(test.get("arguments"), "arguments")
    for i, probe in enumerate(test.get("probes", []) or []):
        if isinstance(probe, dict):
            arguments(probe.get("arguments"), f"probes/{i}/arguments")
    return changes



async def build(context, evidence, jury, registry, call, *, author=None, previous=None,
                construction_rounds=1, reviewers=None, policy=None, record=lambda *_: None, repair_findings=()):
    construction_rounds = validate_rounds(construction_rounds)
    refined = construction_rounds > 1
    bounded = bool(getattr(call, "bounded_stage", False))
    if refined and reviewers is not None and len(reviewers) != 3:
        raise ValueError('Builder construction requires exactly three independent reviewer slots')
    checker_catalog = {k: v for k, v in registry.catalog().items()
                       if not (refined or bounded) or (k not in registry.recipes and '::builder:' not in k)}
    scope = {**requirement_scope(context, evidence),
             "checkers": {k: v for k, v in checker_catalog.items() if k not in registry.recipes},
             "tools": registry.tools.catalog() if registry.tools else {}}
    if refined:
        _, scoped_evidence = requirement_context(context, evidence)
        scoped_evidence.update({r: deepcopy(v) for r, v in evidence.items()
                                if r.startswith('reference:task_review:')})
        scope['construction'] = {'rounds': construction_rounds, 'version': CONSTRUCTION_VERSION,
                                 'audit_version': AUDIT_VERSION,
                                 'evidence': fingerprint(scoped_evidence),
                                 'jury': fingerprint(jury)}
        if (previous and previous.get('build_scope') == scope
                and completed_construction(previous, construction_rounds)
                and all(
                    (t['checker'] in registry.checkers
                     and (t['checker'] not in registry.recipes
                          or registry.recipes[t['checker']] == t.get('execution_recipe')))
                    or (t['checker'] not in registry.checkers and t.get('execution_recipe'))
                    for t in previous['tests'])):
            pack = deepcopy(previous)
            for test in pack['tests']:
                if test['checker'] not in registry.checkers:
                    registry.bind_recipe(test['checker'], test['execution_recipe'])
            record('builder_reuse', {'reused_criteria': [c['id'] for c in jury['criteria']],
                                    'rebuilt_criteria': [], 'reuse_fraction': 1})
            record('pack', pack)
            return pack
    old = {c["id"]: c for c in (previous or {}).get("criteria", [])}
    roles = knowledge_roles(context)

    def meaning(c):
        return {k: v for k, v in c.items() if k not in {"anchors", "evidence_refs"}}

    def safe_binding(test):
        try:
            validate_current_product_refs({k: test.get(k) for k in ("arguments", "probes", "execution_recipe")})
            return test.get("binding_status") == "callable"
        except ValueError:
            return False

    reusable = {c["id"] for c in jury["criteria"] if c["id"] in old
                and meaning(c) == meaning(old[c["id"]])
                and previous.get("build_scope") == scope
                and any(t["criterion_id"] == c["id"] for t in previous["tests"])
                and all(safe_binding(t) for t in previous["tests"]
                        if t["criterion_id"] == c["id"])} if previous and not refined else set()
    record("builder_reuse", {"reused_criteria": sorted(reusable),
        "rebuilt_criteria": [c["id"] for c in jury["criteria"] if c["id"] not in reusable],
        "reuse_fraction": len(reusable) / len(jury["criteria"]) if jury["criteria"] else 0})
    if reusable:
        changed = [c for c in jury["criteria"] if c["id"] not in reusable]
        fresh = await build(context, evidence, {**jury, "criteria": changed}, registry, call,
                            author=author, repair_findings=repair_findings,
                            record=lambda name, row: record("changed_" + name, row)) if changed else None
        pack = deepcopy(fresh or previous)
        fresh_criteria = {c["id"]: c for c in (fresh or {}).get("criteria", [])}
        pack["criteria"] = [{**deepcopy(c), "anchors": deepcopy(old[c["id"]]["anchors"])}
                            if c["id"] in reusable else fresh_criteria[c["id"]] for c in jury["criteria"]]
        pack["tests"] = [deepcopy(t) for t in previous["tests"] if t["criterion_id"] in reusable]
        ids = {t["test_id"] for t in pack["tests"]}
        for test in (fresh or {}).get("tests", []):
            test = deepcopy(test)
            while test["test_id"] in ids:
                test["test_id"] = test["criterion_id"] + ":" + test["test_id"]
            ids.add(test["test_id"])
            pack["tests"].append(test)
        current = {c["id"]: c for c in pack["criteria"]}
        for test in pack["tests"]:
            test["rubric"] = deepcopy(current[test["criterion_id"]])
            bind_knowledge(test, roles)
        if roles:
            pack["knowledge_roles"] = roles
        else:
            pack.pop("knowledge_roles", None)
        # A removed criterion or updated provenance/coverage also changes the published bundle.
        pack["pack_id"] = uuid4().hex
        for target, source in (("independent_expectations", "independent_expectations"),
                               ("proposal_expectations", "proposal_expectations"),
                               ("expectation_coverage", "coverage")):
            pack[target] = deepcopy(jury.get(source, []))
        pack["status"] = "callable" if all(t["binding_status"] == "callable" for t in pack["tests"]) else "needs_implementation"
        pack["build_scope"] = scope
        pack["measurement_profile"] = measurement_profile(pack, registry)
        record("pack", pack)
        return pack
    if author is None and registry.tools is not None:
        from reward_engine.core.outcome.measurement_tools import author_checker

        async def author(test, registry):
            await author_checker(test, registry, context, evidence, call)
    # Compile the accepted definition, not every duplicate draft from its history.
    # The full proposal/coverage ledger is still copied into the resulting Pack.
    # Compile a current-product template, not one set of tests per historical answer.
    compile_context, compile_evidence = requirement_context(context, evidence)
    if refined or bounded:
        compile_evidence.update({r: deepcopy(v) for r, v in evidence.items()
                                 if r.startswith('reference:task_review:')})
        compile_context['available_evidence_refs'] = list(compile_evidence)
    # Historical product handles are provenance, not executable template inputs.
    # Keep the full accepted criteria in the published Pack while compiling only
    # references actually present in the product-blind requirement view.
    definition = {"criteria": deepcopy(jury["criteria"])}
    omitted = {}
    for criterion in definition["criteria"]:
        refs = criterion.get("evidence_refs", [])
        omitted[criterion["id"]] = [ref for ref in refs if ref not in compile_evidence]
        if "evidence_refs" in criterion:
            criterion["evidence_refs"] = [ref for ref in refs if ref in compile_evidence]
    compile_context["products"] = [p for p in compile_context["products"] if p["user_updates"]]
    if any(omitted.values()):
        record("builder_provenance_scope", {"excluded_from_executable_context": omitted,
               "retained_in_published_criteria": True})
    compile_context["current_product_contract"] = {
        "product_id": "product_1", "artifact_refs": sorted({"product_1:artifact:" + field
        for p in context["products"] for field in p.get("artifact", {})}),
        "environment_ref": "product_1:environment",
        "artifact_types": {field: sorted({type(p['artifact'][field]).__name__
            for p in context['products'] if field in p.get('artifact', {})})
            for field in sorted({f for p in context['products'] for f in p.get('artifact', {})})},
        "environment_keys": sorted({k for p in context['products']
            if isinstance(evidence.get(p.get('environment_ref')), dict)
            for k in evidence[p['environment_ref']]}),
        "note": "Types/keys are observed possibilities, not a future guarantee. Missing fields are unknown. "
                "Runtime binds the actual future product; these are handles, not observations."}
    roles = knowledge_roles(compile_context)
    knowledge_instruction = (
        " Task Review priors in skills carry a typed use (see knowledge_roles). Every check is still grounded in "
        "its Jury criterion. A verification_rule/procedure prior may define HOW a check measures; cite it in "
        "knowledge_refs. A reference_implementation/illustrative_example prior is a reference for anchors or "
        "probes, never the standard: do not demand its specific numbers, tools or wording. A limitation_note "
        "prior is a related caution: same-source notes are automatically attached as context, not mandatory "
        "preconditions. Determine whether each note actually limits the chosen method before using it. "
        "Unrelated notes add no requirement. Unclassified refs need interpretation and cannot automatically "
        "define a measurement. conditions_unstated means applicability is not established, not that the "
        "product failed; an independent valid measurement is still allowed."
        if roles else "")
    async def request_builder(repair):
        from reward_engine.core.outcome.clauses import INSTRUCTION as CLAUSE_INSTRUCTION
        from reward_engine.core.outcome.task_inputs import (
            instruction as task_input_instruction,
        )
        return await ask(
            call,
            "outcome_builder",
            compile_context,
            compile_evidence,
            "Task is the primary goal; knowledge is supplemental and must not add obligations or prescribe "
            "one reference solution. Keep EVERY supplied criterion's identity, meaning and kind. "
            "Define rubric anchors for fail/partial/full. Partial fulfillment is distinct from an absent "
            "or unusable outcome; preserve which portions are satisfied and the severity of the gap. "
            "A binary fail from one checker is not an overall outcome rejection. Do not add approval "
            "prerequisites or required presentation details absent from the frozen criterion. "
            "Preserve its stated priority, scope and valid alternatives. "
            "and one reusable check per criterion by default; split only for genuinely distinct execution methods. "
            "Never multiply tests by the number of historical products. These anchors describe evidence, not an absolute grade. "
            "Use available checkers when adequate. Checker keys are open: if a capability is absent, "
            "Prefer an actual parser/calculation/runtime/source lookup over prose judgment when it can "
            "answer the question. Do not claim that evidence.review executed code or verified hardware. "
            "Check the major outcome coverage, not only historical defects. Unmeasurable expectations "
            "remain visible; binding every criterion does not prove the definition is complete. "
            "The full proposal/coverage ledger is retained by code in the Pack, not repeated here. "
            "Compiling accepted criteria does not resolve outstanding proposal coverage. "
            "describe the new implementation required, not imaginary success. "
            "For evidence.review you may supply up to three probes using measurement_tools. "
            "These execute before the semantic judgment and provide real numeric/source observations. "
            "code.run accepts a bounded numerical Python subset, not arbitrary Python or host execution. "
            "The missing-checker author can generate a recipe once; unavailable GPU capability remains unavailable. "
            "Checks run on future current products: do not hardcode historical candidate identifiers/answers. "
            "A single product being measured is always exposed as product_1, regardless of its pool position. "
            "User-specific standards must be conditional on the current user's actual requirements, "
            "not reuse historical answers as obligations for a fresh Actor. "
            "Never pin product_1:public:N in arguments/probes; evaluate current user_updates dynamically. "
            "Use evidence.review for semantic comparisons to sources, json.parse ONLY for exact JSON syntax. "
            "No official scores or full-pool ranking; all-pass samples still need necessary Task checks."
            + knowledge_instruction + CLAUSE_INSTRUCTION + task_input_instruction(context),
            {
                "rubrics": [
                    {
                        "criterion_id": "id",
                        "anchors": {
                            "fail": "what fails",
                            "partial": "partial completion",
                            "full": "what satisfies",
                        },
                    }
                ],
                "tests": [
                    {
                        "test_id": "open unique key",
                        "criterion_id": "id",
                        "checker": "key",
                        "question": "specific measurement",
                        "arguments": {},
                        "probes": [{"tool": "optional measurement tool", "arguments": {}}],
                        "implementation_needed": "if absent",
                        **({"knowledge_refs": ["optional skill:task_review refs this check's measurement relies on"]}
                           if roles else {}),
                    }
                ],
            },
            extra={"jury": definition, "required_criterion_ids": [c['id'] for c in definition['criteria']],
                   "checker_catalog": checker_catalog,
                   **({'executed_counterexample_repairs': deepcopy(list(repair_findings))} if repair_findings else {}),
                   "measurement_tools": registry.tools.catalog() if registry.tools else {},
                   **({"knowledge_roles": roles} if roles else {}),
                   **({"repair": repair} if repair else {})},
            **({'validate': validate_draft, 'max_contract_retries': 1,
                'max_evidence_rounds': 2, 'close_evidence': True} if refined or bounded else {'max_contract_retries': 1}),
        )
    def validate_draft(value):
        if isinstance(value, dict) and 'criteria' in value and value['criteria'] != jury['criteria']:
            raise ValueError('Builder cannot redefine frozen Jury criteria; report upstream concerns instead')
        compile_contract(value, jury, registry, roles=roles)
        return value

    repair = None
    for attempt in range(1 if refined or bounded else 2):
        result = await request_builder(repair)
        record(f"builder_attempt_{attempt + 1}", {"output": result})
        try:
            bound_result = deepcopy(result)
            if isinstance(bound_result, dict) and isinstance(bound_result.get("tests"), list):
                for index, test in enumerate(bound_result["tests"]):
                    if not isinstance(test, dict):
                        continue
                    bindings = bind_current_artifacts(test, compile_context["current_product_contract"]["artifact_refs"])
                    if bindings:
                        record("builder_artifact_bindings/" + str(index),
                               {"test_id": test.get("test_id"), "bindings": bindings,
                                "reason": "Executable template binds the declared artifact of the current product"})
            criteria, tests = compile_contract(bound_result, jury, registry, roles=roles)
            break
        except (ValueError, TypeError, KeyError) as error:
            record(f"builder_error_{attempt + 1}", {"error": str(error)})
            if attempt:
                raise
            repair = {"original_output": result, "contract_error": str(error),
                      "instruction": "Correct the contract once; preserve all supplied criteria and Task scope."}
    construction = None
    if refined:
        result, construction = await refine_draft(
            result, construction_rounds, compile_context, compile_evidence, jury, call,
            reviewers=reviewers, policy=policy, validate=validate_draft, record=record,
            checker_catalog=checker_catalog,
            measurement_tools=registry.tools.catalog() if registry.tools else {},
            knowledge_roles=roles,
        )
        criteria, tests = compile_contract(result, jury, registry, roles=roles)
    if refined or bounded:
        scope_generated_checkers(tests, checker_catalog, scope)
    if isinstance(result.get("rubrics"), dict):
        record("builder_format_notes", {"rubrics": "ID-keyed anchor table converted to rows"})
    attempted = set()
    for test in tests:
        if (
            test["checker"] not in registry.checkers
            and author is not None
            and test["checker"] not in attempted
        ):
            attempted.add(test["checker"])
            try:
                await author(
                    deepcopy(test), registry
                )  # One bounded implementation attempt, not eval(model_text).
            except Exception as error:
                test["author_error"] = type(error).__name__
                test["author_diagnostic"] = str(error)[:1000]
        test["binding_status"] = (
            "callable" if test["checker"] in registry.checkers else "needs_implementation"
        )
        if test["checker"] in registry.recipes:
            test["execution_recipe"] = deepcopy(registry.recipes[test["checker"]])
        if test.get("execution_recipe"):
            catalog = registry.tools.catalog() if registry.tools else {}
            test["required_tools"] = {p.get("tool", ""): bool(catalog.get(p.get("tool"), {}).get("available"))
                                      for p in test["execution_recipe"]["probes"]}
            if not all(test["required_tools"].values()):
                test["binding_status"] = "needs_implementation"
    for test in tests:
        bind_knowledge(test, roles)  # Include refs introduced by the actual authored execution recipe.
    pack = {
        "schema": "outcome-pack-v1",
        "pack_id": uuid4().hex,
        "task_id": context["task"]["task_id"],
        "task_prompt": context["task"]["prompt"],
        "criteria": criteria,
        "tests": tests,
        "status": "callable"
        if all(t["binding_status"] == "callable" for t in tests)
        else "needs_implementation",
        "validated_improvement": False,
        "build_scope": scope,
        "measurement_sources": {r: deepcopy(v) for r, v in evidence.items()
                                if r.startswith(('task_input:', 'reference:task_review:'))},
        "measurement_priors": [deepcopy(s) for s in context.get("skills", [])
                               if s["ref"].startswith("skill:task_review:")],
        **({"knowledge_roles": roles} if roles else {}),
        "independent_expectations": deepcopy(jury.get("independent_expectations", [])),
        "proposal_expectations": deepcopy(jury.get("proposal_expectations", [])),
        "expectation_coverage": deepcopy(jury.get("coverage", [])),
        **({'construction': construction} if construction else {}),
    }
    pack["measurement_profile"] = measurement_profile(pack, registry)
    if construction:
        construction['published_fingerprint'] = fingerprint({'criteria': criteria, 'tests': tests})
    record("pack", pack)
    return pack


def compile_contract(result, jury, registry, *, roles=None):
    """Pure validation/normalization of the ENTIRE bundle before author/tool effects."""
    if not isinstance(result, dict):
        raise ValueError("Builder reply must be an object")
    known_priors = {r for key in ("method_refs", "reference_refs", "condition_refs", "unclassified_refs")
                    for r in (roles or {}).get(key, [])}
    originals = {c["id"]: c for c in jury["criteria"]}
    rubrics = result.get("rubrics", [])
    if isinstance(rubrics, dict):
        # A keyed anchor table is the same content, not a missing rubric.
        rubrics = [{"criterion_id": key, "anchors": deepcopy(value)}
                   for key, value in rubrics.items()]
    if not isinstance(rubrics, list) or any(not isinstance(r, dict) for r in rubrics):
        raise ValueError("Rubrics must be rows or an ID-keyed anchor table")
    if len(rubrics) != len(originals) or {r.get("criterion_id") for r in rubrics} != set(originals):
        received = [r.get('criterion_id') for r in rubrics]
        raise ValueError("Builder must cover every Outcome criterion exactly once: expected="
                         + repr(sorted(originals)) + "; received=" + repr(received))
    criteria = []
    for row in rubrics:
        if not isinstance(row.get("anchors"), dict) or any(
            not isinstance(row.get("anchors", {}).get(k), str) or not row["anchors"][k].strip()
            for k in ("fail", "partial", "full")
        ):
            raise ValueError("Rubric needs explicit anchors")
        criteria.append({**deepcopy(originals[row["criterion_id"]]), "anchors": row["anchors"]})
    lookup = {c["id"]: c for c in criteria}
    tests, ids = [], set()
    if not isinstance(result.get("tests"), list):
        raise ValueError("Tests must be a list")
    for raw in result["tests"]:
        if not isinstance(raw, dict):
            raise ValueError("Each test must be an object")
        test = deepcopy(raw)
        if (
            not isinstance(test.get("test_id"), str)
            or not test["test_id"].strip()
            or test["test_id"] in ids
            or test.get("criterion_id") not in lookup
            or not isinstance(test.get("checker"), str)
            or not test["checker"].strip()
            or not isinstance(test.get("question"), str)
            or not test["question"].strip()
            or not isinstance(test.get("arguments", {}), dict)
        ):
            raise ValueError("Invalid executable test contract")
        ids.add(test["test_id"])
        if re.search(r"\bproduct_(?:[2-9]|[1-9][0-9]+)\b", str(test)):
            raise ValueError("Harness template cannot bind a historical product; use current product_1")
        validate_current_product_refs({k: test.get(k) for k in ("arguments", "probes", "execution_recipe")})
        if "execution_recipe" in test:
            raise ValueError("Builder must declare probes; execution_recipe is installed by code")
        if "knowledge" in test:
            raise ValueError("Builder must not author knowledge bindings; code derives them from knowledge_refs")
        refs = test.get("knowledge_refs", [])
        if not isinstance(refs, list) or any(not isinstance(r, str) or r not in known_priors for r in refs):
            raise ValueError(f"{test['test_id']}: knowledge_refs must name supplied skill:task_review priors only; "
                             f"received {refs!r}. Use refs from knowledge_roles, not an unresolved-needs handle.")
        test["rubric"] = lookup[test["criterion_id"]]
        bind_knowledge(test, roles)
        if registry.tools and test["checker"] in registry.tools.catalog():
            # A real measurement tool is not a quality verdict. Execute it then
            # let the existing semantic checker compare its receipt to the work.
            tool = test["checker"]
            test["checker"] = "evidence.review"
            test["probes"] = [{"tool": tool, "arguments": deepcopy(test.get("arguments", {}))}]
            test["measurement_tool"] = tool
        if test.get("probes"):
            probes = test["probes"]
            if not isinstance(probes, list) or not 1 <= len(probes) <= 3 or any(
                not isinstance(p, dict) or not isinstance(p.get("tool"), str) or not p["tool"].strip()
                or not isinstance(p.get("arguments", {}), dict) for p in probes
            ):
                raise ValueError("Need a bounded measurement recipe")
            if test["checker"] != "evidence.review":
                raise ValueError("Inline probes belong to the tool-assisted semantic checker")
            for probe in probes:
                validate_probe_arguments(probe)
            test["execution_recipe"] = {"probes": deepcopy(probes)}
        tests.append(test)
    if {t["criterion_id"] for t in tests} != set(originals):
        raise ValueError("Harness must cover every criterion, including common requirements")
    return criteria, tests


def measurement_profile(pack, registry):
    """Descriptive binding mix, not a quality or verifiability score."""
    counts = dict.fromkeys(("deterministic", "tool_assisted", "semantic", "unavailable"), 0)
    for test in pack["tests"]:
        test['measurement_kind'] = registry.kinds.get(test['checker'], 'unavailable')
        kind = ("unavailable" if test.get("binding_status") != "callable" else
                "tool_assisted" if test.get("execution_recipe") else
                "deterministic" if registry.kinds.get(test["checker"]) == "deterministic" else "semantic")
        counts[kind] += 1
    return {"counts": counts, "total": len(pack["tests"]), "executed": False}
