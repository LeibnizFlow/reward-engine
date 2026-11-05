"""Open checker registry. Keys are extensible; execution is real registered code."""

# Preserve the established module import surface.
import json as json
from copy import deepcopy
from time import monotonic

from reward_engine.core.batch import BatchPolicy as BatchPolicy
from reward_engine.core.batch import collect_batch as collect_batch
from reward_engine.core.outcome.construction import fingerprint as fingerprint
from reward_engine.core.outcome.context import EvidenceReadLimit as EvidenceReadLimit
from reward_engine.core.outcome.context import (
    ResponseContractError as ResponseContractError,
)
from reward_engine.core.outcome.context import ask as ask
from reward_engine.core.outcome.context import (
    measurement_prior_view as measurement_prior_view,
)
from reward_engine.core.outcome.context import (
    measurement_test_view,
    validate_current_product_refs,
)
from reward_engine.core.outcome.context import require_refs as require_refs
from reward_engine.core.outcome.context import response_payload as response_payload
from reward_engine.core.outcome.context import used_previews as used_previews
from reward_engine.core.outcome.coverage import complete as complete
from reward_engine.core.outcome.coverage import expectations as expectations
from reward_engine.core.outcome.coverage import report as report
from reward_engine.core.outcome.harness_checkers import defaults as defaults
from reward_engine.core.outcome.harness_claims import (
    claim_contract_issues as claim_contract_issues,
)
from reward_engine.core.outcome.harness_claims import claim_locations as claim_locations
from reward_engine.core.outcome.harness_claims import (
    compose_field_correction as compose_field_correction,
)
from reward_engine.core.outcome.harness_claims import (
    compose_response_unwrap as compose_response_unwrap,
)
from reward_engine.core.outcome.harness_claims import (
    field_correction_targets as field_correction_targets,
)
from reward_engine.core.outcome.harness_claims import (
    has_product_evidence as has_product_evidence,
)
from reward_engine.core.outcome.harness_claims import (
    preserve_claim_assessment as preserve_claim_assessment,
)
from reward_engine.core.outcome.harness_claims import (
    unresolved_measurements as unresolved_measurements,
)
from reward_engine.core.outcome.harness_claims import validate_claims as validate_claims
from reward_engine.core.outcome.harness_claims import validate_result as validate_result
from reward_engine.core.outcome.harness_measurement import (
    _bound_measurement_inputs as _bound_measurement_inputs,
)
from reward_engine.core.outcome.harness_measurement import measure as measure
from reward_engine.core.outcome.harness_measurement import (
    measurement_context as measurement_context,
)
from reward_engine.core.outcome.harness_measurement import (
    measurement_input_fingerprint as measurement_input_fingerprint,
)
from reward_engine.core.outcome.harness_measurement import (
    summarize_results as summarize_results,
)

MEASUREMENT_INPUT_SCHEMA = "outcome-measurement-input/v1"


def unknown(reason, *, error=False):
    return {
        "status": "unknown",
        "reason": reason,
        "evidence_refs": [],
        "error": error,
        "gap": "",
    }


# Provider/SDK transport, timeout and budget failures are physical: the check
# never produced a semantic verdict. They must be retried or retained as
# unknown, never treated as evidence of a defective standard or checker.
INFRASTRUCTURE_FAILURES = frozenset(
    {
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "ReadError",
        "ConnectError",
        "RemoteProtocolError",
        "TransportError",
        "ProviderStreamError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "HTTPStatusError",
        "TimeoutError",
        "FrameworkError",
        "BudgetExhausted",
        "RequestBudgetExceeded",
    }
)


def failure_layer(error):
    """Separate a physical/infrastructure failure from an evaluator defect."""
    return (
        "infrastructure"
        if type(error).__name__ in INFRASTRUCTURE_FAILURES
        else "evaluation_execution"
    )


async def smoke(pack, registry):
    """Zero-model infrastructure probe; deliberately NOT a product-quality verdict."""
    from reward_engine.core.outcome.context import build_context

    bindings = {t["test_id"]: t.get("binding_status", "unknown") for t in pack["tests"]}
    receipt = None
    if "json.parse" in registry.checkers:
        context, evidence = build_context(
            {"task_id": "infra-smoke", "prompt": "Parse fixture"},
            [{"artifact": {"final_response": "{}"}}],
        )
        receipt = await registry.run(
            {
                "test_id": "infra-parser",
                "criterion_id": "infra",
                "checker": "json.parse",
                "arguments": {"ref": "product_1:artifact:final_response"},
            },
            context,
            evidence,
        )
    return {
        "bindings": bindings,
        "receipt": receipt,
        "status": "receipt_path_checked"
        if receipt and receipt["status"] == "pass"
        else "binding_only",
        "boundary": "Synthetic parser receipt only; per-check semantic quality and GPU execution NOT validated.",
    }


class Registry:
    def __init__(self):
        self.checkers = {}
        self.kinds = {}
        self.dependencies = {}
        self.recipes = {}
        self.tools = None
        self.review_call = None

    def bind_recipe(self, key, recipe):
        validate_current_product_refs(recipe)
        if key in self.checkers and self.recipes.get(key) != recipe:
            raise ValueError("Recipe cannot replace an existing checker")
        if key not in self.checkers:
            self.register(
                key,
                self.semantic,
                "Tool-assisted generated measurement recipe",
                kind="model_assisted",
            )
        self.recipes[key] = deepcopy(recipe)

    def register(
        self, key, run, description, *, kind="deterministic", dependencies=None
    ):
        if (
            not isinstance(key, str)
            or not key.strip()
            or not callable(run)
            or not description
        ):
            raise ValueError(
                "Checker requires key, executable async function and description"
            )
        if key in self.checkers:
            raise ValueError("Do not silently replace an existing checker")
        if kind not in {"deterministic", "model_assisted"}:
            raise ValueError("Declare whether the check depends on model judgment")
        if dependencies is not None and (
            kind != "deterministic" or not callable(dependencies)
        ):
            raise ValueError(
                "Reuse dependencies must be trusted deterministic checker code"
            )
        self.checkers[key] = (run, description)
        self.kinds[key] = kind
        self.dependencies[key] = dependencies

    def inputs(self, test, context, evidence):
        dependency = self.dependencies.get(test["checker"])
        if dependency is None:
            return {"context": context, "evidence": evidence}
        return {
            "task": context["task"],
            "users": [p["user_updates"] for p in context["products"]],
            "input": dependency(test, context, evidence),
        }

    def catalog(self):
        return {key: description for key, (_, description) in self.checkers.items()}

    def tool_catalog(self, evidence, *, authored=False):
        if self.tools is None:
            return {}
        if authored and hasattr(self.tools, "catalog_for"):
            return self.tools.catalog_for(evidence)
        return self.tools.catalog()

    async def run(self, test, context, evidence):
        from reward_engine.core.outcome.authored_json import enabled

        completed = (
            getattr(self.review_call, "completed_operations", None)
            if enabled(self.review_call)
            else None
        )
        binding = {
            "test": test,
            "context": context,
            "evidence": evidence,
            "tools": self.tool_catalog(evidence, authored=enabled(self.review_call)),
        }
        from reward_engine.core.outcome.measurement_requests import (
            enabled as recovery_enabled,
        )
        from reward_engine.core.outcome.resume_inputs import validate_measurement

        def revalidate(row):
            return validate_measurement(
                row,
                test,
                evidence,
                policy=self.tools.role_contract_policy,
                tools=self.tools,
            )

        if completed is not None:
            cached = completed.load(
                "measurement",
                binding,
                **({"validate": revalidate} if recovery_enabled(self.tools) else {}),
            )
            if cached is not None:
                return revalidate(cached) if recovery_enabled(self.tools) else cached
        result = await self._run(test, context, evidence)
        if (
            completed is not None
            and not result.get("error")
            and not result.get("unresolved_measurement_refs")
            and not result.get("unresolved_evidence_refs")
        ):
            completed.save("measurement", binding, result)
        return result

    async def _run(self, test, context, evidence):
        started = monotonic()
        test = measurement_test_view(test)
        key = test["checker"]
        inquiry = deepcopy(
            context.get("measurement_inquiries", {}).get(test["test_id"])
        )
        if key not in self.checkers:
            return {
                "test_id": test["test_id"],
                "criterion_id": test["criterion_id"],
                **unknown(f"Checker {key} needs implementation", error=True),
                **({"measurement_inquiry": inquiry} if inquiry else {}),
            }
        try:
            validate_current_product_refs(
                {k: test.get(k) for k in ("arguments", "probes", "execution_recipe")}
            )
            observed = evidence
            evidence, context = deepcopy(evidence), deepcopy(context)
            context.pop("measurement_inquiries", None)
            if inquiry:
                context["measurement_inquiry"] = inquiry
            # Supplied standards may be cited, but never count as observed product facts.
            rubric = test.get("rubric", {})
            for ref, value in ((test.get("test_id"), test), (rubric.get("id"), rubric)):
                if ref:
                    if ref in evidence:
                        raise ValueError(
                            "Check/criterion ID collides with an evidence handle"
                        )
                    evidence[ref] = deepcopy(value)
            context["available_evidence_refs"] = list(evidence)
            result = await self.checkers[key][0](
                deepcopy(test), deepcopy(context), deepcopy(evidence)
            )
            # Only trusted registered checker code may introduce measurement receipts.
            receipts = result.pop("_tool_evidence", {})
            evidence.update(receipts)
            result = validate_result(result, evidence)
            if result["status"] != "unknown" and not has_product_evidence(
                result["evidence_refs"], observed, receipts
            ):
                raise ValueError(
                    "Known check results need observed product evidence, not just the Task"
                )
        except Exception as error:
            result = unknown(f"Checker failed: {type(error).__name__}", error=True)
            result["diagnostic"] = str(error)[:400]
            result["failure_layer"] = failure_layer(error)
        return {
            **result,
            "test_id": test["test_id"],
            "criterion_id": test["criterion_id"],
            "checker": key,
            "measurement_kind": self.kinds[key],
            "elapsed_s": monotonic() - started,
            **({"measurement_inquiry": inquiry} if inquiry else {}),
        }
