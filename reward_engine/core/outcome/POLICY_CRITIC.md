# Runtime Policy Critic

Critic works toward the full desired outcome: define the target, identify current gaps, prioritize actions, observe execution, and reassess. Shared `improvement.py` fields include `outcome_goal`, `full_score_gaps`, and `improvements`. Actions carry stable IDs, linked gaps, concrete changes, expected contribution, verification, and evidence. Up to three actions are prioritized while the remaining gaps are preserved.

`global_assessment` describes the current stage target and missing result. `local_guidance` describes next actions or questions and conditions for continuing, changing direction, or seeking help. Plans remain proposals, not observed facts. Critic may choose `observe` when no useful intervention exists.

## Inputs and execution

Construct `PolicyCritic(task, pack, registry, memory=..., config=...)`, then call `await critic(candidate, turn=3, intent=...)`. It uses the existing Outcome Pack and check registry rather than a second set of criteria.

Inputs include the public task, real user supplements, recent actor updates, actual tool observations, relevant permitted memory, and optional exploration intent. Hidden answers, native grader results, and internal reasoning are excluded. Arm routing may omit historical memory; see [arm protocols](ARMS.md).

Custom adapters must record actor statements as `actor_updates`, not `artifact.final_response`. Tool requests and actual results enter `source_documents` with turn IDs. Environment metadata describes the observation scope, not an invented workspace scan. An absent final answer is normal during execution; checks that need it return unknown.

The default inline window contains the latest six tool records within 16 KiB. Larger records retain previews and full evidence handles; originals remain on disk. Unresolved pack questions retain applicability, sources, and uncertainty through `question_context`; they are leads to verify, not new requirements or established defects.

1. Assess progress and select zero or a few existing tests.
2. Execute selected checks only when needed (default maximum three).
3. If measurements ran, synthesize their actual results into `observe`, `suggest`, or `clarify` feedback for the same actor.

There is no runtime A/B resampling or separate process-reward layer. Earlier passes do not substitute for final checks. Internal advice does not expand user authorization.

## State, limits, and degraded results

Each instance retains the last two checkpoint summaries as `prior_feedback` and saves `turn_N_progress`. This is local progress context, not another cross-trial memory system. Generated advice does not prove injection, adoption, or completion; those require receipts and later observations.

One overall deadline covers assessment, bounded contract recovery, measurement, and synthesis. Measurement receives at most half the remaining budget to leave time for synthesis. Failures before completed measurement return unavailable. If measurement completed but synthesis fails or times out, the result is degraded and carries the actual check summary instead of stale initial advice or fabricated success.

A `runtime_check` defect reference requires a valid primary failure or a valid independent local `inquiry_result.fail`. A primary pass/unknown cannot erase a valid local failure. Pure unknown or execution error does not prove a product defect.

Requests default to a 160,000-byte limit, configurable with `policy_critic.max_request_bytes`. Oversized requests record actual UTF-8 size, limit, stage, and field sizes and are rejected before a model call. Repeated coverage details can use evidence indexes; actual measurement facts and complete retained receipts remain intact.

## Scheduling and delivery

```json
{"policy_critic":{"enabled":true,"every_k_turns":3,"max_tests":3,"timeout_s":120,"top_k":3}}
```

This explicitly selects frequency three. `hypothesis_feedback_v1` defaults to one and uses Critic only on D; only B receives Hypothesis. Legacy routing defaults to Critic disabled and frequency three, with explicitly enabled Critic on legacy B/D. The model callable comes from the explicit `call` argument or `registry.review_call`; the integrating application supplies its model and credentials.

Count completed root actor model turns, not tools, Critic calls, failed model calls, or duplicate callbacks. At an eligible k-th turn with tool actions, intervene after the tools complete and before the next model request. A final tool-free response does not trigger periodic review. An integration implementing D must also review the actual final artifact with the same Critic and permit at most one continuation and recheck. D does not additionally run C's DeliveryReview.

The current native OpenCode campaigns do not schedule these periodic callbacks. Custom adapters must count eligible turns, call the Critic with actual observations, and inject returned advice into the same actor session. Configuration alone does not install this integration. Keep internal feedback separate from public user messages and preserve receipts showing whether advice was actually injected.

The core exposes terminal review and recheck methods, including `finish_runtime`, for integrations implementing D's delivery contract. Callers must enforce continuation budgets and report missing or unavailable reviews accurately.
