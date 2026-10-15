const SPEAK = "[REWARD_GUIDED_POLICY:SPEAK]";
const CLARIFY = "[REWARD_GUIDED_POLICY:CLARIFY]";
const DEGRADED = "[REWARD_GUIDED_POLICY:DEGRADED]";
const CONSTRAINT_BLOCK = "[OPERATIONAL_CONSTRAINT:BLOCK]";
const CONSTRAINT_WARN = "[OPERATIONAL_CONSTRAINT:WARN]";
const CONSTRAINT_UNKNOWN = "[OPERATIONAL_CONSTRAINT:UNKNOWN]";

const REQUEST_SCHEMA = "reward_builder_v4.opencode_action.v1";
const DECISION_SCHEMA = "reward_builder_v4.opencode_decision.v1";
const DEFAULT_TIMEOUT_MS = 90_000;
const MAX_TIMEOUT_MS = 300_000;
const MAX_PAYLOAD_BYTES = 8_000;
const MAX_VALUE_CHARS = 480;
const MAX_RESULT_CHARS = 1_600;
const MAX_FEEDBACK_CHARS = 4_000;
const MAX_TRANSITION_TOKEN_CHARS = 1_024;
const MAX_KEYS = 16;
const MAX_ARRAY_ITEMS = 8;
const MAX_DEPTH = 3;
const VALUE_BUDGET = 4_000;

const SENSITIVE_KEY =
  /authorization|api.?key|token|secret|password|credential|cookie|headers?|environment|(?:^|[_-])auth(?:entication)?(?:$|[_-])|^env$/i;
const SECRET_ASSIGNMENT =
  /\b(?:[A-Z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)|authorization)\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)/gi;
const BEARER_TOKEN = /\bbearer\s+[A-Za-z0-9._~+\/-]+=*/gi;
const PROVIDER_KEY = /\b(?:sk|ak)-[A-Za-z0-9_-]{8,}\b/gi;
const QUERY_SECRET =
  /([?&](?:api[_-]?key|token|secret|password|credential)=)[^&#\s]+/gi;

class GatewayFailure extends Error {
  constructor(kind) {
    super(kind);
    this.name = "GatewayFailure";
    this.kind = kind;
  }
}

function endpoint() {
  const raw = process.env.REWARD_BUILDER_V4_CRITIC_URL;
  if (!raw) throw new GatewayFailure("critic_url_missing");
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new GatewayFailure("critic_url_invalid");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username !== "" ||
    url.password !== "" ||
    !["127.0.0.1", "localhost", "[::1]", "::1"].includes(url.hostname)
  ) {
    throw new GatewayFailure("critic_url_not_trusted_loopback");
  }
  return url.toString();
}

function timeoutMs() {
  const value = Number(
    process.env.REWARD_BUILDER_V4_TIMEOUT_MS ?? String(DEFAULT_TIMEOUT_MS),
  );
  if (!Number.isFinite(value) || value <= 0 || value > MAX_TIMEOUT_MS) {
    throw new GatewayFailure("critic_timeout_invalid");
  }
  return Math.floor(value);
}

function redactText(value, limit = MAX_VALUE_CHARS) {
  const text = String(value)
    .replace(BEARER_TOKEN, "[REDACTED]")
    .replace(PROVIDER_KEY, "[REDACTED]")
    .replace(SECRET_ASSIGNMENT, "[REDACTED]")
    .replace(QUERY_SECRET, "$1[REDACTED]")
    .trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 14))}...[truncated]`;
}

function safeIdentifier(value, fallback) {
  if (typeof value !== "string" || value.trim() === "") return fallback;
  return redactText(value, 160);
}

function boundedValue(value, state, depth = 0) {
  if (state.remaining <= 0) return "[truncated]";
  if (value === null || typeof value === "boolean" || typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const limit = Math.min(state.extended ? state.limit : MAX_VALUE_CHARS, state.remaining);
    const result = redactText(value, limit);
    state.remaining -= result.length;
    return result;
  }
  if (depth >= (state.extended ? 6 : MAX_DEPTH)) return `[${Array.isArray(value) ? "array" : "object"}]`;
  if (Array.isArray(value)) {
    return value
      .slice(0, state.extended ? 128 : MAX_ARRAY_ITEMS)
      .map((item) => boundedValue(item, state, depth + 1));
  }
  if (typeof value === "object") {
    const result = {};
    const keys = Object.keys(value)
      .filter((key) => !SENSITIVE_KEY.test(key))
      .sort()
      .slice(0, state.extended ? 32 : MAX_KEYS);
    for (const key of keys) {
      const safeKey = redactText(key, 120);
      result[safeKey] = boundedValue(value[key], state, depth + 1);
      if (state.remaining <= 0) break;
    }
    return result;
  }
  return `[${typeof value}]`;
}

/** A bounded, secret-free action projection; never the Actor's full prompt. */
export function boundedActionProjection(
  phase,
  input,
  args,
  toolResult,
  transitionToken,
) {
  // Runner-configured capacity, never an Actor-supplied option. Office/file
  // edits must not silently use a 480-character preview as the whole action.
  const configured = Number(process.env.REWARD_BUILDER_V4_ACTION_BUDGET_CHARS ?? VALUE_BUDGET);
  if (!Number.isInteger(configured) || configured < VALUE_BUDGET || configured > 48_000) {
    throw new GatewayFailure("invalid_action_budget");
  }
  const state = { remaining: configured, limit: configured, extended: configured > VALUE_BUDGET };
  const projection = {
    schema_version: REQUEST_SCHEMA,
    phase,
    session_id: safeIdentifier(input?.sessionID, "unknown-session"),
    action_id: safeIdentifier(input?.callID, "unknown-action"),
    tool_name: safeIdentifier(input?.tool, "unknown-tool"),
    tool_args: boundedValue(args ?? {}, state),
    ...(["after", "cancel"].includes(phase)
      ? {
          ...(phase === "after"
            ? { tool_result: redactText(toolResult ?? "", MAX_RESULT_CHARS) }
            : {}),
          ...(transitionToken === undefined
            ? {}
            : { transition_token: transitionToken }),
        }
      : {}),
  };
  if (new TextEncoder().encode(JSON.stringify(projection)).byteLength <=
      (state.extended ? 4 * configured + MAX_PAYLOAD_BYTES : MAX_PAYLOAD_BYTES)) {
    return projection;
  }
  return {
    schema_version: REQUEST_SCHEMA,
    phase,
    session_id: projection.session_id,
    action_id: projection.action_id,
    tool_name: projection.tool_name,
    tool_args: { projection_truncated: true },
    ...(["after", "cancel"].includes(phase)
      ? {
          ...(phase === "after"
            ? { tool_result: redactText(toolResult ?? "", 800) }
            : {}),
          ...(transitionToken === undefined
            ? {}
            : { transition_token: transitionToken }),
        }
      : {}),
  };
}

/** Return only the user's bounded answer to a Critic-requested clarification. */
export function boundedClarificationProjection(input, output) {
  const text = (Array.isArray(output?.parts) ? output.parts : [])
    .filter((part) => part && part.type === "text" && typeof part.text === "string")
    .map((part) => part.text.trim())
    .filter(Boolean)
    .join("\n");
  if (!text) return null;
  return {
    schema_version: REQUEST_SCHEMA,
    phase: "clarification_answer",
    session_id: safeIdentifier(input?.sessionID, "unknown-session"),
    answer: redactText(text, 2_000),
  };
}

function validateDecision(value) {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    value.schema_version !== DECISION_SCHEMA ||
    value.status !== "ok" ||
    !["silent", "speak", "clarify", "constraint_block"].includes(value.decision)
  ) {
    throw new GatewayFailure("invalid_decision_schema");
  }
  if (["speak", "clarify", "constraint_block"].includes(value.decision)) {
    if (
      typeof value.feedback !== "string" ||
      value.feedback.trim() === "" ||
      value.feedback.length > MAX_FEEDBACK_CHARS
    ) {
      throw new GatewayFailure("invalid_speak_feedback");
    }
    value = { ...value, feedback: value.feedback.trim() };
  }
  if (!["allow", "warn", "block", "unknown"].includes(value.constraint_outcome)) {
    throw new GatewayFailure("invalid_constraint_outcome");
  }
  if (
    (value.decision === "constraint_block") !==
    (value.constraint_outcome === "block")
  ) {
    throw new GatewayFailure("inconsistent_constraint_block");
  }
  if (
    value.constraint_outcome === "warn" &&
    (typeof value.constraint_feedback !== "string" ||
      value.constraint_feedback.trim() === "" ||
      value.constraint_feedback.length > MAX_FEEDBACK_CHARS)
  ) {
    throw new GatewayFailure("invalid_constraint_warning");
  }
  if (value.constraint_outcome === "warn") {
    value = { ...value, constraint_feedback: value.constraint_feedback.trim() };
  }
  // No ticket (omitted or null) is the current, before-only protocol.
  if (value.transition_token != null) {
    if (
      typeof value.transition_token !== "string" ||
      value.transition_token.trim() === "" ||
      value.transition_token !== value.transition_token.trim() ||
      value.transition_token.length > MAX_TRANSITION_TOKEN_CHARS
    ) {
      throw new GatewayFailure("invalid_transition_token");
    }
  }
  if (Object.hasOwn(value, "global_assessment") &&
      (typeof value.global_assessment !== "string" || value.global_assessment.length > MAX_FEEDBACK_CHARS)) {
    throw new GatewayFailure("invalid_global_assessment");
  }
  if (value.local_guidance != null &&
      (typeof value.local_guidance !== "string" || value.local_guidance.length > MAX_FEEDBACK_CHARS)) {
    throw new GatewayFailure("invalid_local_guidance");
  }
  return value;
}

async function evaluateOnce(payload) {
  let response;
  try {
    response = await fetch(endpoint(), {
      method: "POST",
      headers: { accept: "application/json", "content-type": "application/json" },
      body: JSON.stringify(payload),
      redirect: "error",
      signal: AbortSignal.timeout(timeoutMs()),
    });
  } catch (error) {
    if (error instanceof GatewayFailure) throw error;
    const kind = ["AbortError", "TimeoutError"].includes(error?.name)
      ? "timeout"
      : "network_error";
    throw new GatewayFailure(kind);
  }
  if (!response.ok) {
    throw new GatewayFailure(`http_${response.status}`);
  }
  let value;
  try {
    value = await response.json();
  } catch {
    throw new GatewayFailure("invalid_json");
  }
  return validateDecision(value);
}

async function evaluate(payload) {
  try {
    return await evaluateOnce(payload);
  } catch (error) {
    if (error instanceof GatewayFailure) throw error;
    throw new GatewayFailure("unknown_gateway_failure");
  }
}

function appendMarkedFeedback(output, marker, feedback) {
  output.output = `${output.output ?? ""}\n${marker}\n${feedback}`.trim();
}

/**
 * Thin OpenCode Single-Agent adapter.
 *
 * OpenCode transports only a bounded action projection. The trusted loopback
 * gateway owns task progress, Harness evidence, model calls, and run health.
 */
export const RewardBuilderV4RuntimePolicyCritic = async ({ client }) => {
  const boundSessions = new Set();
  const pendingConstraintNotices = new Map();
  const pendingDegradedNotices = new Map();
  const pendingClarificationSessions = new Set();
  const pendingQuestionCalls = new Map();
  const pendingTransitions = new Map();
  const transitionOwners = new Map();
  const consumedTransitionTokens = new Set();

  const actionKey = (input) =>
    JSON.stringify([input?.sessionID ?? "", input?.callID ?? ""]);
  const pendingAssessments = new Map();

  function failureKind(error) {
    return error instanceof GatewayFailure
      ? error.kind
      : "unknown_gateway_failure";
  }

  function degradedNotice(kind, phase) {
    return {
      marker: DEGRADED,
      text: `Runtime Policy Critic ${phase} review was unavailable (${kind}); the tool result was preserved.`,
    };
  }

  function retirePendingTransition(key) {
    const token = pendingTransitions.get(key);
    pendingTransitions.delete(key);
    if (token === undefined) return;
    transitionOwners.delete(token);
    consumedTransitionTokens.add(token);
  }

  function registerTransition(input, token) {
    if (token === undefined) return null;
    if (
      typeof input?.sessionID !== "string" ||
      input.sessionID === "" ||
      typeof input?.callID !== "string" ||
      input.callID === ""
    ) {
      return "transition_identity_missing";
    }
    const key = actionKey(input);
    if (
      consumedTransitionTokens.has(token) ||
      transitionOwners.has(token)
    ) {
      return "transition_token_replay";
    }
    pendingTransitions.set(key, token);
    transitionOwners.set(token, key);
    return null;
  }

  function consumeTransition(input) {
    const key = actionKey(input);
    const token = pendingTransitions.get(key);
    if (token === undefined) return null;
    pendingTransitions.delete(key);
    const owner = transitionOwners.get(token);
    transitionOwners.delete(token);
    consumedTransitionTokens.add(token);
    return owner === key ? token : null;
  }

  async function bindSession(sessionID) {
    if (typeof sessionID !== "string" || sessionID === "") {
      throw new GatewayFailure("invalid_opencode_session_identity");
    }
    if (boundSessions.has(sessionID)) return;
    let response;
    try {
      response = await client.session.get({ path: { id: sessionID } });
    } catch {
      throw new GatewayFailure("opencode_session_resolution_failed");
    }
    const info = response?.data ?? response;
    if (!info || info.id !== sessionID) {
      throw new GatewayFailure("opencode_session_binding_failed");
    }
    boundSessions.add(sessionID);
  }

  function constraintNotice(result) {
    if (result.constraint_outcome === "warn") {
      return { marker: CONSTRAINT_WARN, text: result.constraint_feedback };
    }
    if (result.constraint_outcome === "unknown") {
      return {
        marker: CONSTRAINT_UNKNOWN,
        text: "The operational constraint check was inconclusive; the action was not blocked.",
      };
    }
    return null;
  }

  function markedNotice(notice) {
    return notice ? `\n${notice.marker}\n${notice.text}` : "";
  }

  function appendDecisionFeedback(output, result) {
    if (result.decision === "constraint_block") {
      appendMarkedFeedback(output, CONSTRAINT_BLOCK, result.feedback);
    } else if (result.decision === "speak") {
      appendMarkedFeedback(output, SPEAK, result.feedback);
    } else if (result.decision === "clarify") {
      appendMarkedFeedback(output, CLARIFY, result.feedback);
    }
    const notice = constraintNotice(result);
    if (notice) appendMarkedFeedback(output, notice.marker, notice.text);
  }

  return {
    "tool.execute.before": async (input, output) => {
      const key = actionKey(input);
      pendingAssessments.delete(key);
      pendingConstraintNotices.delete(key);
      pendingDegradedNotices.delete(key);
      pendingQuestionCalls.delete(key);
      // OpenCode may replay a before event for the same call. Retire the old
      // server ticket before asking for a replacement; otherwise the trusted
      // loop correctly rejects the duplicate while the tool silently runs.
      const staleToken = pendingTransitions.get(key);
      if (staleToken !== undefined) {
        try {
          await evaluate(
            boundedActionProjection(
              "cancel",
              input,
              undefined,
              undefined,
              staleToken,
            ),
          );
        } catch (error) {
          retirePendingTransition(key);
          pendingDegradedNotices.set(
            key,
            degradedNotice(failureKind(error), "before replay cleanup"),
          );
          return;
        }
        retirePendingTransition(key);
      }
      let result;
      try {
        await bindSession(input.sessionID);
        result = await evaluate(
          boundedActionProjection(
            "before",
            input,
            output.args,
            undefined,
          ),
        );
      } catch (error) {
        // Only an explicit Critic decision may stop a tool. Runtime, transport,
        // or protocol failure is visible to the Actor after the real tool runs.
        pendingDegradedNotices.set(
          key,
          degradedNotice(failureKind(error), "before"),
        );
        return;
      }
      if (result.decision === "constraint_block") {
        throw new Error(`${CONSTRAINT_BLOCK}\n${result.feedback}`);
      }
      const notice = constraintNotice(result);
      if (result.decision === "speak") {
        // Throwing here prevents the proposed tool body from executing. OpenCode
        // returns this tool error to the same root Actor session for revision.
        throw new Error(`${SPEAK}\n${result.feedback}${markedNotice(notice)}`);
      }
      if (result.decision === "clarify") {
        // The Critic cannot impersonate the Actor or contact the user itself.
        // Stop this tool and ask the same root Actor session to surface the
        // single clarification question to its user.
        pendingClarificationSessions.add(input.sessionID);
        throw new Error(`${CLARIFY}\n${result.feedback}${markedNotice(notice)}`);
      }
      if (notice) {
        pendingConstraintNotices.set(key, notice);
      }
      if (result.global_assessment?.trim()) {
        pendingAssessments.set(key, result.global_assessment +
          (result.local_guidance ? "\nNon-blocking suggestion: " + result.local_guidance : ""));
      }
      if (input.tool === "question" && typeof result.clarification_question_id === "string") {
        pendingQuestionCalls.set(key, result.clarification_question_id);
        pendingClarificationSessions.add(input.sessionID);
      }
      const transitionFailure = registerTransition(
        input,
        result.transition_token,
      );
      if (transitionFailure) {
        pendingDegradedNotices.set(
          key,
          degradedNotice(transitionFailure, "before"),
        );
      }
    },

    "tool.execute.after": async (input, output) => {
      const key = actionKey(input);
      const assessment = pendingAssessments.get(key);
      pendingAssessments.delete(key);
      if (assessment) appendMarkedFeedback(output, "[RUNTIME_POLICY_CRITIC:BEFORE_ASSESSMENT]",
        "Pre-action assessment, not a verdict on the resulting artifact: " + assessment);
      const questionID = pendingQuestionCalls.get(key);
      pendingQuestionCalls.delete(key);
      if (questionID !== undefined && input.tool === "question") {
        // OpenCode's native tool returns actual user selections in metadata.
        // Never infer an answer from Actor prose, a tool error or "Unanswered".
        const answers = output?.metadata?.answers;
        if (Array.isArray(answers) && answers.length === 1 &&
            Array.isArray(answers[0]) && answers[0].length > 0 &&
            answers[0].every((answer) => typeof answer === "string" && answer.trim())) {
          try {
            const projection = boundedClarificationProjection(input, {
              parts: [{ type: "text", text: answers[0].join(", ") }],
            });
            const result = await evaluate({
              ...projection, clarification_question_id: questionID,
            });
            if (result.decision !== "silent" || result.clarification_recorded !== true) {
              throw new GatewayFailure("invalid_clarification_acknowledgement");
            }
            pendingClarificationSessions.delete(input.sessionID);
          } catch (error) {
            appendMarkedFeedback(output, DEGRADED,
              degradedNotice(failureKind(error), "clarification answer").text);
          }
        }
      }
      const transitionToken = consumeTransition(input);
      if (transitionToken !== null) {
        try {
          const result = await evaluate(
            boundedActionProjection(
              "after",
              input,
              input?.args,
              output?.output,
              transitionToken,
            ),
          );
          // The tool has already run. Feedback can guide the Actor's next move,
          // but it cannot retroactively block or replace the real tool result.
          appendDecisionFeedback(output, result);
          if (
            result.critic_status === "unknown" &&
            typeof result.degraded_notice === "string" &&
            result.degraded_notice.trim() !== ""
          ) {
            appendMarkedFeedback(output, DEGRADED, result.degraded_notice.trim());
          }
        } catch (error) {
          try {
            await evaluate(
              boundedActionProjection(
                "cancel",
                input,
                undefined,
                undefined,
                transitionToken,
              ),
            );
          } catch {
            // Best effort only: the server may already have consumed the token.
          }
          appendMarkedFeedback(
            output,
            DEGRADED,
            degradedNotice(failureKind(error), "after").text,
          );
        }
      }
      const notice = pendingConstraintNotices.get(key);
      pendingConstraintNotices.delete(key);
      if (notice) {
        appendMarkedFeedback(output, notice.marker, notice.text);
      }
      const degraded = pendingDegradedNotices.get(key);
      pendingDegradedNotices.delete(key);
      if (degraded) {
        appendMarkedFeedback(output, degraded.marker, degraded.text);
      }
    },

    "chat.message": async (input, output) => {
      if (!pendingClarificationSessions.has(input?.sessionID)) return;
      try {
        await bindSession(input.sessionID);
        const projection = boundedClarificationProjection(input, output);
        // Images and other non-text parts are not an answer we can faithfully
        // encode in this compact bridge.  Keep the clarification open until a
        // real textual answer arrives instead of fabricating one.
        if (projection === null) return;
        const result = await evaluate(projection);
        if (result.decision !== "silent") {
          throw new GatewayFailure("invalid_clarification_acknowledgement");
        }
        pendingClarificationSessions.delete(input.sessionID);
      } catch {
        // Never suppress a user's message because the optional Critic path is
        // unavailable. Keep the pending marker so a later answer may retry.
      }
    },
  };
};

export default RewardBuilderV4RuntimePolicyCritic;
