import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  RewardBuilderV4RuntimePolicyCritic,
  boundedActionProjection,
  boundedClarificationProjection,
} from "../../reward_engine/legacy/adapters/opencode_plugin.mjs";

const originalFetch = globalThis.fetch;
const originalURL = process.env.REWARD_BUILDER_V4_CRITIC_URL;
const originalTimeout = process.env.REWARD_BUILDER_V4_TIMEOUT_MS;
const originalBudget = process.env.REWARD_BUILDER_V4_ACTION_BUDGET_CHARS;

function decision(decisionName, feedback = undefined, extra = {}) {
  return new Response(
    JSON.stringify({
      schema_version: "reward_builder_v4.opencode_decision.v1",
      status: "ok",
      decision: decisionName,
      constraint_outcome: "allow",
      ...(feedback === undefined ? {} : { feedback }),
      ...extra,
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
}

function clientFor(sessions = {}) {
  return {
    session: {
      get: async ({ path }) => {
        if (!(path.id in sessions)) throw new Error("missing session");
        return { data: { id: path.id, parentID: sessions[path.id] } };
      },
    },
  };
}

async function pluginWith(sessions = { root: null }) {
  process.env.REWARD_BUILDER_V4_CRITIC_URL = "http://127.0.0.1:8765/critic";
  process.env.REWARD_BUILDER_V4_TIMEOUT_MS = "1000";
  return RewardBuilderV4RuntimePolicyCritic({ client: clientFor(sessions) });
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalURL === undefined) delete process.env.REWARD_BUILDER_V4_CRITIC_URL;
  else process.env.REWARD_BUILDER_V4_CRITIC_URL = originalURL;
  if (originalTimeout === undefined) delete process.env.REWARD_BUILDER_V4_TIMEOUT_MS;
  else process.env.REWARD_BUILDER_V4_TIMEOUT_MS = originalTimeout;
  if (originalBudget === undefined) delete process.env.REWARD_BUILDER_V4_ACTION_BUDGET_CHARS;
  else process.env.REWARD_BUILDER_V4_ACTION_BUDGET_CHARS = originalBudget;
});

test("configured long-edit projection preserves the full memo and action rows, while redacting keys", () => {
  process.env.REWARD_BUILDER_V4_ACTION_BUDGET_CHARS = "24000";
  const args = { report_markdown: "A complete decision memo. ".repeat(230),
    actions: Array.from({length: 15}, (_, i) => ({id: `A${i}`, action: "Check real evidence", due: "2026-09-28",
      depends_on: [], evidence: ["source.md"]})), apiKey: "PRIVATE_KEY" };
  const view = boundedActionProjection("before", {sessionID:"root",callID:"long",tool:"edit_artifact"}, args);
  assert.equal(view.tool_args.report_markdown, args.report_markdown.trim());
  assert.deepEqual(view.tool_args.actions, args.actions);
  assert.equal(JSON.stringify(view).includes("PRIVATE_KEY"), false);
});

test("SILENT assessment reaches only the matching tool result and does not block it", async () => {
  const phases = [];
  globalThis.fetch = async (_url, options) => {
    phases.push(JSON.parse(options.body).phase);
    return decision("silent", undefined, {global_assessment:"Complete the remaining handoff.",
      local_guidance:"Check the remaining source before delivery.", transition_token:null});
  };
  const plugin = await pluginWith();
  const input = {sessionID:"root",callID:"edit",tool:"edit_artifact"};
  await plugin["tool.execute.before"](input, {args:{text:"draft"}});
  const wrong = {output:"another result"};
  await plugin["tool.execute.after"]({...input,callID:"other"}, wrong);
  assert.equal(wrong.output,"another result");
  const actual = {output:"saved draft"};
  await plugin["tool.execute.after"](input, actual);
  assert.match(actual.output,/saved draft/);
  assert.match(actual.output,/BEFORE_ASSESSMENT/);
  assert.match(actual.output,/Complete the remaining handoff/);
  assert.match(actual.output,/Check the remaining source/);
  assert.deepEqual(phases, ["before"]); // Native After hook adds cached feedback, zero network work.
});

test("the same call ID in two root sessions cannot exchange Critic feedback", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return decision("silent", undefined, {global_assessment: `Gap for ${request.session_id}`,
      local_guidance: `Advice for ${request.session_id}`, transition_token: null});
  };
  const plugin = await pluginWith({one: null, two: null});
  for (const sessionID of ["one", "two"]) {
    await plugin["tool.execute.before"]({sessionID, callID: "same-call", tool: "edit_artifact"},
      {args: {text: "draft"}});
  }
  for (const sessionID of ["two", "one"]) {
    const result = {output: `Saved ${sessionID}`};
    await plugin["tool.execute.after"]({sessionID, callID: "same-call", tool: "edit_artifact"}, result);
    assert.match(result.output, new RegExp(`Advice for ${sessionID}`));
    assert.equal(result.output.includes(`Advice for ${sessionID === "one" ? "two" : "one"}`), false);
  }
  assert.equal(requests.length, 2); // Only Before; no post-action calls.
});

test("action projection is bounded and omits secret, header, and env fields", () => {
  const projection = boundedActionProjection(
    "before",
    { sessionID: "root", callID: "call-1", tool: "bash" },
    {
      command:
        "curl -H 'Authorization: Bearer TOP_SECRET_TOKEN' 'https://x.test?a=1&token=URL_SECRET' " +
        "x".repeat(20_000),
      nested: { useful: "keep me", API_KEY: "NESTED_SECRET" },
      apiKey: "CAMEL_SECRET",
      headers: { authorization: "Bearer HEADER_SECRET" },
      env: { ORDINARY: "also must not leave the Actor process" },
      YIBU_API_KEY: "DIRECT_SECRET",
      unicode: "猫".repeat(10_000),
    },
  );
  const encoded = JSON.stringify(projection);

  assert.ok(Buffer.byteLength(encoded, "utf8") <= 8_000);
  assert.equal(projection.tool_args.nested.useful, "keep me");
  for (const forbidden of [
    "TOP_SECRET_TOKEN",
    "URL_SECRET",
    "NESTED_SECRET",
    "HEADER_SECRET",
    "DIRECT_SECRET",
    "CAMEL_SECRET",
    "also must not leave",
  ]) {
    assert.equal(encoded.includes(forbidden), false);
  }
  assert.equal(Object.hasOwn(projection.tool_args, "headers"), false);
  assert.equal(Object.hasOwn(projection.tool_args, "env"), false);
  assert.equal(Object.hasOwn(projection, "review_scope"), false);
});

test("before reads output.args and SPEAK blocks the proposed tool in the root session", async () => {
  let request;
  globalThis.fetch = async (_url, options) => {
    request = JSON.parse(options.body);
    return decision("speak", "Keep the global goal; revise this local action.");
  };
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      {
        sessionID: "root",
        callID: "call-1",
        tool: "edit",
        args: { source: "wrong-before-location" },
      },
      { args: { source: "correct-output-args" } },
    ),
    /REWARD_GUIDED_POLICY:SPEAK.*global goal/s,
  );
  assert.equal(request.phase, "before");
  assert.equal(request.tool_args.source, "correct-output-args");
});

test("CLARIFY stops the pending tool and asks the same Actor to question the user", async () => {
  globalThis.fetch = async () =>
    decision(
      "clarify",
      JSON.stringify({
        requested_actor_action: "ask_user",
        user_question: "Which deployment region should be used?",
      }),
    );
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      { sessionID: "root", callID: "clarify-call", tool: "deploy" },
      { args: { region: "unknown" } },
    ),
    /REWARD_GUIDED_POLICY:CLARIFY.*Which deployment region/s,
  );
});

test("the next user answer closes clarification in the same OpenCode session", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return request.phase === "before"
      ? decision(
          "clarify",
          JSON.stringify({
            requested_actor_action: "ask_user",
            user_question: "Which deployment region should be used?",
          }),
        )
      : decision("silent", undefined, {
          clarification_recorded: true,
          clarification_question_id: "clarify:root:clarify-call",
        });
  };
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      { sessionID: "root", callID: "clarify-call", tool: "deploy" },
      { args: { region: "unknown" } },
    ),
    /REWARD_GUIDED_POLICY:CLARIFY/,
  );
  await plugin["chat.message"](
    { sessionID: "root", messageID: "user-answer-1" },
    { message: {}, parts: [{ type: "text", text: "Use Hong Kong." }] },
  );
  await plugin["chat.message"](
    { sessionID: "root", messageID: "user-answer-2" },
    { message: {}, parts: [{ type: "text", text: "Do not resend this." }] },
  );

  assert.equal(requests.length, 2);
  assert.deepEqual(requests[1], {
    schema_version: "reward_builder_v4.opencode_action.v1",
    phase: "clarification_answer",
    session_id: "root",
    answer: "Use Hong Kong.",
  });
});

test("clarification projection is bounded and redacts secrets", () => {
  const payload = boundedClarificationProjection(
    { sessionID: "root" },
    {
      parts: [
        {
          type: "text",
          text: `Use Hong Kong. API_KEY=DO_NOT_LEAK ${"x".repeat(4_000)}`,
        },
      ],
    },
  );

  assert.equal(payload.phase, "clarification_answer");
  assert.ok(payload.answer.length <= 2_000);
  assert.equal(payload.answer.includes("DO_NOT_LEAK"), false);
});

test("native question metadata returns the actual user answer once without a transition", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    if (request.phase === "clarification_answer") {
      return decision("silent", "", { clarification_recorded: true });
    }
    if (request.tool_name === "question") {
      return decision("silent", "", { clarification_question_id: "clarify:root:edit" });
    }
    return decision("clarify", "Ask who the intended audience is.");
  };
  const plugin = await pluginWith();
  await assert.rejects(plugin["tool.execute.before"](
    { sessionID: "root", callID: "edit", tool: "write" }, { args: {} },
  ), /CLARIFY/);
  const call = { sessionID: "root", callID: "ask", tool: "question" };
  await plugin["tool.execute.before"](call, { args: { questions: [{ question: "Who is this for?" }] } });
  const output = { output: "Native question tool result.", metadata: { answers: [["Players"]] } };
  await plugin["tool.execute.after"](call, output);
  await plugin["tool.execute.after"](call, output);
  assert.equal(output.output, "Native question tool result.");
  assert.equal(requests.length, 3);
  assert.deepEqual(requests[2], {
    schema_version: "reward_builder_v4.opencode_action.v1", phase: "clarification_answer",
    session_id: "root", answer: "Players", clarification_question_id: "clarify:root:edit",
  });
});

for (const metadata of [{}, { answers: [[]] }, { answers: [[null]] }]) {
  test(`unanswered native question does not invent a user answer: ${JSON.stringify(metadata)}`, async () => {
    const requests = [];
    globalThis.fetch = async (_url, options) => {
      requests.push(JSON.parse(options.body));
      return decision("silent", "", { clarification_question_id: "clarify:root:edit" });
    };
    const plugin = await pluginWith();
    const call = { sessionID: "root", callID: "ask", tool: "question" };
    await plugin["tool.execute.before"](call, { args: {} });
    await plugin["tool.execute.after"](call, { output: "Actor claims the answer is X", metadata });
    assert.equal(requests.length, 1);
  });
}

test("a non-text message does not fabricate or close a pending clarification", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return request.phase === "before"
      ? decision(
          "clarify",
          JSON.stringify({
            requested_actor_action: "ask_user",
            user_question: "Which deployment region should be used?",
          }),
        )
      : decision("silent", undefined, { clarification_recorded: true });
  };
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      { sessionID: "root", callID: "clarify-image", tool: "deploy" },
      { args: { region: "unknown" } },
    ),
    /REWARD_GUIDED_POLICY:CLARIFY/,
  );
  await plugin["chat.message"](
    { sessionID: "root", messageID: "image-only" },
    { message: {}, parts: [{ type: "image", url: "file:///answer.png" }] },
  );
  assert.equal(requests.length, 1);

  await plugin["chat.message"](
    { sessionID: "root", messageID: "text-answer" },
    { message: {}, parts: [{ type: "text", text: "Use Hong Kong." }] },
  );
  assert.equal(requests.length, 2);
  assert.equal(requests[1].answer, "Use Hong Kong.");
});

test("after never calls the gateway and preserves an unmarked result", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new Error("after must not fetch");
  };
  const plugin = await pluginWith({});
  const output = { output: "TEST RESULT: failed", args: { wrong: true } };

  await plugin["tool.execute.after"](
    {
      sessionID: "root",
      callID: "call-2",
      tool: "bash",
      args: { command: "npm test" },
    },
    output,
  );

  assert.equal(calls, 0);
  assert.equal(output.output, "TEST RESULT: failed");
});

test("Actor payload cannot choose the Critic review scope", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    requests.push(JSON.parse(options.body));
    return decision("silent");
  };
  const plugin = await pluginWith({ root: null, child: "root" });

  await plugin["tool.execute.before"](
    { sessionID: "child", callID: "child-call", tool: "edit" },
    { args: { value: "child" } },
  );
  await plugin["tool.execute.before"](
    { sessionID: "root", callID: "root-call", tool: "edit" },
    { args: { value: "root" } },
  );
  assert.equal(requests.length, 2);
  assert.equal(requests.every((row) => !Object.hasOwn(row, "review_scope")), true);
});

test("an explicit blocking constraint stops the pending action", async () => {
  globalThis.fetch = async () =>
    decision(
      "constraint_block",
      "The pending action violates no-destructive-write.",
      { constraint_outcome: "block" },
    );
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      { sessionID: "root", callID: "blocked-call", tool: "write" },
      { args: { path: "protected" } },
    ),
    /OPERATIONAL_CONSTRAINT:BLOCK.*no-destructive-write/s,
  );
});

test("a warning survives until the real tool result", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return decision("silent", undefined, {
      constraint_outcome: "warn",
      constraint_feedback: "Confirm the generated file before publishing.",
    });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "warned-call",
    tool: "write",
    args: { path: "draft.html" },
  };
  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(calls, 1);
  assert.match(output.output, /real tool result/);
  assert.match(output.output, /OPERATIONAL_CONSTRAINT:WARN/);
  assert.match(output.output, /Confirm the generated file/);
});

test("a warning is retained when the root Critic also SPEAKs", async () => {
  globalThis.fetch = async () =>
    decision("speak", "Revise the pending action.", {
      constraint_outcome: "warn",
      constraint_feedback: "Confirm the target before writing.",
    });
  const plugin = await pluginWith();

  await assert.rejects(
    plugin["tool.execute.before"](
      { sessionID: "root", callID: "warn-speak", tool: "write" },
      { args: { path: "draft.html" } },
    ),
    /REWARD_GUIDED_POLICY:SPEAK.*Revise the pending action.*OPERATIONAL_CONSTRAINT:WARN.*Confirm the target/s,
  );
});

test("unknown is non-blocking but remains explicit after the tool runs", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return decision("silent", undefined, { constraint_outcome: "unknown" });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "unknown-check",
    tool: "write",
    args: { path: "draft.html" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(calls, 1);
  assert.match(output.output, /real tool result/);
  assert.match(output.output, /OPERATIONAL_CONSTRAINT:UNKNOWN/);
  assert.match(output.output, /inconclusive/);
});

test("a transition token binds one bounded after request to the same action", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    if (request.phase === "before") {
      return decision("silent", undefined, {
        transition_token: "opaque-transition-token-1",
      });
    }
    return decision("speak", "Keep the global goal; repair the observed failure.");
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "transition-call",
    tool: "bash",
    args: { command: "npm test" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = {
    output: `API_KEY=AFTER_SECRET\n${"large-result ".repeat(400)}`,
  };
  await plugin["tool.execute.after"](input, output);

  assert.equal(requests.length, 2);
  assert.equal(requests[1].phase, "after");
  assert.equal(requests[1].session_id, "root");
  assert.equal(requests[1].action_id, "transition-call");
  assert.equal(requests[1].transition_token, "opaque-transition-token-1");
  assert.ok(requests[1].tool_result.length <= 1_600);
  assert.equal(requests[1].tool_result.includes("AFTER_SECRET"), false);
  assert.match(output.output, /REWARD_GUIDED_POLICY:SPEAK/);
  assert.match(output.output, /repair the observed failure/);
});

test("a transition token is single-use and cannot cross action IDs", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return request.phase === "before"
      ? decision("silent", undefined, {
          transition_token: "opaque-transition-token-2",
        })
      : decision("silent");
  };
  const plugin = await pluginWith();
  const owner = {
    sessionID: "root",
    callID: "owner-call",
    tool: "write",
    args: { path: "index.html" },
  };
  const other = { ...owner, callID: "other-call" };

  await plugin["tool.execute.before"](owner, { args: owner.args });
  await plugin["tool.execute.after"](other, { output: "other result" });
  assert.equal(requests.length, 1);

  await plugin["tool.execute.after"](owner, { output: "owner result" });
  assert.equal(requests.length, 2);
  assert.equal(requests[1].action_id, "owner-call");

  await plugin["tool.execute.after"](owner, { output: "replayed result" });
  assert.equal(requests.length, 2);
});

test("before gateway failure is fail-open and leaves a degraded notice", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response("busy", { status: 503 });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "call-3",
    tool: "write",
    args: { path: "index.html" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(calls, 1);
  assert.match(output.output, /real tool result/);
  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /before review was unavailable \(http_503\)/);
});

test("invalid gateway decisions fail open with a protocol-degraded notice", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return new Response(JSON.stringify({ decision: "silent" }), { status: 200 });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "call-4",
    tool: "write",
    args: { path: "index.html" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(calls, 1);
  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /invalid_decision_schema/);
});

test("inconsistent block decisions degrade instead of creating a false block", async () => {
  const inconsistent = [
    decision("silent", undefined, { constraint_outcome: "block" }),
    decision("constraint_block", "blocked", { constraint_outcome: "allow" }),
  ];
  let index = 0;
  globalThis.fetch = async () => inconsistent[index++];
  const plugin = await pluginWith();

  for (const callID of ["silent-block", "block-allow"]) {
    const input = { sessionID: "root", callID, tool: "write" };
    await plugin["tool.execute.before"](input, {
      args: { path: "draft.html" },
    });
    const output = { output: "real tool result" };
    await plugin["tool.execute.after"](input, output);
    assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
    assert.match(output.output, /inconsistent_constraint_block/);
  }
});

test("a malformed warning degrades without blocking the tool", async () => {
  globalThis.fetch = async () =>
    decision("silent", undefined, { constraint_outcome: "warn" });
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "warn-without-feedback",
    tool: "write",
  };

  await plugin["tool.execute.before"](input, {
    args: { path: "draft.html" },
  });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /invalid_constraint_warning/);
});

test("OpenCode session lookup failure is fail-open and visible after the tool", async () => {
  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    return decision("silent");
  };
  const plugin = await pluginWith({});
  const input = {
    sessionID: "missing",
    callID: "call-6",
    tool: "edit",
  };

  await plugin["tool.execute.before"](input, { args: {} });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(calls, 0);
  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /opencode_session_resolution_failed/);
});

test("after gateway failure preserves the result and asks the gateway to cancel the ticket", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    if (request.phase === "before") {
      return decision("silent", undefined, {
          transition_token: "opaque-transition-token-3",
        })
    }
    if (request.phase === "cancel") {
      return decision("silent", undefined, { transition_cancelled: true });
    }
    return new Response("busy", { status: 503 });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "after-failure",
    tool: "bash",
    args: { command: "npm test" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);
  await plugin["tool.execute.after"](input, { output: "replay" });

  assert.equal(requests.length, 3);
  assert.equal(requests[2].phase, "cancel");
  assert.equal(requests[2].transition_token, "opaque-transition-token-3");
  assert.match(output.output, /real tool result/);
  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /after review was unavailable \(http_503\)/);
});

test("a replayed before cancels the stale server ticket before replacing it", async () => {
  const requests = [];
  let beforeCount = 0;
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    if (request.phase === "before") {
      beforeCount += 1;
      return decision("silent", undefined, {
        transition_token: `opaque-replay-token-${beforeCount}`,
      });
    }
    if (request.phase === "cancel") {
      return decision("silent", undefined, { transition_cancelled: true });
    }
    return decision("silent", undefined, { critic_status: "measured" });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "replayed-before",
    tool: "edit",
    args: { path: "index.html" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  await plugin["tool.execute.before"](input, { args: input.args });
  await plugin["tool.execute.after"](input, { output: "real result" });

  assert.deepEqual(
    requests.map((request) => request.phase),
    ["before", "cancel", "before", "after"],
  );
  assert.equal(requests[1].transition_token, "opaque-replay-token-1");
  assert.equal(requests[3].transition_token, "opaque-replay-token-2");
});

test("an unknown Process measurement is visible but is not negative Reward", async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return request.phase === "before"
      ? decision("silent", undefined, {
          transition_token: "opaque-transition-token-unknown",
        })
      : decision("silent", undefined, {
          critic_status: "unknown",
          degraded_reason: "browser unavailable",
          degraded_notice:
            "Runtime Policy Critic could not measure the last action; no negative Process Reward was assigned.",
        });
  };
  const plugin = await pluginWith();
  const input = {
    sessionID: "root",
    callID: "unknown-process",
    tool: "bash",
    args: { command: "npm test" },
  };

  await plugin["tool.execute.before"](input, { args: input.args });
  const output = { output: "real tool result" };
  await plugin["tool.execute.after"](input, output);

  assert.equal(requests.length, 2);
  assert.match(output.output, /real tool result/);
  assert.match(output.output, /REWARD_GUIDED_POLICY:DEGRADED/);
  assert.match(output.output, /no negative Process Reward/);
});
