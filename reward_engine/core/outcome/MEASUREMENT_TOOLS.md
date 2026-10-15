# Measurement tools

Builder and delivery review use actual tools before model-assisted interpretation. Tool execution does not add a second judging layer.

| Tool | Capability | Boundary |
|---|---|---|
| `calculate` | Evaluate numeric expressions | Does not establish that an expression models the claim correctly |
| `code.run` | Interpret a restricted numeric Python subset over JSON inputs | No arbitrary Python, dependency installation, or CUDA |
| `document.read` | Read supplied evidence; optionally retrieve observed HTTPS URLs through Tavily | No hidden grader access or arbitrary local-file reading |
| `cuda.run` | Call a trusted, configured isolated GPU backend | Requires an actual backend and execution receipts |

Builder selects checks from the real capability catalog and may include probes. If a checker is missing, a single Author call can produce a tool recipe. It is bound only after execution succeeds; a failed implementation remains unavailable rather than entering an unbounded repair loop. Recipes and their knowledge sources persist with the pack.

A model-assisted check can request tools:

```json
{"need_tools":[{"tool":"calculate","arguments":{"expression":"2+3"}}]}
```

The controller executes the request, saves its raw receipt, and returns a `tool:<test_id>:<index>` evidence handle. The model then provides claims, evidence, and judgments tied to the actual product, not solely to a criterion or computed number.

Tool receipts distinguish observed from unknown; check verdicts distinguish pass, fail, and unknown. Tool failure is not product failure. The normal measurement budget allows two request rounds and four total tool calls. Each receipt is retained separately. Final semantic judgments remain labeled `model_assisted`.

## Configuration

The shared adapter is `reward_engine/adapters/outcome_measurement.py`. To permit retrieval of already observed external documents:

```json
{"measurement":{"external_documents":true}}
```

`TAVILY_API_KEY` is read on demand. Only the observed URL is sent to the retrieval service, not the entire task or product. Stored evidence remains readable with external retrieval disabled.

For GPU execution, supply a trusted asynchronous `run(source, inputs)` implementation:

```json
{"measurement":{"cuda_runner":"trusted_gpu_adapter:run","cuda_isolated":true}}
```

The adapter must actually provide isolation, device access, and execution, returning `executed=true`, the real `device`, `exit_code`, and measurements. A configuration flag alone does not create isolation or prove GPU execution.

The numeric interpreter does not use `eval` or `exec`; imports, filesystem, network, and attribute access are unavailable. Source, inputs, outputs, and operation counts are bounded. It is a restricted interpreter, not a general Python sandbox.
