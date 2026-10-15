# Optional Claw-Eval companion interfaces

`optional_rail.patch` extends the external benchmark's `experiments/scripts/jiuwenswarm_container_runtime.py` and `jiuwenswarm_actor_worker.py` with an optional `rail_extension={source, config}`. It stages trusted Rail source and registers it on the root agent. Default tools, the JiuwenSwarm core, and task-only baseline behavior are unchanged when the extension is absent.

In the Claw-Eval checkout, use `git apply --check` before applying the patch. A successful reverse check means it is already applied; do not apply it twice or overwrite a different version. Deployment bundles must include the updated runtime and worker files alongside current engine source.

When Critic is enabled but the interface is missing, `run_trial` fails before starting containers. Provider credentials remain in the controller; containers receive a temporary trial callback token. The Rail implementation is `reward_engine/adapters/jiuwen_outcome_rail.py`.

## Native trace dependency

`native_trace.patch` preserves the external `jiuwenswarm_trace.py` and `jiuwenswarm_schema_replay.py` interfaces. Check applicability before applying; verify existing files with a reverse check rather than replacing them blindly.

Both `native_trial` and post-hoc `reexport` use `argument_formatter`. If actual MCP arguments differ from the original wire representation, the exporter replays the original actor image's SchemaUtils conversion, verifies image and source hashes, and requires an exact match with dispatched arguments. The replay container has no network, credentials, task execution, or model calls. Original arguments, dispatched arguments, and replay evidence are retained. Matching wire arguments require no replay.

Updating the engine checkout alone does not update the external benchmark checkout; deployment must carry both companion interfaces.
