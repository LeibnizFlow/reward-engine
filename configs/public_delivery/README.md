# Public delivery contract

`api-usage-explain-cli-l3-001.json` binds a native WorkBuddy task to the exact relative delivery path `final_answer.md`. It checks regular-file presence, not content correctness or complete compliance with natural-language instructions.

The companion `.instruction.json` stores the exact native public instruction as Unicode-escaped source data. Decode its `instruction` field to recover the original UTF-8 text. Its SHA-256 and quoted line ranges must match the contract. Translating that input would define a different task and invalidate the binding. The contract's `source.path` refers to the benchmark's own `instruction.md`.

Explicit executor interpretation resolves the instruction's general no-file-generation rule against the later specific delivery requirement. Source identity checks do not establish that this interpretation is correct; inspect `translation` and the original instruction when reusing this contract. Never derive public requirements from private grader data.
