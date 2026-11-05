# Benchmark entry points

These modules prepare immutable campaign configurations, run native tasks, and retain execution evidence. Obtain benchmark datasets and dependencies separately. Set paths explicitly; no archived experiment installation is required.

- [Claw-Eval](claw_eval/README.md)
- [Office](workbuddy/README.md)
- [Model configuration](../docs/configuration.md)
- [JiuwenSwarm runtime](../reward_engine/adapters/jiuwenswarm/README.md)

Both entries support `prepare`, `run`, and reporting commands; use `--help` for the complete interface. Preparation requires fresh output directories. Resume with the same entry's `run --output DIRECTORY`.

## Resume with an explicit implementation revision

Use `python -m experiments.revision --help` to prepare a separately recorded implementation revision. Retain the original frozen configuration and evidence. Revision verification binds code, settings, and completed-stage provenance; editing frozen files in place is not a migration.
