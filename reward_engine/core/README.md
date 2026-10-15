# Core controllers

The [Outcome package](outcome/README.md) implements task-grounded criteria, executable measurements, feedback, candidate comparison, and memory. Benchmark integrations share this implementation.

The core operates on public task context and actual candidate evidence. Runtime adapters own execution and artifact collection. Missing evidence remains unknown; benchmark graders remain independent post-hoc evaluators.

Start with [the runnable example](../../docs/quickstart.md) and [architecture](../../docs/architecture.md).
