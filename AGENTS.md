# Repository working rules

- Put all one-off scripts, diagnostic probes, temporary tests, logs, model-call receipts, generated reports, build outputs and historical experiment materials under the repository's `_local/` directory. Do not create them at the repository root, in maintained packages, or in `/tmp`.
- Use `_local/scripts/` for temporary code, `_local/runs/<unique-run>/` for executions, `_local/reports/` for generated reports, and `_local/tmp/` for scratch space. The entire `_local/` tree is ignored by Git; never force-add it.
- Keep maintained engine and adapter regression tests in `tests/`. Do not archive a test just because it detects a bug or fails. Historical paper analysis and its tests are local-only.
- Preserve archived evidence bytes, including failed attempts. `_local/relocation-2026-10-15.json` records moved paths; do not rewrite historical receipt paths or hashes to make old audits pass. Relocated frozen runs are evidence, not in-place resumable runs.
- Before Python/test/build commands, create `_local/tmp` and `_local/cache/pycache`, then set `TMPDIR="$PWD/_local/tmp"` and `PYTHONPYCACHEPREFIX="$PWD/_local/cache/pycache"`. Use `COVERAGE_FILE="$PWD/_local/cache/coverage/.coverage"` when overriding coverage settings. Pytest and Ruff caches are already configured in `pyproject.toml`.
- Build from a temporary source copy under `_local/build/` and write distributions under `_local/dist/`, so setuptools intermediates stay out of the maintained tree.
- Keep changes focused. Preserve native grading's separation from Reward Engine learning and selection. Never manufacture revisions, remove failed receipts, weaken an audit, or label a scripted run as live validation.

- Name maintained files by function, without temporary version suffixes. Keep versions in package metadata and protocol fields; retain model names and benchmark task IDs.
- Keep captured model responses and their exact replay assertions in `_local/archive/captured-regressions/`. Maintained regressions use small, explicitly synthetic inputs and shared helpers in `tests/support/`. Record original paths and SHA-256 before archiving. Never replace functional coverage with skips.

- Keep testing guides, smoke/pipeline validation instructions, local file-management notes and validation reports under `_local/docs/` or `_local/reports/`, never in maintained documentation or release packages. Maintained docs cover installation, usage, architecture, interfaces and licensing.
