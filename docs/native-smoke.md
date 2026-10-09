# Native benchmark smoke tests

These commands run one fresh model-driven task and its native grader. They check
installation, transport, tool execution, artifact capture, and scoring. They do
not run the three-round reward-construction experiment or establish benchmark-wide
quality. Scores and failed attempts are retained under `runs/`.

## Evidence

Download the assets from the private [evidence release](https://github.com/TheLittleJimmy/reward-engine/releases/tag/evidence-v1.0.0)
into `zenodo/`. Access requires permission to this repository. The data archives
are the original frozen publication; `source-and-reports.tar.gz` is a historical
source snapshot, not the current repaired checkout.

```sh
python -m release.restore --archives zenodo
python -m release.verify --data-root data --reports reports
```

Restore verifies archive checksums, restores only `data/`, and refuses to overwrite
an existing data directory. It never extracts the historical source over current
code. The verifier checks file hashes, native score bindings, and HTML links.

## Runtime setup

Use Python 3.12, Docker with Compose, Node.js, and the `claw-eval-agent:latest`
image described by the restored Claw benchmark. The OpenCode 1.18.16 binary and
WorkBuddy task/build inputs are included in the restored evidence. Models use an
external API; no local GPU is required.

The following setup reuses the host Python packages on this server and installs
the additional native interfaces in the repository's virtual environment:

```sh
python3.12 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[runtime,analysis,dev]'
.venv/bin/python -m pip install --no-deps 'harbor==0.18.0'
.venv/bin/python -m pip install shortuuid toml pathspec jinja2 filelock dirhash tenacity typer PyJWT
```

For a separate host, install Harbor's full dependencies with `.[workbuddy]` and
the restored Claw benchmark's requirements as well. WorkBuddy uses its native
CompositeVerifier; Claw uses the native task grader and freshly started mock services.

Copy `.env.example` to `.env`, keep it private (`chmod 600 .env`), and fill in
`YIBU_API_KEY` and `YIBU_BASE_URL`. Then prepare the published native source layout:

```sh
.venv/bin/python -m experiments.setup_native --credentials .env
```

The setup copies the native source and task inputs into ignored `local/` paths,
and adds those paths to this virtual environment. It does not require the old
`jiuwenswarm-swe-runner` checkout or the retired `swarm_reward` package.

## Run fresh tasks

Each output directory must be new. The runner freezes the current code and input
hashes before launch, executes that snapshot, then records `smoke_status.json` and
the native `RESULT.json` and `OFFICIAL.private.json` receipts.

```sh
.venv/bin/python -m experiments.smoke --benchmark claw_eval --task-id T111zh_expense_email_check --output runs/claw_smoke
.venv/bin/python -m experiments.smoke --benchmark workbuddy --task-id api-usage-explain-cli-l3-001 --output runs/workbuddy_smoke
```

Use `--backend-url`, `--credentials`, and `--key-env` to select another compatible
provider. The configured actor route is `deepseek-v4-flash`; accepted response
aliases are recorded in the frozen configuration. The smoke test exits nonzero
unless both the actor and native grading complete successfully. A valid low or
zero score is retained as measured, rather than retried for a higher score.

## Regression checks

```sh
.venv/bin/python -m pytest -q tests/unit tests/integration
node --test tests/integration/test_opencode_adapter.mjs
python -m ruff check .
python -m ruff format --check .
```

The optional Claw Docker integration witness additionally needs `CLAW_EVAL_ROOT`,
`CLAW_OPENCODE_BINARY`, and `CLAW_ACTOR_IMAGE`; its local model fixture verifies
transport without paid calls. The live tasks above independently use the real
configured model and grader.

## Verified run on 2026-10-10

| Benchmark | Task | Native score (0–1) | Result |
|---|---|---:|---|
| Claw-Eval | `T111zh_expense_email_check` | 0.932 | Actor completed; native grader returned `graded` |
| WorkBuddy Bench | `api-usage-explain-cli-l3-001` | 0.9213 | Actor completed; native grader returned `scored` (`partial_pass`) |

These runs made 7 and 16 successful model requests respectively, including grading.
Neither reused a historical candidate. The WorkBuddy score is a measured partial
pass, not a claim of perfect task performance. Local run paths and receipt hashes,
along with the test/build/evidence checks, are in
[runtime_validation.json](../release/runtime_validation.json).
