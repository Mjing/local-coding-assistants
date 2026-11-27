# Gemma 4 Developer Agent — Local Development Kit

This repository contains the competition starter data plus a complete local workflow for
building, training, validating, and packaging a Gemma 4 developer agent:

- a generic, repo-agnostic **multi-agent submission** template
- a **local evaluation harness** (approximate, Ollama-based)
- **SFT data synthesis** from the competition tasks and mined public repositories
- **per-role LoRA training** and adapter packaging for E2B/E4B
- a cross-repo **task miner** with graph/embedding generation
- submission and data **validators**

The official competition harness reference is preserved in
[`HARNESS_README.md`](HARNESS_README.md). The competition scores submissions against a
hidden **private repository** test set; this repository can only validate against the
public tasks and a 25-task public held-out split.

---

## Competition summary

- Task: fix a real Python repository issue.
- Phase 1: agent edits `/workspace` with nine built-in tools.
- Phase 2: fresh sandbox, apply the agent patch plus the hidden `test_patch`, run pytest,
  and score PASS/FAIL from JUnit XML.
- Public training data: 129 tasks from `fastapi`, `rich`, `requests`, `httpx`.
- Hidden test: about 120 tasks curated from private repositories.
- Submission: declarative `agent.yaml` + prompts/sub-agents/adapters; no arbitrary Python.
- Single base model per submission; different LoRA adapters per agent are allowed.
- Total unpacked submission must be under 3 GiB; context is 32,768 tokens.

Allowed scoring models:

| Alias | Small-model prize |
|---|---|
| `gemma-4-e2b-it-qat-w4a16-ct` | Yes |
| `gemma-4-e4b-it-qat-w4a16-ct` | Yes |
| `gemma-4-12b-it-qat-w4a16-ct` | No |
| `gemma-4-31b-it-qat-w4a16-ct` | No |

BF16 aliases such as `google/gemma-4-E2B-it` and `google/gemma-4-E4B-it` are used for
training/local tooling only. The submitted adapters must be declared against the
quantized competition aliases above.

---

## Repository layout

```text
.
├── HARNESS_README.md              # Official harness reference
├── README.md                      # This file
├── tasks.jsonl                    # 129 public tasks (gitignored starter data)
├── snapshots/                     # Public repo snapshots at base commits (gitignored)
├── graphs/                        # Public graph files (gitignored)
├── embeddings/                    # Public embedding files (gitignored)
├── wheels/                        # Offline dependency wheels (gitignored)
├── sample_submission/             # Competition starter submission
├── docker/  sandbox/              # Starter sandbox material
│
├── submission/                    # Generic LoopAgent template (E2B alias, no adapters)
├── submission_single/             # Single-agent direct-coder baseline
│
├── tools/
│   ├── local_harness.py           # ADK + Ollama Phase 1/Phase 2 approximation
│   ├── validate_submission.py     # Offline YAML/adapter/size validator
│   ├── mine_cross_repo.py         # Mine unrelated public Python repos into tasks
│   └── run_heldout_eval.sh        # Official swegemma eval on the 25 held-out tasks
│
├── sft/
│   ├── synthesize_role_data.py    # Build role-specific SFT examples
│   ├── train_lora.py              # Train one LoRA (shared adapter or one role)
│   ├── train_all_roles.py         # Train four per-role adapters, optionally in parallel
│   ├── package_adapter.py         # Normalize one adapter for the submission
│   ├── package_all_roles.py       # Package/attach all per-role adapters
│   ├── data/                      # Generated role data (gitignored)
│   ├── cross_repo/                # Mined tasks/snapshots/graphs/embeddings
│   └── held_out_ids.txt           # 25 public held-out task IDs
│
├── resources/                     # Optional reference artifacts (gitignored)
└── kaggle_training_bundle.zip     # Generated Kaggle bundle (gitignored)
```

Large starter data files (`tasks.jsonl`, `snapshots/`, `graphs/`, `embeddings/`, `wheels/`)
are gitignored because of their size. Mount or extract the competition dataset before
running local validation or synthesis.

---

## Agent architecture

`submission/agent.yaml` is a fixed `LoopAgent`:

```text
LoopAgent (max_iterations)
  investigator   -> FILE / LINES / FUNCTION / FIX        (read-only)
  implementer    -> edit, parse-check, diff, SPEC/COMMAND
  test_runner    -> execute exactly one command, raw output
  test_checker   -> independent verdict, only submit_patch on PASS
```

Key design points:

- **Repo-agnostic prompts.** No `fastapi`/`rich`/`requests` file hints. The problem
  statement and `grep`/graph tools drive localization.
- **Context isolation by role.** Each subagent has a narrow system prompt, its own tools,
  and `output_key` handoffs (`investigator_report`, `implementer_report`, `runner_report`).
- **Checker-only submission.** Only `test_checker` has `submit_patch`. It re-runs the
  command, parses JUnit XML when available, and checks exit codes, failures, errors,
  skips, no-tests, timeouts, crashes, and a non-empty source diff.
- **No truncation of the final report.** Examples longer than `--max-seq-len` are skipped
  during tokenization instead of truncated.
- **Non-pass handling.** Exit codes `124/137/-11`, `TimeoutExceeded`, `MemoryError`, OOM,
  skipped/xfail-only, import-only, no-tests-collected, and empty assertions are treated
  as FAIL/INCONCLUSIVE; the checker does not submit.

The source template defaults to a slow profile (`60 min`, `120 calls`, `max_iterations: 3`).
For scoring, the packaged fast variants use `5 min / 40 calls / 40 turns / 2 iterations`.

See [`submission/prompts/`](submission/prompts) for the exact prompts.

---

## Training data

Generated with `sft/synthesize_role_data.py` from the public training tasks and from
mined unrelated public repositories.

Current expanded dataset:

| Role | Examples | Tasks | Trainable tokens |
|---|---:|---:|---:|
| investigator | 385 | 385 | 51k |
| implementer | 334 | 334 | 331k |
| test_runner | 201 | 201 | 71k |
| test_checker | 373 | 201 | 79k |
| **Total** | **1,293** | **385** | **532k** |

- 102 tasks are from the four public training repos.
- 283 tasks are from 14 unrelated public repositories (`click`, `itsdangerous`,
  `markupsafe`, `attrs`, `marshmallow`, `tabulate`, `pluggy`, `bottle`, `cerberus`,
  `tomli`, `jinja`, `werkzeug`, `urllib3`, `jsonschema`).
- Total raw tokens: ~2.66M; median trajectory ~1,721 tokens; max ~23,736.
- Long examples: 11 skipped at 8,192; 3 skipped at 16,384.
- The 25 public held-out tasks are excluded from training and listed in
  [`sft/held_out_ids.txt`](sft/held_out_ids.txt).

### Synthesis method

For each task, `synthesize_role_data.py`:

1. Extracts source/test hunks from the gold patch and test patch.
2. Runs the real local tools (`grep`, `read_file`, `edit_file`, `run_command`) on the
   snapshot to produce real tool responses.
3. Builds native Gemma-4 chat examples in the format:
   `<bos><|turn>system ... <|turn>user ... <|turn>model ... <turn|>`.
4. Masks the system/user prompt and `<|tool_response>...</tool_response|>` spans.
5. Generates positive and negative checker examples (PASS + submit vs FAIL + NEXT FIX).

Safety guards:

- Test commands run with a 4 GiB address-space cap and CPU-time cap.
- 60-second test timeout.
- Runs with `TimeoutExceeded`, OOM, `SIGKILL`, or `SIGXCPU` are skipped rather than used
  as negative examples. This prevents infinite-loop bugs from OOM-killing the host.

Known gaps:

- FastAPI runner/checker synthesis is limited by the local dependency environment; it
  produces investigator/implementer data but fewer runner/checker examples.
- Some large multi-hunk or new-file patches skip the implementer example.
- The generated checker commands target the available test files; the agent prompt also
  supports direct `python3 -c "assert ..."` checks.

### Cross-repo mining

```bash
python tools/mine_cross_repo.py \
  --repos pallets/click pallets/itsdangerous pallets/markupsafe \
  --max-per-repo 25 \
  --out-dir sft/cross_repo
```

The miner:

- clones each public repo and filters commits that modify source and tests,
- excludes docs/CI/chore/version-bump style commits,
- writes `tasks.jsonl`, parent-commit snapshots, graph JSON, and 256-d embeddings,
- fingerprints content to avoid duplicate work.

The mined tasks are unrelated public repos and are not the competition's private test set.

---

## Local setup

Python 3.11+ is sufficient for the non-training tooling.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install google-adk litellm pyyaml numpy networkx pytest tokenizers
```

For local agent smoke tests, install Ollama and create a 32k-context variant:

```bash
printf 'FROM gemma4:e2b\nPARAMETER num_ctx 32768\n' > /tmp/Modelfile.e2b
ollama create gemma4-e2b-32k -f /tmp/Modelfile.e2b
```

For offline `--tokenize-only` checks, place any compatible Gemma-4 `tokenizer.json`
at a local path and pass `--tokenizer-json <path>`.

### Validate a submission

```bash
python tools/validate_submission.py submission
```

Checks: exactly one root config, single base model, allowed tools, adapter references,
allowed extensions, and the 3 GiB limit.

### Local harness

`tools/local_harness.py` approximates the official Phase 1/Phase 2 flow with ADK +
LiteLlm + Ollama, the nine tools, budget/time guards, patch extraction, and pytest
verification.

```bash
python tools/local_harness.py \
  --task-id requests_6592 \
  --submission submission \
  --model gemma4-e2b-32k \
  --max-tool-calls 60 --max-turns 8 --time-minutes 15
```

It is **not** the official scorer and it does not load the QAT LoRA adapters. It is for
prompt/workflow debugging only.

---

## Training

Actual training requires a GPU. Kaggle's T4x2/P100/L4x4 notebooks are suitable; the
competition L4x4 is available only for competition-attached notebooks.

Install Unsloth and train with either:

- a **shared** adapter (`sft/train_lora.py`) across all roles, or
- **per-role** adapters (`sft/train_all_roles.py`), one adapter per role.

### Shared adapter

```bash
python sft/train_lora.py \
  --data-dir sft/data \
  --out sft/adapters/main_lora \
  --epochs 2 --load-in-4bit --max-seq-len 8192
```

### Per-role adapters

```bash
python sft/train_all_roles.py \
  --model /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1 \
  --base-alias gemma-4-e2b-it-qat-w4a16-ct \
  --data-dir /kaggle/working/code/sft/data \
  --out-root /kaggle/working/adapters_e2b \
  --epochs 2 --load-in-4bit --max-seq-len 8192 \
  --jobs 2 --gpu-ids 0 1
```

- `--jobs` must be <= the number of visible GPUs. On T4x2 use 2; on L4x4 use 4.
- Roles are trained concurrently, one GPU per role, with a per-role Unsloth cache.
- Use `--skip-existing` and the same `--out-root` to resume a dead session.
- Prefer the local Kaggle model path (for example
  `/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1`) and set
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` to prevent HF downloads.
- `--max-seq-len` skips long examples rather than truncating the final report.

Memory guidance:

- E2B: T4x2 with 4-bit/8k, or L4x4 with bf16/16k.
- E4B: L4x4 with 4-bit/8k is recommended. T4x2 is tight; use 4-bit and 1k–2k sequence
  length, or train E4B on L4.

### Package a submission

```bash
# E2B template already declares the E2B alias
python sft/package_all_roles.py \
  --adapters-root /kaggle/working/adapters_e2b \
  --submission /kaggle/working/code/submission \
  --base-alias gemma-4-e2b-it-qat-w4a16-ct \
  --strip-multimodal

python tools/validate_submission.py submission
(cd submission && zip -qr ../submission_e2b.zip .)
```

For E4B, copy the template, remove old adapters, rewrite every subagent `model:` line to
`gemma-4-e4b-it-qat-w4a16-ct`, then package with the E4B alias. The packaging helper
attaches each `<role>_lora` to the matching subagent YAML.

The source template defaults to the slow budgets. For scoring, set the fast profile:

```yaml
evaluation:
  timeout_seconds: 120
  max_tool_calls: 40
  max_time_minutes: 5
  max_turns: 40
```

and `max_iterations: 2`.

### Kaggle bundle

`kaggle_training_bundle.zip` is generated from the scripts, `sft/data`, and the
submission template. It is gitignored and can be regenerated by zipping those paths. See
the training section above for the Kaggle cell sequence.

---

## Evaluation

### Public held-out tasks

The 25 public held-out tasks are listed in `sft/held_out_ids.txt`. They are a proxy for the
hidden private test set, not the official score.

If the official `swegemma` CLI is available in the notebook:

```bash
swegemma eval \
  --tasks /kaggle/input/competitions/gemma-4-developer-agent/tasks.jsonl \
  --snapshots-dir /kaggle/input/competitions/gemma-4-developer-agent/snapshots \
  --submission-dir /kaggle/working/code/submission \
  --results-dir /kaggle/working/results/e2b_public_heldout \
  --sandbox subprocess \
  --concurrency 2 \
  --task-ids $(cat sft/held_out_ids.txt)
```

The harness writes `summary.json`, `task_results.jsonl`, `patches/`, `test_outputs/`, and
`traces/`.

`swegemma` is not published on PyPI; `which swegemma` being empty means the competition
harness is not installed in that notebook. Use the competition's evaluation environment
or the Kaggle leaderboard for the official score.

### Scoring budgets

The source template allows up to 60 minutes / 120 calls / 500 turns and 3 loop iterations.
That can make a ~120-task evaluation take many hours. The fast profile
(5 min / 40 calls / 40 turns / 2 iterations) is recommended for iteration and was used in
the packaged fast variants.

---

## Known limitations

- The hidden test set is private and cannot be run locally.
- The official `swegemma` harness is not public; local evaluation is approximate.
- E2B/E4B zero-shot orchestration is unreliable; the adapters are intended to improve
  format/tool discipline, not guarantee resolution-rate gains.
- The dataset is 1,293 examples / 532k trainable tokens. This is enough for behavior
  cloning, but not for broad new coding/reasoning ability. More cross-repo tasks,
  additional trajectories per task, and more verification-focused examples are the
  highest-value expansions.
- FastAPI runner/checker data is limited by the local test dependency environment.
- The expanded cross-repo set is 14 public repos, not the competition's private repos.
- Per-role adapters are trained from scratch; E2B and E4B adapters are not interchangeable.
- The checker adapter should be retrained after the JUnit/timeout-aware prompt/data patch
  for best results.

---

## Status

Implemented and validated:

- generic LoopAgent submission template and offline validator
- local ADK/Ollama harness with Phase 1 tools and Phase 2 pytest
- role-specific SFT synthesis with real tool calls and safety guards
- cross-repo task miner with graph/embedding generation
- per-role LoRA training (parallel jobs), adapter packaging, and single-model validation
- JUnit/exit-code/timeout/crash-aware checker prompt
- 25-task public held-out evaluation script

No official leaderboard score is recorded in this repository. Use the competition
evaluation environment or Kaggle leaderboard for the authoritative result.

---

## License and data

The competition dataset and Gemma models are subject to the competition rules and the
Gemma Terms of Use. External reference artifacts under `resources/` are optional and are
not required by the training pipeline. No separate license file is included; add one if
you redistribute the code.
