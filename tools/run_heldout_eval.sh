#!/usr/bin/env bash
set -euo pipefail
# Run the official swegemma harness on the 25 public held-out tasks.
# Adjust these paths for your Kaggle notebook.
DATA_ROOT=${DATA_ROOT:-/kaggle/input/gemma-4-developer-agent}
SUBMISSION=${SUBMISSION:-/kaggle/working/single_sub}
RESULTS=${RESULTS:-/kaggle/working/results/single_adapter}
IDS=$(tr '\n' ' ' < "$(dirname "$0")/../sft/held_out_ids.txt")

swegemma eval \
  --tasks "$DATA_ROOT/tasks.jsonl" \
  --snapshots-dir "$DATA_ROOT/snapshots" \
  --submission-dir "$SUBMISSION" \
  --results-dir "$RESULTS" \
  --sandbox subprocess \
  --task-ids $IDS
