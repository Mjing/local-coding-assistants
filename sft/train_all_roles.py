#!/usr/bin/env python3
"""Train one LoRA adapter per role.

For multi-GPU Kaggle boxes, roles can be trained concurrently, one role per GPU:
  --jobs 2   on T4 x2
  --jobs 4   on L4 x4

  python sft/train_all_roles.py --model google/gemma-4-E2B-it \
      --out-root /kaggle/working/adapters_expanded \
      --epochs 2 --load-in-4bit --max-seq-len 8192 --jobs 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLES = ("investigator", "implementer", "test_runner", "test_checker")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "sft" / "data"))
    ap.add_argument("--out-root", default=str(ROOT / "sft" / "adapters"))
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--jobs", type=int, default=1,
                    help="roles trained concurrently (one GPU each)")
    ap.add_argument("--gpu-ids", nargs="*", default=None,
                    help="GPU ids to assign, e.g. 0 1")
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--base-alias", default="gemma-4-e2b-it-qat-w4a16-ct")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    gpu_ids = args.gpu_ids or [str(i) for i in range(max(1, args.jobs))]
    out_root = Path(args.out_root)
    roles = []
    for role in args.roles:
        out = out_root / f"{role}_lora"
        if args.skip_existing and (out / "adapter_config.json").is_file():
            print(f"skip {role}: {out} already exists")
            continue
        roles.append(role)

    failed = False
    for start in range(0, len(roles), args.jobs):
        wave = roles[start:start + args.jobs]
        procs = []
        for i, role in enumerate(wave):
            gpu = gpu_ids[i % len(gpu_ids)]
            out = out_root / f"{role}_lora"
            cmd = [
                sys.executable, str(ROOT / "sft" / "train_lora.py"),
                "--data-dir", args.data_dir, "--roles", role, "--out", str(out),
                "--model", args.model, "--base-alias", args.base_alias,
                "--epochs", str(args.epochs), "--lr", str(args.lr),
                "--r", str(args.r), "--alpha", str(args.alpha),
                "--batch-size", str(args.batch_size),
                "--grad-accum", str(args.grad_accum),
                "--max-seq-len", str(args.max_seq_len),
                "--val-ratio", str(args.val_ratio), "--seed", str(args.seed),
            ]
            cmd.append("--load-in-4bit" if args.load_in_4bit else "--no-4bit")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env["UNSLOTH_COMPILE_LOCATION"] = str((out / "unsloth_compiled_cache").resolve())
            print(f"\n===== launch {role} on GPU {gpu} -> {out} =====", flush=True)
            procs.append((role, subprocess.Popen(cmd, env=env)))
        for role, proc in procs:
            rc = proc.wait()
            if rc != 0:
                print(f"ERROR: {role} training exited with {rc}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
