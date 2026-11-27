#!/usr/bin/env python3
"""Train one LoRA adapter per role (fresh process per role).

  python sft/train_all_roles.py --model google/gemma-4-E2B-it \
      --out-root sft/adapters --epochs 2 --load-in-4bit

Produces:
  sft/adapters/investigator_lora/
  sft/adapters/implementer_lora/
  sft/adapters/test_runner_lora/
  sft/adapters/test_checker_lora/
"""
from __future__ import annotations

import argparse
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

    out_root = Path(args.out_root)
    for role in args.roles:
        out = out_root / f"{role}_lora"
        if args.skip_existing and (out / "adapter_config.json").is_file():
            print(f"skip {role}: {out} already exists")
            continue
        print(f"\n===== training {role} -> {out} =====", flush=True)
        cmd = [
            sys.executable, str(ROOT / "sft" / "train_lora.py"),
            "--data-dir", args.data_dir, "--roles", role, "--out", str(out),
            "--model", args.model, "--base-alias", args.base_alias,
            "--epochs", str(args.epochs), "--lr", str(args.lr),
            "--r", str(args.r), "--alpha", str(args.alpha),
            "--batch-size", str(args.batch_size), "--grad-accum", str(args.grad_accum),
            "--max-seq-len", str(args.max_seq_len), "--val-ratio", str(args.val_ratio),
            "--seed", str(args.seed),
        ]
        cmd.append("--load-in-4bit" if args.load_in_4bit else "--no-4bit")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
