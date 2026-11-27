#!/usr/bin/env python3
"""Package per-role adapters and attach each to its matching subagent YAML.

  python sft/package_all_roles.py --adapters-root /kaggle/working/adapters
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
    ap.add_argument("--adapters-root", default=str(ROOT / "sft" / "adapters"))
    ap.add_argument("--submission", default=str(ROOT / "submission"))
    ap.add_argument("--base-alias", default="gemma-4-e2b-it-qat-w4a16-ct")
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--strip-multimodal", action="store_true")
    args = ap.parse_args()

    for role in args.roles:
        src = Path(args.adapters_root) / f"{role}_lora"
        if not (src / "adapter_config.json").is_file():
            print(f"skip {role}: no adapter at {src}")
            continue
        cmd = [
            sys.executable, str(ROOT / "sft" / "package_adapter.py"),
            "--adapter", str(src), "--name", f"{role}_lora",
            "--submission", args.submission, "--base-alias", args.base_alias,
            "--role", role, "--attach-role",
        ]
        if args.strip_multimodal:
            cmd.append("--strip-multimodal")
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
