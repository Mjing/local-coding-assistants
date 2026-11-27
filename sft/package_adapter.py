#!/usr/bin/env python3
"""Normalize a trained PEFT LoRA adapter for the swegemma submission layout.

  python sft/package_adapter.py --adapter sft/adapters/main_lora --name main_lora --attach

Produces:
  submission/adapters/<name>/adapter_config.json
  submission/adapters/<name>/adapter_model.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"]


def strip_multimodal(src: Path, dst: Path) -> tuple[int, int]:
    try:
        from safetensors.torch import load_file, save_file
    except Exception:
        shutil.copy2(src, dst)
        return -1, -1
    tensors = load_file(str(src))
    kept = {k: v for k, v in tensors.items() if ".language_model." in k}
    if not kept:
        shutil.copy2(src, dst)
        return len(tensors), len(tensors)
    save_file(kept, str(dst))
    return len(tensors), len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="main_lora")
    ap.add_argument("--submission", default=str(ROOT / "submission"))
    ap.add_argument("--base-alias", default="gemma-4-e2b-it-qat-w4a16-ct")
    ap.add_argument("--strip-multimodal", action="store_true")
    ap.add_argument("--attach", action="store_true",
                    help="add adapter: <name> to each LoRA agent YAML")
    ap.add_argument("--role", default=None,
                    help="subagent role to attach (with --attach-role)")
    ap.add_argument("--attach-role", action="store_true",
                    help="add adapter: <name> only to the YAML for --role")
    args = ap.parse_args()

    src = Path(args.adapter)
    cfg_path = src / "adapter_config.json"
    weights = src / "adapter_model.safetensors"
    if not cfg_path.is_file() or not weights.is_file():
        print(f"missing adapter_config.json or adapter_model.safetensors in {src}")
        return 2

    cfg = json.loads(cfg_path.read_text())
    rank = cfg.get("r")
    if not isinstance(rank, int):
        print(f"cannot determine LoRA rank from {cfg_path}")
        return 2
    if rank > 128:
        print(f"rank {rank} exceeds max_lora_rank 128")
        return 2
    cfg["base_model_name_or_path"] = args.base_alias
    cfg["peft_type"] = "LORA"
    cfg["task_type"] = "CAUSAL_LM"
    cfg["lora_dropout"] = 0.0
    cfg["inference_mode"] = True
    if not cfg.get("target_modules"):
        cfg["target_modules"] = TARGETS

    out = Path(args.submission) / "adapters" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    if args.strip_multimodal:
        total, kept = strip_multimodal(weights, out / "adapter_model.safetensors")
        print(f"tensors: {total} -> {kept}")
    else:
        shutil.copy2(weights, out / "adapter_model.safetensors")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"packaged {out} ({size / 1e6:.1f} MB, rank={rank}, alpha={cfg.get('lora_alpha')})")
    if size >= 3_221_225_472:
        print("ERROR: adapter exceeds the 3 GiB submission budget")
        return 1

    if args.attach:
        for f in sorted(Path(args.submission, "sub_agents").glob("*.yaml")):
            text = f.read_text()
            if "adapter:" in text or "model:" not in text:
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("model:"):
                    lines.insert(i + 1, f"adapter: {args.name}")
                    break
            f.write_text("\n".join(lines) + "\n")
            print(f"attached adapter to {f}")

    if args.attach_role:
        if not args.role:
            print("ERROR: --attach-role requires --role")
            return 2
        found = False
        for f in sorted(Path(args.submission, "sub_agents").glob("*.yaml")):
            text = f.read_text()
            if f.stem != args.role and not re.search(
                    rf"^name:\s*{re.escape(args.role)}\s*$", text, re.M):
                continue
            lines = text.splitlines()
            replaced = False
            for i, line in enumerate(lines):
                if line.startswith("adapter:"):
                    lines[i] = f"adapter: {args.name}"
                    replaced = True
                    break
            if not replaced:
                for i, line in enumerate(lines):
                    if line.startswith("model:"):
                        lines.insert(i + 1, f"adapter: {args.name}")
                        break
            f.write_text("\n".join(lines) + "\n")
            print(f"attached {args.name} to {f}")
            found = True
        if not found:
            print(f"WARNING: no subagent YAML found for role {args.role}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
