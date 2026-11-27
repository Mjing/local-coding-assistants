#!/usr/bin/env python3
"""Train a LoRA adapter for the generic intake -> implement -> test -> check loop.

Designed for a Kaggle GPU. Local CPU can run --stats-only and --tokenize-only
(if tokenizers is installed) to validate the data and masking before training.

Examples:
  python sft/train_lora.py --stats-only
  python sft/train_lora.py --tokenize-only --tokenizer-json /path/tokenizer.json
  python sft/train_lora.py --out sft/adapters/main_lora --epochs 2 --load-in-4bit
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
# Unsloth must be imported before transformers/peft. Also keep its compile cache
# on a writable filesystem when running from a read-only Kaggle dataset.
_CACHE_ROOT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(_CACHE_ROOT / "unsloth_compiled_cache"))
try:
    import unsloth  # noqa: F401
except Exception:
    pass

ROLES = ("investigator", "implementer", "test_runner", "test_checker")
MODEL_MARKER = "<|turn>model\n"
TOOL_RESP_RE = re.compile(r"<\|tool_response>.*?<tool_response\|>", re.S)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Data loading / stats
# ---------------------------------------------------------------------------
def load_rows(data_dir: Path, roles: list[str], max_examples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in roles:
        p = data_dir / f"{role}.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["role"] = role
            rows.append(r)
            if max_examples and len(rows) >= max_examples:
                return rows
    return rows


def task_key(row: dict[str, Any]) -> str:
    return str(row.get("id", "")).removesuffix("_neg")


def stats(rows: list[dict[str, Any]]) -> None:
    print(f"rows: {len(rows)}")
    by_role = {}
    for r in rows:
        by_role[r["role"]] = by_role.get(r["role"], 0) + 1
    print("by role:", by_role)
    print("unique tasks:", len({task_key(r) for r in rows}))
    lens = [len(r["text"]) for r in rows]
    print(f"chars min/median/max: {min(lens)} / {int(statistics.median(lens))} / {max(lens)}")
    prefix_ok = sum(r["text"].startswith(r["prompt"]) for r in rows)
    print(f"text starts with prompt: {prefix_ok}/{len(rows)}")
    tr = [len(TOOL_RESP_RE.findall(r["text"])) for r in rows]
    print(f"tool_response spans per row: min/median/max "
          f"{min(tr)} / {int(statistics.median(tr))} / {max(tr)}")
    # assistant start sanity
    no_marker = sum(MODEL_MARKER not in r["text"] for r in rows)
    print(f"rows missing model marker: {no_marker}")


# ---------------------------------------------------------------------------
# Tokenization + label masking
# ---------------------------------------------------------------------------
def mask_from_offsets(text: str, ids: list[int], offsets: list[tuple[int, int]],
                      prompt: str) -> tuple[list[int], int, int]:
    labels = list(ids)
    marker = text.find(MODEL_MARKER)
    assistant_start = marker + len(MODEL_MARKER) if marker != -1 else len(prompt)
    prompt_masked = 0
    for i, (s, e) in enumerate(offsets):
        if e <= assistant_start:
            labels[i] = -100
            prompt_masked += 1
    resp_spans = [(m.start(), m.end()) for m in TOOL_RESP_RE.finditer(text)]
    resp_masked = 0
    for i, (s, e) in enumerate(offsets):
        if labels[i] == -100:
            continue
        if any(s >= rs and e <= re_ for rs, re_ in resp_spans):
            labels[i] = -100
            resp_masked += 1
    return labels, prompt_masked, resp_masked


def tokenize_transformers(rows: list[dict[str, Any]], tokenizer: Any,
                          max_len: int, limit: int = 0,
                          allow_truncate: bool = False):
    out = []
    skipped = 0
    for r in rows[:limit] if limit else rows:
        enc = tokenizer(r["text"], add_special_tokens=False,
                        return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offsets = [tuple(x) for x in enc["offset_mapping"]]
        if len(ids) > max_len:
            if allow_truncate:
                ids = ids[:max_len]
                offsets = offsets[:max_len]
            else:
                skipped += 1
                continue
        labels, pm, rm = mask_from_offsets(r["text"], ids, offsets, r["prompt"])
        if all(x == -100 for x in labels):
            skipped += 1
            continue
        out.append({"input_ids": ids, "labels": labels,
                    "attention_mask": [1] * len(ids),
                    "role": r["role"], "id": r.get("id"),
                    "prompt_masked": pm, "response_masked": rm})
    return out, skipped


def tokenize_tokenizers_json(rows: list[dict[str, Any]], tokenizer_json: Path,
                             max_len: int, limit: int = 0,
                             allow_truncate: bool = False):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer_json))
    out = []
    skipped = 0
    for r in rows[:limit] if limit else rows:
        enc = tok.encode(r["text"], add_special_tokens=False)
        ids = list(enc.ids)
        offsets = [tuple(x) for x in enc.offsets]
        if len(ids) > max_len:
            if allow_truncate:
                ids = ids[:max_len]
                offsets = offsets[:max_len]
            else:
                skipped += 1
                continue
        labels, pm, rm = mask_from_offsets(r["text"], ids, offsets, r["prompt"])
        if all(x == -100 for x in labels):
            skipped += 1
            continue
        out.append({"input_ids": ids, "labels": labels,
                    "attention_mask": [1] * len(ids),
                    "role": r["role"], "id": r.get("id"),
                    "prompt_masked": pm, "response_masked": rm})
    return out, skipped


def token_stats(tok_rows: list[dict[str, Any]]) -> None:
    lens = [len(r["input_ids"]) for r in tok_rows]
    trainable = [sum(1 for x in r["labels"] if x != -100) for r in tok_rows]
    ratios = [t / max(1, l) for t, l in zip(trainable, lens)]
    print(f"tokenized rows: {len(tok_rows)}")
    print(f"tokens min/median/max: {min(lens)} / {int(statistics.median(lens))} / {max(lens)}")
    print(f"trainable tokens min/median/max: "
          f"{min(trainable)} / {int(statistics.median(trainable))} / {max(trainable)}")
    print(f"trainable ratio min/median/max: "
          f"{min(ratios):.2f} / {statistics.median(ratios):.2f} / {max(ratios):.2f}")


# ---------------------------------------------------------------------------
# Train / val split by task id
# ---------------------------------------------------------------------------
def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keys = sorted({task_key(r) for r in rows})
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio)) if val_ratio > 0 else 0
    val_keys = set(keys[:n_val])
    train = [r for r in rows if task_key(r) not in val_keys]
    val = [r for r in rows if task_key(r) in val_keys]
    return train, val


# ---------------------------------------------------------------------------
# Model + training
# ---------------------------------------------------------------------------
def build_tokenizer(model_name: str, tokenizer_json: Path | None) -> Any:
    if tokenizer_json:
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tokenizer_json))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class ListDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]


def train(args: argparse.Namespace, tok_rows: list[dict[str, Any]]) -> None:
    import torch

    out = Path(args.out)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise SystemExit(
            f"Output directory is not writable: {out}. "
            f"Use --out /kaggle/working/.... Original error: {exc}"
        )
    os.environ.setdefault(
        "UNSLOTH_COMPILE_LOCATION",
        str(out.resolve().parent / "unsloth_compiled_cache"),
    )

    FastLanguageModel = None
    try:
        from unsloth import FastLanguageModel  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Unsloth unavailable: {type(exc).__name__}: {exc}")

    from transformers import Trainer, TrainingArguments

    tok = build_tokenizer(args.model, None)

    class Collator:
        def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
            max_len = max(len(f["input_ids"]) for f in features)
            pad = getattr(tok, "pad_token_id", 0) or 0
            input_ids, labels, attn = [], [], []
            for f in features:
                n = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [pad] * n)
                labels.append(f["labels"] + [-100] * n)
                attn.append(f["attention_mask"] + [0] * n)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

    model = None
    if FastLanguageModel is not None:
        try:
            model, tok = FastLanguageModel.from_pretrained(
                model_name=args.model, max_seq_length=args.max_seq_len,
                dtype=None, load_in_4bit=args.load_in_4bit)
            model = FastLanguageModel.get_peft_model(
                model, r=args.r, lora_alpha=args.alpha,
                target_modules=TARGET_MODULES, lora_dropout=0.0, bias="none",
                use_gradient_checkpointing="unsloth", random_state=args.seed)
            FastLanguageModel.for_training(model)
            print("using Unsloth")
        except Exception as exc:  # noqa: BLE001
            print(f"Unsloth load/train setup failed: {type(exc).__name__}: {exc}")
            model = None
    if model is None:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(r=args.r, lora_alpha=args.alpha,
                          target_modules=TARGET_MODULES, lora_dropout=0.0,
                          bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        print("using Transformers + PEFT")

    train_rows, val_rows = split_rows(tok_rows, args.val_ratio, args.seed)
    print(f"train examples: {len(train_rows)} | val examples: {len(val_rows)}")

    bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    optim = "adamw_torch"
    try:
        import bitsandbytes  # noqa: F401
        optim = "adamw_8bit"
    except Exception:
        pass

    ta_kwargs: dict[str, Any] = dict(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        optim=optim,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    targs = None
    for _ in range(12):
        try:
            targs = TrainingArguments(**ta_kwargs)
            break
        except TypeError as exc:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not m or m.group(1) not in ta_kwargs:
                raise
            dropped = m.group(1)
            print(f"dropping unsupported TrainingArguments argument: {dropped}")
            ta_kwargs.pop(dropped)
    if targs is None:
        raise RuntimeError("could not construct TrainingArguments")
    trainer = Trainer(
        model=model, args=targs, train_dataset=ListDataset(train_rows),
        eval_dataset=ListDataset(val_rows) if val_rows else None,
        data_collator=Collator(),
    )
    trainer.train()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(out)
    patch_adapter_config(out / "adapter_config.json", args.base_alias, args.r, args.alpha)
    print(f"saved adapter -> {out}")


def patch_adapter_config(path: Path, base_alias: str, r: int, alpha: int) -> None:
    if not path.is_file():
        return
    cfg = json.loads(path.read_text())
    cfg["base_model_name_or_path"] = base_alias
    cfg["r"] = r
    cfg["lora_alpha"] = alpha
    cfg["peft_type"] = "LORA"
    cfg["task_type"] = "CAUSAL_LM"
    if not cfg.get("target_modules"):
        cfg["target_modules"] = TARGET_MODULES
    path.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "sft" / "data"))
    ap.add_argument("--roles", nargs="*", default=list(ROLES))
    ap.add_argument("--out", default=str(ROOT / "sft" / "adapters" / "main_lora"))
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--base-alias", default="gemma-4-e2b-it-qat-w4a16-ct")
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--allow-truncate", action="store_true")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--tokenize-only", action="store_true")
    args = ap.parse_args()

    rows = load_rows(Path(args.data_dir), args.roles, args.max_examples)
    if not rows:
        print("no data rows")
        return 2
    stats(rows)
    if args.stats_only:
        return 0

    if args.tokenizer_json and args.tokenize_only:
        tok_rows, skipped = tokenize_tokenizers_json(
            rows, Path(args.tokenizer_json), args.max_seq_len,
            args.max_examples, args.allow_truncate)
    else:
        tok = build_tokenizer(args.model, None)
        tok_rows, skipped = tokenize_transformers(
            rows, tok, args.max_seq_len, args.max_examples, args.allow_truncate)
    print(f"skipped over-length/masked rows: {skipped}")
    if not tok_rows:
        print("no tokenized rows (all labels masked?)")
        return 2
    token_stats(tok_rows)
    if args.tokenize_only:
        return 0
    train(args, tok_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
