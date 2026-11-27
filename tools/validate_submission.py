#!/usr/bin/env python3
"""Offline validator for a swegemma declarative submission directory.

Re-implements the checks described in HARNESS_README.md section 2 so we can catch
schema/packaging mistakes without the (unpublished) adk-submission compiler.

Usage: python tools/validate_submission.py submission
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
ROOT_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
KNOWN_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file",
    "write_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}
AGENT_CLASSES = {"LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}
ALLOWED_EXT = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_TOTAL_BYTES = 3_221_225_472
MAX_INCLUDE_DEPTH = 10
PROVIDER_PREFIXES = ("openai/", "google/", "hosted_vllm/", "custom/")


class IncludeLoader(yaml.SafeLoader):
    pass


def _make_loader(root: Path) -> type[IncludeLoader]:
    class Loader(IncludeLoader):
        def __init__(self, stream: Any, base: Path | None = None) -> None:
            super().__init__(stream)
            self._base = base or root
            self._depth = 0

    def _include(loader: Loader, node: yaml.Node) -> Any:
        rel = loader.construct_scalar(node)
        target = (loader._base / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"!include escapes submission root: {rel}") from exc
        if not target.is_file():
            raise FileNotFoundError(f"!include target missing: {rel} ({target})")
        if target.suffix in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        if target.suffix in (".yaml", ".yml"):
            if loader._depth >= MAX_INCLUDE_DEPTH:
                raise ValueError(f"!include depth exceeded at {rel}")
            sub = Loader(target.read_text(encoding="utf-8"), base=target.parent)
            sub._depth = loader._depth + 1
            return sub.get_single_data()
        raise ValueError(f"!include unsupported extension: {rel}")

    Loader.add_constructor("!include", _include)
    return Loader


def load_yaml(path: Path, root: Path) -> dict[str, Any]:
    Loader = _make_loader(root)
    loader = Loader(path.read_text(encoding="utf-8"), base=path.parent)
    data = loader.get_single_data()
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def normalize_model(name: str) -> str:
    for p in PROVIDER_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "submission").resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 2

    roots = [root / n for n in ROOT_NAMES if (root / n).is_file()]
    if len(roots) != 1:
        print(f"ERROR: expected exactly one root config, found {[p.name for p in roots]}")
        return 2
    root_file = roots[0]

    # size + extension checks
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            if p.suffix not in ALLOWED_EXT:
                errors.append(f"disallowed extension: {p.relative_to(root)}")
    if total >= MAX_TOTAL_BYTES:
        errors.append(f"submission too large: {total} bytes >= {MAX_TOTAL_BYTES}")

    # discover compile tree
    agents: dict[Path, dict[str, Any]] = {}
    queue = [root_file]
    models: dict[str, list[str]] = {}
    seen: set[Path] = set()
    while queue:
        cfg_path = queue.pop(0).resolve()
        if cfg_path in seen:
            continue
        seen.add(cfg_path)
        try:
            cfg = load_yaml(cfg_path, root)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{cfg_path.relative_to(root)}: parse failed: {exc}")
            continue
        agents[cfg_path] = cfg

        agent_class = cfg.get("agent_class", "LlmAgent")
        if agent_class not in AGENT_CLASSES:
            errors.append(f"{cfg_path.relative_to(root)}: unknown agent_class {agent_class!r}")

        model = cfg.get("model")
        if model is not None:
            m = normalize_model(str(model))
            models.setdefault(m, []).append(str(cfg_path.relative_to(root)))

        refs: list[str] = []
        for tool in cfg.get("tools") or []:
            if isinstance(tool, str):
                if tool not in KNOWN_TOOLS:
                    errors.append(f"{cfg_path.relative_to(root)}: unknown tool {tool!r}")
            elif isinstance(tool, dict) and "agent_tool" in tool:
                at = tool["agent_tool"] or {}
                if "config_path" not in at:
                    errors.append(f"{cfg_path.relative_to(root)}: agent_tool missing config_path")
                else:
                    refs.append(str(at["config_path"]))
            else:
                errors.append(f"{cfg_path.relative_to(root)}: bad tools entry {tool!r}")
        for sub in cfg.get("sub_agents") or []:
            if isinstance(sub, dict) and "config_path" in sub:
                refs.append(str(sub["config_path"]))
            else:
                errors.append(f"{cfg_path.relative_to(root)}: bad sub_agents entry {sub!r}")
        for ref in refs:
            p = (cfg_path.parent / ref).resolve()
            try:
                p.relative_to(root)
            except ValueError:
                errors.append(f"{cfg_path.relative_to(root)}: ref escapes root: {ref}")
                continue
            if not p.is_file():
                errors.append(f"{cfg_path.relative_to(root)}: missing ref {ref}")
            else:
                queue.append(p)

    allowed = sorted(models)
    if len(models) != 1:
        errors.append(f"single-model rule violated: {allowed}")
    else:
        m = allowed[0]
        if m not in ALLOWED_MODELS:
            errors.append(f"model not allowed in competition: {m}")

    # adapters referenced must exist with adapter_model.safetensors
    for cfg_path, cfg in agents.items():
        adapter = cfg.get("adapter")
        if adapter:
            d = root / "adapters" / str(adapter)
            if not (d / "adapter_model.safetensors").is_file():
                errors.append(f"{cfg_path.relative_to(root)}: adapter missing: {adapter}")

    print(f"root config       : {root_file.relative_to(root)}")
    print(f"agents discovered : {len(agents)}")
    for cfg_path, cfg in agents.items():
        rel = str(cfg_path.relative_to(root))
        print(f"  - {rel:40} class={cfg.get('agent_class','LlmAgent'):15} "
              f"model={cfg.get('model')} tools={len(cfg.get('tools') or [])}")
    print(f"unique models     : {allowed}")
    print(f"total size        : {total} bytes ({total / 1024 / 1024:.1f} MiB)")
    for w in warnings:
        print(f"WARN : {w}")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1
    print("\nOK: submission looks structurally valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
