#!/usr/bin/env python3
"""Local approximation of the swegemma harness for prompt/agent iteration.

It uses Google ADK + LiteLlm + a local Ollama model to run the declarative
submission, with re-implementations of the 9 sandbox tools, then applies the
produced patch plus the task test_patch in a fresh copy and runs pytest.

This is NOT the official scorer. Use it to iterate on agent behaviour only.

Example:
  python tools/local_harness.py --task-id requests_6644 --submission submission \
    --tasks tasks.jsonl --snapshots snapshots --workdir /tmp/swe_local \
    --model gemma4:e2b
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.genai import types

# ---------------------------------------------------------------------------
# YAML loading with !include (mirrors adk-submission semantics)
# ---------------------------------------------------------------------------
PROVIDER_PREFIXES = ("openai/", "google/", "hosted_vllm/", "custom/")
MAX_INCLUDE_DEPTH = 10


class IncludeLoader(yaml.SafeLoader):
    pass


def _make_loader(root: Path) -> type[IncludeLoader]:
    class Loader(IncludeLoader):
        def __init__(self, stream: Any, base: Path | None = None, depth: int = 0) -> None:
            super().__init__(stream)
            self._base = base or root
            self._depth = depth
            self._root = root

    def _include(loader: Loader, node: yaml.Node) -> Any:
        rel = loader.construct_scalar(node)
        target = (loader._base / rel).resolve()
        try:
            target.relative_to(loader._root)
        except ValueError as exc:
            raise ValueError(f"!include escapes root: {rel}") from exc
        if not target.is_file():
            raise FileNotFoundError(f"!include missing: {rel}")
        if target.suffix in (".md", ".txt"):
            return target.read_text(encoding="utf-8")
        if target.suffix in (".yaml", ".yml"):
            if loader._depth >= MAX_INCLUDE_DEPTH:
                raise ValueError(f"!include depth exceeded: {rel}")
            sub = Loader(target.read_text(encoding="utf-8"), base=target.parent,
                         depth=loader._depth + 1)
            return sub.get_single_data()
        raise ValueError(f"!include unsupported: {rel}")

    Loader.add_constructor("!include", _include)
    return Loader


def load_yaml(path: Path, root: Path) -> dict[str, Any]:
    Loader = _make_loader(root)
    loader = Loader(path.read_text(encoding="utf-8"), base=path.parent)
    data = loader.get_single_data()
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a mapping")
    return data


def normalize_model(name: str) -> str:
    for p in PROVIDER_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


# ---------------------------------------------------------------------------
# Sandbox context and tools
# ---------------------------------------------------------------------------
PROTECTED_RE = re.compile(
    r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|(^|/)[^/]*_test\.py$|"
    r"(^|/)(conftest\.py|pytest\.ini|\.pytest\.ini|pyproject\.toml|tox\.ini|"
    r"setup\.cfg|sitecustomize\.py|usercustomize\.py|_swegemma_stubs\.py)$|"
    r"\.pth$"
)
GIT_EXCLUDE = """__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/
.coverage
"""


def _resolve(ws: Path, filepath: str) -> Path:
    raw = str(filepath).strip()
    if raw.startswith("/workspace/"):
        raw = raw[len("/workspace/"):]
    raw = raw.lstrip("/")
    p = (ws / raw).resolve()
    try:
        p.relative_to(ws.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes workspace: {filepath}") from exc
    return p


def _coerce_int(v: Any, default: int | None = None) -> int | None:
    if v is None:
        return default
    if isinstance(v, bool):
        return int(v)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y")
    return default


@dataclass
class Ctx:
    workspace: Path
    repo: str
    repo_dir: str
    graph_path: Path | None = None
    embedding_path: Path | None = None
    max_tool_calls: int = 120
    max_turns: int = 500
    time_minutes: float = 60.0
    timeout_seconds: int = 300
    tool_calls_used: int = 0
    patch_submitted: bool = False
    submitted_patch: str = ""
    agent_start: float = field(default_factory=time.time)
    _graph: dict[str, Any] | None = None
    _nodes: dict[str, dict[str, Any]] | None = None
    _emb: Any = None
    _emb_keys: list[str] | None = None

    def remaining(self) -> float:
        return self.time_minutes * 60 - (time.time() - self.agent_start)

    def gate(self) -> str | None:
        if self.tool_calls_used >= self.max_tool_calls:
            return self._err("BudgetExhausted",
                             f"tool call budget exhausted ({self.max_tool_calls})")
        if self.remaining() <= 0:
            return self._err("TimeExhausted", "session time budget exhausted")
        self.tool_calls_used += 1
        return None

    @staticmethod
    def _err(etype: str, msg: str, **extra: Any) -> str:
        d = {"status": "error", "error_type": etype, "error_message": msg}
        if extra:
            d["details"] = extra
        return json.dumps(d)

    def load_graph(self) -> None:
        if self._graph is not None or self.graph_path is None or not self.graph_path.is_file():
            return
        data = json.loads(self.graph_path.read_text())
        self._graph = data
        self._nodes = {n["name"]: n for n in data.get("nodes", [])}
        try:
            import numpy as np
            self._emb = np.load(self.embedding_path, allow_pickle=True)
            self._emb_keys = list(self._emb.keys())
        except Exception:
            self._emb = None
            self._emb_keys = None

    def node_text(self, name: str) -> str:
        if self._nodes and name in self._nodes:
            return str(self._nodes[name].get("text", ""))
        return ""


CTX: Ctx | None = None


def _ctx() -> Ctx:
    if CTX is None:
        raise RuntimeError("context not initialized")
    return CTX


def _budget_error() -> str | None:
    ctx = _ctx()
    err = ctx.gate()
    if err:
        return err
    return None


# ---- execution tools ------------------------------------------------------
def get_status() -> str:
    """Return live budget consumption and patch status."""
    c = _ctx()
    return json.dumps({
        "status": "ok",
        "tool_calls_used": c.tool_calls_used,
        "patch_submitted": c.patch_submitted,
        "patch_size": len(c.submitted_patch),
        "tool_calls_remaining": max(0, c.max_tool_calls - c.tool_calls_used),
        "max_tool_calls": c.max_tool_calls,
        "time_seconds_remaining": round(max(0.0, c.remaining()), 1),
        "max_time_minutes": c.time_minutes,
        "agent_elapsed_seconds": round(time.time() - c.agent_start, 1),
        "max_turns": c.max_turns,
        "command_timeout_seconds": c.timeout_seconds,
    })


def _run_bash(c: Ctx, command: str, timeout: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    pp = [str(c.workspace)]
    if (c.workspace / "src").is_dir():
        pp.append(str(c.workspace / "src"))
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TEST_TMPDIR"] = "/tmp"
    # Match the competition sandbox's memory ceiling and prevent runaway tests
    # (e.g. an infinite-loop bug) from OOM-killing the host.
    guarded = (
        f"ulimit -v 4194304 2>/dev/null; ulimit -t {max(1, int(timeout))} 2>/dev/null; "
        f"{command}"
    )
    return subprocess.run(
        ["/bin/bash", "-c", guarded], cwd=str(c.workspace), env=env,
        capture_output=True, text=True, timeout=timeout + 5,
    )


def run_command(command: str) -> str:
    """Execute a shell command in /workspace (300s timeout, output truncated)."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    timeout = int(min(c.timeout_seconds, max(5, int(c.remaining()))))
    try:
        p = _run_bash(c, command, timeout)
    except subprocess.TimeoutExpired:
        return Ctx._err("TimeoutExceeded", f"command exceeded {timeout}s")
    except Exception as exc:  # noqa: BLE001
        return Ctx._err(type(exc).__name__, str(exc))
    out = (p.stdout or "")[-5000:]
    errout = (p.stderr or "")[-5000:]
    if p.returncode == 0:
        return json.dumps({"status": "ok", "stdout": out, "stderr": errout,
                           "exit_code": 0})
    return Ctx._err("CommandError", f"exit code {p.returncode}",
                    stdout=out, stderr=errout, exit_code=p.returncode)


def _strip_protected(diff: str) -> str:
    if not diff:
        return diff
    chunks = re.split(r"(?m)^(?=diff --git )", diff)
    kept = []
    for ch in chunks:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", ch)
        if not m:
            if ch.strip():
                kept.append(ch)
            continue
        path = m.group(2)
        if PROTECTED_RE.search(path):
            continue
        kept.append(ch)
    return "".join(kept)


def submit_patch() -> str:
    """Stage untracked files, capture the workspace git diff, and mark submitted."""
    c = _ctx()
    try:
        _run_bash(c, "git add -N .", 120)
        diff = _run_bash(c, "git diff --binary HEAD", 120)
        patch = _strip_protected(diff.stdout or "")
    except Exception as exc:  # noqa: BLE001
        return Ctx._err(type(exc).__name__, str(exc))
    c.submitted_patch = patch
    c.patch_submitted = True
    files = sorted({m.group(1) for m in re.finditer(r"(?m)^diff --git a/(\S+)", patch)})
    return json.dumps({"status": "ok", "patch_size": len(patch),
                       "files_changed": len(files)})


# ---- workspace tools ------------------------------------------------------
def read_file(filepath: str, start_line: int | None = None,
              end_line: int | None = None) -> str:
    """Read a workspace file with 1-indexed inclusive line slicing (150 lines max)."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    try:
        p = _resolve(c.workspace, filepath)
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return Ctx._err(type(exc).__name__, str(exc))
    lines = text.splitlines()
    total = len(lines)
    start = _coerce_int(start_line, 1) or 1
    start = max(1, start)
    end = _coerce_int(end_line, start + 149) or (start + 149)
    end = min(total, end, start + 149)
    shifted = False
    if start > total:
        start = max(1, total)
        shifted = True
    content = "\n".join(lines[start - 1:end])
    if len(content) > 10000:
        content = content[:10000]
    truncated = len(content) >= 10000 or end < total
    return json.dumps({
        "status": "ok", "filepath": str(filepath), "content": content,
        "start_line": start, "end_line": end, "total_lines": total,
        "is_truncated": truncated, **({"shifted": True} if shifted else {}),
    })


def _reindent(block: str, indent: str) -> str:
    if not block:
        return block
    out = []
    for line in block.splitlines(keepends=True):
        if line.strip():
            out.append(indent + line.lstrip())
        else:
            out.append(line)
    return "".join(out)


def _replace_content(content: str, old: str, new: str,
                     allow_multiple: bool) -> tuple[str, int, str]:
    if old == "":
        raise ValueError("old_string is empty")
    cnt = content.count(old)
    if cnt == 1 or (cnt > 1 and allow_multiple):
        return content.replace(old, new, -1 if allow_multiple else 1), cnt, "exact"
    if cnt > 1:
        raise ValueError(f"old_string matched {cnt} times (allow_multiple=false)")

    # flexible line-based match
    clines = content.splitlines(keepends=True)
    olines = [ln.strip() for ln in old.splitlines()]
    if olines:
        hits = []
        n = len(olines)
        for i in range(len(clines) - n + 1):
            if all(clines[i + j].strip() == olines[j] for j in range(n)):
                hits.append(i)
        if hits and len(hits) == 1 or (hits and allow_multiple):
            i = hits[0]
            indent = re.match(r"\s*", clines[i]).group(0)
            block = _reindent(new, indent)
            first = clines[i]
            prefix = first[: len(first) - len(first.lstrip())]
            del block
            replacement = _reindent(new, prefix)
            if not replacement.endswith("\n"):
                replacement += "\n"
            out = "".join(clines[:i]) + replacement + "".join(clines[i + n:])
            return out, len(hits), "flexible"
        if len(hits) > 1:
            raise ValueError(f"old_string matched {len(hits)} times (allow_multiple=false)")

    # regex token fallback
    toks = re.split(r"([()\[\]{}:>=<])", old)
    pattern = r"\s*".join(re.escape(t) for t in toks if t.strip())
    try:
        matches = list(re.finditer(pattern, content))
    except re.error:
        matches = []
    if len(matches) == 1:
        m = matches[0]
        return content[:m.start()] + new + content[m.end():], 1, "regex"
    if len(matches) > 1 and not allow_multiple:
        raise ValueError(f"old_string matched {len(matches)} times (allow_multiple=false)")
    if matches:
        out = re.sub(pattern, new, content)
        return out, len(matches), "regex"
    raise ValueError("old_string not found")


def edit_file(filepath: str, old_string: str, new_string: str,
              allow_multiple: bool = False) -> str:
    """Replace old_string with new_string in a workspace file (3-tier matching)."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    allow_multiple = _coerce_bool(allow_multiple)
    try:
        p = _resolve(c.workspace, filepath)
        if not p.is_file():
            return Ctx._err("FileEditError", f"file not found: {filepath}")
        content = p.read_text(encoding="utf-8", errors="replace")
        if not content:
            return Ctx._err("FileEditError", f"file is empty: {filepath}")
        new_content, occ, strategy = _replace_content(content, old_string, new_string,
                                                      allow_multiple)
        p.write_text(new_content, encoding="utf-8")
        diff = "".join(difflib.unified_diff(
            content.splitlines(True), new_content.splitlines(True),
            fromfile=f"a/{filepath}", tofile=f"b/{filepath}"))
        return json.dumps({"status": "ok", "filepath": str(filepath),
                           "occurrences": occ, "strategy": strategy,
                           "diff": diff[-5000:], "is_truncated": len(diff) > 5000})
    except Exception as exc:  # noqa: BLE001
        return Ctx._err("FileEditError", str(exc))


def write_file(filepath: str, content: str) -> str:
    """Create or overwrite a workspace file, creating parent directories."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    try:
        p = _resolve(c.workspace, filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({"status": "ok", "filepath": str(filepath),
                           "size": len(content)})
    except Exception as exc:  # noqa: BLE001
        return Ctx._err(type(exc).__name__, str(exc))


# ---- graph tools ----------------------------------------------------------
def _resolve_node(c: Ctx, query: str) -> str | None:
    c.load_graph()
    if not c._nodes:
        return None
    q = query.strip()
    if q in c._nodes:
        return q
    for sep in (".", ";", "/"):
        suffix = sep + q
        for name in c._nodes:
            if name.endswith(suffix) or name == q:
                return name
    low = q.lower()
    for name in c._nodes:
        if name.lower() == low:
            return name
    for name in c._nodes:
        if low in name.lower():
            return name
    return None


def search_similar_code(query: str, k: int = 10) -> str:
    """Find graph nodes most similar to a symbol name using precomputed embeddings."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    c.load_graph()
    k = _coerce_int(k, 10) or 10
    if not c._nodes:
        return Ctx._err("GraphUnavailable", "no graph for this repository")
    resolved = _resolve_node(c, query)
    if resolved is None:
        return Ctx._err("NodeNotFound", f"no node matching {query!r}")
    results: list[dict[str, Any]] = []
    try:
        import numpy as np
        if c._emb is not None and resolved in c._emb:
            qv = c._emb[resolved]
            qn = float(np.linalg.norm(qv)) or 1.0
            scored = []
            for name in c._emb_keys or []:
                v = c._emb[name]
                denom = float(np.linalg.norm(v)) or 1.0
                scored.append((float(np.dot(qv, v) / (qn * denom)), name))
            scored.sort(reverse=True)
            for sim, name in scored[:k]:
                if name == resolved:
                    continue
                results.append({"node_name": name, "code": c.node_text(name)[:600],
                                "similarity": round(sim, 4)})
    except Exception:
        results = []
    if not results:
        toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query.lower()))
        scored = []
        for name in c._nodes:
            nt = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", name.lower()))
            if toks:
                scored.append((len(toks & nt) / len(toks), name))
        scored.sort(reverse=True)
        for sim, name in scored[:k]:
            if sim <= 0:
                continue
            results.append({"node_name": name, "code": c.node_text(name)[:600],
                            "similarity": round(sim, 4)})
    if not results:
        results = [{"node_name": resolved, "code": c.node_text(resolved)[:600],
                    "similarity": 1.0}]
    # cap total code chars
    budget = 5000
    out = []
    per = max(200, budget // max(1, len(results)))
    for r in results:
        r["code"] = r["code"][:per]
        budget -= len(r["code"])
        out.append(r)
    return json.dumps({"status": "ok", "query": resolved, "results": out,
                       "count": len(out), "is_truncated": budget <= 0})


def get_code_neighbors(node: str, edge_type: str | None = None,
                       max_neighbors: int = 50) -> str:
    """Find incoming and outgoing graph neighbors of a symbol node."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    c.load_graph()
    max_neighbors = _coerce_int(max_neighbors, 50) or 50
    if not c._graph:
        return Ctx._err("GraphUnavailable", "no graph for this repository")
    resolved = _resolve_node(c, node)
    if resolved is None:
        return Ctx._err("NodeNotFound", f"no node matching {node!r}")
    et = edge_type.lower() if edge_type else None
    neigh: list[str] = []
    for e in c._graph.get("edges", []):
        if et and str(e.get("type", "")).lower() != et:
            continue
        if e.get("source") == resolved:
            neigh.append(str(e.get("target")))
        elif e.get("target") == resolved:
            neigh.append(str(e.get("source")))
    uniq = sorted(set(neigh))
    trunc = len(uniq) > max_neighbors or sum(map(len, uniq)) > 5000
    uniq = uniq[:max_neighbors]
    return json.dumps({"status": "ok", "node": resolved, "neighbors": uniq,
                       "count": len(uniq), "is_truncated": trunc})


def get_code_subgraph(nodes: list[str]) -> str:
    """Extract the induced subgraph for a list of symbol nodes."""
    c = _ctx()
    err = _budget_error()
    if err:
        return err
    c.load_graph()
    if not c._graph:
        return Ctx._err("GraphUnavailable", "no graph for this repository")
    if isinstance(nodes, str):
        try:
            parsed = json.loads(nodes)
            nodes = parsed if isinstance(parsed, list) else nodes.split(",")
        except Exception:
            nodes = nodes.split(",")
    resolved = [r for r in (_resolve_node(c, str(n)) for n in nodes) if r]
    if not resolved:
        return Ctx._err("NodeNotFound", "none of the nodes resolved")
    rset = set(resolved)
    edges = [e for e in c._graph.get("edges", [])
             if e.get("source") in rset and e.get("target") in rset]
    return json.dumps({"status": "ok", "nodes": resolved, "edges": edges[:200],
                       "node_count": len(resolved), "edge_count": len(edges),
                       "is_truncated": len(edges) > 200})


TOOL_FUNCS = {
    "run_command": run_command,
    "submit_patch": submit_patch,
    "get_status": get_status,
    "read_file": read_file,
    "edit_file": edit_file,
    "write_file": write_file,
    "get_code_neighbors": get_code_neighbors,
    "search_similar_code": search_similar_code,
    "get_code_subgraph": get_code_subgraph,
}


# ---------------------------------------------------------------------------
# Agent compilation
# ---------------------------------------------------------------------------
def _gcc(cfg: dict[str, Any]) -> types.GenerateContentConfig | None:
    raw = cfg.get("generate_content_config") or {}
    if not isinstance(raw, dict):
        return None
    allowed = {k: raw[k] for k in
               ("temperature", "top_p", "top_k", "max_output_tokens",
                "presence_penalty", "frequency_penalty", "seed", "stop_sequences")
               if k in raw}
    return types.GenerateContentConfig(**allowed) if allowed else None


def _thinking_off(cfg: dict[str, Any]) -> bool:
    raw = cfg.get("generate_content_config") or {}
    tc = (raw.get("thinking_config") or {}) if isinstance(raw, dict) else {}
    if tc.get("thinking_budget") == 0:
        return True
    if str(tc.get("thinking_level", "")).upper() == "NONE":
        return True
    return tc.get("include_thoughts") is False and "thinking_budget" not in tc


def build_agent(cfg_path: Path, root: Path, model_id: str, api_base: str) -> Any:
    cfg = load_yaml(cfg_path, root)
    name = str(cfg["name"])
    desc = str(cfg.get("description") or name)
    agent_class = cfg.get("agent_class", "LlmAgent")

    sub_agents = [
        build_agent((cfg_path.parent / s["config_path"]).resolve(), root, model_id, api_base)
        for s in cfg.get("sub_agents") or []
    ]
    if agent_class == "LoopAgent":
        return LoopAgent(name=name, description=desc, sub_agents=sub_agents,
                         max_iterations=int(cfg.get("max_iterations", 500)))
    if agent_class == "SequentialAgent":
        return SequentialAgent(name=name, description=desc, sub_agents=sub_agents)
    if agent_class == "ParallelAgent":
        return ParallelAgent(name=name, description=desc, sub_agents=sub_agents)

    instruction = str(cfg.get("instruction") or cfg.get("global_instruction") or "")
    tools: list[Any] = []
    for t in cfg.get("tools") or []:
        if isinstance(t, str):
            fn = TOOL_FUNCS.get(t)
            if fn is None:
                raise ValueError(f"unknown tool {t!r} in {cfg_path}")
            tools.append(fn)
        elif isinstance(t, dict) and "agent_tool" in t:
            at = t["agent_tool"] or {}
            child = build_agent((cfg_path.parent / at["config_path"]).resolve(),
                                root, model_id, api_base)
            tools.append(AgentTool(agent=child,
                                   skip_summarization=bool(at.get("skip_summarization"))))
        else:
            raise ValueError(f"bad tools entry {t!r} in {cfg_path}")

    model_kwargs: dict[str, Any] = {}
    if _thinking_off(cfg):
        model_kwargs["reasoning_effort"] = "none"
    return LlmAgent(
        name=name,
        description=desc,
        model=LiteLlm(model=f"openai/{model_id}", api_base=api_base,
                      api_key="ollama", **model_kwargs),
        instruction=instruction,
        tools=tools,
        sub_agents=sub_agents,
        output_key=cfg.get("output_key"),
        include_contents=cfg.get("include_contents", "default"),
        generate_content_config=_gcc(cfg),
    )


# ---------------------------------------------------------------------------
# Prompt / snapshot helpers
# ---------------------------------------------------------------------------
def workspace_layout(ws: Path) -> str:
    try:
        p = subprocess.run(["find", ".", "-mindepth", "1", "-maxdepth", "3"],
                           cwd=str(ws), capture_output=True, text=True, timeout=60)
        lines = [ln for ln in p.stdout.splitlines()
                 if not re.search(r"/(\.|__pycache__|\.git)", ln)
                 and "__pycache__" not in ln and not ln.endswith(".pyc")]
    except Exception:
        lines = []
    return "\n".join(lines[:200])


def build_prompt(task: dict[str, Any], ws: Path) -> str:
    parts = [
        f"You are evaluating a software engineering task for repository {task['repo']}.",
        "",
        "Problem Statement:",
        task["problem_statement"].strip(),
    ]
    hints = (task.get("hints_text") or "").strip()
    if hints:
        parts += ["", "## Hints:", hints]
    parts += [
        "",
        "## Execution Environment Rules",
        "- Single command timeout: 300 seconds",
        "- Command output limit: 5000 characters",
        "- File view limit: 150 lines per read_file call, 10000 characters",
        "- Environment is offline. All repository dependencies are already installed.",
        "- Work strictly under /workspace. Put scratch files in /tmp, never in /workspace.",
        "",
        "## Instructions",
        "1. Locate and fix the issue in the repository source code.",
        "2. Make minimal, surgical edits. Do not modify tests or test configuration.",
        "3. Verify with targeted checks only; never run a full test suite.",
        "4. Call submit_patch exactly once when done, then reply with a short summary.",
        "",
        "## Workspace Layout",
        workspace_layout(ws),
    ]
    return "\n".join(parts)


def extract_snapshot(snapshot: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(snapshot), "-C", str(dest)],
                   check=True, timeout=600)


def git_baseline(ws: Path, msg: str = "baseline") -> None:
    if not (ws / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=str(ws), check=True,
                       capture_output=True)
        subprocess.run(["git", "config", "user.email", "agent@eval"], cwd=str(ws),
                       check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "agent"], cwd=str(ws),
                       check=True, capture_output=True)
    exclude = ws / ".git/info/exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    with exclude.open("a", encoding="utf-8") as f:
        f.write("\n" + GIT_EXCLUDE)
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "agent@eval",
        "GIT_COMMITTER_NAME": "agent", "GIT_COMMITTER_EMAIL": "agent@eval",
    })
    subprocess.run(["git", "add", "-A"], cwd=str(ws), env=env, check=True,
                   capture_output=True)
    subprocess.run(["git", "commit", "-m", msg, "--allow-empty", "-q"],
                   cwd=str(ws), env=env, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------------
def _part_summary(part: Any) -> str | None:
    if getattr(part, "function_call", None):
        fc = part.function_call
        args = json.dumps(dict(fc.args or {}), ensure_ascii=False)
        return f"-> {fc.name}({args[:600]})"
    if getattr(part, "function_response", None):
        fr = part.function_response
        resp = fr.response
        try:
            s = json.dumps(resp, ensure_ascii=False)
        except Exception:
            s = str(resp)
        return f"<- {fr.name}: {s[:800]}"
    txt = getattr(part, "text", None)
    if txt:
        return f"[text] {txt[:800]}"
    return None


async def run_phase1(cfg_path: Path, root: Path, task: dict[str, Any], ws: Path,
                     model_id: str, api_base: str, ctx: Ctx,
                     verbose: bool = True) -> dict[str, Any]:
    global CTX
    CTX = ctx
    agent = build_agent(cfg_path, root, model_id, api_base)
    ss = InMemorySessionService()
    runner = Runner(app_name="local_eval", agent=agent, session_service=ss)
    session = await ss.create_session(app_name="local_eval", user_id="eval_user",
                                      state={"problem_description": task["problem_statement"],
                                             "hints": (task.get("hints_text") or "").strip()})
    sid = session.id
    prompt = build_prompt(task, ws)
    if verbose:
        print(f"[phase1] prompt chars={len(prompt)} agent={agent.name}")

    nudges = 0
    turns = 0
    last_text = ""
    trace: list[str] = []
    conversation: list[str] = []
    while turns < ctx.max_turns and ctx.remaining() > 0 and not ctx.patch_submitted:
        turns += 1
        turn_tool = False
        finish_reason = None
        new_message = types.Content(
            role="user",
            parts=[types.Part(text=prompt if turns == 1 else conversation[-1])],
        )
        async for event in runner.run_async(user_id="eval_user", session_id=sid,
                                            new_message=new_message):
            if not isinstance(event, Event):
                continue
            finish_reason = getattr(event, "finish_reason", finish_reason)
            if getattr(event, "error_message", None):
                trace.append(f"[{event.author}] ERROR: {event.error_message}")
                if verbose:
                    print(f"  [{event.author}] ERROR: {event.error_message}")
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "function_call", None):
                        turn_tool = True
                    summ = _part_summary(part)
                    if summ:
                        trace.append(f"[{event.author}] {summ}")
                        if verbose:
                            print(f"  [{event.author}] {summ}")
                    if getattr(part, "text", None):
                        last_text = part.text
                        conversation.append(part.text)
            if ctx.patch_submitted and event.is_final_response():
                break
        if ctx.patch_submitted:
            break
        if turn_tool:
            nudges = 0
        else:
            nudges += 1
            if nudges > 3:
                if verbose:
                    print("[phase1] max nudges reached")
                break
        if verbose:
            print(f"[phase1] turn {turns} ended without submit (nudges={nudges})")
        conversation.append(
            "Please continue your work using the available tools, or call "
            "submit_patch when you have completed and verified your changes.")
        if len(conversation) > 6:
            conversation = conversation[-6:]

    if not ctx.patch_submitted:
        try:
            _run_bash(ctx, "git add -N .", 120)
            d = _run_bash(ctx, "git diff --binary HEAD", 120)
            ctx.submitted_patch = _strip_protected(d.stdout or "")
        except Exception:
            pass

    try:
        (ws.parent / "trace.txt").write_text("\n".join(trace) + "\n")
    except Exception:
        pass
    return {
        "turns": turns,
        "tool_calls_used": ctx.tool_calls_used,
        "patch_submitted": ctx.patch_submitted,
        "patch": ctx.submitted_patch,
        "last_text": last_text,
        "elapsed": round(time.time() - ctx.agent_start, 1),
    }


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
def extract_test_targets(test_patch: str) -> tuple[list[str], list[str]]:
    files = sorted({m.group(1) for m in re.finditer(r"(?m)^--- a/(\S+)", test_patch)})
    names = sorted({m.group(1) for m in
                    re.finditer(r"(?m)^\+\s*(?:async\s+)?def (test_[A-Za-z0-9_]+)", test_patch)})
    return files, names


def apply_patch(ws: Path, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return True, "empty patch"
    pf = Path("/tmp/_local_agent.patch")
    pf.write_text(patch if patch.endswith("\n") else patch + "\n")
    last = ""
    for extra in (["-p1", "--unsafe-paths"], ["-p1", "--unsafe-paths", "-3"],
                  ["-p1", "--unsafe-paths", "--ignore-space-change",
                   "--ignore-whitespace"]):
        p = subprocess.run(["git", "apply", *extra, str(pf)], cwd=str(ws),
                           capture_output=True, text=True)
        if p.returncode == 0:
            return True, " ".join(extra)
        last = p.stderr
    return False, last


def run_phase2(task: dict[str, Any], snapshot: Path, agent_patch: str,
               root: Path, workdir: Path, timeout: int = 300,
               verbose: bool = True) -> dict[str, Any]:
    vws = workdir / "verify_workspace"
    extract_snapshot(snapshot, vws)
    git_baseline(vws, "eval_baseline")
    ok, how = apply_patch(vws, agent_patch)
    if not ok:
        return {"applied": False, "resolved": False,
                "error": f"agent patch failed: {how}"}
    tp = task["test_patch"]
    pf = Path("/tmp/_local_test.patch")
    pf.write_text(tp if tp.endswith("\n") else tp + "\n")
    tp_ok = True
    tp_err = ""
    p = subprocess.run(["git", "apply", "-p1", str(pf)], cwd=str(vws),
                       capture_output=True, text=True)
    if p.returncode != 0:
        p = subprocess.run(["patch", "-p1", "--batch", "--forward", "-i", str(pf)],
                           cwd=str(vws), capture_output=True, text=True)
        tp_ok = p.returncode == 0
        tp_err = p.stderr
    if not tp_ok:
        return {"applied": True, "resolved": False,
                "error": f"test patch failed: {tp_err[:1000]}"}
    files, names = extract_test_targets(tp)
    files = [f for f in files if (vws / f).exists()]
    if not files:
        return {"applied": True, "resolved": False,
                "error": "no test files found after applying test_patch"}
    cmd = [sys.executable, "-m", "pytest", *files, "-q", "--tb=short",
           "-o", "timeout=0", "-p", "no:cacheprovider"]
    if names:
        cmd += ["-k", " or ".join(names)]
    env = os.environ.copy()
    pp = [str(vws)]
    if (vws / "src").is_dir():
        pp.append(str(vws / "src"))
    env["PYTHONPATH"] = os.pathsep.join(pp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    try:
        r = subprocess.run(cmd, cwd=str(vws), env=env, capture_output=True,
                           text=True, timeout=timeout)
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        return {"applied": True, "resolved": r.returncode == 0,
                "exit_code": r.returncode, "targets": files, "names": names,
                "output": out[-8000:]}
    except subprocess.TimeoutExpired:
        return {"applied": True, "resolved": False, "error": "pytest timeout"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--submission", default="submission")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--snapshots", default="snapshots")
    ap.add_argument("--graphs", default="graphs")
    ap.add_argument("--embeddings", default="embeddings")
    ap.add_argument("--workdir", default="/tmp/swe_local")
    ap.add_argument("--model", default="gemma4:e2b")
    ap.add_argument("--api-base", default="http://localhost:11434/v1")
    ap.add_argument("--max-tool-calls", type=int, default=60)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--time-minutes", type=float, default=20.0)
    ap.add_argument("--phase1-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    tasks = [json.loads(l) for l in (base / args.tasks).read_text().splitlines()]
    task = next((t for t in tasks if t["instance_id"] == args.task_id), None)
    if task is None:
        print(f"task not found: {args.task_id}")
        return 2
    root = (base / args.submission).resolve()
    cfg_path = root / "agent.yaml"
    snapshot = (base / args.snapshots) / f"{task['instance_id']}.tgz"
    if not snapshot.is_file():
        print(f"snapshot missing: {snapshot}")
        return 2
    workdir = Path(args.workdir) / args.task_id
    workdir.mkdir(parents=True, exist_ok=True)
    ws = workdir / "workspace"
    print(f"[setup] extracting {snapshot.name} -> {ws}")
    extract_snapshot(snapshot, ws)
    git_baseline(ws)

    repo_key = task["repo"].split("/")[-1]
    gpath = (base / args.graphs) / f"{repo_key}_{task['base_commit']}.json"
    epath = (base / args.embeddings) / f"{repo_key}_{task['base_commit']}.npz"
    ctx = Ctx(workspace=ws, repo=task["repo"], repo_dir=repo_key,
              graph_path=gpath if gpath.is_file() else None,
              embedding_path=epath if epath.is_file() else None,
              max_tool_calls=args.max_tool_calls, max_turns=args.max_turns,
              time_minutes=args.time_minutes)
    print(f"[setup] graph={gpath.is_file()} emb={epath.is_file()}")

    res = asyncio.run(run_phase1(cfg_path, root, task, ws, args.model,
                                 args.api_base, ctx, verbose=not args.quiet))
    patch = res["patch"]
    outdir = base / "local_results" / args.task_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "agent.patch").write_text(patch)
    if not args.quiet:
        print(f"[phase1] submitted={res['patch_submitted']} turns={res['turns']} "
              f"tool_calls={res['tool_calls_used']} elapsed={res['elapsed']}s "
              f"patch_bytes={len(patch)}")
        print(f"[phase1] last_text: {res['last_text'][:1000]}")
    summary: dict[str, Any] = {"task": args.task_id, "phase1": {
        k: res[k] for k in ("turns", "tool_calls_used", "patch_submitted", "elapsed")}}
    if not args.phase1_only:
        v = run_phase2(task, snapshot, patch, root, workdir,
                       verbose=not args.quiet)
        summary["phase2"] = {k: v[k] for k in v if k != "output"}
        if v.get("output") and not args.quiet:
            print("[phase2] output tail:\n" + v["output"][-3000:])
        if not args.quiet:
            print(f"[phase2] resolved={v.get('resolved')} targets={v.get('targets')} "
                  f"names={v.get('names')}")
        (outdir / "phase2.json").write_text(json.dumps(v, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
