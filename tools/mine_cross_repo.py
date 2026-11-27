#!/usr/bin/env python3
"""Mine unrelated public Python repos into SWE-style training tasks.

For each selected commit that modifies source and tests:
  - base = parent commit
  - patch = source diff
  - test_patch = test diff
  - problem_statement = commit message
  - snapshot = git archive of the parent tree (no .git; harness initializes it)
  - graph + embeddings for the parent tree

Usage:
  python tools/mine_cross_repo.py --max-per-repo 6
  python tools/mine_cross_repo.py --repos pallets/click pallets/itsdangerous \
      --out-dir sft/cross_repo --max-per-repo 10
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_REPOS = [
    "pallets/click",
    "pallets/itsdangerous",
    "pallets/markupsafe",
    "python-attrs/attrs",
    "marshmallow-code/marshmallow",
    "astanin/python-tabulate",
    "pytest-dev/pluggy",
    "alecthomas/voluptuous",
    "bottlepy/bottle",
]
EXCLUDE_PARTS = {
    ".git", "tests", "test", "docs", "docs_src", "examples", "benchmarks",
    "scripts", "build", "dist", ".tox", ".venv", "venv", "node_modules",
}
TEST_RE = re.compile(r"(^|/)test_[^/]*\.py$|(^|/)[^/]*_test\.py$")
SKIP_SUBJECT_RE = re.compile(
    r"(?i)\b(merge|revert|wip|release|bump|changelog|version|typo|"
    r"pre-commit|dependabot|lint|dev dependencies|pyright|python versions|"
    r"check types|drop end of life|typing|mypy|ruff|ci|docs?|documentation)\b")


def run(cmd: list[str], cwd: Path | None = None, check: bool = False) -> str:
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))}\n{r.stderr[:1000]}")
    return r.stdout


def repo_short(repo: str) -> str:
    return repo.split("/")[-1]


def is_test(f: str) -> bool:
    return f.startswith("tests/") or f.startswith("test/") or bool(TEST_RE.search(f))


def is_source(f: str) -> bool:
    if not f.endswith(".py") or is_test(f):
        return False
    if set(Path(f).parts) & EXCLUDE_PARTS:
        return False
    return Path(f).name not in {"setup.py", "conftest.py", "noxfile.py"}


def clone(repo: str, clone_dir: Path) -> Path:
    dest = clone_dir / repo.replace("/", "__")
    if dest.exists():
        run(["git", "fetch", "--quiet", "--all"], cwd=dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(dest)],
            check=True)
    return dest


def list_commits(repo_dir: Path) -> list[dict[str, Any]]:
    out = run(["git", "log", "--no-merges", "--date=short",
               "--format=@@@%H\x1f%ad\x1f%s", "--name-only"], cwd=repo_dir)
    commits: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in out.splitlines():
        if line.startswith("@@@"):
            meta = line[3:].split("\x1f")
            if len(meta) < 3:
                continue
            cur = {"hash": meta[0], "date": meta[1], "subject": meta[2], "files": []}
            commits.append(cur)
        elif cur is not None and line.strip():
            cur["files"].append(line.strip())
    return commits


def name_status(repo_dir: Path, base: str, commit: str, files: list[str]) -> dict[str, str]:
    out = run(["git", "diff", "--name-status", base, commit, "--", *files],
              cwd=repo_dir)
    result: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            result[parts[-1]] = parts[0][:1]
    return result


def diff_line_count(diff: str) -> int:
    return sum(1 for line in diff.splitlines()
               if (line.startswith("+") or line.startswith("-"))
               and not line.startswith("+++") and not line.startswith("---"))


# ---------------------------------------------------------------------------
# Graph/embedding generation
# ---------------------------------------------------------------------------
def build_graph(workspace: Path, short: str, base: str,
                graph_dir: Path, emb_dir: Path) -> None:
    nodes: list[dict[str, str]] = []
    node_ast: dict[str, ast.AST] = {}
    for path in workspace.rglob("*.py"):
        rel = path.relative_to(workspace)
        if set(rel.parts) & EXCLUDE_PARTS:
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue
        mod = ".".join(rel.with_suffix("").parts)

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{prefix}.{child.name}" if prefix else child.name
                    full = f"{mod}.{name}"
                    text = ast.get_source_segment(src, child) or ""
                    nodes.append({"id": full, "name": full, "text": text})
                    node_ast[full] = child
                    visit(child, name)

        visit(tree, "")

    edges: list[dict[str, Any]] = []
    for full, node in node_ast.items():
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            fn = child.func
            name = (fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute) else None)
            if not name:
                continue
            for target in node_ast:
                if target == name or target.endswith("." + name):
                    edges.append({"type": "calls", "source": full,
                                  "target": target, "key": 0})
                    break

    graph_dir.mkdir(parents=True, exist_ok=True)
    data = {"directed": True, "multigraph": True,
            "graph": {"repo_name": short}, "nodes": nodes, "edges": edges}
    (graph_dir / f"{short}_{base}.json").write_text(json.dumps(data))

    emb_dir.mkdir(parents=True, exist_ok=True)
    vectors: dict[str, np.ndarray] = {}
    for n in nodes:
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", n["text"].lower())
        vec = np.zeros(256, dtype=np.float32)
        for t in toks:
            vec[int(hashlib.sha1(t.encode()).hexdigest(), 16) % 256] += 1.0
        norm = float(np.linalg.norm(vec)) or 1.0
        vectors[n["id"]] = vec / norm
    np.savez_compressed(emb_dir / f"{short}_{base}.npz", **vectors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--out-dir", default="sft/cross_repo")
    ap.add_argument("--clone-dir", default="/tmp/cross_repo_clones")
    ap.add_argument("--max-per-repo", type=int, default=8)
    ap.add_argument("--min-year", type=int, default=2018)
    ap.add_argument("--max-files", type=int, default=8)
    ap.add_argument("--max-source-lines", type=int, default=160)
    ap.add_argument("--max-test-lines", type=int, default=160)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    snap_dir = out / "snapshots"
    graph_dir = out / "graphs"
    emb_dir = out / "embeddings"
    for d in (snap_dir, graph_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)
    clone_dir = Path(args.clone_dir)
    clone_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = out / "tasks.jsonl"
    existing: set[str] = set()
    if tasks_path.is_file():
        for line in tasks_path.read_text().splitlines():
            try:
                existing.add(json.loads(line)["instance_id"])
            except Exception:
                pass

    manifest: dict[str, Any] = {"repos": {}, "total": 0, "skipped": []}
    with tasks_path.open("a", encoding="utf-8") as fout:
        for repo in args.repos:
            short = repo_short(repo)
            print(f"== {repo}")
            try:
                repo_dir = clone(repo, clone_dir)
            except Exception as exc:
                print(f"  clone failed: {exc}")
                manifest["skipped"].append({"repo": repo, "reason": f"clone: {exc}"})
                continue
            commits = list_commits(repo_dir)
            picked = 0
            for c in commits:
                if picked >= args.max_per_repo:
                    break
                try:
                    year = int(c["date"][:4])
                except Exception:
                    continue
                if year < args.min_year:
                    continue
                if not c["subject"] or SKIP_SUBJECT_RE.search(c["subject"]):
                    continue
                files = c["files"]
                if len(files) > args.max_files:
                    continue
                src = [f for f in files if is_source(f)]
                tests = [f for f in files if is_test(f)]
                if not src or not tests:
                    continue
                commit = c["hash"]
                base = run(["git", "rev-parse", f"{commit}^"], cwd=repo_dir).strip()
                if not base:
                    continue
                task_id = f"{short}_{commit[:8]}"
                if task_id in existing:
                    continue
                status = name_status(repo_dir, base, commit, src + tests)
                if any(status.get(f) != "M" for f in src):
                    continue
                src_diff = run(["git", "diff", "--no-color", base, commit, "--", *src],
                               cwd=repo_dir)
                test_diff = run(["git", "diff", "--no-color", base, commit, "--", *tests],
                                cwd=repo_dir)
                if not src_diff or not test_diff:
                    continue
                if diff_line_count(src_diff) > args.max_source_lines:
                    continue
                if diff_line_count(test_diff) > args.max_test_lines:
                    continue

                tmp = clone_dir / f"tmp_{short}_{commit[:8]}"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                archive = subprocess.run(
                    f"git -C {repo_dir} archive {base} | tar -x -C {tmp}",
                    shell=True, capture_output=True, text=True)
                if archive.returncode != 0:
                    shutil.rmtree(tmp, ignore_errors=True)
                    continue
                snap = snap_dir / f"{task_id}.tgz"
                subprocess.run(["tar", "-czf", str(snap), "-C", str(tmp), "."],
                               check=True, capture_output=True)
                if not args.no_graphs:
                    try:
                        build_graph(tmp, short, base, graph_dir, emb_dir)
                    except Exception as exc:
                        print(f"  graph failed for {task_id}: {exc}")
                shutil.rmtree(tmp, ignore_errors=True)

                problem = run(["git", "show", "-s", "--format=%B", commit],
                              cwd=repo_dir).strip() or c["subject"]
                created = run(["git", "show", "-s", "--format=%cI", commit],
                              cwd=repo_dir).strip()
                task = {
                    "instance_id": task_id,
                    "repo": repo,
                    "base_commit": base,
                    "patch": src_diff,
                    "test_patch": test_diff,
                    "problem_statement": problem[:8000],
                    "hints_text": "",
                    "created_at": created,
                }
                fout.write(json.dumps(task) + "\n")
                fout.flush()
                existing.add(task_id)
                picked += 1
                manifest["total"] += 1
                print(f"  + {task_id} {c['subject'][:70]}")
            manifest["repos"][repo] = picked

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
