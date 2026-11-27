#!/usr/bin/env python3
"""Synthesize role-specific SFT data for the generic LoopAgent orchestra.

For each training task we build native Gemma-4 chat examples for:
  investigator -> implementer -> test_runner -> test_checker

The tool calls and tool responses are produced by the same local tool
implementations used by tools/local_harness.py, so the trajectories are real.

Usage:
  python sft/synthesize_role_data.py --limit 5
  python sft/synthesize_role_data.py --task-ids requests_6592 requests_7309
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import local_harness as lh  # noqa: E402

PROMPT_DIR = ROOT / "submission" / "prompts"

# 25 public tasks held out from the nexaaii split; keep them out of training.
HELD_OUT = {
    "fastapi_15030", "fastapi_14962", "fastapi_14978", "rich_4070", "fastapi_14099",
    "fastapi_14609", "fastapi_14306", "fastapi_14459", "fastapi_14448", "fastapi_14419",
    "fastapi_14262", "fastapi_14186", "fastapi_14077", "rich_3930", "httpx_3672",
    "fastapi_15800", "fastapi_15785", "fastapi_14371", "fastapi_14605", "fastapi_15745",
    "fastapi_14953", "fastapi_13786", "rich_3486", "rich_3052", "rich_3061",
}

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


# ---------------------------------------------------------------------------
# Native Gemma-4 formatting
# ---------------------------------------------------------------------------
def _fmt_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'<|"|>{v}<|"|>'
    return f'<|"|>{json.dumps(v, ensure_ascii=False)}<|"|>'


def thought(text: str) -> str:
    return f"<|channel>thought\n{text}<channel|>"


def tool_call(name: str, args: dict[str, Any]) -> str:
    inner = ",".join(f"{k}:{_fmt_val(v)}" for k, v in args.items())
    return f"<|tool_call>call:{name}{{{inner}}}<tool_call|>"


def tool_response(name: str, response: str) -> str:
    safe = response.replace("<|" , "<\\|")
    return (f"<|tool_response>response:{name}{{value:<|\"|>{safe}<|\"|>}}"
            f"<tool_response|>")


def assistant_text(steps: list[tuple[str, str, dict[str, Any], str]], final: str) -> str:
    parts: list[str] = []
    for th, name, args, resp in steps:
        parts.append(thought(th))
        parts.append(tool_call(name, args))
        parts.append(tool_response(name, resp))
    parts.append(final)
    return "".join(parts)


def render(system: str, user: str, assistant: str) -> tuple[str, str]:
    prompt = (f"<bos><|turn>system\n<|think|>\n{system}<turn|>\n"
              f"<|turn>user\n{user}<turn|>\n<|turn>model\n")
    return prompt, prompt + assistant + "<turn|>"


def task_prompt(task: dict[str, Any]) -> str:
    return (
        f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
        "Problem Statement:\n"
        f"{task['problem_statement'].strip()}\n\n"
        "## Execution Environment Rules\n"
        "- Single command timeout: 300 seconds\n"
        "- Command output limit: 5000 characters\n"
        "- File view limit: 150 lines per read_file call, 10000 characters\n"
        "- Environment is offline. All repository dependencies are pre-installed.\n"
        "- Work strictly under /workspace. Put scratch files in /tmp, never in /workspace.\n"
        "- Never modify tests or test-runner configuration.\n"
    )


# ---------------------------------------------------------------------------
# Patch parsing
# ---------------------------------------------------------------------------
def parse_hunks(patch: str) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    cur_file = ""
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[len("+++ b/"):]
            continue
        m = HUNK_RE.match(line)
        if m:
            cur = {
                "file": cur_file,
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or 1),
                "context": (m.group(5) or "").strip(),
                "lines": [],
            }
            hunks.append(cur)
            continue
        if cur is not None and line[:1] in (" ", "+", "-"):
            cur["lines"].append(line)
    return [h for h in hunks if h["file"] and h["lines"]]


def hunk_old_new(h: dict[str, Any]) -> tuple[str, str]:
    old: list[str] = []
    new: list[str] = []
    for line in h["lines"]:
        if line.startswith(" "):
            old.append(line[1:])
            new.append(line[1:])
        elif line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
    return "\n".join(old), "\n".join(new)


def parse_test_patch(test_patch: str) -> tuple[list[str], list[str]]:
    files: set[str] = set()
    names: set[str] = set()
    for line in test_patch.splitlines():
        if line.startswith("--- a/"):
            files.add(line[len("--- a/"):])
        elif line.startswith("+++ b/"):
            files.add(line[len("+++ b/"):])
        m = re.match(r"^\+\s*(?:async\s+)?def (test_[A-Za-z0-9_]+)", line)
        if m:
            names.add(m.group(1))
    return sorted(f for f in files if f.endswith(".py")), sorted(names)


def enclosing_symbol(text: str, line: int) -> str:
    try:
        tree = ast.parse(text)
    except Exception:
        return ""

    def walk(node: ast.AST, prefix: str) -> str:
        best = ""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                end = getattr(child, "end_lineno", child.lineno)
                if child.lineno <= line <= end:
                    name = f"{prefix}{child.name}"
                    best = walk(child, name + ".") or name
        return best or prefix

    if isinstance(tree, ast.Module):
        return walk(tree, "")
    return ""


def pick_literals(hunks: list[dict[str, Any]], problem: str) -> list[str]:
    cands: list[str] = []
    for h in hunks:
        for line in h["lines"]:
            if line.startswith(" "):
                continue
            body = line[1:]
            for m in re.finditer(r"""["']([^"']{3,60})["']""", body):
                cands.append(m.group(1))
            for m in re.finditer(r"\b\d{2,6}\b", body):
                cands.append(m.group(0))
            for m in re.finditer(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", body):
                cands.append(m.group(0))
    # de-dup, prefer literals that appear in the problem statement
    seen: set[str] = set()
    uniq = [c for c in cands if not (c in seen or seen.add(c))]
    ps = problem.lower()
    first = [c for c in uniq if c.lower() in ps]
    return first + [c for c in uniq if c not in first]


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------
def fresh_ctx(ws: Path, repo: str) -> lh.Ctx:
    ctx = lh.Ctx(workspace=ws, repo=repo, repo_dir=repo.split("/")[-1],
                 max_tool_calls=500, max_turns=500, time_minutes=600)
    lh.CTX = ctx
    return ctx


def apply_diff(ws: Path, diff: str, name: str) -> bool:
    if not diff.strip():
        return True
    pf = ws.parent / f"_{name}.patch"
    pf.write_text(diff if diff.endswith("\n") else diff + "\n")
    for cmd in (
        ["git", "apply", "-p1", str(pf)],
        ["git", "apply", "-p1", "--ignore-space-change", "--ignore-whitespace", str(pf)],
        ["patch", "-p1", "--batch", "--forward", "-i", str(pf)],
    ):
        r = subprocess.run(cmd, cwd=str(ws), capture_output=True, text=True)
        if r.returncode == 0:
            return True
    return False


def reset_workspace(ws: Path) -> None:
    subprocess.run(["git", "checkout", "--", "."], cwd=str(ws),
                   capture_output=True, text=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(ws),
                   capture_output=True, text=True)


def revert_test_files(ws: Path, files: list[str]) -> None:
    if not files:
        return
    subprocess.run(["git", "checkout", "--", *files], cwd=str(ws),
                   capture_output=True, text=True)
    subprocess.run(["git", "clean", "-f", "--", *files], cwd=str(ws),
                   capture_output=True, text=True)


def run_cmd(ws: Path, command: str, timeout: int = 60) -> tuple[int, str]:
    ctx = fresh_ctx(ws, "")
    ctx.timeout_seconds = timeout
    lh.CTX = ctx
    resp = lh.run_command(command)
    try:
        d = json.loads(resp)
    except Exception:
        return 1, resp
    if d.get("status") == "ok":
        return 0, resp
    return int((d.get("details") or {}).get("exit_code", 1)), resp


def run_tool(ws: Path, name: str, **kwargs: Any) -> str:
    fresh_ctx(ws, "")
    fn = lh.TOOL_FUNCS[name]
    return fn(**kwargs)


def unsafe_run(exit_code: int, resp: str) -> bool:
    if exit_code in (124, 137, 152):
        return True
    low = resp.lower()
    return any(s in low for s in (
        "memoryerror", "out of memory", "killed", "timeoutexceeded",
        "cannot allocate", "sigxcpu", "sigkill",
    ))


def choose_command(ws: Path, task: dict[str, Any], test_files: list[str],
                   test_names: list[str]) -> tuple[str, bool]:
    if not test_files:
        return "python3 -c \"print('no test command available')\"", False

    # Names of added tests come from the patch; names of modified tests come from
    # the old side of hunks that actually remove lines.
    names: set[str] = set(test_names)
    for h in parse_hunks(task["test_patch"]):
        if h["file"] not in test_files:
            continue
        if not any(line.startswith("-") for line in h["lines"]):
            continue
        p = ws / h["file"]
        if not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        lo = h["old_start"]
        hi = lo + max(1, h["old_count"]) - 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_end = getattr(node, "end_lineno", node.lineno)
                if node.lineno <= hi and node_end >= lo:
                    names.add(node.name)

    if not apply_diff(ws, task["test_patch"], "test_choose"):
        return f"python3 -m pytest {' '.join(test_files)} -q -x --tb=short", False

    files = " ".join(test_files)
    flags = "-p no:asyncio -p no:anyio --junitxml=/tmp/result.xml -ra"
    candidates: list[str] = []
    if names:
        candidates.append(
            f'python3 -m pytest {files} -q -x --tb=short {flags} -k "{" or ".join(sorted(names))}"')
    candidates.append(f"python3 -m pytest {files} -q -x --tb=short {flags}")

    chosen = None
    for cmd in candidates:
        exit_code, resp = run_cmd(ws, cmd)
        if exit_code == 0 and not unsafe_run(exit_code, resp):
            chosen = cmd
            break
    revert_test_files(ws, test_files)
    reset_workspace(ws)
    if chosen:
        return chosen, True
    return candidates[0], False


def build_investigator(task: dict[str, Any], ws: Path, hunks: list[dict[str, Any]]) -> dict[str, Any]:
    problem = task["problem_statement"]
    first = hunks[0]
    target_file = first["file"]
    line = first["old_start"]
    lines = (ws / target_file).read_text(encoding="utf-8", errors="replace").splitlines()
    symbol = enclosing_symbol("\n".join(lines), line)

    steps: list[tuple[str, str, dict[str, Any], str]] = []
    literals = pick_literals(hunks, problem)[:6]
    found = False
    for lit in literals:
        cmd = f"grep -rn {shlex.quote(lit)} --include=*.py ."
        resp = run_tool(ws, "run_command", command=cmd)
        try:
            d = json.loads(resp)
            out = (d.get("stdout") or "")
        except Exception:
            out = ""
        if target_file in out:
            steps.append(("Find the distinctive literal in the source.", "run_command",
                          {"command": cmd}, resp))
            found = True
            break
    if not found and literals:
        cmd = f"grep -rn {shlex.quote(literals[0])} --include=*.py ."
        resp = run_tool(ws, "run_command", command=cmd)
        steps.append(("Find the distinctive literal in the source.", "run_command",
                      {"command": cmd}, resp))

    start = max(1, line - 6)
    end = min(len(lines), line + first["old_count"] + 4)
    rf = run_tool(ws, "read_file", filepath=target_file, start_line=start, end_line=end)
    steps.append(("Read the exact target region.", "read_file",
                  {"filepath": target_file, "start_line": start, "end_line": end}, rf))

    fix = problem.strip().splitlines()[0][:160]
    report = (f"FILE: {target_file}\n"
              f"LINES: {line}-{max(line, line + first['old_count'] - 1)}\n"
              f"FUNCTION: {symbol or first['context'] or 'module'}\n"
              f"FIX: {fix}\n"
              f"CONFIDENCE: high")
    system = (PROMPT_DIR / "investigator.md").read_text()
    user = task_prompt(task)
    prompt, text = render(system, user, assistant_text(steps, report))
    return {"id": task["instance_id"], "role": "investigator", "text": text,
            "prompt": prompt, "meta": {"file": target_file, "lines": [line, end]}}


def build_implementer(task: dict[str, Any], ws: Path, hunks: list[dict[str, Any]],
                      investigator_report: str, test_files: list[str],
                      test_names: list[str]) -> dict[str, Any] | None:
    steps: list[tuple[str, str, dict[str, Any], str]] = []
    changed_files: list[str] = []
    for h in hunks:
        f = h["file"]
        old, new = hunk_old_new(h)
        if not old or old == new:
            continue
        lines = (ws / f).read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, h["old_start"] - 3)
        end = min(len(lines), h["old_start"] + h["old_count"] + 3)
        rf = run_tool(ws, "read_file", filepath=f, start_line=start, end_line=end)
        steps.append(("Read the target region.", "read_file",
                      {"filepath": f, "start_line": start, "end_line": end}, rf))
        resp = run_tool(ws, "edit_file", filepath=f, old_string=old, new_string=new)
        try:
            ok = json.loads(resp).get("status") == "ok"
        except Exception:
            ok = False
        if not ok:
            return None
        steps.append(("Apply the minimal fix.", "edit_file",
                      {"filepath": f, "old_string": old, "new_string": new}, resp))
        changed_files.append(f)

    if not changed_files:
        return None

    for f in changed_files:
        cmd = f'python3 -c "import ast; ast.parse(open({f!r}).read())"'
        resp = run_tool(ws, "run_command", command=cmd)
        steps.append((f"Parse-check {f}.", "run_command", {"command": cmd}, resp))

    diff_cmd = "git diff --stat -- " + " ".join(shlex.quote(f) for f in changed_files)
    diff_resp = run_tool(ws, "run_command", command=diff_cmd)
    steps.append(("Confirm the source diff is non-empty.", "run_command",
                  {"command": diff_cmd}, diff_resp))

    command, positive_ready = choose_command(ws, task, test_files, test_names)

    spec = task["problem_statement"].strip().splitlines()[0][:200]
    report = (f"SPEC: {spec}\n"
              f"FILE: {', '.join(changed_files)}\n"
              f"COMMAND: {command}\n"
              f"NOTES: Minimal source edit applied; verify with the command above.")
    system = (PROMPT_DIR / "implementer.md").read_text().replace(
        "{investigator_report}", investigator_report)
    user = task_prompt(task)
    prompt, text = render(system, user, assistant_text(steps, report))
    return {"id": task["instance_id"], "role": "implementer", "text": text,
            "prompt": prompt, "meta": {"files": changed_files, "command": command,
                                       "positive_ready": positive_ready}}


def build_runner_checker(task: dict[str, Any], ws: Path, hunks: list[dict[str, Any]],
                         implementer_meta: dict[str, Any], test_files: list[str],
                         test_names: list[str], source_files: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    command = implementer_meta["command"]
    if not implementer_meta.get("positive_ready"):
        return None, None, None
    diff_cmd = "git diff --stat -- " + " ".join(shlex.quote(f) for f in source_files)
    impl_report = implementer_meta["report"]

    # Negative: base + test_patch -> command must fail, source diff empty.
    neg_runner_resp = None
    if apply_diff(ws, task["test_patch"], "test_neg"):
        exit_code, run_resp = run_cmd(ws, command)
        diff_resp = run_tool(ws, "run_command", command=diff_cmd)
        if exit_code != 0 and not unsafe_run(exit_code, run_resp):
            neg_runner_resp = (exit_code, run_resp, diff_resp)
        revert_test_files(ws, test_files)
    reset_workspace(ws)

    # Positive: gold source + test_patch -> command passes; revert tests before diff.
    if not apply_diff(ws, task["patch"], "gold_pos"):
        return None, None, None
    if not apply_diff(ws, task["test_patch"], "test_pos"):
        revert_test_files(ws, test_files)
        return None, None, None
    exit_code, run_resp = run_cmd(ws, command)
    if exit_code != 0 or unsafe_run(exit_code, run_resp):
        revert_test_files(ws, test_files)
        return None, None, None
    revert_test_files(ws, test_files)
    diff_resp = run_tool(ws, "run_command", command=diff_cmd)

    runner_steps = [
        ("Run the verification command from the report.", "run_command",
         {"command": command}, run_resp),
        ("Record the source diff.", "run_command", {"command": diff_cmd}, diff_resp),
    ]
    runner_final = (f"COMMAND: {command}\n"
                    f"EXIT: {exit_code}\n"
                    f"OUTPUT: {run_resp[:3000]}\n"
                    f"DIFF: {diff_resp}")
    runner_system = (PROMPT_DIR / "test_runner.md").read_text().replace(
        "{implementer_report}", impl_report)
    prompt, text = render(runner_system, task_prompt(task),
                          assistant_text(runner_steps, runner_final))
    runner_ex = {"id": task["instance_id"], "role": "test_runner", "text": text,
                 "prompt": prompt, "meta": {"command": command, "exit": exit_code}}

    # Positive checker
    submit_resp = ""
    if exit_code == 0:
        fresh_ctx(ws, task["repo"])
        submit_resp = lh.submit_patch()
    chk_steps = [
        ("Re-run the command and inspect the diff.", "run_command",
         {"command": command}, run_resp),
        ("Confirm the source diff.", "run_command", {"command": diff_cmd}, diff_resp),
    ]
    if exit_code == 0:
        chk_steps.append(("The command passed and the diff is non-empty; submit.",
                          "submit_patch", {}, submit_resp))
        chk_final = (f"STATUS: PASS\nSUBMITTED: yes\nEXIT: {exit_code}\n"
                     f"DIFF: {diff_resp}\nEVIDENCE: command exited 0\nNEXT FIX: NONE")
    else:
        chk_final = (f"STATUS: FAIL\nSUBMITTED: no\nEXIT: {exit_code}\n"
                     f"DIFF: {diff_resp}\nEVIDENCE: command did not exit 0\n"
                     f"NEXT FIX: apply the source fix and re-run the command")
    chk_system = ((PROMPT_DIR / "test_checker.md").read_text()
                  .replace("{implementer_report}", impl_report)
                  .replace("{runner_report}", runner_final))
    prompt, text = render(chk_system, task_prompt(task),
                          assistant_text(chk_steps, chk_final))
    checker_ex = {"id": task["instance_id"], "role": "test_checker", "text": text,
                  "prompt": prompt, "meta": {"exit": exit_code, "submitted": exit_code == 0}}

    # Negative checker from the negative runner run
    neg_checker = None
    if neg_runner_resp is not None:
        n_exit, n_run, n_diff = neg_runner_resp
        n_runner_final = (f"COMMAND: {command}\nEXIT: {n_exit}\n"
                          f"OUTPUT: {n_run[:3000]}\nDIFF: {n_diff}")
        n_steps = [
            ("Re-run the command before the fix and inspect the diff.", "run_command",
             {"command": command}, n_run),
            ("Confirm the source diff.", "run_command", {"command": diff_cmd}, n_diff),
        ]
        n_final = (f"STATUS: FAIL\nSUBMITTED: no\nEXIT: {n_exit}\n"
                   f"DIFF: {n_diff}\nEVIDENCE: command failed before the fix\n"
                   f"NEXT FIX: apply the source fix and re-run the command")
        n_system = ((PROMPT_DIR / "test_checker.md").read_text()
                    .replace("{implementer_report}", impl_report)
                    .replace("{runner_report}", n_runner_final))
        prompt, text = render(n_system, task_prompt(task),
                              assistant_text(n_steps, n_final))
        neg_checker = {"id": task["instance_id"] + "_neg", "role": "test_checker",
                       "text": text, "prompt": prompt,
                       "meta": {"exit": n_exit, "submitted": False}}
    return runner_ex, checker_ex, neg_checker


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "tasks.jsonl"))
    ap.add_argument("--snapshots", default=str(ROOT / "snapshots"))
    ap.add_argument("--out", default=str(ROOT / "sft" / "data"))
    ap.add_argument("--workdir", default="/tmp/sft_ws")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--task-ids", nargs="*", default=None)
    ap.add_argument("--repos", nargs="*", default=None)
    ap.add_argument("--keep-workspaces", action="store_true")
    args = ap.parse_args()

    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines()]
    if args.task_ids:
        wanted = set(args.task_ids)
        tasks = [t for t in tasks if t["instance_id"] in wanted]
    else:
        tasks = [t for t in tasks if t["instance_id"] not in HELD_OUT]
        if args.repos:
            wanted = set(args.repos)
            tasks = [t for t in tasks
                     if t["repo"] in wanted or t["repo"].split("/")[-1] in wanted]
    if args.offset:
        tasks = tasks[args.offset:]
    if args.limit:
        tasks = tasks[:args.limit]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    files = {r: (out / f"{r}.jsonl").open(mode, encoding="utf-8")
             for r in ("investigator", "implementer", "test_runner", "test_checker")}
    counts = {r: 0 for r in files}
    skipped: list[dict[str, str]] = []
    workroot = Path(args.workdir)
    workroot.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(tasks, 1):
        tid = task["instance_id"]
        snapshot = Path(args.snapshots) / f"{tid}.tgz"
        if not snapshot.is_file():
            skipped.append({"id": tid, "reason": "missing snapshot"})
            continue
        ws = workroot / tid / "workspace"
        print(f"[{i}/{len(tasks)}] {tid}", flush=True)
        try:
            lh.extract_snapshot(snapshot, ws)
            lh.git_baseline(ws)
            fresh_ctx(ws, task["repo"])
            hunks = parse_hunks(task["patch"])
            if not hunks:
                skipped.append({"id": tid, "reason": "no source hunks"})
                continue
            test_files, test_names = parse_test_patch(task["test_patch"])
            source_files = sorted({h["file"] for h in hunks})

            inv = build_investigator(task, ws, hunks)
            files["investigator"].write(json.dumps(inv) + "\n")
            counts["investigator"] += 1

            inv_report = inv["text"].rsplit("<|turn>model\n", 1)[-1].rsplit("<turn|>", 1)[0]
            inv_report = re.sub(r"<\|channel>thought.*?<channel\|>", "", inv_report, flags=re.S)
            inv_report = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", inv_report, flags=re.S)
            inv_report = re.sub(r"<\|tool_response>.*?<tool_response\|>", "", inv_report, flags=re.S)
            inv_report = inv_report.strip()

            reset_workspace(ws)
            impl = build_implementer(task, ws, hunks, inv_report, test_files, test_names)
            if impl is not None:
                files["implementer"].write(json.dumps(impl) + "\n")
                counts["implementer"] += 1
                impl_meta = dict(impl["meta"])
                impl_meta["report"] = impl["text"].rsplit("<|turn>model\n", 1)[-1].rsplit("<turn|>", 1)[0]
                impl_meta["report"] = re.sub(r"<\|channel>thought.*?<channel\|>", "", impl_meta["report"], flags=re.S)
                impl_meta["report"] = re.sub(r"<\|tool_call>.*?<tool_call\|>", "", impl_meta["report"], flags=re.S)
                impl_meta["report"] = re.sub(r"<\|tool_response>.*?<tool_response\|>", "", impl_meta["report"], flags=re.S)
                impl_meta["report"] = impl_meta["report"].strip()

                reset_workspace(ws)
                runner, checker, neg = build_runner_checker(
                    task, ws, hunks, impl_meta, test_files, test_names, source_files)
                if runner is not None:
                    files["test_runner"].write(json.dumps(runner) + "\n")
                    counts["test_runner"] += 1
                if checker is not None:
                    files["test_checker"].write(json.dumps(checker) + "\n")
                    counts["test_checker"] += 1
                if neg is not None:
                    files["test_checker"].write(json.dumps(neg) + "\n")
                    counts["test_checker"] += 1
            else:
                skipped.append({"id": tid, "reason": "implementer edit failed"})
        except Exception as exc:  # noqa: BLE001
            skipped.append({"id": tid, "reason": f"{type(exc).__name__}: {exc}"})
        finally:
            if not args.keep_workspaces:
                shutil.rmtree(workroot / tid, ignore_errors=True)

    for f in files.values():
        f.close()
    manifest = {"tasks": len(tasks), "counts": counts, "skipped": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
