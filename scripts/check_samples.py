#!/usr/bin/env python3
"""Run the code samples of the documentation (docs/) and lint the markdown.

    python scripts/check_samples.py                    # every page
    python scripts/check_samples.py guide/sources.md   # some pages
    python scripts/check_samples.py --list             # show the blocks, run nothing
    python scripts/check_samples.py --no-run           # lint only
    python scripts/check_samples.py --outputs          # also compare the quoted outputs

What is checked

* ``python`` blocks are executed. The blocks of one page run in order in one
  namespace (a page is one session, so a worked example can span several
  blocks), each page in its own process and in an empty temporary directory.
* ``json`` blocks must parse; a block that is a whole simulation spec (it has
  ``region`` and ``mesh``) is also given to the core, which builds its grid.
* ``bash``, ``rust`` and ``text`` blocks are not executed. Signatures and
  program output therefore go in ``text`` blocks.
* ``reference/python-api.md`` is generated: it is compared with a fresh run
  of ``gen_api.py`` instead.
* With ``--outputs``, what a block prints is compared with the ``text`` block
  that directly follows it on the page, and differences are listed. They are
  not failures: thread counts, timings and device names depend on the machine.
* Lint: one ``# Title`` on the first line, a language on every fence, no HTML,
  relative links and images that resolve, and ``site.json`` listing every
  page exactly once.

Skipping. A block is skipped, not failed, when it needs something this
machine does not have. The needs are read from the code itself:

* ``import gdsfactory`` / ``klayout`` / ``torch`` / ``jax``  -> that package,
* ``lightsim.invdes``                             -> a build that has the module,
* a device string ``"cuda:0"``, ``"cuda:0,1"``    -> that many GPUs,
* ``shutil.which("lightsim")``                    -> the command-line program
  (on ``PATH``, or built in ``target/release`` of this repository).

A later block that uses a name defined only in a skipped block is skipped as
well. Everything else must run. The exit status is 1 if a block or a lint
rule fails.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
# a program built in this repository counts as installed
os.environ["PATH"] = os.pathsep.join([str(ROOT / "target" / "release"), os.environ.get("PATH", "")])
GENERATED = {"reference/python-api.md"}
LANGUAGES = {"python", "bash", "json", "rust", "text"}
OPTIONAL_PACKAGES = ("gdsfactory", "klayout", "torch", "jax")
FENCE = re.compile(r"^(`{3,})\s*([^`\s]*)\s*$")


# ----------------------------------------------------------------------------
# markdown
# ----------------------------------------------------------------------------


def blocks(text: str):
    """Fenced blocks of a page: (language, first line number, code)."""
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        fence, lang, start = m.group(1), m.group(2), i + 1
        j = start
        while j < len(lines) and lines[j].strip() != fence:
            j += 1
        out.append((lang, start + 1, "\n".join(lines[start:j])))
        i = j + 1
    return out


def quoted_outputs(text: str) -> dict:
    """{first line of a python block: the ``text`` block that directly follows it}."""
    out, lines, prev = {}, text.split("\n"), None
    for lang, line, code in blocks(text):
        if lang == "text" and prev is not None:
            between = lines[prev[1] : line - 2]  # after the closing fence of the python block
            if not any(l.strip() for l in between):
                out[prev[0]] = code
        end = line - 1 + len(code.split("\n")) + 1  # index after the closing fence
        prev = (line, end) if lang == "python" else None
    return out


def prose(text: str) -> str:
    """The page without fenced blocks and inline code (for the lint rules)."""
    keep, inside = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            keep.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(keep)


def lint(pages) -> list[str]:
    errors = []
    for rel in pages:
        text = (DOCS / rel).read_text()
        if not re.match(r"# \S", text):
            errors.append(f"{rel}: the first line must be '# Title'")
        if len(re.findall(r"^# ", prose(text), flags=re.M)) != 1:
            errors.append(f"{rel}: exactly one '# ' heading is allowed")
        for lang, line, _ in blocks(text):
            if lang not in LANGUAGES:
                errors.append(f"{rel}:{line - 1}: fence language {lang!r} is not one of {sorted(LANGUAGES)}")
        body = prose(text)
        if text.startswith("---"):
            errors.append(f"{rel}: front matter is not allowed")
        for m in re.finditer(r"(?<!\\)</?[A-Za-z][A-Za-z0-9]*(\s[^<>]*)?/?>", body):
            errors.append(f"{rel}: HTML is not allowed: {m.group(0)!r}")
        for m in re.finditer(r"!?\[[^\]]*\]\(([^)\s]+)\)", body):
            target = m.group(1)
            if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            path = (DOCS / rel).parent / target.split("#")[0]
            if not path.is_file():
                errors.append(f"{rel}: broken link {target!r}")
            elif DOCS not in path.resolve().parents:
                errors.append(f"{rel}: link {target!r} leaves docs/")
    site = DOCS / "site.json"
    if site.is_file():
        listed = [p["file"] for s in json.loads(site.read_text())["sections"] for p in s["pages"]]
        every = sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md"))
        for f in every:
            if listed.count(f) != 1:
                errors.append(f"site.json: {f} is listed {listed.count(f)} times")
        for f in listed:
            if f not in every:
                errors.append(f"site.json: {f} does not exist")
        for s in json.loads(site.read_text())["sections"]:
            for p in s["pages"]:
                if len(p.get("summary", "")) >= 110:
                    errors.append(f"site.json: the summary of {p['file']} has {len(p['summary'])} characters (limit 109)")
    return errors


# ----------------------------------------------------------------------------
# requirements of a block
# ----------------------------------------------------------------------------


def requirements(code: str) -> dict:
    need = {}
    for mod in OPTIONAL_PACKAGES:
        if re.search(rf"^\s*(import|from)\s+{mod}\b", code, flags=re.M):
            need[mod] = 1
    if "density_polygons" in code:
        need["contourpy"] = 1
    if re.search(r"\binvdes\b", code):
        need["lightsim.invdes"] = 1
    if re.search(r"""which\(["']lightsim["']\)""", code):
        need["the lightsim program"] = 1
    gpus = 0
    for m in re.finditer(r"""["']cuda(?::([0-9,]+|all))?["']""", code):
        ids = m.group(1)
        gpus = max(gpus, 2 if ids == "all" else 1 if not ids else max(int(i) for i in ids.split(",")) + 1)
    if gpus:
        need["gpu"] = gpus
    return need


def defined_and_used(code: str):
    """Names a block binds at top level, and names it reads."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(), set()
    bound, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else used).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
    return bound, used


# ----------------------------------------------------------------------------
# the per-page runner (executed in a subprocess, see run_page)
# ----------------------------------------------------------------------------

RUNNER = r"""
import contextlib, io, json, sys, time, traceback, __future__

job = json.load(open(sys.argv[1]))
ns = {"__name__": "__main__"}
results = []
for b in job["blocks"]:
    if b["skip"]:
        results.append({"status": "skipped", "reason": b["skip"]})
        continue
    out, t0 = io.StringIO(), time.time()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(b["code"], f"{job['page']}:{b['line']}", "exec"), ns)
        results.append({"status": "ran", "seconds": time.time() - t0, "output": out.getvalue()})
    except BaseException:
        results.append({"status": "failed", "seconds": time.time() - t0, "output": out.getvalue(), "error": traceback.format_exc()})
        break
json.dump(results, open(sys.argv[2], "w"))
"""


def available() -> dict:
    have = {m: importlib.util.find_spec(m) is not None for m in OPTIONAL_PACKAGES + ("contourpy",)}
    try:
        have["lightsim.invdes"] = importlib.util.find_spec("lightsim.invdes") is not None
    except ModuleNotFoundError:
        have["lightsim.invdes"] = False
    have["the lightsim program"] = shutil.which("lightsim") is not None
    try:
        import lightsim

        have["gpu"] = sum(d.startswith("cuda") for d in lightsim.devices())
    except Exception as exc:  # no lightsim at all: nothing can run
        sys.exit(f"cannot import lightsim: {exc}\nbuild it first: maturin develop --release")
    return have


def plan_page(text: str, have: dict):
    """Decide, block by block, what runs."""
    plan, missing_names, defined = [], {}, set()
    for lang, line, code in blocks(text):
        if lang != "python":
            continue
        skip = None
        for what, n in requirements(code).items():
            if have.get(what, 0) < n:
                skip = f"needs {n} GPU(s)" if what == "gpu" else f"needs {what}"
                break
        bound, used = defined_and_used(code)
        if skip is None:
            # a name the block binds itself (its own import or assignment) is not taken from earlier
            lost = sorted(n for n in used if n in missing_names and n not in bound)
            if lost:
                skip = f"uses {lost[0]!r} from a skipped block"
        if skip:
            for n in bound - defined:  # names an earlier block already made stay usable
                missing_names.setdefault(n, skip)
        else:
            defined |= bound
            for n in bound:
                missing_names.pop(n, None)
        plan.append({"line": line, "code": code, "skip": skip})
    return plan


def run_page(rel: str, plan, timeout: float):
    with tempfile.TemporaryDirectory() as tmp:
        job, res = Path(tmp) / "job.json", Path(tmp) / "result.json"
        job.write_text(json.dumps({"page": rel, "blocks": plan}))
        work = Path(tmp) / "work"
        work.mkdir()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
        try:
            proc = subprocess.run([sys.executable, "-c", RUNNER, str(job), str(res)], cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return [{"status": "failed", "error": f"timed out after {timeout:.0f} s"}]
        if not res.is_file():
            return [{"status": "failed", "error": proc.stderr[-2000:] or "the runner died"}]
        return json.loads(res.read_text())


def check_json(rel: str, text: str) -> list[str]:
    errors = []
    for lang, line, code in blocks(text):
        if lang != "json":
            continue
        try:
            obj = json.loads(code)
        except json.JSONDecodeError as exc:
            errors.append(f"{rel}:{line}: invalid JSON: {exc}")
            continue
        if isinstance(obj, dict) and "region" in obj and "mesh" in obj:
            try:
                from lightsim import _core

                _core.time_info(json.dumps(obj))
            except Exception as exc:
                errors.append(f"{rel}:{line}: the core rejects this spec: {exc}")
    return errors


def check_generated() -> list[str]:
    gen = ROOT / "scripts" / "gen_api.py"
    if not gen.is_file():
        return []
    proc = subprocess.run([sys.executable, str(gen), "--check"], capture_output=True, text=True)
    return [] if proc.returncode == 0 else [f"reference/python-api.md is stale: {(proc.stdout + proc.stderr).strip()[-400:]}"]


# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the documentation's code samples and lint the markdown.")
    ap.add_argument("pages", nargs="*", help="pages relative to docs/ (default: all)")
    ap.add_argument("--list", action="store_true", help="list the blocks and what would run")
    ap.add_argument("--no-run", action="store_true", help="lint only")
    ap.add_argument("--no-lint", action="store_true", help="run only")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per page (default 1800)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print what the blocks print")
    ap.add_argument("--outputs", action="store_true", help="report where the output quoted after a block differs from what it printed (timings and thread counts differ between machines)")
    a = ap.parse_args()

    every = sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md"))
    pages = [str(Path(p)).removeprefix("docs/") for p in a.pages] or every
    failures = []
    if not a.no_lint:
        failures += lint(pages)
    counts = {"ran": 0, "skipped": 0, "failed": 0, "not reached": 0}
    differing = []
    if not a.no_run or a.list:
        have = available()
        print("available: " + ", ".join(f"{k}={v}" for k, v in have.items()))
        if not a.list:
            failures += check_generated()
        for rel in pages:
            text = (DOCS / rel).read_text()
            if not a.list:
                failures += check_json(rel, text)
            if rel in GENERATED:
                continue
            plan = plan_page(text, have)
            if not plan:
                continue
            if a.list:
                for b in plan:
                    print(f"{rel}:{b['line']}: {'skip: ' + b['skip'] if b['skip'] else 'run'}")
                    counts["skipped" if b["skip"] else "ran"] += 1  # "ran" counts what would run
                continue
            t0 = time.time()
            results = run_page(rel, plan, a.timeout)
            line = []
            for b, r in zip(plan, results):
                counts[r["status"]] += 1
                line.append({"ran": ".", "skipped": "s", "failed": "F"}[r["status"]])
                if r["status"] == "failed":
                    failures.append(f"{rel}:{b['line']}: sample failed\n{r.get('error', '')}")
                if a.verbose and r.get("output"):
                    print(f"--- {rel}:{b['line']}\n{r['output'].rstrip()}")
                if a.verbose and r["status"] == "skipped":
                    print(f"--- {rel}:{b['line']} skipped: {r['reason']}")
                quoted = quoted_outputs(text).get(b["line"]) if a.outputs else None
                if quoted is not None and r["status"] == "ran":
                    got = [l.rstrip() for l in r.get("output", "").strip().split("\n")]
                    want = [l.rstrip() for l in quoted.strip().split("\n")]
                    if got != want:
                        differing.append(f"{rel}:{b['line']}")
                        print(f"--- {rel}:{b['line']}: quoted output differs")
                        for g, w in zip(got + [""] * len(want), want + [""] * len(got)):
                            if g != w:
                                print(f"    printed: {g}\n    quoted:  {w}")
            counts["not reached"] += len(plan) - len(results)
            print(f"{rel:34s} {''.join(line):24s} {time.time() - t0:6.1f} s")
            for b, r in zip(plan, results):
                if r["status"] == "skipped" and not a.verbose:
                    print(f"    line {b['line']}: skipped, {r['reason']}")
    if a.list:
        print(f"\npython blocks: {counts['ran']} to run, {counts['skipped']} to skip")
    elif not a.no_run:
        print("\npython blocks: " + ", ".join(f"{v} {k}" for k, v in counts.items() if v or k in ("ran", "skipped", "failed")))
    if a.outputs:
        print(f"quoted outputs that differ from this run: {len(differing)}")
    for f in failures:
        print("\nFAIL " + f)
    print(f"\n{len(failures)} problem(s)" if failures else "\nall good")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
