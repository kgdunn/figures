#!/usr/bin/env python3
"""Run every figure script that uses ``process_improve``, and report what breaks.

The scripts in this repository draw the figures for *Process Improvement using
Data*. They call the companion package, and the package moves: over one release
cycle five of them stopped running, on renames and changed return types, and
nothing noticed until someone tried to regenerate a figure. Four of the five
drew every figure on one page of the book, so that page's illustrations could
not be reproduced from its own source.

This runs them. Which scripts are in scope is not a list to maintain: it is
every ``*.py`` in the repository whose source imports ``process_improve``, so a
new script joins the set by existing.

Each runs in its own subprocess, from its own directory, with ``savefig`` and
``show`` disabled, so a check writes no image and cannot dirty the tree. A
traceback fails the script. So does a ``DeprecationWarning``, or any warning
class the library defines, raised from a line of the script itself: that is the
library saying a rename is coming, and it is the signal that was missed.

A comment anywhere in a script changes how it is treated::

    # check-scripts: skip <reason>              never run; give the reason
    # check-scripts: slow <reason>              run only with --slow (or --all)
    # check-scripts: requires <mod> -- <why>    run only where the module imports

Usage::

    python tools/check_figure_scripts.py              # every script, in parallel
    python tools/check_figure_scripts.py --list       # what would run
    python tools/check_figure_scripts.py doe          # only paths under doe/
    python tools/check_figure_scripts.py --jobs 1     # serially, for a clean traceback

Exit status is 1 if any script failed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent / "_script_harness.py"
RESULT_PREFIX = "@@FIGURE-SCRIPT-RESULT@@ "

IMPORTS_LIBRARY_RE = re.compile(r"^\s*(?:from|import)\s+process_improve\b", re.MULTILINE)
MARKER_RE = re.compile(
    r"^\s*#\s*check-scripts:\s*(?P<kind>skip|slow|requires)\b\s*(?P<reason>.*?)\s*$", re.MULTILINE
)
SIBLING_IMPORT_RE = re.compile(r"^\s*(?:from\s+(?P<from>[A-Za-z_]\w*)\s+import|import\s+(?P<plain>[A-Za-z_]\w*))", re.MULTILINE)
MODULE_NAME_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "tools", "node_modules"}


@dataclass
class Script:
    path: Path
    marker: str | None = None
    reason: str = ""

    @property
    def rel(self) -> str:
        return str(self.path.relative_to(ROOT))


@dataclass
class Result:
    script: Script
    status: str  # passed, failed, skipped
    detail: str = ""
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


def discover(root: Path = ROOT) -> list[Script]:
    """Every script that reaches the library, directly or through a sibling module.

    Several scripts here never name ``process_improve``: they import a sibling in the
    same directory that does, ``colour_case_study`` and ``omnibus_designs`` being the
    two. They break in exactly the same way when the library changes, so the set is
    closed over sibling imports rather than stopping at the direct users.
    """
    candidates: dict[Path, str] = {}
    for path in sorted(root.rglob("*.py")):
        if set(path.relative_to(root).parts) & EXCLUDED_DIRS:
            continue
        candidates[path] = path.read_text(encoding="utf-8", errors="replace")

    in_scope = {path for path, source in candidates.items() if IMPORTS_LIBRARY_RE.search(source)}
    # A sibling import can only resolve within its own directory, which is what goes on
    # sys.path when the script runs. Repeat until the set stops growing.
    by_directory: dict[Path, dict[str, Path]] = {}
    for path in candidates:
        by_directory.setdefault(path.parent, {})[path.stem] = path
    growing = True
    while growing:
        growing = False
        for path, source in candidates.items():
            if path in in_scope:
                continue
            siblings = by_directory.get(path.parent, {})
            for match in SIBLING_IMPORT_RE.finditer(source):
                name = match["from"] or match["plain"]
                if siblings.get(name) in in_scope:
                    in_scope.add(path)
                    growing = True
                    break

    found = []
    for path in sorted(in_scope):
        marker = MARKER_RE.search(candidates[path])
        found.append(
            Script(
                path=path,
                marker=marker["kind"] if marker else None,
                reason=marker["reason"] if marker else "",
            )
        )
    return found


def run_one(script: Script, *, timeout: int) -> Result:
    started = time.perf_counter()
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="always")
    try:
        completed = subprocess.run(  # noqa: S603 - arguments are our own
            [sys.executable, str(HARNESS), str(script.path)],
            cwd=script.path.parent,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        detail = f"timed out after {timeout}s; mark it `# check-scripts: slow` if that is expected"
        return Result(script, "failed", detail, elapsed)

    elapsed = time.perf_counter() - started
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            payload = json.loads(line[len(RESULT_PREFIX) :])
    if payload is None:
        detail = (completed.stdout + completed.stderr).strip() or "the harness produced no result"
        return Result(script, "failed", detail, elapsed)
    if payload["status"] == "failed":
        return Result(script, "failed", payload["detail"], elapsed)
    if payload["status"] == "warned":
        detail = "warnings from the script's own lines:\n  " + "\n  ".join(payload["warnings"])
        return Result(script, "failed", detail, elapsed, payload["warnings"])
    return Result(script, "passed", "", elapsed)


def required_modules(reason: str) -> list[str]:
    """Module names at the head of a ``requires`` marker; anything after ``--`` is prose."""
    return reason.split("--", 1)[0].split()


def missing_modules(reason: str) -> list[str]:
    return [name for name in required_modules(reason) if importlib.util.find_spec(name) is None]


def select(scripts: list[Script], patterns: list[str], *, include_slow: bool) -> list[Script]:
    if patterns:
        scripts = [s for s in scripts if any(p in s.rel for p in patterns)]
    return [s for s in scripts if s.marker != "slow" or include_slow]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("patterns", nargs="*", help="only scripts whose path contains one of these")
    parser.add_argument("--list", action="store_true", help="list what would run, then stop")
    parser.add_argument("--slow", "--all", dest="slow", action="store_true", help="include scripts marked slow")
    parser.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)), help="scripts to run at once")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds allowed per script")
    args = parser.parse_args(argv)

    scripts = select(discover(), args.patterns, include_slow=args.slow)
    if not scripts:
        print("No script imports process_improve under those paths.")
        return 0

    if args.list:
        for s in scripts:
            flag = f"  [{s.marker} {s.reason}]".rstrip() if s.marker else ""
            print(f"{s.rel}{flag}")
        print(f"\n{len(scripts)} script(s).")
        return 0

    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {}
        for s in scripts:
            if s.marker == "skip":
                results.append(Result(s, "skipped", s.reason or "no reason given"))
                continue
            if s.marker == "requires":
                names = required_modules(s.reason)
                bad = [n for n in names if not MODULE_NAME_RE.match(n)]
                if not names or bad:
                    detail = (
                        "malformed `requires` marker: expected module names, optionally followed "
                        f"by `-- <reason>`; got {s.reason!r}"
                    )
                    results.append(Result(s, "failed", detail))
                    continue
                absent = missing_modules(s.reason)
                if absent:
                    results.append(Result(s, "skipped", f"requires {' '.join(absent)} (not installed)"))
                    continue
            futures[pool.submit(run_one, s, timeout=args.timeout)] = s
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  [{result.status:7s}] {result.script.rel}  ({result.seconds:.0f}s)", flush=True)

    results.sort(key=lambda r: r.script.rel)
    failures = [r for r in results if r.status == "failed"]
    for r in results:
        if r.status == "skipped":
            print(f"\nSKIPPED {r.script.rel}: {r.detail}")
    for r in failures:
        print(f"\nFAILED {r.script.rel}")
        print("\n".join("    " + line for line in r.detail.rstrip().splitlines()))

    passed = sum(1 for r in results if r.status == "passed")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(f"\n{len(results)} scripts: {passed} passed, {skipped} skipped, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
