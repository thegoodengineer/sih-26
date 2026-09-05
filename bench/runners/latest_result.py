#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pick the most recent results CSV for a glob, ordered by GIT HISTORY.

WHY NOT THE OBVIOUS THINGS. Two of them were tried here and both are wrong:

  Sorting by FILENAME sorts the commit sha as text, which is meaningless. It put
  `netlib-medium-e71ad03.csv` after `netlib-medium-a90db47.csv` purely because `e` follows
  `a`, so the demo reported 40/50 from a superseded run when the current one said 41/50.

  Sorting by MODIFICATION TIME is right on the machine that produced the files and wrong
  everywhere else: git does not record mtimes, so a fresh clone stamps every file with the
  checkout time and the order becomes arbitrary. That is precisely the situation a judge is
  in, and the failure is silent - a plausible number from the wrong run.

Every results CSV records the commit it was produced at, and `git log` orders those exactly.
That ordering is identical on every clone, which is the property the other two lack.

    python bench/runners/latest_result.py "netlib-medium-*.csv"
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def commit_order() -> list[str]:
    """Full commit hashes, newest first. Empty when git is unavailable."""
    try:
        out = subprocess.run(["git", "log", "--format=%H"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return out.stdout.split()


def first_row(path: Path) -> dict:
    """The first data row of a results CSV, or an empty dict if it cannot be read."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                return row
    except (OSError, ValueError):
        pass
    return {}


def commit_of(path: Path) -> str:
    """The commit recorded in a results CSV's first row."""
    return (first_row(path).get("git_commit") or "").strip()


def is_default_run(path: Path) -> bool:
    """False when the run was made with a non-default solver option.

    netlib.py records --solver-option values in a `solver_options` column. Such a run is a
    measurement OF an option, not the tier's evidence: the #66 re-measurement committed four
    medium-tier CSVs at one commit, three with an option set, and they would otherwise tie
    on git position and be ordered by mtime - which on a fresh clone is the checkout order.
    A CSV without the column predates the option and counts as a default run.
    """
    return not (first_row(path).get("solver_options") or "").strip()


def latest(pattern: str) -> Path | None:
    candidates = [path for path in RESULTS_DIR.glob(pattern) if is_default_run(path)]
    if not candidates:
        return None

    order = commit_order()
    # Position in `git log`, so smaller is newer. A CSV whose commit is not in this history -
    # produced on another branch, or rebased away - sorts last rather than being dropped: a
    # stale number is better than no number, and the caller prints which commit it came from.
    index = {sha: i for i, sha in enumerate(order)}

    def rank(path: Path) -> tuple[int, float]:
        recorded = commit_of(path)
        position = len(order)
        if recorded:
            for sha, i in index.items():
                if sha.startswith(recorded):
                    position = i
                    break
        # mtime only breaks ties among commits git cannot order.
        return (position, -path.stat().st_mtime)

    return sorted(candidates, key=rank)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help='e.g. "netlib-medium-*.csv"')
    parser.add_argument("--summary", action="store_true",
                        help="print 'PASSED, measured on commit SHA' instead of the path")
    parser.add_argument("--status-counts", action="store_true",
                        help="print a breakdown of the `status` column instead. --summary "
                             "counts a `passed` column, which the netlib CSVs carry and the "
                             "MIPLIB ones do not - asking for --summary there silently "
                             "reports 0, which reads as a total failure rather than as the "
                             "wrong question.")
    args = parser.parse_args()

    path = latest(args.pattern)
    if path is None:
        if args.summary:
            print(f"an unknown number - no CSV matching {args.pattern} in bench/results/")
            return 0
        return 1

    if not args.summary and not args.status_counts:
        print(path)
        return 0

    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))

    if args.status_counts:
        # Ordered most-to-least conclusive, so the sentence reads as a claim getting weaker
        # rather than as an arbitrary tally. Any status not in this list is appended, so a
        # new one shows up rather than vanishing from the count.
        commit = rows[0].get("git_commit", "?") if rows else "?"
        seen: dict[str, int] = {}
        for row in rows:
            key = (row.get("status") or "unknown").strip()
            seen[key] = seen.get(key, 0) + 1
        preferred = ["optimal", "feasible", "node_limit", "time_limit", "infeasible"]
        order = [k for k in preferred if k in seen] + sorted(k for k in seen
                                                            if k not in preferred)
        parts = [f"{seen[k]} {k.replace('_', ' ')}" for k in order]
        print(f"{len(rows)} instances: {', '.join(parts)}, measured on commit {commit}")
        return 0
    passed = sum(1 for row in rows if row.get("passed") == "1")
    commit = rows[0].get("git_commit", "?") if rows else "?"
    print(f"{passed}, measured on commit {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
