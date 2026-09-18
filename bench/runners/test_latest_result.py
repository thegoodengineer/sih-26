#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic benchmark selection in latest_result.py.

Hand-rolled rather than pytest, matching the repository convention.
Run directly:

    python bench/runners/test_latest_result.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Add the directory containing latest_result.py to sys.path,
# consistent with other tests if run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import latest_result

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


def test_deterministic_tie_break() -> None:
    with tempfile.TemporaryDirectory() as td:
        old_results_dir = latest_result.RESULTS_DIR
        latest_result.RESULTS_DIR = Path(td)
        try:
            content1 = "instance,published_objective,git_commit,timestamp_utc\ntest,1,f7ca7e9,2026-08-26T00:00:01+00:00\n"
            content2 = "instance,published_objective,git_commit,timestamp_utc\ntest,1,f7ca7e9,2026-08-26T00:00:02+00:00\n"
            content3 = "instance,published_objective,git_commit,timestamp_utc\ntest,1,f7ca7e9,2026-08-26T00:00:00+00:00\n"

            path1 = Path(td) / "compare-highs-medium-f7ca7e9.csv"
            path2 = Path(td) / "compare-highs-medium-f7ca7e9-second.csv"
            path3 = Path(td) / "compare-highs-medium-f7ca7e9-third.csv"

            path1.write_text(content1)
            os.utime(path1, (time.time(), time.time() + 10))

            path2.write_text(content2)
            os.utime(path2, (time.time(), time.time() + 5))

            path3.write_text(content3)
            os.utime(path3, (time.time(), time.time() + 15))

            winner = latest_result.latest("compare-highs-medium-*.csv")
            check(winner == path2, "newest timestamp wins for same commit",
                  f"got {winner}, expected {path2}")
        finally:
            latest_result.RESULTS_DIR = old_results_dir


def test_different_commits_ordering() -> None:
    with tempfile.TemporaryDirectory() as td:
        old_results_dir = latest_result.RESULTS_DIR
        latest_result.RESULTS_DIR = Path(td)
        try:
            order = latest_result.commit_order()
            check(len(order) >= 2, "sufficient git history exists",
                  f"found {len(order)} commits, need at least 2")
            if len(order) < 2:
                return

            newest = order[0]
            older = order[1]

            content1 = f"instance,published_objective,git_commit,timestamp_utc\ntest,1,{newest},\n"
            content2 = f"instance,published_objective,git_commit,timestamp_utc\ntest,1,{older},\n"

            path1 = Path(td) / "test-newest.csv"
            path2 = Path(td) / "test-older.csv"

            path1.write_text(content1)
            path2.write_text(content2)

            # Make older commit have newer mtime
            os.utime(path2, (time.time(), time.time() + 10))
            os.utime(path1, (time.time(), time.time()))

            winner = latest_result.latest("test-*.csv")
            check(winner == path1, "newer git commit wins despite older mtime",
                  f"got {winner}, expected {path1}")
        finally:
            latest_result.RESULTS_DIR = old_results_dir


def main() -> int:
    print("test_deterministic_tie_break")
    test_deterministic_tie_break()
    print("test_different_commits_ordering")
    test_different_commits_ordering()

    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
