#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-check our Netlib objectives against HiGHS, a third-party solver.

WHY THIS EXISTS.

bench/runners/netlib.py judges an answer against the optimum published in Netlib's `readme`.
That table is the standard reference and it is what the pass rate is measured on - but it
dates from the era the instances do, and on some instances it disagrees with what modern
solvers compute. When it does, "we do not match the published optimum" is ambiguous between
two very different situations:

    our answer is wrong          - a solver bug, the thing the benchmark exists to find
    the published value is stale - a reference problem, and nothing to fix in the solver

Our own verifier cannot separate those. It re-derives the answer from the same MPS file we
read, so if we agree with it we have only shown that our two readers agree - and both were
written by this project. Issue #75 said exactly this, and it is the one failure mode an
in-house independent check cannot cover.

A third-party solver can. HiGHS reads the same file with its own reader and solves it with
its own simplex; if HiGHS lands on our number rather than the readme's, the readme is the
outlier.

WHAT THIS IS NOT.

It is NOT a pass criterion, and nothing here feeds the benchmark's pass rate. Netlib's table
stays the reference netlib.py measures against, because a project cannot be allowed to grade
itself against a solver it chose. This produces EVIDENCE, in a CSV, for reading alongside the
benchmark - so a disagreement can be attributed rather than merely counted.

HiGHS is used here as a reference solver for comparison, which is what CLAUDE.md's dependency
policy permits and what bench/runners/compare.py already does. No HiGHS source is read,
vendored, or derived from.

Usage:
    python bench/runners/cross_check_highs.py                 # every fetched instance
    python bench/runners/cross_check_highs.py scrs8 nesm      # named instances
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

# Agreement threshold, matching netlib.py's PASS_RELATIVE_TOLERANCE so the two documents
# describe the same notion of "the same answer".
AGREEMENT_TOLERANCE = 1e-6

FIELDS = [
    "instance",
    "our_objective",
    "highs_objective",
    "published_objective",
    "relative_ours_vs_highs",
    "relative_highs_vs_published",
    "verdict",
    "highs_status",
    "git_commit",
    "machine",
    "timestamp_utc",
]


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=REPO_ROOT)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def relative(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instances", nargs="*")
    parser.add_argument("--binary", type=Path, default=None,
                        help="our solver; defaults to build/sankhya[.exe]")
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    try:
        import highspy
    except ImportError:
        print("highspy is not installed; pip install highspy", file=sys.stderr)
        return 2

    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        raise SystemExit("no reference data; run bench/runners/fetch_data.py first")
    reference = json.loads(reference_path.read_text())["instances"]

    binary = args.binary
    if binary is None:
        for candidate in ("build/sankhya", "build/sankhya.exe"):
            if (REPO_ROOT / candidate).exists():
                binary = REPO_ROOT / candidate
                break
    if binary is None:
        raise SystemExit("no solver binary found; pass --binary")

    names = sorted(args.instances or reference)
    rows = []

    print(f"{'instance':<10} {'ours vs HiGHS':>14} {'HiGHS vs readme':>16}  verdict")
    print("-" * 68)

    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists() or name not in reference:
            continue
        published = reference[name]["published_optimal"]

        # --- our answer -------------------------------------------------------------------
        completed = subprocess.run(
            [str(binary), "solve", str(mps), "--option", f"time_limit={args.time_limit}"],
            capture_output=True, text=True)
        ours = None
        for line in completed.stdout.splitlines():
            if line.startswith("Result:"):
                parts = line.split()
                if "objective" in parts:
                    try:
                        ours = float(parts[parts.index("objective") + 1])
                    except (ValueError, IndexError):
                        ours = None

        # --- HiGHS ------------------------------------------------------------------------
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        highs.setOptionValue("time_limit", args.time_limit)
        highs.readModel(str(mps))
        highs.run()
        highs_objective = highs.getObjectiveValue()
        highs_status = str(highs.getModelStatus())

        d_highs = None if ours is None else relative(ours, highs_objective)
        d_published = relative(highs_objective, published)

        # The verdict names WHICH of the three disagrees, which is the whole point of the
        # exercise. "stale reference" is only claimed when we and HiGHS agree closely AND
        # both differ from the readme - two independent solvers landing on the same number
        # is a far stronger statement than either one alone.
        if ours is None:
            verdict = "we produced no answer"
        elif d_highs > AGREEMENT_TOLERANCE:
            verdict = "WE DIFFER FROM HIGHS"
        elif d_published > AGREEMENT_TOLERANCE:
            verdict = "reference appears stale"
        else:
            verdict = "all three agree"

        rows.append({
            "instance": name,
            "our_objective": "" if ours is None else repr(ours),
            "highs_objective": repr(highs_objective),
            "published_objective": repr(published),
            "relative_ours_vs_highs": "" if d_highs is None else repr(d_highs),
            "relative_highs_vs_published": repr(d_published),
            "verdict": verdict,
            "highs_status": highs_status,
            "git_commit": git_commit(),
            "machine": f"{platform.system()}-{platform.machine()}",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        shown = "-" if d_highs is None else f"{d_highs:.1e}"
        print(f"{name:<10} {shown:>14} {d_published:>16.1e}  {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"cross-check-highs-{git_commit()}.csv")
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    stale = [r["instance"] for r in rows if r["verdict"] == "reference appears stale"]
    differ = [r["instance"] for r in rows if r["verdict"] == "WE DIFFER FROM HIGHS"]
    print("-" * 68)
    print(f"{len(rows)} instance(s) compared")
    if stale:
        print(f"reference appears stale on {len(stale)}: {', '.join(stale)}")
    if differ:
        print(f"WE DIFFER FROM HIGHS on {len(differ)}: {', '.join(differ)}")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")

    # Exit non-zero only when WE are the outlier. A stale reference is a finding to read,
    # not a build failure - and this script does not gate anything in CI.
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
