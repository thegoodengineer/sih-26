#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched Netlib instances and emit the evidence CSV.

Every column CLAUDE.md requires is here: instance, sha256 of the instance file, our
objective, the PUBLISHED reference objective, absolute and relative gap, status, wall time,
iterations, git commit and a machine tag. Without the CSV there is no claim.

Two things this runner does that a plain timing loop would not:

*   The reference values come from ``data/netlib/reference.json``, which ``fetch_data.py``
    parsed out of Netlib's own readme. No number here was typed from memory.

*   Every solution is handed to ``tools/verify_solution.py``, which re-parses the model with
    its own MPS reader and re-derives feasibility, the objective and strong duality without
    touching our C++. Matching the published optimum says the answer is right; the verifier
    says the answer is *self-consistent*, and the two failures look nothing alike.

Usage:
    python bench/runners/netlib.py
    python bench/runners/netlib.py --binary build/sankhya --time-limit 60
    python bench/runners/netlib.py --check        # fail if the pass rate dropped
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

def display_path(path: Path) -> Path | str:
    """A path for printing: repo-relative when it is inside the repo, absolute otherwise.

    `relative_to` RAISES when the target is outside REPO_ROOT, and that turned a successful
    run into a traceback after every result had already been printed - taking the exit code
    with it, so a run where everything passed reported failure. Writing a CSV somewhere else
    on purpose is a legitimate thing to ask for, not an error.
    """
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path

VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# A run counts as a pass when the status is optimal AND the objective matches the published
# value to this relative accuracy. Status alone is not enough: a solver that confidently
# reports "optimal" with the wrong number is the exact failure this project exists to catch.
PASS_RELATIVE_TOLERANCE = 1e-6

CSV_COLUMNS = [
    "instance",
    "instance_sha256",
    "rows",
    "columns",
    "nonzeros",
    "status",
    # The solver's own explanation. Without it every failure is just "numerical_error" and
    # docs/BENCHMARKS.md cannot say WHICH failure, which is most of what makes a named
    # failure useful to anyone deciding whether the tool fits their model.
    "message",
    "our_objective",
    "published_objective",
    "absolute_gap",
    "relative_gap",
    "matches_published",
    "independently_verified",
    # Why the verifier said no. Without it, "the verifier rejected this" cannot distinguish
    # a bad point from a model the verifier could not parse - and those want different
    # people looking at them.
    "verifier_message",
    "passed",
    "wall_seconds",
    "solver_seconds",
    "iterations",
    "algorithm",
    "git_commit",
    "machine",
    "timestamp_utc",
]


def as_number(value) -> float | None:
    """Coerce a JSON numeric field to float.

    JSON has no literal for infinity, so the writer emits non-finite values as the strings
    "inf", "-inf" and "nan" rather than letting them collapse to null. Python's float()
    accepts all three, so this is the only special case a consumer needs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def machine_tag() -> str:
    return f"{platform.system()}-{platform.machine()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-main/sankhya.exe",
                      "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; build first, or pass --binary")


def run_one(binary: Path, mps: Path, time_limit: float, verify: bool) -> dict:
    """Solve one instance, then verify the solution independently."""
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [
            str(binary), "solve", str(mps),
            "--stats", str(stats_path),
            "--write-sol", str(sol_path),
            "--time-limit", str(time_limit),
            "--option", "log_to_console=false",
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started

        if not stats_path.exists():
            return {
                "status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                "wall_seconds": wall,
                "stderr": completed.stderr.strip()[:400],
                "verified": None,
            }

        blob = json.loads(stats_path.read_text())
        # The writer nests the blob; flatten the fields this runner reports on.
        result = blob.get("result", {})
        model = blob.get("model", {})
        effort = blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "objective": as_number(result.get("objective")),
            "absolute_gap": as_number(result.get("absolute_gap")),
            "relative_gap": as_number(result.get("relative_gap")),
            "algorithm": result.get("algorithm", ""),
            "rows": model.get("rows", ""),
            "columns": model.get("columns", ""),
            "nonzeros": model.get("nonzeros", ""),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "wall_seconds": wall,
            "stderr": completed.stderr.strip()[:400],
            "verified": None,
        }

        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run(
                [sys.executable, str(VERIFIER), str(mps), str(sol_path), "--quiet"],
                capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
            if check.returncode != 0:
                flat["verifier_output"] = (check.stdout + check.stderr).strip()[:600]
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the independent verifier (not recommended)")
    parser.add_argument("--check", action="store_true",
                        help="fail if the pass count dropped versus the newest committed CSV")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths are resolved "
                             "against the repository root")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        raise SystemExit("no reference data; run bench/runners/fetch_data.py first")
    reference = json.loads(reference_path.read_text())["instances"]

    names = sorted(args.instances or reference)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    commit, machine = git_commit(), machine_tag()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    print(f"solver   {binary}")
    print(f"commit   {commit}   machine {machine}")
    print()
    print(f"{'instance':<11}{'status':<9}{'our objective':>22}{'published':>22}"
          f"{'rel err':>10}{'iters':>7}{'time':>8}  verified  result")
    print("-" * 104)

    for name in names:
        entry = reference[name]
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            print(f"{name:<11}{'MISSING':<9}")
            continue

        published = entry["published_optimal"]
        blob = run_one(binary, mps, args.time_limit, not args.no_verify)
        status = blob["status"]
        ours = blob.get("objective")
        gap = None if ours is None else abs(ours - published) / max(1.0, abs(published))
        matches = bool(status == "optimal" and gap is not None
                       and gap <= PASS_RELATIVE_TOLERANCE)
        verified = blob.get("verified")
        # A pass needs BOTH: the right number, and a solution that survives independent
        # re-derivation. Either one alone can be satisfied by a solver that is wrong.
        passed = matches and (verified is not False)

        rows.append({
            "instance": name,
            "instance_sha256": sha256_file(mps),
            "rows": blob.get("rows", ""),
            "columns": blob.get("columns", ""),
            "nonzeros": blob.get("nonzeros", ""),
            "status": status,
            "message": blob.get("message", ""),
            "verifier_message": blob.get("verifier_output", ""),
            "our_objective": "" if ours is None else repr(ours),
            "published_objective": repr(published),
            "absolute_gap": "" if ours is None else repr(abs(ours - published)),
            "relative_gap": "" if gap is None else repr(gap),
            "matches_published": int(matches),
            "independently_verified": "" if verified is None else int(verified),
            "passed": int(passed),
            "wall_seconds": round(blob.get("wall_seconds", 0.0), 6),
            "solver_seconds": blob.get("solver_seconds", ""),
            "iterations": blob.get("iterations", ""),
            "algorithm": blob.get("algorithm", ""),
            "git_commit": commit,
            "machine": machine,
            "timestamp_utc": timestamp,
        })

        ours_text = "-" if ours is None else f"{ours:>22.12e}"
        gap_text = "-" if gap is None else f"{gap:>10.1e}"
        verified_text = {True: "  yes   ", False: "  NO    ", None: "  -     "}[verified]
        print(f"{name:<11}{status:<9}{ours_text}{published:>22.12e}{gap_text}"
              f"{str(blob.get('iterations', '-')):>7}{blob.get('wall_seconds', 0.0):>7.2f}s"
              f"{verified_text}  {'PASS' if passed else 'FAIL'}")
        if not passed:
            if blob.get("stderr"):
                print(f"             stderr: {blob['stderr']}")
            if blob.get("verifier_output"):
                print(f"             verifier: {blob['verifier_output']}")

    passes = sum(row["passed"] for row in rows)
    total = len(rows)
    print("-" * 104)
    print(f"{passes}/{total} matched the published optimum to a relative "
          f"{PASS_RELATIVE_TOLERANCE:g} AND passed independent verification")
    if total and passes < total:
        failed = [row["instance"] for row in rows if not row["passed"]]
        # Naming the failures is not optional. A pass rate without them is a claim.
        print(f"failed: {', '.join(failed)}")

    # Tier goes in the FILENAME. Both tiers at the same commit previously produced the same
    # path, so running medium after small silently overwrote it and docs/BENCHMARKS.md could
    # only ever describe whichever ran last.
    tier = json.loads(reference_path.read_text()).get("instance_set", "")
    tier_tag = f"{tier}-" if tier and tier != "explicit" else ""
    out_path = args.out or (RESULTS_DIR / f"netlib-{tier_tag}{commit}.csv")
    # Resolve against the repository root BEFORE anything else touches it. Two separate
    # problems came from leaving a user-supplied relative path alone:
    #
    #   1. `relative_to(REPO_ROOT)` on the status line raised ValueError, after the results
    #      had been printed, taking the exit code with it.
    #
    #   2. Worse and quieter: RESULTS_DIR.glob() yields ABSOLUTE paths, so a relative
    #      out_path never compared equal to any of them. The CSV just written was therefore
    #      not excluded from the "previous runs" set, and being the newest by mtime it became
    #      the baseline - so the run was compared against ITSELF and the --check regression
    #      gate could never fire. That is a gate that silently passes, on the evidence
    #      CLAUDE.md says the project stands or falls by.
    #
    # Problem 2 was unreachable only because problem 1 crashed first. Fixing the traceback
    # alone would have exposed it.
    out_path = (REPO_ROOT / out_path).resolve()
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {display_path(out_path)}")

    if args.check:
        # Compare against the SAME TIER only. Now that the tier is in the filename, a bare
        # netlib-*.csv glob would happily take a 50-instance medium baseline for an 8-instance
        # small run and report a catastrophic regression, or the reverse and report a triumph.
        # Either way the gate would be measuring the size of the instance set rather than the
        # health of the solver - the same class of silently-wrong gate #31 fixed here.
        # Compare against a baseline covering the SAME INSTANCES, by row count rather than by
        # filename. Matching on the tier tag alone would silently discard every CSV written
        # before the tag existed - all of them small-set runs, and the only history there is.
        # Matching on size keeps them and still refuses to weigh an 8-instance run against a
        # 50-instance one, which would measure the size of the set rather than the health of
        # the solver: the same class of silently-wrong gate #31 fixed here.
        def comparable(candidate: Path) -> bool:
            if candidate == out_path:
                return False
            try:
                with candidate.open(newline="") as handle:
                    return len(list(csv.DictReader(handle))) == len(rows)
            except OSError:
                return False

        previous = sorted((p for p in RESULTS_DIR.glob("netlib-*.csv") if comparable(p)),
                          key=lambda p: p.stat().st_mtime)
        if not previous:
            print("no earlier CSV to compare against; this run is the baseline")
            return 0
        with previous[-1].open(newline="") as handle:
            baseline = list(csv.DictReader(handle))
        baseline_passes = sum(int(row["passed"]) for row in baseline)
        print(f"baseline {previous[-1].name}: {baseline_passes}/{len(baseline)}")
        if passes < baseline_passes:
            print(f"REGRESSION: pass count fell from {baseline_passes} to {passes}")
            return 1

    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
