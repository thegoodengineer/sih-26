#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched MIPLIB subset and record what happened.

The MILP counterpart of netlib.py. Same CSV contract from CLAUDE.md - instance, sha256, our
objective, published optimum, gaps, status, time, git commit, machine - plus the two columns
a MILP needs and an LP does not: NODES, and the bound.

THE DISTINCTION THIS FILE EXISTS TO KEEP. On an LP there is one question: is the objective
right. On a MILP there are two, and they come apart constantly:

    matched   - our objective equals the published optimum
    proved    - we also closed the bound against it and know it is optimal

`flugpl` is the case in point. We return exactly 1201500, which IS the published optimum, and
we report `feasible` rather than `optimal` because the search stopped on the relative gap
target instead of proving the bound. Collapsing those two into one "passed" column would
either throw away a correct answer or launder a tolerance stop into a proof. Both columns are
recorded, and the summary reports them separately.

A third column, `verified`, is the independent check: tools/verify_solution.py re-reads the
model with its own MPS reader and confirms the point is feasible AND integral. An incumbent
that is not integral is a relaxation, whatever the status says.

    python bench/runners/miplib.py --time-limit 600
    python bench/runners/miplib.py --instances flugpl gen-ip002
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# An objective within this relative distance of the published optimum counts as MATCHED.
# Looser than the LP set's 1e-6 on purpose: MIPLIB objectives run to eight and nine figures,
# and the published values in the .solu file are themselves given to about ten.
MATCH_RELATIVE_TOLERANCE = 1e-6

CSV_COLUMNS = [
    "instance",
    "instance_sha256",
    "rows",
    "columns",
    "nonzeros",
    "integer_columns",
    "status",
    "message",
    "our_objective",
    "published_objective",
    "dual_bound",
    "absolute_gap",
    "relative_gap",
    "matched_published",
    "proved_optimal",
    "independently_verified",
    "nodes",
    "wall_seconds",
    "solver_seconds",
    "git_commit",
    "machine",
    "timestamp_utc",
    "solver_options",
]


def as_number(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("inf", "+inf", "infinity"):
            return math.inf
        if text in ("-inf", "-infinity"):
            return -math.inf
        if text == "nan":
            return math.nan
        try:
            return float(text)
        except ValueError:
            return None
    return float(value)


def find_binary(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    for candidate in ("build/sankhya", "build/sankhya.exe", "build/Release/sankhya.exe"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    return None


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def solve(binary: Path, instance: Path, time_limit: float, verify: bool,
          solver_options: list | None = None) -> dict:
    import time

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(instance),
                   "--time-limit", str(time_limit),
                   "--stats", str(stats_path),
                   "--write-sol", str(sol_path),
                   "--option", "log_to_console=false"]
        for option in solver_options or []:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started

        if not stats_path.exists():
            return {"status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                    "message": completed.stderr.strip()[:300],
                    "wall_seconds": wall, "verified": None}

        blob = json.loads(stats_path.read_text())
        result = blob.get("result", {})
        model = blob.get("model", {})
        effort = blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", "")[:300],
            "objective": as_number(result.get("objective")),
            "dual_bound": as_number(result.get("dual_bound")),
            "absolute_gap": as_number(result.get("absolute_gap")),
            "relative_gap": as_number(result.get("relative_gap")),
            "rows": model.get("rows", ""),
            "columns": model.get("columns", ""),
            "nonzeros": model.get("nonzeros", ""),
            "integer_columns": model.get("integer_columns", ""),
            "nodes": effort.get("nodes", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "wall_seconds": wall,
            "verified": None,
        }

        # The independent check runs on any point we claim, proved or not. An incumbent that
        # is not integral is a relaxation whatever the status says, and that is precisely
        # what this catches.
        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run(
                [sys.executable, str(VERIFIER), str(instance), str(sol_path), "--quiet"],
                capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="pass --option KEY=VALUE to the solver and record it in the CSV. "
                             "A run made with one is a measurement OF that option, not the "
                             "tier's evidence, and latest_result.py skips it for that reason.")
    args = parser.parse_args()

    binary = find_binary(args.binary)
    if binary is None:
        print("no solver binary; build first", file=sys.stderr)
        return 1

    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        print("no data/miplib/reference.json; run bench/runners/fetch_miplib.py first",
              file=sys.stderr)
        return 1
    manifest = json.loads(reference_path.read_text())
    reference = manifest["instances"]

    names = args.instances or sorted(reference)
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    print(f"commit   {commit}   machine {machine}   time limit {args.time_limit:g}s")
    print()
    print(f"{'instance':<24}{'status':<14}{'our objective':>20}{'published':>20}"
          f"{'gap':>10}{'nodes':>9}{'time':>9}  match proved ver")
    print("-" * 124)

    rows = []
    for name in names:
        entry = reference.get(name)
        instance = DATA_DIR / f"{name}.mps.gz"
        if entry is None or not instance.exists():
            print(f"{name:<24}{'MISSING':<14}", flush=True)
            continue

        published = float(entry["published_optimal"])
        blob = solve(binary, instance, args.time_limit, not args.no_verify,
                     args.solver_option)
        ours = blob.get("objective")
        status = blob["status"]

        matched = (ours is not None and math.isfinite(ours)
                   and abs(ours - published) <= MATCH_RELATIVE_TOLERANCE * max(1.0, abs(published)))
        # PROVED means the solver closed the bound itself, not that the number happens to be
        # right. Only kOptimal asserts that, and #29's guard has already re-measured it.
        proved = status == "optimal" and matched

        relative = blob.get("relative_gap")
        rows.append({
            "instance": name,
            "instance_sha256": entry.get("gz_sha256", ""),
            "rows": blob.get("rows", ""),
            "columns": blob.get("columns", ""),
            "nonzeros": blob.get("nonzeros", ""),
            "integer_columns": blob.get("integer_columns", ""),
            "status": status,
            "message": blob.get("message", ""),
            "our_objective": "" if ours is None else repr(ours),
            "published_objective": repr(published),
            "dual_bound": "" if blob.get("dual_bound") is None else repr(blob["dual_bound"]),
            "absolute_gap": "" if blob.get("absolute_gap") is None else repr(blob["absolute_gap"]),
            "relative_gap": "" if relative is None else repr(relative),
            "matched_published": int(matched),
            "proved_optimal": int(proved),
            "independently_verified": "" if blob["verified"] is None else int(blob["verified"]),
            "nodes": blob.get("nodes", ""),
            "wall_seconds": f"{blob['wall_seconds']:.6f}",
            "solver_seconds": blob.get("solver_seconds", ""),
            "git_commit": commit,
            "machine": machine,
            "timestamp_utc": stamp,
            "solver_options": " ".join(args.solver_option),
        })

        gap_text = "-" if relative is None or not math.isfinite(relative) else f"{relative:.2e}"
        ours_text = "-" if ours is None else f"{ours:.10g}"
        mark = lambda flag: " yes " if flag else " NO  "  # noqa: E731
        ver = blob["verified"]
        print(f"{name:<24}{status:<14}{ours_text:>20}{published:>20.10g}"
              f"{gap_text:>10}{str(blob.get('nodes','')):>9}{blob['wall_seconds']:>8.1f}s"
              f" {mark(matched)}{mark(proved)}"
              f"{'  -  ' if ver is None else mark(ver)}", flush=True)

    print("-" * 124, flush=True)
    matched_count = sum(r["matched_published"] for r in rows)
    proved_count = sum(r["proved_optimal"] for r in rows)
    print(f"{matched_count}/{len(rows)} reached the published optimum; "
          f"{proved_count}/{len(rows)} also PROVED it optimal")
    unproved = [r["instance"] for r in rows if not r["proved_optimal"]]
    if unproved:
        # Naming them is not optional. A rate without its failures is a claim, not evidence.
        print(f"not proved: {', '.join(unproved)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"miplib-{commit}.csv")
    out_path = (REPO_ROOT / out_path).resolve()
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    try:
        shown = out_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = out_path
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
