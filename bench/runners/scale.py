#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How far up does this solver actually go? (#34)

PS26119 asks for "thousands to millions of variables", and until this script there was no
CSV behind any answer to that - only one 5000x5000 instance solved live in the demo, which
is a demonstration, not a benchmark. `docs/PS26119_COVERAGE.md` has called this the project's
biggest single gap for weeks, and the reason it stayed a gap is that a large RANDOM instance
proves nothing: nobody knows its optimum, so a wrong answer and a right one look identical.

bench/runners/generate_large_lp.py solves that by building the instance BACKWARDS from a
chosen primal-dual pair that already satisfies the KKT conditions, out of integer data. The
optimal objective is therefore known exactly before the solver sees the file, at any size.
This runner walks a family of sizes, hands each instance to each engine under one time
limit, and records what came back next to what was true by construction.

WHAT THE NUMBERS HERE MEAN, AND WHAT THEY DO NOT. The accuracy columns are properties of the
solver: an objective is right or it is not, whatever machine measured it. The seconds are a
property of this laptop on the day, and the STATUS column depends on both - an engine that
runs out of time at one size on a slow machine may finish there on a fast one. Read the
error column first and the clock second.

Usage:
    python bench/runners/scale.py --binary build/sankhya
    python bench/runners/scale.py --sizes 1000 5000 20000 --engines pdhg --time-limit 300
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
GENERATOR = REPO_ROOT / "bench" / "runners" / "generate_large_lp.py"

DEFAULT_SIZES = [1_000, 5_000, 20_000, 100_000]
DEFAULT_ENGINES = ["dual-simplex", "pdhg", "ipm"]

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "columns", "nonzeros", "analytic_optimum",
    "engine", "status", "our_objective", "absolute_error", "relative_error",
    "reached_optimum", "iterations", "wall_seconds", "solver_seconds", "time_limit",
    "git_commit", "machine", "timestamp_utc", "solver_options",
]

# An objective this close to one known exactly by construction is the right answer; the
# construction's data are integers, so there is no reference error to allow for.
MATCH_RELATIVE_TOLERANCE = 1e-6


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() or "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-dbg/sankhya.exe"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; pass --binary")


def generate(size: int, nnz_per_col: int, seed: int, directory: Path) -> tuple[Path, float]:
    """Write the instance and return its path and the optimum that is true by construction."""
    path = directory / f"scale-{size}.mps"
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--rows", str(size), "--cols", str(size),
         "--nnz-per-col", str(nnz_per_col), "--seed", str(seed), "--out", str(path)],
        capture_output=True, text=True, check=True)
    optimum = None
    for line in result.stdout.splitlines():
        if "analytic optimum:" in line:
            optimum = float(line.split("analytic optimum:")[1].strip())
    if optimum is None:
        raise SystemExit(f"the generator printed no analytic optimum for size {size}")
    return path, optimum


# A solve is stopped from outside at this multiple of its own limit. The solver checks the
# clock BETWEEN iterations, so one very expensive iteration overruns by however long that
# iteration takes: measured here, the interior-point method spent 813 s on a single iteration
# of a 20,000-row model under a 120 s limit, and Mittelmann's bdry2 did the same thing at
# 648 s against 300 s. That is worth recording rather than waiting out, and a run that cannot
# finish is a result too.
OVERRUN_FACTOR = 3.0


def solve(binary: Path, instance: Path, engine: str, time_limit: float,
          extra: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        command = [str(binary), "solve", str(instance), "--stats", str(stats),
                   "--time-limit", str(time_limit),
                   "--option", "log_to_console=false",
                   "--option", f"algorithm={engine}"]
        for option in extra:
            command += ["--option", option]
        started = time.perf_counter()
        try:
            subprocess.run(command, capture_output=True, text=True,
                           timeout=time_limit * OVERRUN_FACTOR)
        except subprocess.TimeoutExpired:
            wall = time.perf_counter() - started
            return {"status": "overran_its_limit", "objective": None, "iterations": "",
                    "wall": wall, "solver": ""}
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "wall": wall, "solver": ""}
        blob = json.loads(stats.read_text())
        result = blob.get("result", {})
        objective = result.get("objective")
        return {
            "status": result.get("status", "unknown"),
            "objective": None if objective is None else float(objective),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "solver": blob.get("effort", {}).get("solve_seconds", ""),
            "wall": wall,
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="square instances of this many rows and columns")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES)
    parser.add_argument("--nnz-per-col", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="pass --option KEY=VALUE to every solve and record it in the "
                             "CSV's solver_options column, so a run made to measure an "
                             "option is never mistaken for the engine's own evidence")
    parser.add_argument("--keep", type=Path, default=None,
                        help="write the generated instances here instead of a temporary "
                             "directory (a 100k instance is about 26 MB)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    binary = args.binary or default_binary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    options = " ".join(args.solver_option)
    rows: list[dict] = []

    print(f"Scale, against optima known exactly by construction. Limit {args.time_limit:g}s "
          f"per solve, commit {commit}.\n")
    print(f"{'size':>8}  {'engine':<13}{'status':<13}{'objective':>18}{'rel err':>10}"
          f"{'iters':>10}{'seconds':>9}")
    print("-" * 82)

    with tempfile.TemporaryDirectory() as tmp:
        directory = args.keep or Path(tmp)
        directory.mkdir(parents=True, exist_ok=True)
        for size in args.sizes:
            instance, optimum = generate(size, args.nnz_per_col, args.seed, directory)
            digest = sha256(instance)
            nonzeros = size * args.nnz_per_col
            for engine in args.engines:
                result = solve(binary, instance, engine, args.time_limit, args.solver_option)
                objective = result["objective"]
                if objective is None or not math.isfinite(objective):
                    absolute = relative = None
                    matched = False
                else:
                    absolute = abs(objective - optimum)
                    relative = absolute / max(1.0, abs(optimum))
                    matched = relative <= MATCH_RELATIVE_TOLERANCE
                row = {
                    "instance": instance.name,
                    "instance_sha256": digest,
                    "rows": size,
                    "columns": size,
                    "nonzeros": nonzeros,
                    "analytic_optimum": repr(optimum),
                    "engine": engine,
                    "status": result["status"],
                    "our_objective": "" if objective is None else repr(objective),
                    "absolute_error": "" if absolute is None else repr(absolute),
                    "relative_error": "" if relative is None else repr(relative),
                    "reached_optimum": int(matched),
                    "iterations": result["iterations"],
                    "wall_seconds": f"{result['wall']:.6f}",
                    "solver_seconds": result["solver"],
                    "time_limit": args.time_limit,
                    "git_commit": commit,
                    "machine": machine,
                    "timestamp_utc": timestamp,
                    "solver_options": options,
                }
                rows.append(row)
                # WRITTEN AFTER EVERY SOLVE. The whole family takes the better part of an
                # hour and one solve can overrun badly; a runner that only writes at the end
                # loses every measurement it already made the first time something has to be
                # stopped. Rewriting the file each time costs nothing at this row count.
                write_csv(args.out or (RESULTS_DIR / f"scale-{commit}.csv"), rows)
                shown = "-" if objective is None else f"{objective:.10g}"
                error = "-" if relative is None else f"{relative:.1e}"
                print(f"{size:>8}  {engine:<13}{result['status']:<13}{shown:>18}{error:>10}"
                      f"{str(result['iterations']):>10}{result['wall']:>8.1f}s", flush=True)
            print("-" * 82, flush=True)

    out = args.out or (RESULTS_DIR / f"scale-{commit}.csv")
    write_csv(out, rows)

    reached = [r for r in rows if r["reached_optimum"] == 1]
    print(f"\n{len(reached)} of {len(rows)} solves reached the analytic optimum to a relative "
          f"{MATCH_RELATIVE_TOLERANCE:g}.")
    if reached:
        largest = max(int(r["rows"]) for r in reached)
        by_engine = sorted({r["engine"] for r in reached if int(r["rows"]) == largest})
        print(f"Largest size reached: {largest:,} rows and columns, by {', '.join(by_engine)}.")
    try:
        shown_path = out.relative_to(REPO_ROOT)
    except ValueError:
        # --out may point outside the repository, which is what a trial run should do.
        shown_path = out
    print(f"wrote {shown_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
