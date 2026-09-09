#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evidence for the first-order engine: PDHG against the simplex, at two tolerances.

The plan for this phase asks for four things, and this script produces all four from live
runs rather than from anybody's recollection:

1.  PDHG's objective next to the simplex's, on every Netlib instance we solve.
2.  Results reported at 1e-4 and 1e-8 SEPARATELY. A first-order method's cost depends
    enormously on the requested accuracy, so a single blended number would be meaningless.
3.  Iteration counts with restarts on and off, so the claim that restarts help is measured
    rather than asserted.
4.  An honest list of the instances PDHG cannot drive to 1e-8.

Usage:
    python bench/runners/pdhg_report.py --binary build/sankhya
"""

from __future__ import annotations

import argparse
import csv
import datetime
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

CSV_COLUMNS = [
    "instance", "algorithm", "tolerance", "restarts_enabled", "status",
    "objective", "published_objective", "relative_error", "iterations", "seconds",
    "reached_tolerance", "git_commit", "machine", "timestamp_utc", "solver_options",
]


def git_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def as_number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-gate/sankhya.exe",
                      "build-gate/sankhya", "build-main/sankhya.exe", "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; pass --binary")


def run(binary: Path, mps: Path, algorithm: str, tolerance: float | None,
        restarts: bool, time_limit: float, extra_options: list[str] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "s.json"
        command = [str(binary), "solve", str(mps), "--stats", str(stats),
                   "--time-limit", str(time_limit),
                   "--option", "log_to_console=false",
                   "--option", f"algorithm={algorithm}"]
        if tolerance is not None:
            command += ["--option", f"pdhg_tolerance={tolerance:g}"]
        if not restarts:
            command += ["--option", "pdhg_restart=false"]
        for option in extra_options or []:
            command += ["--option", option]
        started = time.perf_counter()
        subprocess.run(command, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "iterations": "",
                    "seconds": seconds}
        blob = json.loads(stats.read_text())
        return {
            "status": blob.get("result", {}).get("status", "unknown"),
            "objective": as_number(blob.get("result", {}).get("objective")),
            "iterations": blob.get("effort", {}).get("iterations", ""),
            "seconds": seconds,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*", metavar="NAME",
                        help="run only these instances. Without it every instance in "
                             "data/netlib/reference.json is run, which is the whole tier the "
                             "last fetch left there - 89 instances at four settings each, "
                             "hours of solving for a report whose point is the committed nine.")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE",
                        help="pass --option KEY=VALUE to every PDHG solve and record it in the "
                             "CSV's solver_options column. A run made with one is a measurement "
                             "OF that option, not the engine's evidence, and latest_result.py "
                             "skips it for that reason.")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the CSV here instead of bench/results/pdhg-<commit>.csv")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    reference = json.loads((DATA_DIR / "reference.json").read_text())["instances"]
    names = sorted(args.instances) if args.instances else sorted(reference)
    unknown = [n for n in names if n not in reference]
    if unknown:
        raise SystemExit("not in data/netlib/reference.json: " + ", ".join(unknown))
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []

    def record(name, algorithm, tolerance, restarts, result, published):
        error = (None if result["objective"] is None
                 else abs(result["objective"] - published) / max(1.0, abs(published)))
        rows.append({
            "instance": name, "algorithm": algorithm,
            "tolerance": "" if tolerance is None else f"{tolerance:g}",
            "restarts_enabled": "" if restarts is None else int(restarts),
            "status": result["status"],
            "objective": "" if result["objective"] is None else repr(result["objective"]),
            "published_objective": repr(published),
            "relative_error": "" if error is None else repr(error),
            "iterations": result["iterations"], "seconds": round(result["seconds"], 6),
            "reached_tolerance": int(result["status"] == "optimal"),
            "git_commit": commit, "machine": machine, "timestamp_utc": timestamp,
            "solver_options": " ".join(args.solver_option),
        })
        return error

    # ---- 1 & 2: agreement with the simplex, at both tolerances -------------------------
    print("PDHG vs the simplex. Relative error is against the PUBLISHED optimum.\n")
    print(f"{'instance':<11}{'simplex obj':>21}{'simplex err':>12}"
          f"{'PDHG 1e-4 obj':>21}{'err':>10}{'iters':>8}"
          f"{'PDHG 1e-8 obj':>21}{'err':>10}{'iters':>8}  1e-8?")
    print("-" * 124)

    missed_tight: list[str] = []
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]

        simplex = run(binary, mps, "simplex", None, True, args.time_limit, args.solver_option)
        simplex_error = record(name, "simplex", None, None, simplex, published)

        loose = run(binary, mps, "pdhg", 1e-4, True, args.time_limit, args.solver_option)
        loose_error = record(name, "pdhg", 1e-4, True, loose, published)

        tight = run(binary, mps, "pdhg", 1e-8, True, args.time_limit, args.solver_option)
        tight_error = record(name, "pdhg", 1e-8, True, tight, published)
        if tight["status"] != "optimal":
            missed_tight.append(name)

        def fmt(value, width, digits):
            return "-".rjust(width) if value is None else f"{value:>{width}.{digits}e}"

        print(f"{name:<11}{fmt(simplex['objective'], 21, 12)}{fmt(simplex_error, 12, 1)}"
              f"{fmt(loose['objective'], 21, 12)}{fmt(loose_error, 10, 1)}"
              f"{str(loose['iterations']):>8}"
              f"{fmt(tight['objective'], 21, 12)}{fmt(tight_error, 10, 1)}"
              f"{str(tight['iterations']):>8}"
              f"  {'yes' if tight['status'] == 'optimal' else 'NO'}")

    print("-" * 124)
    if missed_tight:
        print(f"did NOT reach 1e-8 within the time limit: {', '.join(missed_tight)}")
    else:
        print("every instance reached the 1e-8 relative tolerance")

    # ---- 3: restarts on versus off ------------------------------------------------------
    print("\nRestarts on vs off, at 1e-8. This is the measurement behind the claim that "
          "restarts help.\n")
    print(f"{'instance':<11}{'restarts on':>14}{'restarts off':>15}{'speedup':>10}  status off")
    print("-" * 62)
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]
        on = run(binary, mps, "pdhg", 1e-8, True, args.time_limit, args.solver_option)
        off = run(binary, mps, "pdhg", 1e-8, False, args.time_limit, args.solver_option)
        record(name, "pdhg", 1e-8, True, on, published)
        record(name, "pdhg", 1e-8, False, off, published)
        on_iters = on["iterations"] if isinstance(on["iterations"], int) else 0
        off_iters = off["iterations"] if isinstance(off["iterations"], int) else 0
        speedup = (off_iters / on_iters) if on_iters else 0.0
        print(f"{name:<11}{on_iters:>14}{off_iters:>15}{speedup:>9.2f}x  {off['status']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or (RESULTS_DIR / f"pdhg-{commit}.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
