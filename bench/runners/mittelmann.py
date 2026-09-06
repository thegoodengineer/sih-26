#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched Mittelmann LP instances and emit the evidence CSV (#60).

The third of the three libraries PS26119 names. Unlike Netlib there is no published optimum
to pass against: Mittelmann's benchmark page publishes solver times. So a row here carries
the status, the objective, the independent verifier's verdict on the certificate, and - when
`highspy` is installed - HiGHS's objective on the same file as a cross-check, with the
relative difference. A row is a PASS when the status is optimal, the verifier accepts the
answer, and (if HiGHS ran) the two objectives agree to 1e-6 relative.

These instances are far larger than Netlib's; the benchmark page allows 15,000 s each.
This runner states the limit it used in every row and in the summary, reports the shifted
geometric mean of the solve times over the instances that finished, and NAMES every one
that did not. A table of failures is honest evidence about scale; see docs/BENCHMARKS.md.

    python bench/runners/fetch_mittelmann.py
    python bench/runners/mittelmann.py --time-limit 600
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
DATA_DIR = REPO_ROOT / "data" / "mittelmann"
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"
SHIFT_SECONDS = 10.0  # Mittelmann's own convention for LP benchmarks
AGREEMENT_TOLERANCE = 1e-6

CSV_COLUMNS = [
    "instance", "instance_sha256", "rows", "columns", "nonzeros", "status", "message",
    "our_objective", "highs_objective", "relative_difference", "independently_verified",
    "verifier_message", "passed", "wall_seconds", "solver_seconds", "iterations", "algorithm",
    "time_limit", "git_commit", "machine", "timestamp_utc", "solver_options",
]


def git_commit() -> str:
    """Short commit hash, with "-dirty" appended when tracked files other than the tier
    manifests are modified. The manifests (data/netlib/reference.json and
    data/mittelmann/reference.json) are rewritten by the fetch scripts as part of the
    runner's own workflow and say nothing about what was measured; untracked files are
    ignored for the same reason (fetched instances are untracked by design)."""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        commit = result.stdout.strip() or "unknown"
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no", "--",
             ".", ":!data/netlib/reference.json", ":!data/mittelmann/reference.json"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if status.stdout.strip():
            commit += "-dirty"
        return commit
    except OSError:
        return "unknown"

def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-main/sankhya.exe",
                      "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; build first, or pass --binary")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def highs_objective(mps: Path, time_limit: float) -> tuple[float | None, str]:
    """HiGHS as a SEPARATE PROCESS over the same file; nothing of it is linked or read."""
    script = (
        "import sys, highspy\n"
        "h = highspy.Highs(); h.setOptionValue('output_flag', False)\n"
        f"h.setOptionValue('time_limit', {float(time_limit)!r})\n"
        f"h.readModel({str(mps)!r}); h.run()\n"
        "print(h.modelStatusToString(h.getModelStatus())); print(repr(h.getInfo().objective_function_value))\n")
    try:
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              timeout=time_limit + 60)
    except (OSError, subprocess.TimeoutExpired):
        return None, "highs did not finish"
    lines = done.stdout.strip().splitlines()
    if done.returncode != 0 or len(lines) < 2:
        return None, "highs unavailable"
    status = lines[0].strip()
    try:
        value = float(lines[1])
    except ValueError:
        return None, status
    return (value if status == "Optimal" else None), status


def run_one(binary: Path, mps: Path, time_limit: float, verify: bool,
            solver_options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats_path), "--write-sol",
                   str(sol_path), "--time-limit", str(time_limit), "--option",
                   "log_to_console=false"]
        for option in solver_options:
            command += ["--option", option]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats_path.exists():
            return {"status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                    "wall_seconds": wall, "message": completed.stderr.strip()[:300]}
        blob = json.loads(stats_path.read_text())
        result, model, effort = blob.get("result", {}), blob.get("model", {}), blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"), "message": result.get("message", ""),
            "objective": result.get("objective"), "rows": model.get("rows", ""),
            "columns": model.get("columns", ""), "nonzeros": model.get("nonzeros", ""),
            "iterations": effort.get("iterations", ""), "solver_seconds": effort.get("solve_seconds", ""),
            "algorithm": result.get("algorithm", ""), "wall_seconds": wall, "verified": "",
            "verifier_message": "",
        }
        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol_path)],
                                   capture_output=True, text=True)
            flat["verified"] = 1 if check.returncode == 0 else 0
            if check.returncode != 0:
                failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
                flat["verifier_message"] = "; ".join(failing)[:300]
        return flat


def shifted_geometric_mean(values: list[float], shift: float) -> float:
    if not values:
        return float("nan")
    return math.exp(sum(math.log(v + shift) for v in values) / len(values)) - shift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--no-highs", action="store_true", help="skip the HiGHS cross-check")
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        raise SystemExit("no data/mittelmann/reference.json; run bench/runners/fetch_mittelmann.py")
    manifest = json.loads(manifest_path.read_text())
    instances = manifest["instances"]
    names = args.instances or sorted(instances)
    binary = args.binary or default_binary()
    commit, machine = git_commit(), f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    solver_options = " ".join(args.solver_option)
    have_highs = False
    if not args.no_highs:
        have_highs = subprocess.run([sys.executable, "-c", "import highspy"],
                                    capture_output=True).returncode == 0

    print(f"solver   {binary}\ncommit   {commit}   machine {machine}   time limit {args.time_limit:g}s"
          + (f"   options {solver_options}" if solver_options else "")
          + ("   HiGHS cross-check on" if have_highs else "   HiGHS cross-check off (highspy not installed)"))
    print(f"\n{'instance':22s}{'rows':>9s}{'cols':>9s}{'status':>16s}{'our objective':>20s}"
          f"{'HiGHS':>20s}{'rel diff':>10s}{'time':>9s}  ver  result")
    print("-" * 122)
    rows: list[dict] = []
    for name in names:
        entry = instances.get(name)
        mps = DATA_DIR / f"{name}.mps"
        if entry is None or not mps.exists():
            print(f"{name:22s} MISSING (run fetch_mittelmann.py)")
            continue
        sha = entry.get("mps_sha256") or sha256_of(mps)
        flat = run_one(binary, mps, args.time_limit, not args.no_verify, args.solver_option)
        ours = flat.get("objective")
        highs_value, highs_status = (None, "")
        if have_highs:
            highs_value, highs_status = highs_objective(mps, args.time_limit)
        rel = ""
        agree = True
        if isinstance(ours, (int, float)) and highs_value is not None:
            rel = abs(ours - highs_value) / max(1.0, abs(highs_value))
            agree = rel <= AGREEMENT_TOLERANCE
        passed = (flat["status"] == "optimal" and flat.get("verified") == 1 and agree)
        rows.append({
            "instance": name, "instance_sha256": sha, "rows": flat.get("rows", ""),
            "columns": flat.get("columns", ""), "nonzeros": flat.get("nonzeros", ""),
            "status": flat["status"], "message": flat.get("message", "")[:200],
            "our_objective": ours if ours is not None else "",
            "highs_objective": highs_value if highs_value is not None else highs_status,
            "relative_difference": rel, "independently_verified": flat.get("verified", ""),
            "verifier_message": flat.get("verifier_message", ""), "passed": 1 if passed else 0,
            "wall_seconds": flat.get("wall_seconds", ""), "solver_seconds": flat.get("solver_seconds", ""),
            "iterations": flat.get("iterations", ""), "algorithm": flat.get("algorithm", ""),
            "time_limit": args.time_limit, "git_commit": commit, "machine": machine,
            "timestamp_utc": timestamp, "solver_options": solver_options,
        })
        ours_text = f"{ours:.12e}" if isinstance(ours, (int, float)) else "-"
        highs_text = (f"{highs_value:.12e}" if highs_value is not None else (highs_status or "-"))
        rel_text = f"{rel:.1e}" if rel != "" else "-"
        print(f"{name:22s}{str(flat.get('rows', '')):>9s}{str(flat.get('columns', '')):>9s}"
              f"{flat['status']:>16s}{ours_text:>20s}{highs_text:>20s}{rel_text:>10s}"
              f"{float(flat.get('wall_seconds', 0)):>8.1f}s  {str(flat.get('verified', '-')):>3s}  "
              f"{'PASS' if passed else 'FAIL'}")
        if flat.get("verifier_message"):
            print(f"{'':22s}verifier: {flat['verifier_message'][:90]}")

    finished = [float(r["solver_seconds"]) for r in rows if r["status"] == "optimal" and r["solver_seconds"] != ""]
    failed = [r["instance"] for r in rows if r["passed"] != 1]
    print("-" * 122)
    print(f"{sum(r['passed'] == 1 for r in rows)}/{len(rows)} optimal, verified"
          + (" and agreeing with HiGHS" if have_highs else "") + f" within {args.time_limit:g}s")
    if finished:
        print(f"shifted geometric mean solve time over the {len(finished)} that finished "
              f"(shift {SHIFT_SECONDS:g}s): {shifted_geometric_mean(finished, SHIFT_SECONDS):.2f}s")
    if failed:
        print(f"failed: {', '.join(failed)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"mittelmann-{commit}.csv")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0 if rows and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
