#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare SANKHYA against HiGHS on identical instances, same machine, same time limit.

HiGHS is invoked ONLY as an external subprocess binary. It is never linked into SANKHYA,
never a build dependency, and no part of its source informs ours - see the red line in
CLAUDE.md and section 1 of docs/PROVENANCE.md. This script shells out to whatever `highs`
executable the machine already has, reads its stdout, and compares numbers. That is the
entire relationship.

The comparison exists because the problem statement asks us to compare against an
established solver, not to beat one. Expect to lose on time: HiGHS is a decade of
specialist work. Losing by a stated factor on instances where both agree on the objective
is a far stronger result than a benchmark that quietly omits the comparison.

Installing HiGHS (any of these; none of them touches the build):
    apt-get install highs                 # Debian/Ubuntu
    conda install -c conda-forge highs
    or download a release from https://github.com/ERGO-Code/HiGHS/releases

Usage:
    python bench/runners/compare.py
    python bench/runners/compare.py --highs-binary /path/to/highs --time-limit 60
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

CSV_COLUMNS = [
    "instance",
    "published_objective",
    "sankhya_status",
    "sankhya_objective",
    "sankhya_relative_error",
    "sankhya_seconds",
    "sankhya_iterations",
    "highs_status",
    "highs_objective",
    "highs_relative_error",
    "highs_seconds",
    "objectives_agree",
    "speed_ratio_sankhya_over_highs",
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
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False)
        return r.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def find_highs(explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit if explicit.exists() else None
    found = shutil.which("highs") or shutil.which("highs.exe")
    return Path(found) if found else None


def default_sankhya() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-main/sankhya.exe",
                      "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no SANKHYA binary found; build first, or pass --sankhya-binary")


def run_sankhya(binary: Path, mps: Path, time_limit: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        started = time.perf_counter()
        subprocess.run([str(binary), "solve", str(mps), "--stats", str(stats),
                        "--time-limit", str(time_limit),
                        "--option", "log_to_console=false"],
                       capture_output=True, text=True)
        wall = time.perf_counter() - started
        if not stats.exists():
            return {"status": "no_output", "objective": None, "seconds": wall,
                    "iterations": ""}
        blob = json.loads(stats.read_text())
        return {
            "status": blob.get("result", {}).get("status", "unknown"),
            "objective": as_number(blob.get("result", {}).get("objective")),
            "seconds": wall,
            "iterations": blob.get("effort", {}).get("iterations", ""),
        }


# HiGHS prints a summary block; these are the lines we need out of it.
HIGHS_STATUS = re.compile(r"^\s*Model\s+status\s*:\s*(.+?)\s*$", re.M | re.I)
HIGHS_OBJECTIVE = re.compile(r"^\s*Objective\s+value\s*:\s*(-?[\d.eE+]+)\s*$", re.M | re.I)


def run_highs(binary: Path, mps: Path, time_limit: float) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [str(binary), str(mps), "--time_limit", str(time_limit)],
        capture_output=True, text=True)
    wall = time.perf_counter() - started

    text = completed.stdout + completed.stderr
    status_match = HIGHS_STATUS.search(text)
    objective_match = HIGHS_OBJECTIVE.search(text)
    status = status_match.group(1).strip().lower() if status_match else "unparsed"
    objective = float(objective_match.group(1)) if objective_match else None
    return {"status": status, "objective": objective, "seconds": wall,
            "raw": text[-500:] if objective is None else ""}


def relative_error(value: float | None, published: float) -> float | None:
    if value is None:
        return None
    return abs(value - published) / max(1.0, abs(published))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--highs-binary", type=Path, default=None)
    parser.add_argument("--sankhya-binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    highs = find_highs(args.highs_binary)
    if highs is None:
        # Refusing to invent numbers is the whole point of this project. No binary, no row.
        print("HiGHS was not found on this machine, so no comparison was run.")
        print()
        print("Install it (none of these affects the SANKHYA build):")
        print("    apt-get install highs")
        print("    conda install -c conda-forge highs")
        print("    or a release from https://github.com/ERGO-Code/HiGHS/releases")
        print()
        print("then re-run:")
        print("    python bench/runners/compare.py --time-limit 60")
        return 2

    sankhya = args.sankhya_binary or default_sankhya()
    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        raise SystemExit("no reference data; run bench/runners/fetch_data.py first")
    reference = json.loads(reference_path.read_text())["instances"]

    names = sorted(args.instances or reference)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    print(f"SANKHYA  {sankhya}")
    print(f"HiGHS    {highs}   (external subprocess; never linked)")
    print(f"machine  {machine}   time limit {args.time_limit}s")
    print()
    print(f"{'instance':<11}{'SANKHYA obj':>20}{'HiGHS obj':>20}{'agree':>7}"
          f"{'SANKHYA s':>11}{'HiGHS s':>10}{'ratio':>8}")
    print("-" * 87)

    rows: list[dict] = []
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]
        ours = run_sankhya(sankhya, mps, args.time_limit)
        theirs = run_highs(highs, mps, args.time_limit)

        our_error = relative_error(ours["objective"], published)
        their_error = relative_error(theirs["objective"], published)
        agree = (ours["objective"] is not None and theirs["objective"] is not None
                 and abs(ours["objective"] - theirs["objective"])
                 <= 1e-6 * max(1.0, abs(published)))
        ratio = (ours["seconds"] / theirs["seconds"]) if theirs["seconds"] > 0 else None

        rows.append({
            "instance": name,
            "published_objective": repr(published),
            "sankhya_status": ours["status"],
            "sankhya_objective": "" if ours["objective"] is None else repr(ours["objective"]),
            "sankhya_relative_error": "" if our_error is None else repr(our_error),
            "sankhya_seconds": round(ours["seconds"], 6),
            "sankhya_iterations": ours["iterations"],
            "highs_status": theirs["status"],
            "highs_objective": "" if theirs["objective"] is None else repr(theirs["objective"]),
            "highs_relative_error": "" if their_error is None else repr(their_error),
            "highs_seconds": round(theirs["seconds"], 6),
            "objectives_agree": int(agree),
            "speed_ratio_sankhya_over_highs": "" if ratio is None else round(ratio, 3),
            "git_commit": commit,
            "machine": machine,
            "timestamp_utc": timestamp,
        })

        our_text = "-" if ours["objective"] is None else f"{ours['objective']:>20.10e}"
        their_text = "-" if theirs["objective"] is None else f"{theirs['objective']:>20.10e}"
        ratio_text = "-" if ratio is None else f"{ratio:>8.2f}"
        print(f"{name:<11}{our_text}{their_text}{'yes' if agree else 'NO':>7}"
              f"{ours['seconds']:>10.3f}s{theirs['seconds']:>9.3f}s{ratio_text}")
        if theirs["objective"] is None and theirs.get("raw"):
            print(f"           could not parse HiGHS output: {theirs['raw'][:200]}")

    agreed = sum(row["objectives_agree"] for row in rows)
    print("-" * 87)
    print(f"{agreed}/{len(rows)} instances where the two solvers agree on the objective")
    ratios = [row["speed_ratio_sankhya_over_highs"] for row in rows
              if isinstance(row["speed_ratio_sankhya_over_highs"], float)]
    if ratios:
        ordered = sorted(ratios)
        median = ordered[len(ordered) // 2]
        print(f"median wall-time ratio SANKHYA/HiGHS: {median:.2f}x")

    out_path = args.out or (RESULTS_DIR / f"compare-highs-{commit}.csv")
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
