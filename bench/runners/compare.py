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


CSV_COLUMNS = [
    "instance",
    "published_objective",
    "sankhya_status",
    "sankhya_objective",
    "sankhya_relative_error",
    "sankhya_seconds",
    # The envelope, not just the point estimate. A reader checking whether the ratio above
    # is trustworthy needs to see how far the samples it came from were spread.
    "sankhya_seconds_min",
    "sankhya_seconds_max",
    "sankhya_iterations",
    "highs_status",
    "highs_objective",
    "highs_relative_error",
    "highs_seconds",
    "highs_seconds_min",
    "highs_seconds_max",
    "timing_repeats",
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
    """A HiGHS command-line binary, if the machine has one."""
    if explicit is not None:
        return explicit if explicit.exists() else None
    found = shutil.which("highs") or shutil.which("highs.exe")
    return Path(found) if found else None


def find_highspy():
    """The `highspy` package, as a fallback when no HiGHS binary is installed.

    Still an external solver, still never linked into SANKHYA: `highspy` is a pip package
    that this benchmark script imports, and nothing in `src/` knows it exists. It is not a
    build dependency and no part of HiGHS informs our code. See the red line in CLAUDE.md
    and section 1 of docs/PROVENANCE.md.
    """
    try:
        import highspy  # noqa: PLC0415 - optional, only needed for the comparison
        return highspy
    except ImportError:
        return None


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
        effort = blob.get("effort", {})
        # Solver-internal time, to match what HiGHS reports. Process wall time is kept
        # alongside it because it is the number a user actually waits through, but it is not
        # the number the two solvers are compared on - it would be dominated by start-up on
        # instances this small.
        internal = as_number(effort.get("solve_seconds"))
        return {
            "status": blob.get("result", {}).get("status", "unknown"),
            "objective": as_number(blob.get("result", {}).get("objective")),
            "seconds": wall if internal is None else internal,
            "wall_seconds": wall,
            "iterations": effort.get("iterations", ""),
        }


# HiGHS prints a summary block; these are the lines we need out of it.
HIGHS_STATUS = re.compile(r"^\s*Model\s+status\s*:\s*(.+?)\s*$", re.M | re.I)
HIGHS_OBJECTIVE = re.compile(r"^\s*Objective\s+value\s*:\s*(-?[\d.eE+]+)\s*$", re.M | re.I)
HIGHS_ITERATIONS = re.compile(r"^\s*Simplex\s+iterations?\s*:\s*(\d+)\s*$", re.M | re.I)
HIGHS_RUNTIME = re.compile(r"^\s*HiGHS\s+run\s+time\s*:\s*([\d.eE+-]+)\s*$", re.M | re.I)


def run_highs_binary(binary: Path, mps: Path, time_limit: float) -> dict:
    started = time.perf_counter()
    completed = subprocess.run([str(binary), str(mps), "--time_limit", str(time_limit)],
                               capture_output=True, text=True)
    wall = time.perf_counter() - started
    text = completed.stdout + completed.stderr

    status = HIGHS_STATUS.search(text)
    objective = HIGHS_OBJECTIVE.search(text)
    iterations = HIGHS_ITERATIONS.search(text)
    runtime = HIGHS_RUNTIME.search(text)
    return {
        "status": status.group(1).strip().lower() if status else "unparsed",
        "objective": float(objective.group(1)) if objective else None,
        "iterations": int(iterations.group(1)) if iterations else "",
        # Prefer HiGHS's own reported run time; fall back to wall clock.
        "seconds": float(runtime.group(1)) if runtime else wall,
        "wall_seconds": wall,
        "raw": text[-500:] if objective is None else "",
    }


def run_highs_python(highspy, mps: Path, time_limit: float) -> dict:
    """Solve through the `highspy` package, reporting HiGHS's OWN run time.

    Timing note, and it matters for honesty: a Python process pays roughly 50 ms of
    interpreter start-up, which is more than ten times what HiGHS takes to solve `afiro`.
    Timing the process would therefore make our solver look good for a reason that has
    nothing to do with either solver. Both sides are measured by their own internal solve
    time instead - HiGHS's `getRunTime()` against our `effort.solve_seconds` - which is also
    what published benchmark tables report.
    """
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("time_limit", float(time_limit))
    started = time.perf_counter()
    solver.readModel(str(mps))
    solver.run()
    wall = time.perf_counter() - started

    info = solver.getInfo()
    status = solver.modelStatusToString(solver.getModelStatus()).strip().lower()
    optimal = status == "optimal"
    return {
        "status": status,
        "objective": info.objective_function_value if optimal else None,
        "iterations": info.simplex_iteration_count,
        "seconds": solver.getRunTime(),
        "wall_seconds": wall,
        "raw": "",
    }


def repeat(solve, repeats: int, warmup: int) -> dict:
    """Run one solver on one instance several times and keep the MEDIAN solve time.

    WHY THIS EXISTS. The instances in the small set solve in one to fifteen MILLISECONDS.
    At that scale a single timing measures the operating system's scheduling luck at least
    as much as it measures either solver, and the reported figure moved run to run without
    a line of code changing: across five consecutive runs of an unchanged binary the median
    ratio came out 0.89x, 0.93x, 0.97x, 0.92x and 0.71x, and one instance swung by a factor
    of three. A benchmark number that changes when nothing changed is not evidence, and this
    project's whole claim is that its numbers are reproducible.

    The median, not the mean: timing noise on a loaded desktop is one-sided - a run can be
    delayed arbitrarily but cannot finish faster than the work takes - so the distribution
    has a long right tail that drags a mean around. The median ignores it.

    The first `warmup` runs are discarded. The first solve of an instance pays for reading
    the file into the OS page cache, and for our process, for loading the binary itself.
    That cost is real but it is not solve time, and charging it to whichever solver happens
    to run first is exactly the kind of accident this function exists to remove.

    The objective is checked for stability across repeats and reported if it moves. Both
    solvers are deterministic on a fixed input, so a wandering objective is a bug worth
    surfacing loudly rather than averaging into silence.
    """
    samples: list[dict] = []
    for index in range(warmup + repeats):
        result = solve()
        if index >= warmup:
            samples.append(result)

    seconds = sorted(sample["seconds"] for sample in samples)
    median_seconds = seconds[len(seconds) // 2]

    # Report the run whose time IS the median, so the objective, status and iteration count
    # all come from one real solve rather than being stitched together across runs.
    chosen = min(samples, key=lambda sample: abs(sample["seconds"] - median_seconds))

    objectives = {None if sample["objective"] is None else round(sample["objective"], 9)
                  for sample in samples}

    out = dict(chosen)
    out["seconds"] = median_seconds
    out["seconds_min"] = seconds[0]
    out["seconds_max"] = seconds[-1]
    out["repeats"] = len(samples)
    out["objective_stable"] = len(objectives) == 1
    return out


def envelopes_overlap(ours: dict, theirs: dict) -> bool:
    """Do the two solvers' observed timing ranges intersect?

    This is the question the ratio column silently pretends to have answered. Repeating the
    measurement was supposed to stabilise that ratio; it did not, and the raw envelopes say
    why. On adlittle our 31 samples spanned 1.5-4.3 ms and HiGHS's spanned 1.5-4.1 ms - the
    two ranges sit almost exactly on top of each other, so the medians can land either side
    of 1.0x depending on nothing but which process got descheduled more. Taking more samples
    narrows the error on each median but does not separate distributions that genuinely
    overlap, which is why the headline ratio kept moving at 5, 15 and 31 repeats alike.

    When this returns True the honest reading of that row is "we cannot tell these apart on
    this instance", and printing a two-decimal ratio without saying so invites exactly the
    wrong conclusion.
    """
    return ours["seconds_min"] <= theirs["seconds_max"] and         theirs["seconds_min"] <= ours["seconds_max"]


def spread_fraction(sample: dict) -> float | None:
    """(max - min) / median, the honest measure of how much a timing still wobbles."""
    median = sample.get("seconds")
    if not median or sample.get("seconds_min") is None:
        return None
    return (sample["seconds_max"] - sample["seconds_min"]) / median


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
    parser.add_argument("--repeats", type=int, default=15,
                        help="timed runs per solver per instance; the MEDIAN is reported. "
                             "1 reproduces the old single-shot behaviour and its noise.")
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed runs before the timed ones, to pay the page-cache "
                             "and process-load cost once rather than charging it to "
                             "whichever solver ran first")
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths are resolved "
                             "against the repository root")
    args = parser.parse_args()

    # Prefer a real command-line binary; fall back to the highspy package. Either way HiGHS
    # runs as a separate solver and is never linked into SANKHYA.
    highs = find_highs(args.highs_binary)
    highspy = None if highs is not None else find_highspy()
    if highs is None and highspy is None:
        # Refusing to invent numbers is the whole point of this project. No HiGHS, no row.
        print("HiGHS was not found on this machine, so no comparison was run.")
        print()
        print("Install it (none of these affects the SANKHYA build):")
        print("    pip install highspy")
        print("    apt-get install highs")
        print("    conda install -c conda-forge highs")
        print()
        print("then re-run:")
        print("    python bench/runners/compare.py --time-limit 60")
        return 2
    backend = str(highs) if highs is not None else "highspy (pip package, separate process)"

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
    print(f"HiGHS    {backend}   (external solver; never linked)")
    print("times are SOLVER-INTERNAL on both sides: HiGHS getRunTime() against our")
    print("         effort.solve_seconds. Process start-up is excluded for both.")
    print(f"each time is the MEDIAN of {args.repeats} run(s) after {args.warmup} discarded"
          f" warm-up; these instances")
    print("         solve in milliseconds, so a single timing measures scheduling luck.")
    print(f"machine  {machine}   time limit {args.time_limit}s")
    print()
    print(f"{'instance':<11}{'SANKHYA obj':>20}{'HiGHS obj':>20}{'agree':>7}"
          f"{'SANKHYA s':>11}{'HiGHS s':>10}{'ratio':>8}")
    print("-" * 87)

    rows: list[dict] = []
    timing_spread: list[tuple[str, dict]] = []
    resolved: list[bool] = []
    for name in names:
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            continue
        published = reference[name]["published_optimal"]
        ours = repeat(lambda: run_sankhya(sankhya, mps, args.time_limit),
                      args.repeats, args.warmup)
        theirs = repeat(
            lambda: (run_highs_binary(highs, mps, args.time_limit) if highs is not None
                     else run_highs_python(highspy, mps, args.time_limit)),
            args.repeats, args.warmup)

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
            "sankhya_seconds_min": round(ours["seconds_min"], 6),
            "sankhya_seconds_max": round(ours["seconds_max"], 6),
            "sankhya_iterations": ours["iterations"],
            "highs_status": theirs["status"],
            "highs_objective": "" if theirs["objective"] is None else repr(theirs["objective"]),
            "highs_relative_error": "" if their_error is None else repr(their_error),
            "highs_seconds": round(theirs["seconds"], 6),
            "highs_seconds_min": round(theirs["seconds_min"], 6),
            "highs_seconds_max": round(theirs["seconds_max"], 6),
            "timing_repeats": ours["repeats"],
            "objectives_agree": int(agree),
            "speed_ratio_sankhya_over_highs": "" if ratio is None else round(ratio, 3),
            "git_commit": commit,
            "machine": machine,
            "timestamp_utc": timestamp,
        })

        timing_spread.append((name, ours))
        if ours["repeats"] > 1 and theirs["repeats"] > 1:
            resolved.append(not envelopes_overlap(ours, theirs))

        our_text = "-" if ours["objective"] is None else f"{ours['objective']:>20.10e}"
        their_text = "-" if theirs["objective"] is None else f"{theirs['objective']:>20.10e}"
        ratio_text = "-" if ratio is None else f"{ratio:>8.2f}"
        overlap = (ours["repeats"] > 1 and theirs["repeats"] > 1
                   and envelopes_overlap(ours, theirs))
        print(f"{name:<11}{our_text}{their_text}{'yes' if agree else 'NO':>7}"
              f"{ours['seconds']:>10.3f}s{theirs['seconds']:>9.3f}s{ratio_text}"
              f"{'  ~' if overlap else ''}")
        if not ours["objective_stable"]:
            print(f"           SANKHYA's objective MOVED across repeats on {name} - "
                  f"the solver is not deterministic on this input")
        if not theirs["objective_stable"]:
            print(f"           HiGHS's objective moved across repeats on {name}")
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
        print(f"median solve-time ratio SANKHYA/HiGHS: {median:.2f}x  (>1 means we are slower)")
        print(f"  ratios across instances span {ordered[0]:.2f}x to {ordered[-1]:.2f}x")

    # How much each individual timing still wobbles, stated rather than hidden. If this is
    # large the median above is still the best estimate available, but the reader deserves
    # to know how much to trust the last digit.
    worst = None
    for name, sample in timing_spread:
        fraction = spread_fraction(sample)
        if fraction is not None and (worst is None or fraction > worst[1]):
            worst = (name, fraction)
    if worst is not None:
        print(f"  worst per-instance timing spread (max-min)/median: "
              f"{worst[1] * 100:.0f}% on {worst[0]}")

    # THE HEADLINE NUMBER IS NOT ALLOWED TO STAND ALONE. Rows marked `~` are ones where the
    # two solvers' timing ranges overlap, meaning this measurement did not separate them.
    # Reporting a ratio without that count is how "0.9x" gets read as "we are faster" when
    # the data supports nothing of the kind.
    if resolved:
        indistinguishable = resolved.count(False)
        if indistinguishable:
            print(f"  ~ marks the {indistinguishable} of {len(resolved)} instance(s) whose "
                  f"timing envelopes OVERLAP. On those this")
            print("    measurement does not separate the two solvers: read the ratio beside "
                  "them as")
            print("    'indistinguishable', not as a speed result.")
        print("  MORE REPEATS DO NOT FIX THIS. On an unchanged binary the median ratio came "
              "out")
        print("    1.60x and 1.03x at 1 repeat, 0.98x and 1.50x at 5, 1.04x and 0.83x at 15,")
        print("    0.78x and 0.95x at 31. Averaging narrows each median but cannot separate")
        print("    distributions that genuinely overlap. These models solve in single-digit")
        print("    milliseconds; deciding which solver is faster needs instances big enough")
        print("    to time at all - see issue #34.")

    # Tier in the filename, for the same reason netlib.py carries one: running the medium
    # comparison after the small one at the same commit otherwise overwrites it, and
    # docs/BENCHMARKS.md can then only ever describe whichever ran last.
    tier = json.loads((DATA_DIR / "reference.json").read_text()).get("instance_set", "")
    tier_tag = f"{tier}-" if tier and tier != "explicit" else ""
    out_path = args.out or (RESULTS_DIR / f"compare-highs-{tier_tag}{commit}.csv")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
