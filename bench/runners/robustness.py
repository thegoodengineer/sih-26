#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The robustness sweep (#71): push each numerical hazard until the answer moves, and record
where that happens.

PS26119 asks for "a clear demonstration of numerical robustness ... involving degeneracy,
weak LP relaxations or ill-conditioned constraint matrices". data/casestudies/ demonstrates
each hazard on one chosen instance. A demonstration proves we handle a case; a sweep finds
the case we do not, and the number it produces - "reliable to an entry spread of 1e9,
wrong at 1e12, and here is why" - is the honest form of the claim.

Every instance is built BACKWARDS from a chosen primal-dual pair (the KKT construction the
oracle fuzz uses, tests/oracles/lp_generator.hpp), so its optimal objective is known
exactly before anything is solved and the sweep needs no second solver. Each family then
applies one transformation with a parameter that grows until something breaks:

  conditioning   row scales 10^(+-k/2) and column scales 10^(-+k/2), alternating, so the
                 entries spread over 10^k. An exact change of variables: the optimum is
                 unchanged, and the solver has no excuse.
  near_parallel  every active row gets a twin that differs in one coefficient by a
                 relative 10^-k; the twin is active at the optimum with a zero multiplier,
                 so the optimum is unchanged and the basis is nearly singular.
  cost_ratio     the dual multipliers and reduced costs used in the construction span
                 1 .. 10^k, so the cost vector does too, with the optimum still known.
  redundancy     k times the row count of extra rows, each a positive combination of two
                 active rows with the matching right-hand side: implied, active, useless.
  degeneracy     k times the column count of rows, all active at the optimum: a vertex
                 with k*n tight constraints in n dimensions.

Each point is solved through the CLI exactly as a judge would run it, verified by
tools/verify_solution.py, and written to bench/results/robustness-<commit>.csv. The summary
names, per family, the last parameter that passed and the first that failed, and the
reason. docs/BENCHMARKS.md reports that table from the CSV.

    python bench/runners/robustness.py            # the full sweep, a few minutes
    python bench/runners/robustness.py --quick    # the CI size: one instance per point,
                                                  # shorter sweeps
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

CSV_COLUMNS = [
    "family", "parameter", "value", "seed", "rows", "columns", "status", "objective",
    "expected_objective", "relative_error", "primal_infeasibility", "dual_infeasibility",
    "verified", "iterations", "solver_seconds", "message", "git_commit", "machine",
    "timestamp_utc", "solver_options",
]

# An answer counts as right when it is optimal and within this of the known objective.
PASS_RELATIVE_TOLERANCE = 1e-6


# =========================================================================================
# The KKT construction: an LP whose optimum is known before it is solved
# =========================================================================================

class Instance:
    """min c'x  s.t.  A x >= b (rows), l <= x <= u. Rows are (coefficients, lower, upper)
    with upper = +inf for a >= row and lower == upper for an equality."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cost: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.rows: list[tuple[dict[int, float], float, float]] = []
        self.expected = 0.0

    def num_cols(self) -> int:
        return len(self.cost)

    def num_rows(self) -> int:
        return len(self.rows)


def kkt_instance(rng: random.Random, cols: int, rows: int, active_fraction: float,
                 multiplier_span: int = 0, name: str = "KKT") -> Instance:
    """Build an LP around a chosen x* and y*.

    x* >= 0 with some zeros; A integer and sparse; the first `active` rows are tight at x*
    with multipliers y* > 0, the rest are slack with y* = 0; c = A'y* + d with d >= 0 and
    d_j = 0 wherever x*_j > 0. Then x* is optimal by the KKT conditions and c'x* is the
    optimum. `multiplier_span` = k makes y* and d range over 1 .. 10^k (the cost_ratio
    family); 0 keeps them small.
    """
    x_star = [0 if rng.random() < 0.4 else rng.randint(1, 5) for _ in range(cols)]
    if all(v == 0 for v in x_star):
        x_star[0] = 3
    a: list[list[int]] = []
    for _ in range(rows):
        row = [0] * cols
        nonzeros = 0
        for j in range(cols):
            if rng.random() < 0.6:
                row[j] = rng.randint(-6, 6)
                if row[j] != 0:
                    nonzeros += 1
        if nonzeros == 0:
            row[rng.randrange(cols)] = rng.randint(1, 6)
        a.append(row)
    active = max(1, int(round(rows * active_fraction)))

    def magnitude() -> int:
        if multiplier_span <= 0:
            return rng.randint(1, 5)
        return rng.randint(1, 5) * (10 ** rng.randint(0, multiplier_span))

    y_star = [magnitude() if i < active else 0 for i in range(rows)]
    d = [0 if x_star[j] > 0 else (0 if rng.random() < 0.3 else magnitude()) for j in range(cols)]
    cost = [sum(a[i][j] * y_star[i] for i in range(rows)) + d[j] for j in range(cols)]
    instance = Instance(name)
    instance.cost = [float(c) for c in cost]
    instance.lower = [0.0] * cols
    instance.upper = [float("inf")] * cols
    for i in range(rows):
        activity = sum(a[i][j] * x_star[j] for j in range(cols))
        rhs = activity if i < active else activity - rng.randint(1, 5)
        entries = {j: float(a[i][j]) for j in range(cols) if a[i][j] != 0}
        instance.rows.append((entries, float(rhs), float("inf")))
    instance.expected = float(sum(cost[j] * x_star[j] for j in range(cols)))
    instance.x_star = x_star  # type: ignore[attr-defined]
    instance.active = active  # type: ignore[attr-defined]
    return instance


# =========================================================================================
# The families: one transformation each, parameter k
# =========================================================================================

def conditioning(rng: random.Random, k: int) -> Instance:
    base = kkt_instance(rng, cols=12, rows=10, active_fraction=0.6, name="CONDITIONING")
    n, m = base.num_cols(), base.num_rows()
    row_scale = [10.0 ** ((1 if i % 2 == 0 else -1) * k / 2.0) for i in range(m)]
    col_scale = [10.0 ** ((-1 if j % 2 == 0 else 1) * k / 2.0) for j in range(n)]
    out = Instance(base.name)
    out.cost = [base.cost[j] * col_scale[j] for j in range(n)]
    out.lower = [base.lower[j] / col_scale[j] for j in range(n)]
    out.upper = [base.upper[j] / col_scale[j] if base.upper[j] != float("inf") else float("inf")
                 for j in range(n)]
    for i, (entries, lo, hi) in enumerate(base.rows):
        scaled = {j: v * row_scale[i] * col_scale[j] for j, v in entries.items()}
        out.rows.append((scaled, lo * row_scale[i], hi * row_scale[i] if hi != float("inf") else hi))
    out.expected = base.expected
    return out


def near_parallel(rng: random.Random, k: int) -> Instance:
    base = kkt_instance(rng, cols=10, rows=8, active_fraction=0.75, name="NEARPARALLEL")
    out = Instance(base.name)
    out.cost, out.lower, out.upper, out.expected = base.cost, base.lower, base.upper, base.expected
    out.rows = list(base.rows)
    x_star = base.x_star  # type: ignore[attr-defined]
    for i in range(base.active):  # type: ignore[attr-defined]
        entries, lo, hi = base.rows[i]
        twin = dict(entries)
        j = min(twin, key=lambda key: key)
        twin[j] = twin[j] * (1.0 + 10.0 ** (-k))
        activity = sum(v * x_star[c] for c, v in twin.items())
        out.rows.append((twin, activity, float("inf")))
    return out


def cost_ratio(rng: random.Random, k: int) -> Instance:
    return kkt_instance(rng, cols=12, rows=10, active_fraction=0.6, multiplier_span=k,
                        name="COSTRATIO")


def redundancy(rng: random.Random, k: int) -> Instance:
    base = kkt_instance(rng, cols=10, rows=8, active_fraction=0.75, name="REDUNDANCY")
    out = Instance(base.name)
    out.cost, out.lower, out.upper, out.expected = base.cost, base.lower, base.upper, base.expected
    out.rows = list(base.rows)
    active = base.active  # type: ignore[attr-defined]
    for t in range(k * base.num_rows()):
        i, j = t % active, (t + 1 + t // active) % active
        wi, wj = rng.randint(1, 3), rng.randint(1, 3)
        entries: dict[int, float] = {}
        for c, v in base.rows[i][0].items():
            entries[c] = entries.get(c, 0.0) + wi * v
        for c, v in base.rows[j][0].items():
            entries[c] = entries.get(c, 0.0) + wj * v
        out.rows.append((entries, wi * base.rows[i][1] + wj * base.rows[j][1], float("inf")))
    return out


def degeneracy(rng: random.Random, k: int) -> Instance:
    cols = 6
    return kkt_instance(rng, cols=cols, rows=max(1, k) * cols, active_fraction=1.0,
                        name="DEGENERACY")


FAMILIES = {
    "conditioning": (conditioning, "entry spread 10^k", list(range(0, 31, 2)), list(range(0, 13, 4))),
    "near_parallel": (near_parallel, "twin rows differing by a relative 10^-k",
                      list(range(0, 17, 1)), [0, 4, 8, 12]),
    "cost_ratio": (cost_ratio, "costs spanning 10^k", list(range(0, 17, 1)), [0, 4, 8, 12]),
    "redundancy": (redundancy, "k times the row count of implied rows", [0, 1, 2, 4, 8, 16, 32],
                   [0, 4, 16]),
    "degeneracy": (degeneracy, "k times the column count of rows, all active",
                   [1, 2, 4, 8, 16, 32, 64], [1, 8, 32]),
}


# =========================================================================================
# MPS, with every digit: the writer must not be the limit the sweep measures
# =========================================================================================

def write_mps(instance: Instance, path: Path) -> None:
    out = [f"NAME          {instance.name}", "ROWS", " N  OBJ"]
    kinds = []
    for i, (_, lo, hi) in enumerate(instance.rows):
        if lo == hi:
            kinds.append("E")
        elif hi == float("inf"):
            kinds.append("G")
        elif lo == float("-inf"):
            kinds.append("L")
        else:
            kinds.append("G")  # ranged: G with a RANGES entry below
        out.append(f" {kinds[-1]}  R{i}")
    out.append("COLUMNS")
    by_col: dict[int, list[tuple[str, float]]] = {j: [] for j in range(instance.num_cols())}
    for j in range(instance.num_cols()):
        if instance.cost[j] != 0.0:
            by_col[j].append(("OBJ", instance.cost[j]))
    for i, (entries, _, _) in enumerate(instance.rows):
        for j, v in entries.items():
            if v != 0.0:
                by_col[j].append((f"R{i}", v))
    for j in range(instance.num_cols()):
        for row, v in by_col[j]:
            out.append(f"    X{j:<9} {row:<9} {v!r}")
        if not by_col[j]:
            out.append(f"    X{j:<9} OBJ       0")
    out.append("RHS")
    for i, (_, lo, hi) in enumerate(instance.rows):
        rhs = lo if kinds[i] in ("E", "G") else hi
        if rhs != 0.0:
            out.append(f"    RHS       R{i:<8} {rhs!r}")
    ranged = [(i, hi - lo) for i, (_, lo, hi) in enumerate(instance.rows)
              if kinds[i] == "G" and hi != float("inf")]
    if ranged:
        out.append("RANGES")
        for i, width in ranged:
            out.append(f"    RNG       R{i:<8} {width!r}")
    bounds = []
    for j in range(instance.num_cols()):
        lo, hi = instance.lower[j], instance.upper[j]
        if lo == float("-inf") and hi == float("inf"):
            bounds.append(f" FR BND       X{j}")
            continue
        if lo != 0.0:
            bounds.append(f" LO BND       X{j:<8} {lo!r}" if lo != float("-inf") else f" MI BND       X{j}")
        if hi != float("inf"):
            bounds.append(f" UP BND       X{j:<8} {hi!r}")
    if bounds:
        out.append("BOUNDS")
        out.extend(bounds)
    out.append("ENDATA")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


# =========================================================================================
# Running and recording
# =========================================================================================

def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-main/sankhya.exe",
                      "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; build first, or pass --binary")


def solve_one(binary: Path, instance: Instance, mps: Path, time_limit: float,
              solver_options: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        stats = Path(tmp) / "stats.json"
        sol = Path(tmp) / "solution.sol"
        command = [str(binary), "solve", str(mps), "--stats", str(stats), "--write-sol",
                   str(sol), "--time-limit", str(time_limit), "--option",
                   "log_to_console=false"]
        for option in solver_options:
            command += ["--option", option]
        completed = subprocess.run(command, capture_output=True, text=True)
        if not stats.exists():
            return {"status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                    "message": completed.stderr.strip()[:300]}
        blob = json.loads(stats.read_text())
        result, effort = blob.get("result", {}), blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "objective": result.get("objective"),
            "primal_infeasibility": result.get("primal_infeasibility", ""),
            "dual_infeasibility": result.get("dual_infeasibility", ""),
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "verified": "",
        }
        if sol.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run([sys.executable, str(VERIFIER), str(mps), str(sol)],
                                   capture_output=True, text=True)
            flat["verified"] = 1 if check.returncode == 0 else 0
            if check.returncode != 0:
                # The failing check, so the CSV says WHICH condition the point violates.
                failing = [line.strip() for line in check.stdout.splitlines() if "[FAIL]" in line]
                flat["verifier"] = "; ".join(failing)[:200]
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--quick", action="store_true",
                        help="the CI size: one instance per point, shorter sweeps")
    parser.add_argument("--instances", type=int, default=3,
                        help="instances per sweep point (default 3; --quick forces 1)")
    parser.add_argument("--families", nargs="*", choices=sorted(FAMILIES), default=None)
    parser.add_argument("--solver-option", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verifier", type=Path, default=None,
                        help="a different tools/verify_solution.py to judge the certificates "
                             "with (the verifier's own tolerances are part of what a sweep "
                             "measures)")
    args = parser.parse_args()
    global VERIFIER
    if args.verifier is not None:
        VERIFIER = args.verifier.resolve()

    binary = args.binary or default_binary()
    commit = git_commit()
    machine = f"{platform.system()}-{platform.machine()}"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    instances_per_point = 1 if args.quick else args.instances
    solver_options = " ".join(args.solver_option)
    families = args.families or list(FAMILIES)

    rows: list[dict] = []
    print(f"solver   {binary}\ncommit   {commit}   machine {machine}"
          + (f"   options {solver_options}" if solver_options else ""))
    print(f"{'family':14s} {'k':>4s} {'seed':>5s} {'rows':>5s} {'cols':>5s} {'status':10s} "
          f"{'rel err':>9s} {'iters':>6s} verified")
    summary: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for family in families:
            build, parameter, full_sweep, quick_sweep = FAMILIES[family]
            sweep = quick_sweep if args.quick else full_sweep
            last_pass, first_fail, first_fail_reason = None, None, ""
            for k in sweep:
                point_ok = True
                for seed_index in range(instances_per_point):
                    seed = 71000 + 1000 * seed_index + k
                    rng = random.Random(seed)
                    instance = build(rng, k)
                    mps = Path(tmp) / f"{family}_{k}_{seed}.mps"
                    write_mps(instance, mps)
                    flat = solve_one(binary, instance, mps, args.time_limit, args.solver_option)
                    ours = flat.get("objective")
                    expected = instance.expected
                    rel = (abs(ours - expected) / max(1.0, abs(expected))
                           if isinstance(ours, (int, float)) else float("inf"))
                    # A pass needs all three: the status, the known objective, and the
                    # independent verifier's acceptance of the certificate. An answer that
                    # is right but cannot be verified is reported as exactly that.
                    passed = (flat["status"] == "optimal" and rel <= PASS_RELATIVE_TOLERANCE
                              and flat.get("verified") == 1)
                    if not passed:
                        point_ok = False
                        if first_fail is None:
                            first_fail = k
                            detail = flat.get("verifier") or flat.get("message", "")
                            first_fail_reason = (f"{flat['status']}, relative error {rel:.2e}, "
                                                 f"verified {flat.get('verified', '')}"
                                                 + (f": {detail[:140]}" if detail else ""))
                    rows.append({
                        "family": family, "parameter": parameter, "value": k, "seed": seed,
                        "rows": instance.num_rows(), "columns": instance.num_cols(),
                        "status": flat["status"], "objective": ours, "expected_objective": expected,
                        "relative_error": rel, "primal_infeasibility": flat.get("primal_infeasibility", ""),
                        "dual_infeasibility": flat.get("dual_infeasibility", ""),
                        "verified": flat.get("verified", ""), "iterations": flat.get("iterations", ""),
                        "solver_seconds": flat.get("solver_seconds", ""),
                        "message": (flat.get("verifier") or flat.get("message", ""))[:200],
                        "git_commit": commit,
                        "machine": machine, "timestamp_utc": timestamp,
                        "solver_options": solver_options,
                    })
                    print(f"{family:14s} {k:>4d} {seed:>5d} {instance.num_rows():>5d} "
                          f"{instance.num_cols():>5d} {flat['status']:10s} {rel:>9.2e} "
                          f"{str(flat.get('iterations', '')):>6s} {flat.get('verified', '')}")
                if point_ok and first_fail is None:
                    last_pass = k
            summary[family] = {"parameter": parameter, "last_pass": last_pass,
                               "first_fail": first_fail, "reason": first_fail_reason,
                               "sweep": sweep}

    print("\nwhere each family breaks:")
    for family, s in summary.items():
        if s["first_fail"] is None:
            print(f"  {family:14s} passes the whole sweep (k up to {s['sweep'][-1]}; {s['parameter']})")
        else:
            print(f"  {family:14s} last pass k={s['last_pass']}, first failure k={s['first_fail']} "
                  f"({s['parameter']}): {s['reason']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or (RESULTS_DIR / f"robustness-{'quick-' if args.quick else ''}{commit}.csv")
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
