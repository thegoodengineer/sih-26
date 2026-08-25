#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently verify a SANKHYA solution. Trusts nothing the solver said.

    python tools/verify_solution.py model.mps solution.sol

This script exists so that "our answer is correct" is something a judge can CHECK rather
than believe. Three rules make that real, and all three are load-bearing:

1.  IT LINKS NO PART OF OUR C++. Pure Python, standard library only. It reads the model
    file and the .sol file as text.

2.  IT PARSES THE MODEL ITSELF. The MPS reader below is a second, independent
    implementation written from the IBM specification. It does NOT reuse our C++ reader's
    interpretation of anything. That is the entire point: if src/io/mps_reader.cpp gets a
    RANGES sign or the negative-UP bound convention wrong, this script reads the file
    differently and the disagreement surfaces as a failed check. A verifier that shared the
    reader with the solver would confirm the bug instead of catching it.

3.  IT RECOMPUTES EVERY NUMBER. Row activities, the objective, reduced costs and the dual
    objective are all recomputed from (model, x, y) rather than read from the .sol header.
    The solver's own reported infeasibilities are printed next to ours purely so that a
    disagreement between them is visible.

Exit codes:  0 every check passed   1 a check failed   2 could not read the inputs
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
from pathlib import Path

INF = math.inf
MPS_INFINITY = 1e30

# Defaults mirror include/sankhya/tolerances.hpp. They are CLI-overridable because a judge
# should be able to tighten them and watch what happens.
DEFAULT_PRIMAL_TOL = 1e-7
DEFAULT_DUAL_TOL = 1e-7
DEFAULT_INTEGER_TOL = 1e-6
DEFAULT_DUALITY_TOL = 1e-9


def normalize_infinity(value: float) -> float:
    """MPS uses 1e30 for infinity. Normalise so only true infinities reach the checks."""
    if value >= MPS_INFINITY:
        return INF
    if value <= -MPS_INFINITY:
        return -INF
    return value


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


# =========================================================================================
# An independent MPS reader
# =========================================================================================


class Model:
    def __init__(self) -> None:
        self.name = ""
        self.maximize = False
        self.objective_offset = 0.0
        self.col_names: list[str] = []
        self.col_index: dict[str, int] = {}
        self.col_cost: list[float] = []
        self.col_lower: list[float] = []
        self.col_upper: list[float] = []
        self.col_integer: list[bool] = []
        self.row_names: list[str] = []
        self.row_index: dict[str, int] = {}
        self.row_lower: list[float] = []
        self.row_upper: list[float] = []
        # Column-wise entries: entries[j] is a list of (row, value).
        self.entries: list[list[tuple[int, float]]] = []

    @property
    def num_rows(self) -> int:
        return len(self.row_names)

    @property
    def num_cols(self) -> int:
        return len(self.col_names)

    def add_column(self, name: str, integer: bool) -> int:
        j = self.col_index.get(name)
        if j is not None:
            return j
        j = len(self.col_names)
        self.col_index[name] = j
        self.col_names.append(name)
        self.col_cost.append(0.0)
        self.col_lower.append(0.0)  # MPS default
        self.col_upper.append(INF)  # MPS default
        self.col_integer.append(integer)
        self.entries.append([])
        return j


def parse_mps(path: Path) -> Model:
    """Read an MPS file per the IBM specification. Free-format tokenisation."""
    model = Model()
    section = ""
    objective_row = None
    dropped_rows: set[str] = set()
    row_kind: list[str] = []
    row_rhs: list[float] = []
    row_range: list[float | None] = []
    integer_marker = False
    lower_set: list[bool] = []

    with open_text(path) as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line or line.startswith("*"):
                continue
            if not line[0].isspace():
                head = line.split()
                key = head[0].upper()
                if key == "NAME":
                    model.name = head[1] if len(head) > 1 else ""
                    section = "NAME"
                elif key in ("OBJSENSE", "OBJSENS"):
                    section = "OBJSENSE"
                    if len(head) > 1:
                        model.maximize = head[1].upper().startswith("MAX")
                elif key in ("ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS"):
                    section = key
                elif key == "ENDATA":
                    break
                else:
                    raise ValueError(f"{path}:{lineno}: unknown section {head[0]}")
                continue

            fields = line.split()
            if not fields:
                continue

            if section == "OBJSENSE":
                model.maximize = fields[0].upper().startswith("MAX")
                continue

            if section == "ROWS":
                kind = fields[0].upper()
                name = fields[1]
                if kind == "N":
                    if objective_row is None:
                        objective_row = name
                    else:
                        dropped_rows.add(name)
                    continue
                if kind not in ("L", "G", "E"):
                    raise ValueError(f"{path}:{lineno}: unknown row type {kind}")
                model.row_index[name] = len(model.row_names)
                model.row_names.append(name)
                row_kind.append(kind)
                row_rhs.append(0.0)
                row_range.append(None)
                continue

            if section == "COLUMNS":
                upper_fields = [f.upper().replace("'", "").replace('"', "") for f in fields]
                if "MARKER" in upper_fields:
                    if "INTORG" in upper_fields:
                        integer_marker = True
                    elif "INTEND" in upper_fields:
                        integer_marker = False
                    continue
                col = model.add_column(fields[0], integer_marker)
                while len(lower_set) < model.num_cols:
                    lower_set.append(False)
                for k in range(1, len(fields) - 1, 2):
                    row_name, value = fields[k], float(fields[k + 1])
                    if row_name == objective_row:
                        model.col_cost[col] += value
                    elif row_name in dropped_rows:
                        continue
                    elif row_name in model.row_index:
                        model.entries[col].append((model.row_index[row_name], value))
                    else:
                        raise ValueError(f"{path}:{lineno}: unknown row {row_name}")
                continue

            if section in ("RHS", "RANGES"):
                # The set name is optional: if the first token already names a row, it was
                # omitted. Counting tokens alone cannot tell a set name from a row name.
                start = 0 if (fields[0] in model.row_index or fields[0] == objective_row
                              or fields[0] in dropped_rows) else 1
                for k in range(start, len(fields) - 1, 2):
                    row_name, value = fields[k], float(fields[k + 1])
                    if row_name == objective_row:
                        if section == "RHS":
                            # An RHS on the objective row is the NEGATIVE of the constant.
                            model.objective_offset += -value
                        continue
                    if row_name in dropped_rows:
                        continue
                    if row_name not in model.row_index:
                        raise ValueError(f"{path}:{lineno}: unknown row {row_name}")
                    i = model.row_index[row_name]
                    if section == "RHS":
                        row_rhs[i] = value
                    else:
                        row_range[i] = value
                continue

            if section == "BOUNDS":
                kind = fields[0].upper()
                # As for RHS, the bound-set name is optional.
                name_pos = 1 if fields[1] in model.col_index else 2
                col_name = fields[name_pos]
                if col_name not in model.col_index:
                    raise ValueError(f"{path}:{lineno}: unknown column {col_name}")
                j = model.col_index[col_name]
                value = (
                    float(fields[name_pos + 1]) if len(fields) > name_pos + 1 else 0.0
                )
                if kind == "UP":
                    model.col_upper[j] = value
                    # An UP bound with a negative value and no explicit lower bound implies
                    # a lower bound of -infinity. Applied to integer columns too, matching
                    # what src/io/mps_reader.cpp settled on - see judgement call 5 in
                    # docs/PROVENANCE.md. Recorded here so the two agree deliberately rather
                    # than by luck.
                    if value < 0.0 and not lower_set[j]:
                        model.col_lower[j] = -INF
                elif kind == "LO":
                    model.col_lower[j] = value
                    lower_set[j] = True
                elif kind == "FX":
                    model.col_lower[j] = value
                    model.col_upper[j] = value
                    lower_set[j] = True
                elif kind == "FR":
                    model.col_lower[j] = -INF
                    model.col_upper[j] = INF
                    lower_set[j] = True
                elif kind == "MI":
                    model.col_lower[j] = -INF
                    lower_set[j] = True
                elif kind == "PL":
                    model.col_upper[j] = INF
                elif kind == "BV":
                    model.col_integer[j] = True
                    model.col_lower[j] = 0.0
                    model.col_upper[j] = 1.0
                    lower_set[j] = True
                elif kind == "LI":
                    model.col_integer[j] = True
                    model.col_lower[j] = value
                    lower_set[j] = True
                elif kind == "UI":
                    model.col_integer[j] = True
                    model.col_upper[j] = value
                else:
                    raise ValueError(f"{path}:{lineno}: unsupported bound type {kind}")
                continue

    # Resolve (kind, rhs, range) into two-sided row bounds. Written from the IBM table:
    #   G  ->  [b, b + |R|]        L  ->  [b - |R|, b]
    #   E  ->  [b, b + R] if R >= 0 else [b + R, b]
    for i in range(model.num_rows):
        b = row_rhs[i]
        r = row_range[i]
        kind = row_kind[i]
        if r is None:
            lo, hi = (-INF, b) if kind == "L" else (b, INF) if kind == "G" else (b, b)
        else:
            magnitude = abs(r)
            if kind == "G":
                lo, hi = b, b + magnitude
            elif kind == "L":
                lo, hi = b - magnitude, b
            else:
                lo, hi = (b, b + r) if r >= 0.0 else (b + r, b)
        model.row_lower.append(normalize_infinity(lo))
        model.row_upper.append(normalize_infinity(hi))

    model.col_lower = [normalize_infinity(v) for v in model.col_lower]
    model.col_upper = [normalize_infinity(v) for v in model.col_upper]
    return model


# =========================================================================================
# The .sol reader
# =========================================================================================


class Solution:
    def __init__(self) -> None:
        self.header: dict[str, str] = {}
        self.col_value: dict[str, float] = {}
        self.col_dual: dict[str, float] = {}
        self.col_status: dict[str, str] = {}
        self.row_activity: dict[str, float] = {}
        self.row_dual: dict[str, float] = {}
        self.row_status: dict[str, str] = {}

    @property
    def status(self) -> str:
        return self.header.get("status", "unknown")

    def header_float(self, key: str) -> float | None:
        raw = self.header.get(key)
        if raw is None:
            return None
        try:
            return float(raw)
        except ValueError:
            return None


def parse_sol(path: Path) -> Solution:
    solution = Solution()
    block = ""
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if fields[0] == "begin":
                block = fields[1]
                continue
            if fields[0] == "end":
                block = ""
                continue
            if block == "columns" and len(fields) >= 3:
                solution.col_value[fields[0]] = float(fields[1])
                solution.col_dual[fields[0]] = float(fields[2])
                solution.col_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif block == "rows" and len(fields) >= 3:
                solution.row_activity[fields[0]] = float(fields[1])
                solution.row_dual[fields[0]] = float(fields[2])
                solution.row_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif not block and len(fields) >= 2:
                solution.header[fields[0]] = " ".join(fields[1:])
    return solution


# =========================================================================================
# The checks
# =========================================================================================


class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[bool, str, str]] = []
        self.failures = 0

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.lines.append((ok, name, detail))
        if not ok:
            self.failures += 1
        return ok

    def note(self, name: str, detail: str) -> None:
        self.lines.append((True, name, detail))

    def render(self) -> str:
        width = max((len(n) for _, n, _ in self.lines), default=10)
        out = []
        for ok, name, detail in self.lines:
            mark = "PASS" if ok else "FAIL"
            out.append(f"  [{mark}] {name:<{width}}  {detail}")
        return "\n".join(out)


def bound_contribution(multiplier: float, lower: float, upper: float, tol: float) -> float:
    """The Lagrangian term a bound contributes to the dual objective, in minimize space."""
    if multiplier > tol:
        return multiplier * lower
    if multiplier < -tol:
        return multiplier * upper
    return 0.0


def verify(model: Model, solution: Solution, primal_tol: float, dual_tol: float,
           integer_tol: float, duality_tol: float) -> Report:
    report = Report()
    sigma = -1.0 if model.maximize else 1.0

    # ---- Structure ----------------------------------------------------------------------
    missing_cols = [n for n in model.col_names if n not in solution.col_value]
    missing_rows = [n for n in model.row_names if n not in solution.row_activity]
    report.check(
        not missing_cols and not missing_rows,
        "structure",
        f"{model.num_rows} rows, {model.num_cols} columns present"
        if not missing_cols and not missing_rows
        else f"missing {len(missing_cols)} columns, {len(missing_rows)} rows",
    )
    if missing_cols or missing_rows:
        return report

    x = [solution.col_value[n] for n in model.col_names]

    # ---- Column bounds ------------------------------------------------------------------
    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        violation = max(model.col_lower[j] - x[j], x[j] - model.col_upper[j], 0.0)
        if violation > worst:
            worst, where = violation, name
    report.check(worst <= primal_tol, "column bounds",
                 f"worst violation {worst:.3e}" + (f" on {where}" if where else ""))

    # ---- Integrality --------------------------------------------------------------------
    integer_columns = sum(1 for flag in model.col_integer if flag)
    if integer_columns:
        worst, where = 0.0, ""
        for j, name in enumerate(model.col_names):
            if not model.col_integer[j]:
                continue
            violation = abs(x[j] - round(x[j]))
            if violation > worst:
                worst, where = violation, name
        report.check(worst <= integer_tol, "integrality",
                     f"{integer_columns} integer columns, worst {worst:.3e}"
                     + (f" on {where}" if where else ""))
    else:
        report.note("integrality", "no integer columns")

    # ---- Row activity, recomputed from the matrix ----------------------------------------
    activity = [0.0] * model.num_rows
    for j in range(model.num_cols):
        if x[j] == 0.0:
            continue
        for i, value in model.entries[j]:
            activity[i] += value * x[j]

    worst, where = 0.0, ""
    for i, name in enumerate(model.row_names):
        violation = max(model.row_lower[i] - activity[i], activity[i] - model.row_upper[i], 0.0)
        if violation > worst:
            worst, where = violation, name
    report.check(worst <= primal_tol, "row activity",
                 f"worst violation {worst:.3e}" + (f" on {where}" if where else ""))

    # The solver also reports its own activities. A disagreement means one of us computed
    # A*x differently, which is worth surfacing even when both satisfy the bounds.
    worst_activity_gap = max(
        (abs(activity[i] - solution.row_activity[n]) for i, n in enumerate(model.row_names)),
        default=0.0,
    )
    report.check(worst_activity_gap <= 1e-6, "activity agreement",
                 f"max |ours - solver's| = {worst_activity_gap:.3e}")

    # ---- Objective, recomputed -----------------------------------------------------------
    objective = model.objective_offset + sum(model.col_cost[j] * x[j]
                                             for j in range(model.num_cols))
    claimed = solution.header_float("objective")
    if claimed is None:
        report.check(False, "objective", "the .sol file states no objective")
    else:
        scale = max(1.0, abs(objective))
        report.check(abs(objective - claimed) <= 1e-9 * scale, "objective",
                     f"recomputed {objective:.12e}, solver said {claimed:.12e}, "
                     f"difference {abs(objective - claimed):.3e}")

    if solution.status not in ("optimal", "feasible"):
        report.note("duality", f"skipped: status is {solution.status}")
        return report

    if integer_columns:
        # LP duality does not apply to a MILP: the reported duals belong to some node
        # relaxation, not to the integer problem.
        report.note("duality", "skipped: LP duality does not apply to a MILP")
        return report

    # ---- Dual feasibility ----------------------------------------------------------------
    # Work in minimize space so one set of sign conventions covers both senses.
    y = [sigma * solution.row_dual[n] for n in model.row_names]
    d = [sigma * solution.col_dual[n] for n in model.col_names]
    cost = [sigma * c for c in model.col_cost]

    # Reduced costs must satisfy d = c - A^T y. Recomputing catches a solver that reports a
    # dual vector inconsistent with the reduced costs it also reports.
    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        expected = cost[j] - sum(value * y[i] for i, value in model.entries[j])
        difference = abs(expected - d[j])
        if difference > worst:
            worst, where = difference, name
    report.check(worst <= 1e-6, "reduced costs",
                 f"max |c - A^T y - d| = {worst:.3e}" + (f" on {where}" if where else ""))

    def sign_violation(multiplier: float, value: float, lower: float, upper: float) -> float:
        """How badly a multiplier's sign contradicts which bound the value sits at."""
        if lower == upper:
            return 0.0  # fixed: any multiplier is admissible
        at_lower = math.isfinite(lower) and abs(value - lower) <= primal_tol
        at_upper = math.isfinite(upper) and abs(value - upper) <= primal_tol
        if at_lower and not at_upper:
            return max(0.0, -multiplier)
        if at_upper and not at_lower:
            return max(0.0, multiplier)
        if not at_lower and not at_upper:
            return abs(multiplier)  # strictly between: the multiplier must vanish
        return 0.0

    worst, where = 0.0, ""
    for j, name in enumerate(model.col_names):
        violation = sign_violation(d[j], x[j], model.col_lower[j], model.col_upper[j])
        if violation > worst:
            worst, where = violation, name
    report.check(worst <= dual_tol, "dual feasibility (columns)",
                 f"worst {worst:.3e}" + (f" on {where}" if where else ""))

    worst, where = 0.0, ""
    for i, name in enumerate(model.row_names):
        violation = sign_violation(y[i], activity[i], model.row_lower[i], model.row_upper[i])
        if violation > worst:
            worst, where = violation, name
    report.check(worst <= dual_tol, "dual feasibility (rows)",
                 f"worst {worst:.3e}" + (f" on {where}" if where else ""))

    # ---- Complementary slackness ----------------------------------------------------------
    # A multiplier may only be nonzero where its constraint is tight.
    worst, where = 0.0, ""
    for i, name in enumerate(model.row_names):
        if model.row_lower[i] == model.row_upper[i]:
            continue
        slack_lower = (activity[i] - model.row_lower[i]) if math.isfinite(model.row_lower[i]) else INF
        slack_upper = (model.row_upper[i] - activity[i]) if math.isfinite(model.row_upper[i]) else INF
        product = abs(y[i]) * min(slack_lower, slack_upper)
        if product > worst:
            worst, where = product, name
    for j, name in enumerate(model.col_names):
        if model.col_lower[j] == model.col_upper[j]:
            continue
        slack_lower = (x[j] - model.col_lower[j]) if math.isfinite(model.col_lower[j]) else INF
        slack_upper = (model.col_upper[j] - x[j]) if math.isfinite(model.col_upper[j]) else INF
        product = abs(d[j]) * min(slack_lower, slack_upper)
        if product > worst:
            worst, where = product, name
    report.check(worst <= 1e-6, "complementary slackness",
                 f"worst |multiplier| * slack = {worst:.3e}"
                 + (f" on {where}" if where else ""))

    # ---- Strong duality -------------------------------------------------------------------
    dual_objective_min_space = 0.0
    for i in range(model.num_rows):
        dual_objective_min_space += bound_contribution(
            y[i], model.row_lower[i], model.row_upper[i], dual_tol)
    for j in range(model.num_cols):
        dual_objective_min_space += bound_contribution(
            d[j], model.col_lower[j], model.col_upper[j], dual_tol)
    dual_objective = sigma * dual_objective_min_space + model.objective_offset

    gap = abs(objective - dual_objective)
    scale = max(1.0, abs(objective))
    report.check(gap <= duality_tol * scale, "strong duality",
                 f"primal {objective:.12e}  dual {dual_objective:.12e}  "
                 f"gap {gap:.3e} (relative {gap / scale:.3e})")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="the .mps model (optionally .gz)")
    parser.add_argument("solution", type=Path, help="the .sol file SANKHYA wrote")
    parser.add_argument("--primal-tolerance", type=float, default=DEFAULT_PRIMAL_TOL)
    parser.add_argument("--dual-tolerance", type=float, default=DEFAULT_DUAL_TOL)
    parser.add_argument("--integer-tolerance", type=float, default=DEFAULT_INTEGER_TOL)
    parser.add_argument("--duality-tolerance", type=float, default=DEFAULT_DUALITY_TOL)
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args()

    try:
        model = parse_mps(args.model)
    except (OSError, ValueError) as error:
        print(f"cannot read the model: {error}", file=sys.stderr)
        return 2
    try:
        solution = parse_sol(args.solution)
    except (OSError, ValueError) as error:
        print(f"cannot read the solution: {error}", file=sys.stderr)
        return 2

    report = verify(model, solution, args.primal_tolerance, args.dual_tolerance,
                    args.integer_tolerance, args.duality_tolerance)

    if not args.quiet:
        print(f"model     {args.model}")
        print(f"solution  {args.solution}")
        print(f"problem   {model.name or '(unnamed)'}  "
              f"{model.num_rows} rows x {model.num_cols} columns  "
              f"{'maximize' if model.maximize else 'minimize'}")
        print(f"status    {solution.status}")
        print()
        print("Independent checks (this script shares no code with the solver):")
        print(report.render())
        print()

    if report.failures == 0:
        print(f"VERIFIED: {len(report.lines)} checks passed")
        return 0
    print(f"REJECTED: {report.failures} of {len(report.lines)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
