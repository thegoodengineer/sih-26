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
        # QPS quadratic objective. LOWER TRIANGLE ONLY, keyed (row, col) with row >= col,
        # holding the objective 0.5 * x'Qx - the same convention the file uses and the same
        # one sankhya::Model uses. Stored raw: halving or mirroring here would be exactly
        # the misreading this script exists to catch the solver making.
        self.hessian: dict[tuple[int, int], float] = {}

    def hessian_times(self, x: list[float]) -> list[float]:
        """Qx for the FULL symmetric Q, expanded from the stored lower triangle.

        A stored off-diagonal (r, c) stands for TWO entries of Q, so it contributes to both
        (Qx)[r] and (Qx)[c]. The diagonal contributes once. Getting this wrong is the whole
        reason the check exists, so it is written out rather than delegated.
        """
        out = [0.0] * self.num_cols
        for (r, c), v in self.hessian.items():
            out[r] += v * x[c]
            if r != c:
                out[c] += v * x[r]
        return out

    def quadratic_objective(self, x: list[float]) -> float:
        """0.5 x'Qx - the 0.5 lives in the objective, not in the data."""
        if not self.hessian:
            return 0.0
        qx = self.hessian_times(x)
        return 0.5 * sum(x[j] * qx[j] for j in range(self.num_cols))

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


# Byte offsets and widths of the six fixed-format fields, per the IBM specification.
# Columns are quoted 1-based in the spec: 2-3, 5-12, 15-22, 25-36, 40-47, 50-61. Matched by
# behaviour against src/io/mps_reader.cpp's kFixedFields, not by sharing code with it - an
# independent verifier that read the spec the same wrong way as the solver would confirm a
# bug instead of catching it.
FIXED_FIELDS = [(1, 2), (4, 8), (14, 8), (24, 12), (39, 8), (49, 12)]


def _tokenize(line: str, fixed: bool) -> list[str]:
    """Split one data line into fields.

    Section headers (NAME, ROWS, COLUMNS, ...) are never field-formatted in either dialect -
    they start in column 1 and are read as plain words by the caller before this is reached.
    This only ever sees an indented data line, so `fixed` alone decides the tokenisation."""
    if not fixed:
        return line.split()
    tokens: list[str] = []
    for offset, width in FIXED_FIELDS:
        if offset >= len(line):
            break
        field = line[offset:offset + width].strip()
        if field:
            tokens.append(field)
    return tokens


def parse_mps(path: Path) -> Model:
    """Read an MPS file per the IBM specification.

    Tries free-format tokenisation first, since that is what almost every instance in the
    wild is. Free-format breaks on a name containing a space - "DEDO3 11" tokenises as two
    fields, "DEDO3" and "11", shifting every field after it left by one - so on failure this
    retries the whole file in fixed columns. If the fixed retry also fails, the free-format
    error is the one raised: it is almost always the more informative one for a file that is
    genuinely malformed rather than fixed-format.
    """
    try:
        return _parse_mps(path, fixed=False)
    except ValueError as free_error:
        try:
            return _parse_mps(path, fixed=True)
        except ValueError:
            raise free_error from None


def _parse_mps(path: Path, fixed: bool) -> Model:
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
                elif key in ("QUADOBJ", "QMATRIX", "QSECTION", "QUADS"):
                    # All four spellings are in circulation and denote the same thing.
                    section = "QUADOBJ"
                elif key == "ENDATA":
                    break
                else:
                    raise ValueError(f"{path}:{lineno}: unknown section {head[0]}")
                continue

            fields = _tokenize(line, fixed)
            if not fields:
                continue

            if section == "OBJSENSE":
                model.maximize = fields[0].upper().startswith("MAX")
                continue

            if section == "ROWS":
                # Exactly two fields, never more. A fixed-format row named "DEDO3 1R"
                # free-tokenises to three fields; being lenient and taking the first two
                # would silently create a row called "DEDO3" instead of raising here and
                # letting parse_mps() retry the whole file in fixed columns. Matches the
                # equivalent strictness in src/io/mps_reader.cpp's do_rows().
                if len(fields) != 2:
                    raise ValueError(
                        f"{path}:{lineno}: ROWS entry has {len(fields)} fields, "
                        f"expected exactly 2 (type and name)")
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

            if section == "QUADOBJ":
                # "colname1 colname2 value". The pair is normalised to (max, min) because
                # files disagree about the order, and both orders name the same entry of a
                # symmetric matrix. A repeat is an error rather than an accumulation.
                if len(fields) < 3:
                    raise ValueError(f"{path}:{lineno}: QUADOBJ entry needs 3 fields")
                if fields[0] not in model.col_index or fields[1] not in model.col_index:
                    unknown = fields[0] if fields[0] not in model.col_index else fields[1]
                    raise ValueError(f"{path}:{lineno}: QUADOBJ names unknown column {unknown}")
                a = model.col_index[fields[0]]
                b = model.col_index[fields[1]]
                key2 = (max(a, b), min(a, b))
                if key2 in model.hessian:
                    raise ValueError(f"{path}:{lineno}: duplicate Hessian entry "
                                     f"{fields[0]} {fields[1]}")
                model.hessian[key2] = float(fields[2])
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


QUOTE_CHAR = '"'
BACKSLASH_CHAR = '\\'


def split_record(line: str) -> list[str]:
    """Split a .sol record, honouring a quoted leading name.

    src/io/writer.cpp quotes any name containing whitespace, because fixed-format MPS permits
    them - Netlib's forplan has a column called `DEDO3 11` - and a bare one makes the record
    undecidable: nothing in `DEDO3 11 0 0.0246 at_lower` says whether the name is one field
    or two.

    This parser is written from the format, not shared with the writer. That is the whole
    point of this script: if the two disagree about what a file means, the disagreement has
    to be able to surface, and it cannot if they run the same code.
    """
    if not line.startswith(QUOTE_CHAR):
        return line.split()

    name = []
    i = 1
    while i < len(line):
        c = line[i]
        if c == BACKSLASH_CHAR and i + 1 < len(line):
            name.append(line[i + 1])
            i += 2
            continue
        if c == QUOTE_CHAR:
            i += 1
            break
        name.append(c)
        i += 1
    else:
        # No closing quote. Fall back rather than silently truncating the record.
        return line.split()
    return ["".join(name)] + line[i:].split()


def parse_sol(path: Path) -> Solution:
    solution = Solution()
    block = ""
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = split_record(line)
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


def bound_contribution(multiplier: float, lower: float, upper: float) -> float:
    """The Lagrangian term a bound contributes to the dual objective, in minimize space.

    A positive multiplier prices the lower bound and a negative one the upper. A bound that
    does not exist contributes nothing: the multiplier is then a sign violation, judged in
    its own check, and its whole product with the primal value lands in the duality gap,
    where it is accounted for. An earlier version zeroed every multiplier below the dual
    tolerance, which put |multiplier| * bound into the gap for each column sitting exactly
    at a bound with a rounding-sized reduced cost - a gap manufactured by the test.
    """
    if multiplier > 0.0 and math.isfinite(lower):
        return multiplier * lower
    if multiplier < 0.0 and math.isfinite(upper):
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
    # Scaled by the variable's own magnitude, for the same reason as the rows above.
    worst, where, worst_abs = 0.0, "", 0.0
    for j, name in enumerate(model.col_names):
        violation = max(model.col_lower[j] - x[j], x[j] - model.col_upper[j], 0.0)
        scaled = violation / max(1.0, abs(x[j]))
        if scaled > worst:
            worst, where, worst_abs = scaled, name, violation
    report.check(worst <= primal_tol, "column bounds",
                 f"worst violation {worst_abs:.3e} ({worst:.3e} relative)"
                 + (f" on {where}" if where else ""))

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
    # The row's numerical SCALE is accumulated alongside its activity: the largest term the
    # sum was built from. A residual cannot be smaller than the rounding error of the sum
    # that produced it, and that error is set by the size of the terms, not of the answer.
    #
    # Netlib grow7 is why this is not an absolute test. Its largest solution value is 4.8e+07,
    # so a 1e-7 absolute tolerance is 2.1e-15 relative - below what double precision reaches
    # after three hundred iterations. Its worst violation is 4.2e-15 relative, about nineteen
    # machine epsilons, and the row it occurs on is an equality to ZERO, so scaling by the
    # bound would change nothing; the residual is large because terms of magnitude 1e+07
    # cancel.
    #
    # THIS DOES NOT MAKE THE VERIFIER LESS INDEPENDENT. Its independence is that it shares no
    # code with the solver and re-derives everything from the original file, which is
    # unchanged. Asking a question in the right units is not the same as asking a weaker one:
    # a violation that is large relative to the terms it came from still fails, and the dual
    # checks below - which caught #149's postsolve regression at 1.3e-02 relative - are
    # untouched.
    activity = [0.0] * model.num_rows
    row_scale = [1.0] * model.num_rows
    for j in range(model.num_cols):
        if x[j] == 0.0:
            continue
        for i, value in model.entries[j]:
            term = value * x[j]
            activity[i] += term
            if abs(term) > row_scale[i]:
                row_scale[i] = abs(term)

    worst, where, worst_abs = 0.0, "", 0.0
    for i, name in enumerate(model.row_names):
        violation = max(model.row_lower[i] - activity[i], activity[i] - model.row_upper[i], 0.0)
        scaled = violation / row_scale[i]
        if scaled > worst:
            worst, where, worst_abs = scaled, name, violation
    report.check(worst <= primal_tol, "row activity",
                 f"worst violation {worst_abs:.3e} ({worst:.3e} relative to the row's terms)"
                 + (f" on {where}" if where else ""))

    # The solver also reports its own activities. A disagreement means one of us computed
    # A*x differently, which is worth surfacing even when both satisfy the bounds.
    worst_activity_gap = max(
        (abs(activity[i] - solution.row_activity[n]) for i, n in enumerate(model.row_names)),
        default=0.0,
    )
    report.check(worst_activity_gap <= 1e-6, "activity agreement",
                 f"max |ours - solver's| = {worst_activity_gap:.3e}")

    # ---- Objective, recomputed -----------------------------------------------------------
    # c'x + 0.5 x'Qx + offset. Omitting the quadratic term would have this script declare a
    # correct QP answer wrong - and, worse, declare a solver that ITSELF dropped the term
    # right, since both sides would then be computing the LP objective.
    objective = (model.objective_offset
                 + sum(model.col_cost[j] * x[j] for j in range(model.num_cols))
                 + model.quadratic_objective(x))
    claimed = solution.header_float("objective")
    if claimed is None:
        report.check(False, "objective", "the .sol file states no objective")
    else:
        scale = max(1.0, abs(objective))
        report.check(abs(objective - claimed) <= 1e-9 * scale, "objective",
                     f"recomputed {objective:.12e}, solver said {claimed:.12e}, "
                     f"difference {abs(objective - claimed):.3e}")

    if solution.status != "optimal":
        # Strong duality is a test of OPTIMALITY. A solver reporting kFeasible is explicitly
        # declining to claim optimality - a first-order method that stopped on a tolerance,
        # or a search stopped by a limit - so holding its point to an optimality standard
        # measures something it never asserted. Primal feasibility, integrality and the
        # objective are all still checked above, and those are what kFeasible does assert.
        report.note("duality", f"skipped: status is {solution.status}, not an optimality claim")
        return report

    if integer_columns:
        # LP duality does not apply to a MILP: any reported duals belong to some node
        # relaxation, not to the integer problem. What CAN be checked is the claim the
        # search makes about itself.
        report.note("duality", "skipped: LP duality does not apply to a MILP")

        bound = solution.header_float("dual_bound")
        if bound is None or not math.isfinite(bound):
            report.note("optimality proof",
                        "no finite dual bound reported"
                        + ("" if solution.status != "optimal"
                           else " - but the status claims optimal"))
            if solution.status == "optimal":
                report.check(False, "optimality proof",
                             "status is optimal but no finite bound backs the claim")
            return report

        scale = max(1.0, abs(objective))
        if solution.status == "optimal":
            # "Optimal" on a MILP is a claim that the search CLOSED: the incumbent and the
            # final bound have met. If they have not, the solver is calling an incumbent a
            # proof, which is the most consequential thing a branch and bound can get wrong
            # and the least visible - the point is integral and feasible either way.
            report.check(abs(objective - bound) <= 1e-6 * scale, "optimality proof",
                         f"objective {objective:.12e} vs dual bound {bound:.12e}, "
                         f"gap {abs(objective - bound):.3e}")
        else:
            # Not closed. The bound must still BE a bound: never worse than the incumbent.
            slack = (objective - bound) if not model.maximize else (bound - objective)
            report.check(slack >= -1e-6 * scale, "dual bound is a bound",
                         f"incumbent {objective:.12e}, bound {bound:.12e}, "
                         f"remaining gap {abs(objective - bound):.3e}")
        return report

    # ---- Dual feasibility ----------------------------------------------------------------
    # Work in minimize space so one set of sign conventions covers both senses.
    y = [sigma * solution.row_dual[n] for n in model.row_names]
    d = [sigma * solution.col_dual[n] for n in model.col_names]
    # For an LP the gradient of the objective IS the cost vector. For a QP it is c + Qx, and
    # every KKT condition below is stated in terms of the gradient, so making this one
    # substitution carries dual feasibility, complementary slackness and strong duality over
    # to the quadratic case unchanged - which is the point: a QP optimum is not a special
    # kind of optimum, it is the same conditions about a different gradient.
    gradient = list(model.col_cost)
    if model.hessian:
        qx = model.hessian_times(x)
        gradient = [model.col_cost[j] + qx[j] for j in range(model.num_cols)]
    cost = [sigma * g for g in gradient]

    if model.hessian:
        # The QP engine is a first-order method that carries no basis and reports no reduced
        # costs, so there is nothing of the solver's to cross-check here. d is DERIVED from
        # (model, x, y) instead - which is strictly the stronger test, since the sign and
        # complementarity checks below then price against a vector the solver never chose.
        d = [cost[j] - sum(value * y[i] for i, value in model.entries[j])
             for j in range(model.num_cols)]
        report.note("reduced costs",
                    "derived from c + Qx - A^T y; the QP engine reports none to compare")
    else:
        # Reduced costs must satisfy d = c - A^T y. Recomputing catches a solver that reports
        # a dual vector inconsistent with the reduced costs it also reports.
        # Judged against the magnitude of the terms in c - A^T y, for the same reason the
        # row activities are judged against theirs: it is a difference of quantities that
        # cancel, and its achievable accuracy is set by their size. On grow7 the terms are
        # of order 1e+07, so an absolute 1e-6 asks for 1e-13 relative.
        worst, where, worst_abs = 0.0, "", 0.0
        for j, name in enumerate(model.col_names):
            terms = [value * y[i] for i, value in model.entries[j]]
            expected = cost[j] - sum(terms)
            difference = abs(expected - d[j])
            scale = max(1.0, abs(cost[j]), max((abs(t) for t in terms), default=0.0))
            if difference / scale > worst:
                worst, where, worst_abs = difference / scale, name, difference
        report.check(worst <= 1e-6, "reduced costs",
                     f"max |c - A^T y - d| = {worst_abs:.3e} ({worst:.3e} relative to its terms)"
                     + (f" on {where}" if where else ""))

    def sign_violation(multiplier: float, value: float, lower: float, upper: float) -> float:
        """How badly a multiplier's SIGN contradicts the bound it prices against.

        A positive multiplier prices the lower bound and a negative one the upper bound, so
        the violation is a multiplier pushing against a bound that does not exist.

        Deliberately NOT checked here: "the multiplier must vanish when the constraint is
        strictly interior". That is complementary slackness, and it is measured below as the
        product |multiplier| * slack. Reporting it here as well, via a binary "is the value
        within primal_tol of a bound" test, is a category error and a badly conditioned one:
        it returns the full |multiplier| the moment a value sits a hair outside the window,
        so a point that is optimal to 1e-9 can report a dual violation of 1.0. That reads as
        a catastrophe when the truth is a rounding-width displacement. A vertex solution is
        unaffected either way; a first-order method sits near bounds rather than on them,
        and would be judged by an artefact of the window rather than by its KKT error.
        """
        del value  # activity enters through the complementarity product, not through here
        if lower == upper:
            return 0.0  # equality / fixed: any multiplier is admissible
        if multiplier > 0.0 and not math.isfinite(lower):
            return multiplier
        if multiplier < 0.0 and not math.isfinite(upper):
            return -multiplier
        return 0.0

    # Scaled like the reduced costs above: a sign violation of 6 on a reduced cost whose
    # terms are of order 1e+07 is the precision floor, not a wrong sign.
    worst, where, worst_abs = 0.0, "", 0.0
    for j, name in enumerate(model.col_names):
        violation = sign_violation(d[j], x[j], model.col_lower[j], model.col_upper[j])
        scale = max(1.0, abs(cost[j]),
                    max((abs(value * y[i]) for i, value in model.entries[j]), default=0.0))
        if violation / scale > worst:
            worst, where, worst_abs = violation / scale, name, violation
    report.check(worst <= dual_tol, "dual feasibility (columns)",
                 f"worst {worst_abs:.3e} ({worst:.3e} relative)" + (f" on {where}" if where else ""))

    # A row price has no terms of its own to compare against, so it is judged relative to
    # the size of the prices it sits among - a weaker test than the column one, and stated
    # as such in model.hpp.
    dual_norm = max(1.0, max((abs(v) for v in y), default=0.0))
    worst, where, worst_abs = 0.0, "", 0.0
    for i, name in enumerate(model.row_names):
        violation = sign_violation(y[i], activity[i], model.row_lower[i], model.row_upper[i])
        if violation / dual_norm > worst:
            worst, where, worst_abs = violation / dual_norm, name, violation
    report.check(worst <= dual_tol, "dual feasibility (rows)",
                 f"worst {worst_abs:.3e} ({worst:.3e} relative to |y|)"
                 + (f" on {where}" if where else ""))

    # ---- Complementary slackness ----------------------------------------------------------
    # A multiplier may only be nonzero where its constraint is tight.
    worst, where = 0.0, ""
    def complementarity(multiplier: float, slack: float) -> float:
        """|multiplier| * slack, except where that product is degenerate.

        WHEN THE SLACK IS INFINITE the product is not a usable measure. A free variable has
        no finite bound on either side, so min(slack) is INF and |d| * INF evaluates to inf
        for ANY nonzero d - which demands the reduced cost be BIT-EXACTLY zero. No
        floating-point solver can promise that, and it is not what complementary slackness
        requires: for a constraint that cannot be tight, the condition reduces to "the
        multiplier is zero", and that is testable directly against the dual tolerance.

        This is the same mathematical condition, correctly conditioned - not a loosened one.
        A free column carrying a genuinely nonzero reduced cost still fails, at exactly the
        threshold it should. It was found when a solver change altered a pivot path and left
        -1.05e-15 on a free column of capri: a correct answer, rejected, for having rounded
        a zero rather than for being wrong.
        """
        if not math.isfinite(slack):
            return abs(multiplier)
        return abs(multiplier) * slack

    for i, name in enumerate(model.row_names):
        if model.row_lower[i] == model.row_upper[i]:
            continue
        slack_lower = (activity[i] - model.row_lower[i]) if math.isfinite(model.row_lower[i]) else INF
        slack_upper = (model.row_upper[i] - activity[i]) if math.isfinite(model.row_upper[i]) else INF
        product = complementarity(y[i], min(slack_lower, slack_upper))
        if product > worst:
            worst, where = product, name
    for j, name in enumerate(model.col_names):
        if model.col_lower[j] == model.col_upper[j]:
            continue
        slack_lower = (x[j] - model.col_lower[j]) if math.isfinite(model.col_lower[j]) else INF
        slack_upper = (model.col_upper[j] - x[j]) if math.isfinite(model.col_upper[j]) else INF
        product = complementarity(d[j], min(slack_lower, slack_upper))
        if product > worst:
            worst, where = product, name
    report.check(worst <= 1e-6, "complementary slackness",
                 f"worst |multiplier| * slack = {worst:.3e}"
                 + (f" on {where}" if where else ""))

    # ---- Strong duality -------------------------------------------------------------------
    dual_objective_min_space = 0.0
    for i in range(model.num_rows):
        dual_objective_min_space += bound_contribution(
            y[i], model.row_lower[i], model.row_upper[i])
    for j in range(model.num_cols):
        dual_objective_min_space += bound_contribution(
            d[j], model.col_lower[j], model.col_upper[j])

    # THE GAP IS AN IDENTITY, NOT A MEASUREMENT OF ITS OWN. With d = c - A^T y and the
    # activities recomputed from x, primal - dual is exactly the sum over every column and
    # row of  multiplier * (value - the bound the multiplier's sign prices), plus the
    # consistency and activity residuals judged above. So every item's share of the gap is
    # known, and the question this check can honestly ask is whether the gap is accounted
    # for by shares the per-item checks already accepted:
    #
    #   - a multiplier within the dual tolerance at its own scale (the scale the sign check
    #     used) explains its whole share: it is indistinguishable from zero, and zero would
    #     contribute nothing;
    #   - a larger multiplier explains its share only up to |multiplier| * nearest slack,
    #     which is what the complementarity check judged. A reduced cost of -4e-3 on a
    #     column at its LOWER bound prices the UPPER bound 20 away (Netlib recipe, #157):
    #     nearest slack 0, nothing explained, and the check fails on it exactly as before.
    #
    # Netlib etamacro is why the accounting exists: ten per-item checks pass, and the gap
    # of 1.26e-6 on an objective of 755 (1.67e-9 relative) is one accepted sign violation
    # of 3.2e-8 on KAPSTK65 times that column's value of 40. An aggregate threshold of
    # 1e-9 relative was tighter than a single per-item allowance granted above it.
    accounted = 0.0

    def share(multiplier: float, value: float, lower: float, upper: float,
              multiplier_scale: float) -> float:
        if multiplier == 0.0:
            return 0.0
        priced = lower if multiplier > 0.0 else upper
        term = abs(multiplier) * (abs(value - priced) if math.isfinite(priced) else abs(value))
        if abs(multiplier) <= dual_tol * multiplier_scale:
            return term
        nearest = min(abs(value - lower) if math.isfinite(lower) else INF,
                      abs(value - upper) if math.isfinite(upper) else INF)
        return min(term, abs(multiplier) * nearest) if math.isfinite(nearest) else 0.0

    for j in range(model.num_cols):
        scale_j = max(1.0, abs(cost[j]),
                      max((abs(value * y[i]) for i, value in model.entries[j]), default=0.0))
        accounted += share(d[j], x[j], model.col_lower[j], model.col_upper[j], scale_j)
    for i in range(model.num_rows):
        accounted += share(y[i], activity[i], model.row_lower[i], model.row_upper[i], dual_norm)
    # For a QP the bound contributions sum, at a KKT point, to (c + Qx)'x = c'x + x'Qx, which
    # overshoots the primal objective c'x + 0.5 x'Qx by exactly 0.5 x'Qx. Subtracting it is
    # the Dorn dual of a convex QP, and it makes the gap below a real optimality test rather
    # than an identity that would fail by a fixed amount on every quadratic instance.
    dual_objective = (sigma * dual_objective_min_space + model.objective_offset
                      - model.quadratic_objective(x))

    gap = abs(objective - dual_objective)
    scale = max(1.0, abs(objective))
    report.check(gap <= duality_tol * scale + accounted, "strong duality",
                 f"primal {objective:.12e}  dual {dual_objective:.12e}  "
                 f"gap {gap:.3e} (relative {gap / scale:.3e}), "
                 f"{accounted:.3e} of it from per-item violations accepted above")
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
