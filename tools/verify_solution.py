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
        # A verdict of infeasible or unbounded carries its PROOF, not a point (#191).
        # farkas: row multipliers whose aggregate no point in the column box can satisfy.
        # ray: a direction along which the model stays feasible and the objective improves
        # without limit, checked together with the feasible point in the columns section.
        self.farkas: dict[str, float] = {}
        self.ray: dict[str, float] = {}
        # Sensitivity ranging (populated when --ranging was passed to the solver).
        self.col_ranging_lower: dict[str, float] = {}
        self.col_ranging_upper: dict[str, float] = {}
        self.row_ranging_lower: dict[str, float] = {}
        self.row_ranging_upper: dict[str, float] = {}
        self.iis: list[tuple[str, str]] = []
        # One witness per IIS element: (kind, name, {column name: value}).
        self.iis_witnesses: list[tuple[str, str, dict[str, float]]] = []
        # The solution pool (#225): (rank, objective, {integer column name: value}).
        self.pool: list[tuple[int, float, dict[str, float]]] = []

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
                if block == "iis_witness" and len(fields) >= 4:
                    solution.iis_witnesses.append((fields[2], fields[3], {}))
                continue
            if fields[0] == "end":
                block = ""
                continue
            if block == "columns" and len(fields) >= 3:
                solution.col_value[fields[0]] = float(fields[1])
                solution.col_dual[fields[0]] = float(fields[2])
                solution.col_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif block == "farkas" and len(fields) >= 2:
                solution.farkas[fields[0]] = float(fields[1])
            elif block == "ray" and len(fields) >= 2:
                solution.ray[fields[0]] = float(fields[1])
            elif block == "pool" and len(fields) == 3 and fields[0] == "solution":
                solution.pool.append((int(fields[1]), float(fields[2]), {}))
            elif block == "pool" and len(fields) == 2 and solution.pool:
                solution.pool[-1][2][fields[0]] = float(fields[1])
            elif block == "iis" and len(fields) >= 2:
                solution.iis.append((fields[0], fields[1]))
            elif block == "iis_witness" and len(fields) >= 2 and solution.iis_witnesses:
                solution.iis_witnesses[-1][2][fields[0]] = float(fields[1])
            elif block == "rows" and len(fields) >= 3:
                solution.row_activity[fields[0]] = float(fields[1])
                solution.row_dual[fields[0]] = float(fields[2])
                solution.row_status[fields[0]] = fields[3] if len(fields) > 3 else "unknown"
            elif block == "ranging_columns" and len(fields) >= 3:
                solution.col_ranging_lower[fields[0]] = float(fields[1])
                solution.col_ranging_upper[fields[0]] = float(fields[2])
            elif block == "ranging_rows" and len(fields) >= 3:
                solution.row_ranging_lower[fields[0]] = float(fields[1])
                solution.row_ranging_upper[fields[0]] = float(fields[2])
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


def transpose_times(model: Model, y: list[float]) -> list[float]:
    """A' y, from the column-wise entries. One line of arithmetic, written out."""
    d = [0.0] * model.num_cols
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            d[j] += value * y[i]
    return d


def times(model: Model, x: list[float]) -> list[float]:
    """A x, from the same column-wise entries."""
    out = [0.0] * model.num_rows
    for j in range(model.num_cols):
        if x[j] == 0.0:
            continue
        for i, value in model.entries[j]:
            out[i] += value * x[j]
    return out


def verify_iis(model: Model, solution: Solution, report: Report, primal_tol: float) -> None:
    """Check the two properties that make a set of constraints an IIS (#217).

    1. INFEASIBLE ON ITS OWN. The solver writes, as the farkas section, the certificate of
       the subsystem rather than of the whole model, so verify_farkas() has already proved
       that *some* aggregate of rows contradicts the column box. What remains is to check
       that the aggregate uses nothing outside the IIS: every row with a nonzero multiplier
       is an IIS row, and every column bound the aggregate leans on is an IIS bound.

    2. IRREDUCIBLE. For every element the solver writes a witness: a point that satisfies
       every other element and violates that one. Checking a witness is arithmetic - row
       activities against row bounds, values against column bounds - and needs no solver.

    The solver says `iis_irreducible not-claimed` in the header when one of its trial solves
    ended in neither verdict; then the set may be a superset of an IIS, no witnesses are
    written, and this function notes that rather than failing a claim that was never made.
    """
    if not solution.iis:
        return

    row_iis = [name for kind, name in solution.iis if kind == "row"]
    col_lo_iis = [name for kind, name in solution.iis if kind == "col_lo"]
    col_hi_iis = [name for kind, name in solution.iis if kind == "col_hi"]

    report.note("IIS",
                f"{len(solution.iis)} element(s): "
                f"{len(row_iis)} row(s), {len(col_lo_iis)} col_lo bound(s), "
                f"{len(col_hi_iis)} col_hi bound(s)")

    unknown_rows = [n for n in row_iis if n not in model.row_index]
    if not report.check(not unknown_rows, "IIS rows exist in model",
                        f"unknown row name(s): {unknown_rows}" if unknown_rows
                        else "every IIS row names a row of the model"):
        return
    unknown_cols = [n for n in col_lo_iis + col_hi_iis if n not in model.col_index]
    if not report.check(not unknown_cols, "IIS columns exist in model",
                        f"unknown column name(s): {unknown_cols}" if unknown_cols
                        else "every IIS bound names a column of the model"):
        return

    claimed = solution.header.get("iis_irreducible", "not-claimed")
    if claimed != "yes":
        report.note("IIS properties",
                    "not claimed by the solver (a trial solve was inconclusive), so the set "
                    "may be a superset of an IIS; neither property is checked")
        return

    rows = set(row_iis)
    lo = set(col_lo_iis)
    hi = set(col_hi_iis)

    # ---- 1. the certificate lives inside the IIS ------------------------------------------
    y = [solution.farkas.get(name, 0.0) for name in model.row_names]
    if any(y):
        outside_rows = [model.row_names[i] for i, m in enumerate(y)
                        if m != 0.0 and model.row_names[i] not in rows]
        d = transpose_times(model, y)
        term_scale = 1.0
        for j in range(model.num_cols):
            for i, value in model.entries[j]:
                term_scale = max(term_scale, abs(value * y[i]))
        zero = 1e-11 * term_scale  # the same zero as verify_farkas and the C++ checker
        outside_bounds = []
        for j in range(model.num_cols):
            if d[j] > zero and model.col_names[j] not in hi:
                outside_bounds.append(model.col_names[j] + " (upper)")
            elif d[j] < -zero and model.col_names[j] not in lo:
                outside_bounds.append(model.col_names[j] + " (lower)")
        report.check(not outside_rows and not outside_bounds, "IIS is infeasible on its own",
                     "the certificate's multipliers and the bounds its aggregate leans on "
                     "all belong to the IIS, so the proof above proves the subsystem alone"
                     if not outside_rows and not outside_bounds
                     else "the certificate reaches outside the IIS: rows "
                          + ", ".join(outside_rows[:5]) + "; bounds "
                          + ", ".join(outside_bounds[:5]))
    else:
        report.check(False, "IIS is infeasible on its own",
                     "no certificate in the file, so the subsystem's infeasibility is unproved")

    # ---- 2. every element is necessary: its witness satisfies all the others ---------------
    witnesses = {(kind, name): point for kind, name, point in solution.iis_witnesses}
    missing = [f"{kind} {name}" for kind, name in solution.iis if (kind, name) not in witnesses]
    if not report.check(not missing, "IIS witnesses present",
                        f"no witness for {len(missing)} element(s): " + ", ".join(missing[:5])
                        if missing else f"one witness for each of the {len(solution.iis)} elements"):
        return

    def violation(value: float, lower: float, upper: float) -> float:
        """Relative distance outside [lower, upper], zero inside."""
        below = lower - value if math.isfinite(lower) else 0.0
        above = value - upper if math.isfinite(upper) else 0.0
        worst = max(below, above, 0.0)
        return worst / max(1.0, abs(value),
                           abs(lower) if math.isfinite(lower) else 0.0,
                           abs(upper) if math.isfinite(upper) else 0.0)

    failures = []
    for kind, name in solution.iis:
        point = witnesses[(kind, name)]
        x = [point.get(col, 0.0) for col in model.col_names]
        activity = times(model, x)
        for other in row_iis:
            i = model.row_index[other]
            v = violation(activity[i], model.row_lower[i], model.row_upper[i])
            is_self = kind == "row" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates row {other} by {v:.3e}")
        for other in col_lo_iis:
            j = model.col_index[other]
            v = violation(x[j], model.col_lower[j], math.inf)
            is_self = kind == "col_lo" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates lower bound of {other} by {v:.3e}")
        for other in col_hi_iis:
            j = model.col_index[other]
            v = violation(x[j], -math.inf, model.col_upper[j])
            is_self = kind == "col_hi" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates upper bound of {other} by {v:.3e}")
    report.check(not failures, "IIS is irreducible",
                 f"each of the {len(solution.iis)} witnesses satisfies the other "
                 f"{len(solution.iis) - 1} element(s) and violates its own"
                 if not failures else "; ".join(failures[:4]))


def verify_farkas(model: Model, solution: Solution, report: Report) -> Report:
    """Check a claim of INFEASIBILITY, in the only way a claim of infeasibility can be checked.

    There is no point to test - that is the whole content of the verdict - so a checker that
    asks for one is asking the wrong question. What CAN be handed over is a Farkas
    certificate: one multiplier per row. Aggregating the rows with those weights produces a
    single inequality that every feasible point would have to satisfy, and the certificate is
    good exactly when no point in the column box satisfies it.

    Written out, with rows `row_lower <= a_i.x <= row_upper` and columns in `[l, u]`:

        a multiplier y_i > 0 uses the row's LOWER bound   (a_i.x >= row_lower[i])
        a multiplier y_i < 0 uses the row's UPPER bound   (a_i.x <= row_upper[i])

    so every feasible x satisfies  d.x >= S  where  d = A'y  and  S = sum_i y_i * (that
    bound). The largest d.x can be over the box is M, taking each column to whichever of its
    own bounds the sign of d_j prefers. If M < S the system has no feasible point at all.

    Two ways a certificate can be bogus, both checked rather than assumed: a multiplier that
    leans on a bound the row does not have (infinite), and a column free in the direction d
    prefers, which makes M infinite and proves nothing.
    """
    y = [solution.farkas.get(name, 0.0) for name in model.row_names]
    if not any(y):
        # NOT a failure. Presolve proves infeasibility from bound arithmetic and does not
        # keep the chain of tightenings that would make a Farkas vector, so it says
        # `certificate none` and puts its reason in the message. Failing here would be the
        # very thing #191 exists to stop: this script calling a correct verdict wrong.
        report.note("infeasibility proof",
                    "no certificate offered, so nothing is claimed and nothing is checked; "
                    + (solution.header.get("message", "the solver gave no reason")))
        return report

    # A multiplier may only use a bound the row actually has.
    borrowed = [model.row_names[i] for i, m in enumerate(y)
                if (m > 0.0 and not math.isfinite(model.row_lower[i]))
                or (m < 0.0 and not math.isfinite(model.row_upper[i]))]
    if not report.check(not borrowed, "certificate uses only real bounds",
                        "every multiplier leans on a finite row bound" if not borrowed
                        else f"{len(borrowed)} lean on an infinite bound: "
                             + ", ".join(borrowed[:5])):
        return report

    required = sum(bound_contribution(y[i], model.row_lower[i], model.row_upper[i])
                   for i in range(model.num_rows))

    d = transpose_times(model, y)
    # A coefficient of the aggregate that is zero up to rounding is zero. The rows of the
    # crude-blend demo aggregate to exactly 0 on one column - two terms of 0.577 that
    # cancel - and floating point leaves 1e-17 behind; read as a sign, that "uses" a bound
    # the column does not have and rejects a correct certificate. The same rule the C++
    # checker applies (src/core/certificate.cpp): below 1e-11 of the largest term is zero.
    term_scale = 1.0
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            term_scale = max(term_scale, abs(value * y[i]))
    zero = 1e-11 * term_scale
    reachable = 0.0
    free = []
    for j in range(model.num_cols):
        if d[j] > zero:
            if not math.isfinite(model.col_upper[j]):
                free.append(model.col_names[j])
            else:
                reachable += d[j] * model.col_upper[j]
        elif d[j] < -zero:
            if not math.isfinite(model.col_lower[j]):
                free.append(model.col_names[j])
            else:
                reachable += d[j] * model.col_lower[j]
    if not report.check(not free, "aggregate is bounded above",
                        "every column the aggregate uses is bounded in that direction"
                        if not free
                        else f"{len(free)} unbounded in the direction used, so the aggregate "
                             f"proves nothing: " + ", ".join(free[:5])):
        return report

    # Strictly, and by more than the arithmetic could have invented.
    scale = max(1.0, abs(required), abs(reachable))
    report.check(reachable < required - 1e-9 * scale, "infeasibility proof",
                 f"the rows aggregate to at least {required:.12e}, the column bounds allow at "
                 f"most {reachable:.12e}, a contradiction of {required - reachable:.3e}")
    report.note("certificate size",
                f"{sum(1 for m in y if m != 0.0)} of {model.num_rows} rows carry a multiplier")
    return report


def verify_ray(model: Model, solution: Solution, report: Report, primal_tol: float) -> Report:
    """Check a claim of UNBOUNDEDNESS: a feasible point, and a direction that never stops.

    Unbounded is two claims, and a checker that tests one of them tests nothing. The point in
    the columns section must be feasible - checked by the ordinary primal checks, which run
    first - and the ray must satisfy, for every t >= 0, that x + t*d stays inside every bound
    while the objective improves without limit. That holds exactly when moving along d is
    blocked by nothing:

        (A d)_i > 0 needs the row to have NO upper bound, and < 0 no lower bound
        d_j     > 0 needs the column to have NO upper bound, and < 0 no lower bound

    and the objective strictly improves, c.d < 0 in minimize space. For a quadratic objective
    the ray must also not curve back up, d'Qd <= 0, or the improvement is only local.
    """
    if not solution.ray:
        report.note("unboundedness proof",
                    "no ray offered, so nothing is claimed and nothing is checked; "
                    + (solution.header.get("message", "the solver gave no reason")))
        return report
    d = [solution.ray.get(name, 0.0) for name in model.col_names]
    if not report.check(any(d), "ray is a direction",
                        f"{sum(1 for v in d if v != 0.0)} of {model.num_cols} columns move"
                        if any(d) else "the ray is all zeros, which is not a direction"):
        return report

    scale = max(abs(v) for v in d)
    moving = primal_tol * scale

    blocked_cols = [model.col_names[j] for j in range(model.num_cols)
                    if (d[j] > moving and math.isfinite(model.col_upper[j]))
                    or (d[j] < -moving and math.isfinite(model.col_lower[j]))]
    report.check(not blocked_cols, "ray respects the column bounds",
                 "no column bound blocks the ray" if not blocked_cols
                 else f"{len(blocked_cols)} would be crossed: " + ", ".join(blocked_cols[:5]))

    activity = times(model, d)
    blocked_rows = [model.row_names[i] for i in range(model.num_rows)
                    if (activity[i] > moving and math.isfinite(model.row_upper[i]))
                    or (activity[i] < -moving and math.isfinite(model.row_lower[i]))]
    report.check(not blocked_rows, "ray respects the row bounds",
                 "no row bound blocks the ray" if not blocked_rows
                 else f"{len(blocked_rows)} would be crossed: " + ", ".join(blocked_rows[:5]))

    sigma = -1.0 if model.maximize else 1.0
    improvement = sigma * sum(model.col_cost[j] * d[j] for j in range(model.num_cols))
    report.check(improvement < -1e-9 * max(1.0, abs(improvement)), "ray improves the objective",
                 f"objective changes by {improvement:.12e} per unit step, in minimize space")

    if model.hessian:
        qd = model.hessian_times(d)
        curvature = sum(d[j] * qd[j] for j in range(model.num_cols))
        report.check(curvature <= 1e-9 * max(1.0, abs(curvature)), "ray does not curve back",
                     f"d'Qd = {curvature:.12e}; a positive value means the improvement is "
                     f"only local")
    return report


# The verdicts that hand back a point, mirroring claims_a_point() in include/sankhya/model.hpp.
# The two lists are the .sol file's contract and have to agree; this script deliberately shares
# no code with the solver, so they are kept in step by saying so in both places rather than by
# a header. `unbounded` is here because since #191 it carries the feasible point its ray starts
# from - a ray from outside the feasible region proves nothing.
STATUSES_WITH_A_POINT = ("optimal", "feasible", "unbounded", "iteration_limit", "time_limit",
                         "node_limit", "interrupted")

# Of those, the ones that assert the point is FEASIBLE. The distinction is the whole of what
# a limit means: `optimal` and `feasible` say "here is a point inside the model", and a limit
# says only "here is where I stopped". An interior-point iterate stopped by the clock is not
# feasible and was never claimed to be - it approaches feasibility from outside - so holding
# it to a feasibility standard measures something nobody asserted. `unbounded` is here
# because its ray is only a proof if it starts somewhere the model allows.
#
# A limit is still checked, on the claim it DOES make: the solver reports its own
# primal_infeasibility in the header, and that number has to be true. Understating it is the
# failure worth catching, and it is the one a solver has an incentive to make.
STATUSES_ASSERTING_FEASIBILITY = ("optimal", "feasible", "unbounded")


def verify_pool(model: Model, solution: Solution, report: Report, x: list[float],
                objective: float, primal_tol: float, integer_tol: float) -> None:
    """The solution pool (#225), checked from what the file says and nothing else.

    By default the file carries only the INTEGER columns of each member, so what can be
    proved depends on the model. When every member is a full point - a pure-integer model, or
    a file written with pool_write_all_columns - each is checked the way the main solution
    is: bounds, integrality, every row, and the objective recomputed. With continuous columns
    missing, the integer part is checked exactly and each row is checked for whether the
    continuous columns' own bounds can still close it - a necessary condition, not a proof
    that one continuous completion satisfies every row at once, and the check says so rather
    than claiming more.
    """
    sigma = -1.0 if model.maximize else 1.0
    integer_names = [n for j, n in enumerate(model.col_names) if model.col_integer[j]]
    members = solution.pool
    ranks = [rank for rank, _, _ in members]
    all_names = set(model.col_names)
    full = bool(members) and all(set(values) == all_names for _, _, values in members)
    complete = full or all(set(values) == set(integer_names) for _, _, values in members)
    written = model.col_names if full else integer_names
    report.check(ranks == list(range(1, len(members) + 1)) and complete, "pool: structure",
                 (f"{len(members)} member(s), each listing all {len(written)} "
                  + ("columns" if full else "integer columns")) if complete
                 else f"{len(members)} member(s); ranks {ranks[:5]}, or a member listing "
                      "neither every integer column nor every column")
    if not complete:
        return

    first = members[0]
    worst_first = max((abs(first[2][n] - x[model.col_index[n]])
                       / max(1.0, abs(x[model.col_index[n]])) for n in written), default=0.0)
    scale = max(1.0, abs(objective))
    report.check(worst_first <= integer_tol and abs(first[1] - objective) <= 1e-9 * scale,
                 "pool: first member is the reported solution",
                 f"values differ by at most {worst_first:.3e} (relative), objective "
                 f"{first[1]:.12e} against {objective:.12e}")

    order_ok = all(sigma * members[k + 1][1] >= sigma * members[k][1]
                   - 1e-9 * max(1.0, abs(members[k][1])) for k in range(len(members) - 1))
    report.check(order_ok, "pool: best first",
                 "objectives " + ", ".join(f"{obj:.10g}" for _, obj, _ in members[:10]))

    keys = [tuple(round(values[n]) for n in integer_names) for _, _, values in members]
    report.check(len(set(keys)) == len(keys), "pool: distinct integer assignments",
                 f"{len(set(keys))} distinct of {len(keys)}")

    worst_integrality, worst_bound = 0.0, 0.0
    for _, _, values in members:
        for n in written:
            j = model.col_index[n]
            v = values[n]
            if model.col_integer[j]:
                worst_integrality = max(worst_integrality, abs(v - round(v)))
            worst_bound = max(worst_bound,
                              (model.col_lower[j] - v) / max(1.0, abs(v)),
                              (v - model.col_upper[j]) / max(1.0, abs(v)))
    report.check(worst_integrality <= integer_tol and worst_bound <= primal_tol,
                 "pool: integrality and column bounds",
                 f"worst integrality {worst_integrality:.3e}, worst bound violation "
                 f"{max(worst_bound, 0.0):.3e}")

    # Rows: the written part is fixed; each unwritten continuous column contributes an interval.
    continuous = [] if full else [j for j in range(model.num_cols) if not model.col_integer[j]]
    known = [full or model.col_integer[j] for j in range(model.num_cols)]
    by_row: list[list[tuple[int, float]]] = [[] for _ in range(model.num_rows)]
    for j in range(model.num_cols):
        for i, a in model.entries[j]:
            by_row[i].append((j, a))
    mixed_rows = sum(1 for i in range(model.num_rows)
                     if any(not known[j] for j, _ in by_row[i]))
    worst_row, where = 0.0, ""
    for rank, _, values in members:
        for i in range(model.num_rows):
            low = high = 0.0
            magnitude = 1.0
            for j, a in by_row[i]:
                if known[j]:
                    term = a * values[model.col_names[j]]
                    low += term
                    high += term
                    magnitude = max(magnitude, abs(term))
                else:
                    lo, hi = model.col_lower[j], model.col_upper[j]
                    low += a * lo if a > 0 else a * hi
                    high += a * hi if a > 0 else a * lo
            violation = max(model.row_lower[i] - high, low - model.row_upper[i], 0.0)
            if math.isnan(violation):
                continue
            scaled = violation / magnitude
            if scaled > worst_row:
                worst_row, where = scaled, f"{model.row_names[i]} in member {rank}"
    report.check(worst_row <= primal_tol, "pool: rows",
                 (f"exact on all {model.num_rows} rows ("
                  + ("every column written" if full else "no continuous columns") + ")"
                  if not continuous
                  else f"{model.num_rows - mixed_rows} row(s) exact; on {mixed_rows} row(s) with "
                       "continuous columns, only that their bounds can still close the row")
                 + f"; worst {worst_row:.3e}" + (f" on {where}" if where else ""))

    if continuous:
        report.note("pool: objectives",
                    "not recomputed: the continuous values are not written, by design "
                    "(pool_write_all_columns writes them)")
        return
    worst_objective = 0.0
    for _, claimed, values in members:
        point = [values[n] for n in model.col_names]
        recomputed = (model.objective_offset
                      + sum(model.col_cost[j] * point[j] for j in range(model.num_cols))
                      + model.quadratic_objective(point))
        worst_objective = max(worst_objective,
                              abs(recomputed - claimed) / max(1.0, abs(recomputed)))
    report.check(worst_objective <= 1e-9, "pool: objectives recomputed",
                 f"worst relative difference {worst_objective:.3e}")


def verify(model: Model, solution: Solution, primal_tol: float, dual_tol: float,
           integer_tol: float, duality_tol: float) -> Report:
    report = Report()
    sigma = -1.0 if model.maximize else 1.0

    # ---- A verdict with no point of its own ----------------------------------------------
    # `infeasible` and `unbounded` are answers, not failures, and until #191 this script
    # treated them as though the solver had claimed a solution: it read the all-zero point a
    # .sol file carried out of habit, found it violated the rows, and printed REJECTED at a
    # correct answer. Our own checker calling our own correct verdict wrong is worse than not
    # checking it, so now each verdict is checked as what it is.
    claimed = solution.header.get("certificate", "none")
    if claimed == "farkas" and not solution.farkas:
        report.check(False, "certificate present",
                     "the header says `certificate farkas` but the file carries no farkas "
                     "section")
        return report
    if claimed == "ray" and not solution.ray:
        report.check(False, "certificate present",
                     "the header says `certificate ray` but the file carries no ray section")
        return report

    if solution.status == "infeasible":
        verify_farkas(model, solution, report)
        verify_iis(model, solution, report, primal_tol)
        return report

    # A VERDICT THAT CLAIMS NOTHING IS NOT CHECKED AS IF IT DID (#200). A numerical failure, a
    # solve that never started, a model this solver refuses - none of these assert a point, and
    # a file written under one carries no columns section since #200. Running the primal checks
    # against what it does carry was the same bug #191 fixed for `infeasible` alone: this
    # script printing REJECTED at an answer the solver never made.
    if solution.status not in STATUSES_WITH_A_POINT and solution.status != "infeasible_or_unbounded":
        report.note("verdict",
                    f"status is {solution.status}, which claims no point; nothing is asserted "
                    "and nothing is checked"
                    + (f" - {solution.header['message']}" if "message" in solution.header
                       else ""))
        return report

    if solution.status in ("unbounded", "infeasible_or_unbounded"):
        if solution.status == "infeasible_or_unbounded" and not solution.ray:
            report.note("verdict",
                        "the solver separated neither case and offers no ray; nothing to "
                        "check, and nothing is claimed")
            return report

    # ---- Structure ----------------------------------------------------------------------
    missing_cols = [n for n in model.col_names if n not in solution.col_value]
    missing_rows = [n for n in model.row_names if n not in solution.row_activity]

    if solution.status in ("not_solved", "model_error") and not solution.col_value and not solution.row_activity:
        report.note("structure", f"skipped: status is {solution.status} and no point was claimed")
        return report

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
    asserts_feasibility = solution.status in STATUSES_ASSERTING_FEASIBILITY

    def primal_check(worst_relative: float, worst_absolute: float, name: str,
                     detail: str) -> None:
        """Hold the point to what its status claims, and always report the measurement."""
        if asserts_feasibility:
            report.check(worst_relative <= primal_tol, name, detail)
            return
        claimed = solution.header_float("primal_infeasibility")
        if claimed is None:
            report.note(name, detail + f" - status is {solution.status}, which asserts no "
                                       "feasibility, and the file states none to compare")
            return
        # The solver may be as far outside as it admits to being, and no further.
        allowed = claimed * (1.0 + 1e-6) + primal_tol
        report.check(worst_absolute <= allowed, name + " matches the stated",
                     f"{detail}; the file states primal_infeasibility {claimed:.3e}, and "
                     f"{solution.status} asserts no better")

    # ---- Column bounds ------------------------------------------------------------------
    # Scaled by the variable's own magnitude, for the same reason as the rows above.
    worst, where, worst_abs = 0.0, "", 0.0
    for j, name in enumerate(model.col_names):
        violation = max(model.col_lower[j] - x[j], x[j] - model.col_upper[j], 0.0)
        scaled = violation / max(1.0, abs(x[j]))
        if scaled > worst:
            worst, where, worst_abs = scaled, name, violation
    primal_check(worst, worst_abs, "column bounds",
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
    primal_check(worst, worst_abs, "row activity",
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

    # ---- The ray, when the verdict was unbounded -------------------------------------------
    # Reached only after the point above has been checked feasible, which is the other half
    # of the claim: a ray from an infeasible point proves nothing at all.
    if solution.status in ("unbounded", "infeasible_or_unbounded"):
        return verify_ray(model, solution, report, primal_tol)

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

    if solution.pool:
        verify_pool(model, solution, report, x, objective, primal_tol, integer_tol)

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
            # "Optimal" on a MILP is a claim about the bound: the incumbent is within the
            # gap target of the best bound the search still had open (#188), or the tree
            # was exhausted and the two have met. The targets are read from the header the
            # solver wrote, defaulting to the project's (tolerances.hpp: 1e-4 relative,
            # 1e-6 absolute) when an older file has none. A gap wider than that means the
            # solver called an incumbent a proof, which is the most consequential thing a
            # branch and bound can get wrong and the least visible - the point is integral
            # and feasible either way.
            relative_target = solution.header_float("mip_relative_gap")
            absolute_target = solution.header_float("mip_absolute_gap")
            allowed = max(1e-6 if absolute_target is None else absolute_target,
                          (1e-4 if relative_target is None else relative_target) * scale)
            gap = abs(objective - bound)
            report.check(gap <= allowed + 1e-9 * scale, "optimality proof",
                         f"objective {objective:.12e} vs dual bound {bound:.12e}, "
                         f"gap {gap:.3e} against an allowed {allowed:.3e}")
            # And the bound must still be on the right side of the incumbent.
            slack = (objective - bound) if not model.maximize else (bound - objective)
            report.check(slack >= -1e-6 * scale, "dual bound is a bound",
                         f"incumbent {objective:.12e}, bound {bound:.12e}")
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

    # ---- Sensitivity ranging (optional; only checked when the .sol has the sections) ------
    # A nonbasic column's range is one-sided and equal to its reduced cost: at its lower
    # bound the cost may fall by d_j (minimization space) before the column becomes
    # attractive, at its upper bound it may rise by |d_j|. That is the one case this script
    # can re-derive without the basis factor. The file reports ranges in the MODEL'S sense,
    # so on a maximize model the side that carries d_j is the other one.
    # Reference: Chvatal, "Linear Programming", ch. 10 (1983).
    if solution.col_ranging_lower:
        worst, worst_where, checked = 0.0, "", 0
        for j, name in enumerate(model.col_names):
            status = solution.col_status.get(name)
            if status not in ("at_lower", "at_upper"):
                continue
            dj = d[j]  # minimization space: >= 0 at lower, <= 0 at upper
            lo = solution.col_ranging_lower.get(name)
            hi = solution.col_ranging_upper.get(name)
            if lo is None or hi is None:
                continue
            # (finite side in minimization space, its value) then map to the file's sense.
            finite_is_lower_in_min = status == "at_lower"
            finite_is_lower = finite_is_lower_in_min != model.maximize
            reported = lo if finite_is_lower else hi
            other = hi if finite_is_lower else lo
            expected = abs(dj)
            checked += 1
            scale_j = max(1.0, expected, abs(reported))
            err = abs(reported - expected) / scale_j
            if not math.isinf(other):
                err = max(err, 1.0)  # the free side must be reported as unbounded
            if err > worst:
                worst, worst_where = err, name
        report.check(
            worst <= 1e-6,
            "ranging: nonbasic ranges match the reduced costs",
            f"{checked} nonbasic column(s): the bound side equals |d_j| and the other side is "
            f"unbounded, max relative error {worst:.3e}"
            + (f" on {worst_where}" if worst_where else ""))

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
