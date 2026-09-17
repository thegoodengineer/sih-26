#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The independent MPS reader verify_solution.py checks against (issue #262 split this
out of one 1,529-line file). It does NOT reuse src/io/mps_reader.cpp's interpretation of
anything - see verify_solution.py's own docstring for why that independence matters."""
from __future__ import annotations

from pathlib import Path

from verify_solution_io import INF, normalize_infinity, open_text

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


