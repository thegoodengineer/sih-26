#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The .sol reader verify_solution.py checks a solve against (issue #262 split this out
of one 1,529-line file). Reads only the SANKHYA .sol format written by src/io/writer.cpp's
documented layout - it does not link that C++, only reads what it wrote.
"""
from __future__ import annotations

from pathlib import Path

from verify_solution_io import open_text

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


