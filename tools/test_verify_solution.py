#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for verify_solution.py's MPS reader.

Standalone, no test framework dependency - consistent with the rest of bench/ and tools/.
Run directly:

    python tools/test_verify_solution.py

Exit codes: 0 all tests passed, 1 at least one failed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_solution as vs  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES += 1


# =============================================================================================
# Issue #48: a row or column name containing a space is legal fixed-format MPS. Free-format
# tokenisation splits "DEDO3 1R" into two fields ("DEDO3", "1R"), shifting every field after it
# left by one - on the COLUMNS line below that turns the row name "OB1PNW20" into a value field,
# and float("OB1PNW20") raises exactly the error this issue reports. parse_mps() must retry the
# whole file in fixed columns when free-format tokenisation fails, and recover the real names.
#
# These lines are built at the exact 1-based IBM column positions (2-3, 5-12, 15-22, 25-36,
# 40-47, 50-61), the same positions FIXED_FIELDS in verify_solution.py encodes.
# =============================================================================================

FIXED_FORMAT_MPS_WITH_SPACES_IN_NAMES = """\
NAME          FORPLAN-LIKE
ROWS
 N  OB1PNW20
 E  DEDO3 1R
COLUMNS
    DEDO3 11  OB1PNW20  .02466         DEDO3 1R  -1.
RHS
    RHS1      DEDO3 1R  0.0
ENDATA
"""


def test_fixed_format_row_name_with_space() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "forplan_like.mps"
        path.write_text(FIXED_FORMAT_MPS_WITH_SPACES_IN_NAMES)

        # Free-format alone must fail on this file - if it didn't, the fixture wouldn't be
        # testing anything. This pins the bug report itself, not just the fix.
        raised = False
        try:
            vs._parse_mps(path, fixed=False)
        except ValueError:
            raised = True
        check(raised, "free-format tokenisation fails on this file",
              "confirms the fixture reproduces the reported bug")

        # parse_mps() is the public entry point: free-format first, fixed-column retry on
        # failure. This must succeed and recover the names byte-for-byte, spaces included.
        model = vs.parse_mps(path)
        check(model.num_rows == 1, "row count", f"got {model.num_rows}, expected 1")
        check(model.num_cols == 1, "column count", f"got {model.num_cols}, expected 1")
        check("DEDO3 1R" in model.row_index, "row name recovered with embedded space",
              f"row_names={model.row_names}")
        check("DEDO3 11" in model.col_index, "column name recovered with embedded space",
              f"col_names={model.col_names}")

        if "DEDO3 11" in model.col_index:
            col = model.col_index["DEDO3 11"]
            check(abs(model.col_cost[col] - 0.02466) < 1e-12, "objective coefficient",
                  f"got {model.col_cost[col]}, expected 0.02466")
            entries = dict(model.entries[col])
            if "DEDO3 1R" in model.row_index:
                row = model.row_index["DEDO3 1R"]
                check(row in entries and abs(entries[row] - (-1.0)) < 1e-12,
                      "row coefficient", f"entries={entries}")


# =============================================================================================
# Issue #86: a fixed-format MPS name may legally contain a space ("DEDO3 11"), and the .sol
# columns/rows tables are one whitespace-delimited record per line. src/io/writer.cpp
# double-quotes such a name - with '\' and '"' backslash-escaped inside the quotes - and
# parse_sol()'s _split_name_field() must undo exactly that, or the recovered name will not
# match the one the independent MPS reader parsed from the .mps file, and the column/row
# will show up as structurally missing rather than merely misnamed.
# =============================================================================================

SOL_FILE_WITH_QUOTED_NAMES = """\
# SANKHYA solution file
model QUOTETEST
source in-memory
sense minimize
status optimal
algorithm simplex-primal
objective 8
dual_bound 8
objective_offset 0
rows 2
columns 2
iterations 1
nodes 0
solve_seconds 0.001

# name value reduced_cost basis_status
begin columns 2
"DEDO3 11" 5 0 basic
PLAIN 3 0 basic
end columns

# name activity dual basis_status
begin rows 2
"DEDO3 1R" 4 0.5 at_upper
"HAS\\"QUOTE" 1 0 basic
end rows
"""


def test_sol_reader_recovers_quoted_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quoted.sol"
        path.write_text(SOL_FILE_WITH_QUOTED_NAMES)
        solution = vs.parse_sol(path)

        check("DEDO3 11" in solution.col_value, "quoted column name with a space recovered",
              f"col_value keys={list(solution.col_value)}")
        check("DEDO3 1R" in solution.row_activity, "quoted row name with a space recovered",
              f"row_activity keys={list(solution.row_activity)}")
        check('HAS"QUOTE' in solution.row_activity,
              "backslash-escaped quote inside a name recovered",
              f"row_activity keys={list(solution.row_activity)}")
        check("PLAIN" in solution.col_value, "an unquoted name is unaffected",
              f"col_value keys={list(solution.col_value)}")

        if "DEDO3 11" in solution.col_value:
            check(solution.col_value["DEDO3 11"] == 5.0, "value for the quoted column",
                  f"got {solution.col_value.get('DEDO3 11')}")
            check(solution.col_status["DEDO3 11"] == "basic", "status for the quoted column",
                  f"got {solution.col_status.get('DEDO3 11')}")
        if "DEDO3 1R" in solution.row_activity:
            check(solution.row_dual["DEDO3 1R"] == 0.5, "dual for the quoted row",
                  f"got {solution.row_dual.get('DEDO3 1R')}")


def test_known_bad_solution_is_rejected() -> None:
    """A sanity check on the pass/fail contract itself: a solution violating a bound must
    fail verification, not pass it."""
    model = vs.Model()
    model.row_names = []
    model.col_names = ["x"]
    model.col_index = {"x": 0}
    model.col_cost = [1.0]
    model.col_lower = [0.0]
    model.col_upper = [1.0]
    model.col_integer = [False]
    model.entries = [[]]

    solution = vs.Solution()
    solution.header = {"status": "optimal", "objective": "5.0"}
    solution.col_value = {"x": 5.0}  # violates upper bound 1.0
    solution.col_dual = {"x": 0.0}

    report = vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                       vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
    check(report.failures > 0, "bound-violating solution is rejected",
          f"{report.failures} check(s) failed, as expected")


def main() -> int:
    print("test_fixed_format_row_name_with_space")
    test_fixed_format_row_name_with_space()
    print("test_known_bad_solution_is_rejected")
    test_known_bad_solution_is_rejected()
    print("test_sol_reader_recovers_quoted_names")
    test_sol_reader_recovers_quoted_names()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
