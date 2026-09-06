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
# Reading back the quoted names that src/io/writer.cpp emits (#86).
#
# A fixed-format MPS name may legally contain a space - Netlib forplan has a column
# `DEDO3 11` - and the .sol columns/rows tables are one whitespace-delimited record per line.
# The writer double-quotes such a name, backslash-escaping a backslash or a quote inside it.
# parse_sol has to undo exactly that, or the recovered name will not match the one the
# independent MPS reader parsed, and the column shows up as structurally MISSING rather than
# merely misnamed - a confusing failure a long way from its cause.
#
# The two readers are written from the format independently and deliberately share no code,
# so this is the only place the pairing between them is actually checked.
# =============================================================================================

QUOTE, BACKSLASH = chr(34), chr(92)
ESCAPED_QUOTE_NAME = "HAS" + QUOTE + "QUOTED"

# Built from chr() rather than written as a literal. A quoted name containing an ESCAPED
# quote cannot be written straightforwardly inside a Python string - the source parser
# consumes the backslash first, so the file under test ends up with no escape in it and
# the test silently checks the wrong thing. That happened twice while writing this.
SOL_WITH_QUOTED_NAMES = "\n".join([
    "# SANKHYA solution file",
    "model QTEST",
    "status optimal",
    "",
    "begin columns 2",
    QUOTE + "DEDO3 11" + QUOTE + " 5 0 basic",
    "PLAIN 3 0 basic",
    "end columns",
    "",
    "begin rows 2",
    QUOTE + "DEDO3 1R" + QUOTE + " 4 0.5 at_upper",
    QUOTE + "HAS" + BACKSLASH + QUOTE + "QUOTED" + QUOTE + " 1 0 basic",
    "end rows",
    "",
]) + "\n"



def test_sol_reader_recovers_quoted_names() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quoted.sol"
        path.write_text(SOL_WITH_QUOTED_NAMES)
        # parse_sol RAISES rather than returning something wrong when it mis-splits a
        # record, so catch it here: an uncaught exception aborts the whole runner and every
        # later test silently never runs, which reads as "no failures" in CI.
        try:
            solution = vs.parse_sol(path)
        except Exception as error:  # noqa: BLE001 - reporting it IS the test
            check(False, "parse_sol reads a file containing quoted names", f"raised {error!r}")
            return

        check("DEDO3 11" in solution.col_value, "quoted column name with a space recovered",
              f"col_value keys={list(solution.col_value)}")
        check("DEDO3 1R" in solution.row_activity, "quoted row name with a space recovered",
              f"row_activity keys={list(solution.row_activity)}")
        check(ESCAPED_QUOTE_NAME in solution.row_activity,
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


def _one_row_lp(cost, lower, upper, row_lower, row_upper, entries) -> "vs.Model":
    """A one-row LP with columns x0, x1, ... and row r0."""
    model = vs.Model()
    model.name = "ACCOUNTING"
    model.col_names = [f"x{j}" for j in range(len(cost))]
    model.col_index = {name: j for j, name in enumerate(model.col_names)}
    model.col_cost = list(cost)
    model.col_lower = list(lower)
    model.col_upper = list(upper)
    model.col_integer = [False] * len(cost)
    model.row_names = ["r0"]
    model.row_index = {"r0": 0}
    model.row_lower = [row_lower]
    model.row_upper = [row_upper]
    model.entries = [[(0, a)] for a in entries]
    return model


def _certificate(model, x, d, y) -> "vs.Solution":
    solution = vs.Solution()
    activity = sum(entries[0][1] * v for entries, v in zip(model.entries, x))
    solution.header = {"status": "optimal",
                       "objective": repr(sum(c * v for c, v in zip(model.col_cost, x)))}
    solution.col_value = dict(zip(model.col_names, x))
    solution.col_dual = dict(zip(model.col_names, d))
    solution.row_activity = {"r0": activity}
    solution.row_dual = {"r0": y}
    return solution


def _strong_duality(report):
    """The (ok, name, detail) record of the strong-duality check."""
    return next(line for line in report.lines if line[1] == "strong duality")


def test_strong_duality_still_rejects_a_reduced_cost_pricing_the_wrong_bound() -> None:
    """Netlib recipe (#157) at unit scale. x0 + x1 = 0 with both columns in [0, 20] forces
    both to their lower bounds, so the point is optimal for any costs; with costs (1, 2)
    the row price must satisfy y <= 1. y = 1.004 gives d0 = -0.004: a column at its LOWER
    bound with a reduced cost that prices the UPPER bound, 20 away. The sign check passes
    (both bounds exist), complementarity passes (nearest slack 0), consistency passes (d
    is exactly c - A^T y). Only strong duality sees the 0.08, and the accounting must not
    explain it away: a multiplier above the tolerance explains its share of the gap only
    up to the nearest slack, which is zero."""
    model = _one_row_lp(cost=[1.0, 2.0], lower=[0.0, 0.0], upper=[20.0, 20.0],
                        row_lower=0.0, row_upper=0.0, entries=[1.0, 1.0])
    solution = _certificate(model, x=[0.0, 0.0], d=[1.0 - 1.004, 2.0 - 1.004], y=1.004)
    report = vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                       vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
    ok, _, detail = _strong_duality(report)
    check(report.failures == 1 and not ok,
          "a wrong-bound reduced cost fails strong duality and nothing else", detail)


def test_strong_duality_accepts_a_gap_made_of_accepted_per_item_violations() -> None:
    """Netlib etamacro at unit scale. min x0 - 5e-8 x1 s.t. x0 + x1 >= 40, x0 in [0, 20],
    x1 in [40, 100]. The certificate x = (0, 40), y = 0, d = c is exactly consistent, and
    d1 = -5e-8 is half the dual tolerance: the sign check accepts it (both bounds exist)
    and complementarity is 0 (x1 sits at a bound). The true optimum is x1 = 100, better by
    3e-6 - which is what a reduced cost the dual tolerance calls zero, on a range of 60,
    can hide. The old aggregate test rejected this at a relative gap of 2e-6 against 1e-9,
    tighter than the per-item tolerance that had just accepted the cause. The gap is now
    accounted for by that accepted item."""
    model = _one_row_lp(cost=[1.0, -5e-8], lower=[0.0, 40.0], upper=[20.0, 100.0],
                        row_lower=40.0, row_upper=vs.INF, entries=[1.0, 1.0])
    solution = _certificate(model, x=[0.0, 40.0], d=[1.0, -5e-8], y=0.0)
    report = vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                       vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
    check(report.failures == 0, "accepted per-item violations account for the gap",
          "; ".join(f"{name}: {detail}" for ok, name, detail in report.lines if not ok)
          or _strong_duality(report)[2])


QPS_WITH_OFF_DIAGONAL = """NAME          QCONV
ROWS
 N  COST
 G  R1
COLUMNS
    X         COST        -1.0   R1           1.0
    Y         COST        -1.0   R1           1.0
RHS
    RHS       R1           0.0
BOUNDS
 FR BND       X
 FR BND       Y
QUADOBJ
    X         X            2.0
    X         Y            1.0
    Y         Y            2.0
ENDATA
"""


def test_qps_convention_is_read_as_qps_means_it() -> None:
    """The 0.5 / lower-triangle convention, pinned on the EVALUATED objective.

    QPS states the objective as c'x + 0.5 x'Qx and lists only the lower triangle of the
    symmetric Q, so a stored off-diagonal stands for TWO entries of Q. The two ways to get
    this wrong - halving the off-diagonal, or mirroring it into both triangles - both produce
    a script that reads the file back plausibly and then certifies the solver's answer to a
    DIFFERENT problem. Checking a stored number would not catch either; checking the value of
    the objective at a known point does.

    Q = [[2, 1], [1, 2]], c = (-1, -1). At x = (1, 1):
        c'x        = -2
        0.5 x'Qx   = 0.5 * (2 + 1 + 1 + 2) = 3
        objective  = 1
    A mirrored reading gives 4 for the quadratic term; a halved one gives 2.5.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qconv.qps"
        path.write_text(QPS_WITH_OFF_DIAGONAL)
        model = vs.parse_mps(path)

        check(len(model.hessian) == 3, "three Hessian entries stored",
              f"got {len(model.hessian)}")
        check(model.hessian.get((1, 0)) == 1.0, "the off-diagonal is stored lower-triangular",
              f"hessian={model.hessian}")

        x = [1.0, 1.0]
        quadratic = model.quadratic_objective(x)
        check(abs(quadratic - 3.0) < 1e-12, "0.5 x'Qx at (1, 1)",
              f"got {quadratic}, expected 3.0 (4.0 would mean mirrored, 2.5 halved)")

        # Qx = (2*1 + 1*1, 1*1 + 2*1) = (3, 3). This is the gradient term every KKT check
        # below depends on, so it is pinned separately from the objective.
        qx = model.hessian_times(x)
        check(qx == [3.0, 3.0], "Qx expands the stored triangle symmetrically", f"got {qx}")


def test_qp_optimum_verifies_and_a_wrong_one_does_not() -> None:
    """The KKT conditions, on the QP above, at the true optimum and at a near miss.

    min -x - y + 0.5(2x^2 + 2xy + 2y^2)  s.t. x + y >= 0, x and y free.
    Gradient c + Qx vanishes where 2x + y = 1 and x + 2y = 1, i.e. x = y = 1/3. The row is
    then slack (2/3 > 0) so its multiplier is zero, and the objective is -1/3.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qconv.qps"
        path.write_text(QPS_WITH_OFF_DIAGONAL)
        model = vs.parse_mps(path)

        third = 1.0 / 3.0
        solution = vs.Solution()
        solution.header = {"status": "optimal", "objective": repr(-third)}
        solution.col_value = {"X": third, "Y": third}
        solution.col_status = {"X": "basic", "Y": "basic"}
        solution.col_dual = {"X": 0.0, "Y": 0.0}
        solution.row_activity = {"R1": 2.0 * third}
        solution.row_dual = {"R1": 0.0}

        report = vs.verify(model, solution, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                           vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
        check(report.failures == 0, "the true QP optimum verifies",
              f"{report.failures} check(s) failed")

        # A point that is primal FEASIBLE and whose objective is reported consistently, but
        # which is not stationary. Only the quadratic-aware KKT checks can tell the two
        # apart - to an LP-shaped verifier this point looks exactly as good as the optimum.
        objective = -1.0 + 0.5 * (2.0 + 2.0 * 0.5 + 2.0 * 0.25)
        near_miss = vs.Solution()
        near_miss.header = {"status": "optimal", "objective": repr(objective)}
        near_miss.col_value = {"X": 1.0, "Y": 0.5}
        near_miss.col_status = {"X": "basic", "Y": "basic"}
        near_miss.col_dual = {"X": 0.0, "Y": 0.0}
        near_miss.row_activity = {"R1": 1.5}
        near_miss.row_dual = {"R1": 0.0}

        report = vs.verify(model, near_miss, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                           vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
        check(report.failures > 0, "a feasible non-stationary point is rejected",
              f"{report.failures} check(s) failed, as expected")


QPS_MAXIMIZE = """NAME          QMAX
OBJSENSE
    MAX
ROWS
 N  COST
 L  R1
COLUMNS
    X         COST         4.0   R1           1.0
RHS
    RHS       R1          10.0
BOUNDS
 FR BND       X
QUADOBJ
    X         X           -2.0
ENDATA
"""


def test_quadratic_maximization_keeps_its_sign() -> None:
    """The sign path, which is where a quadratic objective is easiest to get wrong.

    Every KKT check runs in MINIMIZE space, reached by multiplying through by sigma. The
    linear cost was already handled that way; the quadratic term has to follow it, and it
    enters in two places - the gradient c + Qx, and the -0.5 x'Qx the Dorn dual subtracts.
    Miss sigma on either and a maximization QP is checked against the conditions for its
    negation, which rejects correct answers and, on a symmetric enough instance, accepts
    wrong ones.

    max 4x - x^2  s.t. x <= 10, x free. Stored as c = 4 and Q = -2, since the file's
    objective is c'x + 0.5 x'Qx. The gradient 4 - 2x vanishes at x = 2, the row is slack
    there (2 < 10) so its multiplier is zero, and the objective is 8 - 4 = 4.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qmax.qps"
        path.write_text(QPS_MAXIMIZE)
        model = vs.parse_mps(path)

        check(model.maximize, "OBJSENSE MAX survives the quadratic section", "")
        check(model.hessian == {(0, 0): -2.0}, "negative Hessian stored as written",
              f"got {model.hessian}")

        optimum = vs.Solution()
        optimum.header = {"status": "optimal", "objective": "4.0"}
        optimum.col_value = {"X": 2.0}
        optimum.col_status = {"X": "basic"}
        optimum.col_dual = {"X": 0.0}
        optimum.row_activity = {"R1": 2.0}
        optimum.row_dual = {"R1": 0.0}

        report = vs.verify(model, optimum, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                           vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
        check(report.failures == 0, "the true maximum verifies",
              f"{report.failures} check(s) failed")

        # x = 3 is feasible and its objective is reported correctly (12 - 9 = 3). It is
        # simply not the maximum. Only a check that knows the gradient is 4 - 2x can say so.
        near_miss = vs.Solution()
        near_miss.header = {"status": "optimal", "objective": "3.0"}
        near_miss.col_value = {"X": 3.0}
        near_miss.col_status = {"X": "basic"}
        near_miss.col_dual = {"X": 0.0}
        near_miss.row_activity = {"R1": 3.0}
        near_miss.row_dual = {"R1": 0.0}

        report = vs.verify(model, near_miss, vs.DEFAULT_PRIMAL_TOL, vs.DEFAULT_DUAL_TOL,
                           vs.DEFAULT_INTEGER_TOL, vs.DEFAULT_DUALITY_TOL)
        check(report.failures > 0, "a feasible non-maximal point is rejected",
              f"{report.failures} check(s) failed, as expected")


def main() -> int:
    print("test_fixed_format_row_name_with_space")
    test_fixed_format_row_name_with_space()
    print("test_sol_reader_recovers_quoted_names")
    test_sol_reader_recovers_quoted_names()
    print("test_known_bad_solution_is_rejected")
    test_known_bad_solution_is_rejected()
    test_strong_duality_still_rejects_a_reduced_cost_pricing_the_wrong_bound()
    test_strong_duality_accepts_a_gap_made_of_accepted_per_item_violations()
    print("test_qps_convention_is_read_as_qps_means_it")
    test_qps_convention_is_read_as_qps_means_it()
    print("test_qp_optimum_verifies_and_a_wrong_one_does_not")
    test_qp_optimum_verifies_and_a_wrong_one_does_not()
    print("test_quadratic_maximization_keeps_its_sign")
    test_quadratic_maximization_keeps_its_sign()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
