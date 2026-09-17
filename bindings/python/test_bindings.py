#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SANKHYA Python bindings.

Hand-rolled rather than pytest, matching tools/test_verify_solution.py, so that running the
bindings' tests never requires installing anything the solver does not already need.

Every expected value here is derived by hand in the comment above it. A binding is a
translation layer, and the failure it can introduce that the core cannot is reading the RIGHT
number out of the WRONG field - which a test that captured its expectations from a previous
run would happily freeze in.

    PYTHONPATH=bindings/python python bindings/python/test_bindings.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sankhya  # noqa: E402

FAILURES = 0


def check(condition: bool, name: str, detail: str = "") -> None:
    global FAILURES
    if condition:
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILURES += 1
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


def near(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_lp() -> None:
    #   maximise 3x + 2y  s.t.  x + y <= 4,  x + 3y <= 6,  0 <= x <= 3,  y >= 0
    # Both rows tight at x = 3, y = 1 -> objective 11. x sits on its upper bound there, so a
    # boxed column is exercised rather than only the origin cone.
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=3.0, upper=3.0, name="x")
    y = model.add_column(cost=2.0, name="y")
    model.add_row({x: 1.0, y: 1.0}, upper=4.0, name="c0")
    model.add_row({x: 1.0, y: 3.0}, upper=6.0, name="c1")

    check(model.num_cols == 2 and model.num_rows == 2, "dimensions",
          f"{model.num_rows} x {model.num_cols}")
    check(model.num_nonzeros == 4, "nonzeros", str(model.num_nonzeros))
    model.validate()

    result = model.solve(log_to_console=False)
    check(result.status == "optimal", "status", result.status)
    check(result.optimal, "optimal flag")
    check(near(result.objective, 11.0), "objective", f"{result.objective}")
    check(near(result.x[0], 3.0) and near(result.x[1], 1.0), "primal values", str(result.x))
    check(near(result.row_activities[0], 4.0), "row activity", str(result.row_activities))
    check(result.primal_infeasibility <= 1e-7, "measured primal infeasibility",
          f"{result.primal_infeasibility:.3e}")
    check(result.iterations > 0, "iterations reported", str(result.iterations))
    check(len(result.row_duals) == 2, "duals have the right length", str(result.row_duals))


def test_milp() -> None:
    #   maximise x + y  s.t.  2x + 2y <= 3,  x, y in {0, 1}
    # The relaxation gives x = y = 0.75 for 1.5; the integer optimum is a single unit, 1.
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=1.0, upper=1.0, integer=True, name="x")
    y = model.add_column(cost=1.0, upper=1.0, integer=True, name="y")
    model.add_row({x: 2.0, y: 2.0}, upper=3.0)

    result = model.solve(log_to_console=False)
    check(result.status == "optimal", "MILP status", result.status)
    check(near(result.objective, 1.0), "MILP objective", f"{result.objective}")
    check(result.integrality_violation <= 1e-6, "integrality",
          f"{result.integrality_violation:.3e}")
    check(all(abs(v) < 1e-6 or abs(v - 1.0) < 1e-6 for v in result.x), "values are integral",
          str(result.x))
    check(0.0 <= result.relative_gap <= 1e-4, "optimal MILP finished within its gap target",
          f"{result.relative_gap:.3e}")
    check(near(result.absolute_gap, abs(result.objective - result.dual_bound), 1e-9),
          "absolute gap is objective minus bound", f"{result.absolute_gap:.3e}")


def test_qp() -> None:
    # minimise 0.5*(2x^2 + 2y^2) - 2x - 6y  s.t.  x + y <= 3,  x, y >= 0.
    # The unconstrained stationary point (1, 3) violates the row, so the optimum lies on
    # x + y = 3. With y = 3 - x: f = 2x^2 - 2x - 9, minimised at x = 0.5, y = 2.5, f = -9.5.
    model = sankhya.Model()
    x = model.add_column(cost=-2.0, name="x")
    y = model.add_column(cost=-6.0, name="y")
    model.add_row({x: 1.0, y: 1.0}, upper=3.0)
    model.set_quadratic(x, x, 2.0)
    model.set_quadratic(y, y, 2.0)

    result = model.solve(log_to_console=False, qp_tolerance=1e-11, iteration_limit=500000)
    check(result.status == "optimal", "QP status", result.status + " " + result.message)
    check(near(result.objective, -9.5, 1e-5), "QP objective", f"{result.objective}")
    check(near(result.x[0], 0.5, 1e-4) and near(result.x[1], 2.5, 1e-4), "QP primal values",
          str([round(v, 6) for v in result.x]))


def test_coefficient_replaces_rather_than_accumulates() -> None:
    # The MPS reader treats a repeated entry as an error; through an API, overwriting a cell
    # is ordinary. Summing would silently double a coefficient, which the answer does not
    # reveal. minimise x s.t. 2x >= 6 gives 3; a summed 3x >= 6 would give 2.
    model = sankhya.Model()
    x = model.add_column(cost=1.0, name="x")
    row = model.add_row(lower=6.0)
    model.set_coefficient(row, x, 1.0)
    model.set_coefficient(row, x, 2.0)
    check(model.num_nonzeros == 1, "the entry was replaced, not duplicated",
          str(model.num_nonzeros))
    result = model.solve(log_to_console=False)
    check(near(result.objective, 3.0), "objective after replacement", f"{result.objective}")


def test_bool_is_not_routed_to_the_int_setter() -> None:
    # bool is a subclass of int in Python, so a naive isinstance(value, int) check routes
    # True to the integer setter, which the C API then rejects as the wrong type for a bool
    # option. The dispatch order in Options.set exists for this and nothing else.
    options = sankhya.Options(presolve=False, log_to_console=False)
    check(True, "bool options are accepted", "presolve=False, log_to_console=False")

    model = sankhya.Model()
    x = model.add_column(cost=1.0, name="x")
    model.add_row({x: 1.0}, lower=2.0)
    result = model.solve(options)
    check(near(result.objective, 2.0), "solve honours a bool option", f"{result.objective}")


def test_options_reject_a_typo_instead_of_aborting() -> None:
    # The C++ typed accessors call std::abort() on an unknown option name - correct for C++,
    # fatal for an FFI caller, since it would take the interpreter down with it. The C API
    # checks the registry first. If that check regresses, this test does not fail: the whole
    # process dies, which is itself unmistakable.
    raised = False
    try:
        sankhya.Options(no_such_option_at_all=1.0)
    except sankhya.SankhyaError as error:
        raised = "unknown option" in str(error).lower()
    check(raised, "an unknown option raises rather than aborting the process")


def test_reads_a_file() -> None:
    text = ("NAME          TINY\n"
            "ROWS\n"
            " N  COST\n"
            " G  R1\n"
            "COLUMNS\n"
            "    X         COST         1.0   R1           1.0\n"
            "RHS\n"
            "    RHS       R1           4.0\n"
            "ENDATA\n")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "tiny.mps")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        model = sankhya.Model.read(path)
        check(model.num_cols == 1 and model.num_rows == 1, "file dimensions", repr(model))
        result = model.solve(log_to_console=False)
        check(near(result.objective, 4.0), "objective from file", f"{result.objective}")

        raised = False
        try:
            sankhya.Model.read(os.path.join(directory, "does_not_exist.mps"))
        except sankhya.SankhyaError:
            raised = True
        check(raised, "a missing file raises with the reader's message")


def test_infeasible_returns_rather_than_raising() -> None:
    # A CALL that fails raises; a model the solver PROVES infeasible is a successful call
    # with an infeasible answer. Conflating the two would make an ordinary modelling outcome
    # indistinguishable from a bug in the caller's code.
    model = sankhya.Model()
    x = model.add_column(cost=1.0, upper=1.0, name="x")
    model.add_row({x: 1.0}, lower=5.0)
    result = model.solve(log_to_console=False)
    check(result.status in ("infeasible", "infeasible_or_unbounded"),
          "infeasible is returned, not raised", result.status)


def test_certificates_reach_python() -> None:
    # #207: the proofs behind `infeasible` and `unbounded` are returned, and each is None
    # (not []) when the solver has no proof to attach. The checks below are the proofs
    # themselves, worked by hand for these two tiny models, so a binding wired to the wrong
    # field fails here rather than handing back a plausible vector of the right length.
    infinity = sankhya.INFINITY

    # x free, x >= 5 and x <= 2. Presolve off: with it on, presolve settles this from bound
    # arithmetic and no engine runs, so no Farkas vector exists.
    model = sankhya.Model()
    x = model.add_column(cost=1.0, lower=-infinity, name="x")
    model.add_row({x: 1.0}, lower=5.0, name="lo")
    model.add_row({x: 1.0}, upper=2.0, name="hi")
    result = model.solve(log_to_console=False, presolve=False)
    check(result.status == "infeasible", "certificate model is infeasible", result.status)
    check(not result.claims_a_point, "infeasible claims no point")
    check(result.primal_ray is None, "no ray on an infeasible model", str(result.primal_ray))
    y = result.farkas_dual
    check(y is not None and len(y) == 2, "one Farkas multiplier per row", str(y))
    if y is not None and len(y) == 2:
        # y_lo > 0 selects row lo's lower bound, y_hi < 0 row hi's upper bound. x is free, so
        # the aggregate's column coefficient must vanish, and the bound sum must be positive.
        check(y[0] > 0.0 and y[1] < 0.0, "multiplier signs select existing bounds", str(y))
        check(abs(y[0] + y[1]) <= 1e-9 * max(1.0, abs(y[0])), "A'y = 0 on the free column",
              str(y))
        check(y[0] * 5.0 + y[1] * 2.0 > 0.0, "aggregated bounds contradict", str(y))

    with_presolve = model.solve(log_to_console=False)
    check(with_presolve.status == "infeasible", "infeasible with presolve",
          with_presolve.status)
    check(with_presolve.farkas_dual is None or len(with_presolve.farkas_dual) == 2,
          "no certificate reads as None, never []", str(with_presolve.farkas_dual))

    # minimise -x, x >= 1: unbounded along +x from a feasible start.
    model = sankhya.Model()
    model.add_column(cost=-1.0, lower=1.0, name="x")
    result = model.solve(log_to_console=False)
    check(result.status == "unbounded", "ray model is unbounded", result.status)
    check(result.claims_a_point, "unbounded carries its feasible starting point")
    check(result.farkas_dual is None, "no Farkas vector on an unbounded model",
          str(result.farkas_dual))
    d = result.primal_ray
    check(d is not None and len(d) == 1 and d[0] > 0.0, "ray points along +x", str(d))
    check(result.x[0] >= 1.0 - 1e-7, "ray starts from a feasible point", str(result.x))


def test_handles_are_released() -> None:
    # Every handle is owned by a Python object. If __del__ did not free them this would leak
    # a few thousand models; the check is that it completes and stays responsive rather than
    # any assertion about memory, which Python cannot observe portably.
    for _ in range(2000):
        model = sankhya.Model()
        model.add_column(cost=1.0, name="x")
        del model
    with sankhya.Model() as model:
        x = model.add_column(cost=1.0, name="x")
        model.add_row({x: 1.0}, lower=1.0)
        with model.solve(log_to_console=False) as result:
            check(near(result.objective, 1.0), "context managers work",
                  f"{result.objective}")
    check(True, "2000 create/destroy cycles completed")

def test_interrupt() -> None:
    import threading
    import time

    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=3.0, upper=3.0)
    y = model.add_column(cost=2.0)
    model.add_row({x: 1.0, y: 1.0}, upper=4.0)

    def worker():
        time.sleep(0.05)
        model.interrupt()

    threading.Thread(target=worker).start()

    def stall(p):
        time.sleep(0.1)
        return 0

    result = model.solve(iteration_limit=500000000, algorithm="simplex", callback=stall)
    check(result.status == "interrupted", "solver interrupted async", result.status)


def test_callback() -> None:
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=3.0, upper=3.0)
    y = model.add_column(cost=2.0)
    model.add_row({x: 1.0, y: 1.0}, upper=4.0)

    # 1. callback=None
    result = model.solve(callback=None, log_to_console=False)
    check(result.status == "optimal", "callback=None works", result.status)

    # 2. callback returns 0
    calls = []
    def cb_zero(p):
        calls.append(p.phase)
        return 0
    result = model.solve(callback=cb_zero, log_to_console=False)
    check(result.status == "optimal", "callback=cb_zero works", result.status)

    # 3. callback returns 1
    def cb_one(p):
        return 1
    result = model.solve(callback=cb_one, log_to_console=False, iteration_limit=500000000)
    check(result.status == "interrupted", "callback=cb_one interrupts", result.status)

    # 4. callback raises exception
    def cb_exc(p):
        raise ValueError("test exception")
    try:
        model.solve(callback=cb_exc, log_to_console=False, iteration_limit=500000000)
        check(False, "callback exception was swallowed")
    except ValueError as e:
        check(str(e) == "test exception", "callback exception propagated correctly")


def test_executable_discovery() -> None:
    import tempfile
    import os
    import sys
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        # Mock repository root
        original_root = sankhya._executable._repository_root
        sankhya._executable._repository_root = lambda: temp_dir

        original_platform = sys.platform

        # We need to save environ
        old_env = dict(os.environ)

        try:
            os.environ.pop("SANKHYA_EXECUTABLE", None)
            os.environ.pop("SANKHYA_BIN", None)

            # Unix-like test
            sys.platform = "linux"
            build_dir = temp_dir / "build"
            build_dir.mkdir()
            exe = build_dir / "sankhya"
            exe.touch(mode=0o755)

            found = sankhya.locate_executable()
            check(found == exe, "discovered build/sankhya on unix", str(found))

            # Remove unix exe
            exe.unlink()

            # Windows test with build-cuda
            sys.platform = "win32"
            build_cuda_dir = temp_dir / "build-cuda"
            build_cuda_dir.mkdir()
            exe_win = build_cuda_dir / "sankhya.exe"
            exe_win.touch(mode=0o755)

            found = sankhya.locate_executable()
            check(found == exe_win, "discovered build-cuda/sankhya.exe on windows", str(found))

            # Test 4: explicit SANKHYA_EXECUTABLE override
            override_exe = temp_dir / "override.exe"
            override_exe.touch(mode=0o755)
            os.environ["SANKHYA_EXECUTABLE"] = str(override_exe)

            found = sankhya.locate_executable()
            check(found == override_exe, "SANKHYA_EXECUTABLE override works", str(found))

            # Test 5 & 6: SANKHYA_BIN compatibility and precedence
            bin_exe = temp_dir / "bin_override.exe"
            bin_exe.touch(mode=0o755)
            os.environ["SANKHYA_BIN"] = str(bin_exe)

            # Both set, SANKHYA_EXECUTABLE wins
            found = sankhya.locate_executable()
            check(found == override_exe, "SANKHYA_EXECUTABLE takes precedence over SANKHYA_BIN")

            # Only SANKHYA_BIN set
            os.environ.pop("SANKHYA_EXECUTABLE")
            found = sankhya.locate_executable()
            check(found == bin_exe, "SANKHYA_BIN compatibility works")

            # Test 7: Missing executable diagnostic
            os.environ.pop("SANKHYA_BIN")
            exe_win.unlink() # remove the build-cuda exe

            try:
                sankhya.locate_executable()
                check(False, "missing executable did not raise SankhyaError")
            except sankhya.SankhyaError as e:
                msg = str(e)
                check("was not found" in msg, "error mentions not found")
                check("sankhya.exe" in msg, "error mentions requested name")
                check("build-cuda" in msg, "error mentions searched paths")
                check("SANKHYA_EXECUTABLE" in msg, "error mentions override vars")

        finally:
            sankhya._executable._repository_root = original_root
            sys.platform = original_platform
            os.environ.clear()
            os.environ.update(old_env)


def main() -> int:
    print(f"SANKHYA Python bindings, against solver version {sankhya.version()}\n")
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            print(name)
            function()
    print()
    if FAILURES == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{FAILURES} check(s) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
