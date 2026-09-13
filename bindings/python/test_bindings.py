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
import threading
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


def test_progress_callback_reports_fields() -> None:
    # #223. Every solve installs an internal callback (for Ctrl-C) regardless of whether
    # the caller supplies one, and SolveControl always fires at least once (its own first
    # call is never throttled) - so even this one-iteration LP is enough to see a snapshot.
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=3.0, upper=3.0, name="x")
    y = model.add_column(cost=2.0, name="y")
    model.add_row({x: 1.0, y: 1.0}, upper=4.0)
    model.add_row({x: 1.0, y: 3.0}, upper=6.0)

    seen: list[sankhya.Progress] = []
    result = model.solve(log_to_console=False, callback=lambda p: seen.append(p) and False)
    check(result.status == "optimal", "the solve still finished normally", result.status)
    check(len(seen) >= 1, "the callback was called at least once", str(len(seen)))
    progress = seen[0]
    check(isinstance(progress, sankhya.Progress), "the callback receives a Progress")
    check(progress.phase in ("presolve", "lp", "tree"), "phase is one of the known names",
          progress.phase)
    check(progress.elapsed_seconds >= 0.0, "elapsed_seconds is non-negative",
          str(progress.elapsed_seconds))
    check(progress.iterations >= 0, "iterations is non-negative", str(progress.iterations))
    check(repr(progress).startswith("<sankhya.Progress"), "Progress has a readable repr",
          repr(progress))


def test_callback_returning_true_interrupts_the_solve() -> None:
    # A callback that stops on its very first call is deterministic regardless of machine
    # speed or how many nodes/iterations the problem would otherwise take, because
    # SolveControl's own first call is never throttled (#223).
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=1.0, upper=1.0, integer=True, name="x")
    y = model.add_column(cost=1.0, upper=1.0, integer=True, name="y")
    model.add_row({x: 2.0, y: 2.0}, upper=3.0)

    calls = []
    result = model.solve(log_to_console=False, callback=lambda p: calls.append(p) or True)
    check(len(calls) >= 1, "the callback fired before the solve stopped", str(len(calls)))
    check(result.status == "interrupted", "status is interrupted", result.status)
    # kInterrupted claims a point (model.hpp's claims_a_point): the incumbent, or the last
    # feasible iterate, comes back sized like a normal answer rather than an empty vector.
    check(len(result.x) == model.num_cols, "the interrupted result still carries a point",
          str(result.x))


def test_callback_exception_propagates() -> None:
    # A bug in the CALLER's callback is not a KeyboardInterrupt and must not be swallowed
    # the way that is - it has to reach the caller as the exception it actually was.
    #
    # presolve=False: x >= 1 (minimise x) is small enough that presolve alone could fix the
    # column and hand the simplex an empty 0x0 reduced model, which solves without ever
    # running an iteration - and so without ever calling the callback this test is about.
    # Forcing the raw model through the simplex loop is what makes this deterministic.
    model = sankhya.Model()
    x = model.add_column(cost=1.0, name="x")
    model.add_row({x: 1.0}, lower=1.0)

    def bad_callback(_progress: sankhya.Progress) -> bool:
        raise ValueError("boom")

    raised = False
    try:
        model.solve(log_to_console=False, presolve=False, callback=bad_callback)
    except ValueError as error:
        raised = str(error) == "boom"
    check(raised, "an exception raised by the callback propagates out of solve()")


def test_interrupt_from_another_thread() -> None:
    # model.interrupt() (#223) is meant to be called from a thread OTHER than the one
    # running solve() - the whole point, since the solving thread is by definition busy.
    # The callback signals (via a threading.Event, set on its first call, before it sleeps)
    # that the solve is under way; the main thread waits for that and then interrupts. The
    # deliberate sleep inside the callback - not a race against how fast the machine solves
    # the model - is what keeps this deterministic: interrupt() lands somewhere in the
    # first callback's sleep or the next poll() right after, never after the solve has
    # already finished on its own.
    model = sankhya.Model(maximize=True)
    x = model.add_column(cost=1.0, upper=1.0, integer=True, name="x")
    y = model.add_column(cost=1.0, upper=1.0, integer=True, name="y")
    model.add_row({x: 2.0, y: 2.0}, upper=3.0)

    started = threading.Event()
    result_box: list[sankhya.Result] = []

    def slow_callback(_progress: sankhya.Progress) -> bool:
        started.set()
        threading.Event().wait(0.2)  # give the main thread time to call interrupt()
        return False

    def run() -> None:
        result_box.append(model.solve(log_to_console=False, callback=slow_callback))

    thread = threading.Thread(target=run)
    thread.start()
    check(started.wait(timeout=5.0), "the solving thread reported it had started")
    model.interrupt()
    thread.join(timeout=10.0)
    check(not thread.is_alive(), "the solving thread finished")
    check(len(result_box) == 1, "the solve returned a result")
    if result_box:
        check(result_box[0].status == "interrupted", "status is interrupted",
              result_box[0].status)


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
