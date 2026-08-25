#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the demo solve against a hand-verified reference value.

This is a REGRESSION GUARD, not a verifier. It knows one answer for one instance and
notices when that answer moves. The real independent checker is tools/verify_solution.py
(Phase 3), which re-parses the model and recomputes everything without trusting the solver.

The reference below was not taken from any solver, ours or anyone else's. It was derived by
hand from demo/crude_blend.mps and checked three ways: every constraint is satisfied, the
reduced costs equal c - A'y recomputed from the reported duals, and the dual objective
rebuilt from the active bounds alone closes against the primal objective to 2.8e-14. See
docs/PROVENANCE.md for the working.

Usage: check_demo_result.py <stats.json>
"""

import json
import sys

# demo/crude_blend.mps, maximizing margin over three crudes.
EXPECTED_OBJECTIVE = 214.14594594594595

# Tighter than the solver's own feasibility tolerance on purpose. The point of a regression
# guard is to fire on a change far too small to matter numerically but large enough to mean
# something in the algorithm moved.
OBJECTIVE_TOLERANCE = 1e-9

# These are measured by Solution::recompute_quality from the primal and dual vectors, not
# asserted by the engine about itself, so they are evidence rather than opinion.
QUALITY_TOLERANCE = 1e-7


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    with open(argv[1], encoding="utf-8") as handle:
        blob = json.load(handle)

    status = blob["result"]["status"]
    objective = blob["result"]["objective"]
    primal = blob["quality"]["primal_infeasibility"]
    dual = blob["quality"]["dual_infeasibility"]
    iterations = blob["effort"]["iterations"]

    print(
        f"status={status} objective={objective!r} "
        f"primal_inf={primal} dual_inf={dual} iterations={iterations}"
    )

    problems = []
    if status != "optimal":
        problems.append(f"expected status 'optimal', got {status!r}")
    if abs(objective - EXPECTED_OBJECTIVE) > OBJECTIVE_TOLERANCE:
        problems.append(
            f"objective drifted: got {objective!r}, "
            f"expected {EXPECTED_OBJECTIVE!r}, "
            f"difference {abs(objective - EXPECTED_OBJECTIVE):.3e}"
        )
    if primal > QUALITY_TOLERANCE:
        problems.append(f"primal infeasibility {primal:.3e} exceeds {QUALITY_TOLERANCE:.0e}")
    if dual > QUALITY_TOLERANCE:
        problems.append(f"dual infeasibility {dual:.3e} exceeds {QUALITY_TOLERANCE:.0e}")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    print("demo result matches the hand-verified reference")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
