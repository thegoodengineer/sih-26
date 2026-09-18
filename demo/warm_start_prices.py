#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The demo's next scene (#218): one blending model, five crude prices, five plans.

A planner does not solve once. They solve, move a price, and solve again. Each re-solve
here starts from the previous answer's basis, so the simplex resumes with a handful of
pivots instead of from scratch; the pivot count is printed beside each plan because it is
the number that proves the restart, where a wall-clock figure proves nothing.

Run from the repository root with the library built:

    PYTHONPATH=bindings/python python demo/warm_start_prices.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))
import sankhya  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "demo" / "crude_blend.mps"


def main() -> int:
    model = sankhya.Model.read(str(MODEL))
    crude = 0  # the first crude's cost coefficient is the price we move
    prices = [5.0, 9.0, 14.0, 2.5, 0.5]
    previous = None
    print(f"{'price':>8s} {'objective':>14s} {'pivots':>7s}  route")
    for price in prices:
        model.set_cost(crude, price)
        # A cost edit keeps the old basis primal feasible, so the re-solve asks for the
        # primal simplex and starts from the previous answer.
        result = (model.solve(log_to_console=False) if previous is None else
                  model.solve(log_to_console=False, start=previous, algorithm="simplex"))
        route = "cold" if previous is None else "warm, from the previous basis"
        print(f"{price:8.2f} {result.objective:14.4f} {result.iterations:7d}  {route}")
        if result.status != "optimal":
            print(f"  status {result.status}: {result.message}")
            return 1
        previous = result
    return 0


if __name__ == "__main__":
    sys.exit(main())
