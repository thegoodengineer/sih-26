#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent oracle for data/casestudies/power_dispatch.mps.

verify_solution.py can confirm that a MILP answer is feasible, integral and that the bound
closed - but for a MILP it cannot confirm that no BETTER answer exists, because that is the
whole content of the search. A branch and bound that fathoms a node it should have explored
produces a point which passes every one of those checks and is still wrong.

The unit commitment instance is small enough to settle by exhaustion, so this settles it.
Four units means sixteen on/off patterns. For a FIXED pattern the remaining problem is

    min sum_g c_g p_g   s.t.   sum_g p_g = D,   Pmin_g <= p_g <= Pmax_g  (committed only)

which is a continuous problem over a single equality and a box, so the greedy merit order is
exactly optimal: hold every committed unit at its minimum, then buy the shortfall from the
cheapest unit with headroom left. No LP solver is involved and no SANKHYA code is imported,
so agreement between this and the solver is genuine corroboration rather than a tautology.

    python data/casestudies/dispatch_oracle.py
"""
from __future__ import annotations

import itertools
import sys

# Restated here on purpose. Importing them from generate.py would mean a typo in the shared
# constants produced a matching typo in the oracle, and the two would agree while both wrong.
UNITS = [
    # name, marginal cost $/MWh, Pmin, Pmax, start-up cost $
    ("GA", 10.0, 20.0, 100.0, 100.0),
    ("GB", 12.0, 30.0, 120.0, 80.0),
    ("GC", 20.0, 10.0, 150.0, 50.0),
    ("GD", 8.0, 50.0, 60.0, 450.0),
]
DEMAND = 250.0
RESERVE = 1.15 * DEMAND


def dispatch(pattern: tuple[int, ...]) -> tuple[float, dict[str, float]] | None:
    """Cheapest dispatch for one commitment pattern, or None if it cannot serve the load."""
    on = [(name, cost, pmin, pmax, start)
          for (name, cost, pmin, pmax, start), u in zip(UNITS, pattern) if u]
    if sum(pmax for _, _, _, pmax, _ in on) < DEMAND - 1e-9:
        return None  # not enough capacity
    if sum(pmin for _, _, pmin, _, _ in on) > DEMAND + 1e-9:
        return None  # minimum stable generation already overshoots the load
    if sum(pmax for _, _, _, pmax, _ in on) < RESERVE - 1e-9:
        return None  # fails the spinning reserve margin

    output = {name: pmin for name, _, pmin, _, _ in on}
    shortfall = DEMAND - sum(output.values())
    for name, _cost, pmin, pmax, _start in sorted(on, key=lambda u: u[1]):
        if shortfall <= 1e-12:
            break
        take = min(shortfall, pmax - pmin)
        output[name] += take
        shortfall -= take

    cost = sum(c * output[name] for name, c, _, _, _ in on)
    cost += sum(start for _, _, _, _, start in on)
    return cost, output


def main() -> int:
    best_cost = float("inf")
    best = None
    considered = 0
    for pattern in itertools.product((0, 1), repeat=len(UNITS)):
        result = dispatch(pattern)
        if result is None:
            continue
        considered += 1
        cost, output = result
        if cost < best_cost - 1e-9:
            best_cost, best = cost, (pattern, output)

    if best is None:
        print("oracle: no commitment pattern can serve the load")
        return 1

    pattern, output = best
    committed = [name for (name, *_), u in zip(UNITS, pattern) if u]
    print("Exhaustive oracle over all {} commitment patterns "
          "({} of them can serve the load):".format(2 ** len(UNITS), considered))
    print("    committed        {}".format(" ".join(committed)))
    for name, *_ in UNITS:
        print("    {:<16} {:8.2f} MW".format(name, output.get(name, 0.0)))
    print("    total cost       {:.6f}".format(best_cost))
    return 0


if __name__ == "__main__":
    sys.exit(main())
