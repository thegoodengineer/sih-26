#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a sparse multi-period production LP of a chosen size, as MPS.

The Netlib instances SANKHYA currently carries top out around 120 rows, which is far too
small to show what the basis factorization costs: a dense refactorization of a 120x120 basis
is genuinely cheap. This generator produces the shape the problem statement actually cares
about - thousands of rows, a handful of nonzeros per column, and a banded structure that
looks like a real planning model rather than a random matrix.

Structure, for P periods and G goods:

    produce[g, t]   how much of good g is made in period t
    store[g, t]     how much is carried from period t into t + 1

    inventory balance   store[g, t-1] + produce[g, t] - store[g, t] = demand[g, t]
    shared capacity     sum_g produce[g, t] <= capacity[t]

That gives P * (G + 1) rows and P * 2G columns, three nonzeros in a typical column, and a
banded pattern with real coupling between periods. It is feasible by construction: demand is
drawn low enough that producing to order always fits inside the capacity row.

Usage:
    make_scaling_instance.py --periods 200 --goods 3 --out big.mps
"""

import argparse
import random
import sys


def build(periods: int, goods: int, seed: int) -> tuple[list[str], int, int, int]:
    rng = random.Random(seed)

    demand = {(g, t): rng.uniform(5.0, 20.0) for g in range(goods) for t in range(periods)}
    # Capacity always exceeds the largest possible period demand, so produce-to-order is
    # feasible and the instance is guaranteed solvable.
    capacity = [sum(demand[(g, t)] for g in range(goods)) * 1.6 + 25.0 for t in range(periods)]

    produce_cost = {(g, t): rng.uniform(1.0, 4.0) for g in range(goods) for t in range(periods)}
    store_cost = {(g, t): rng.uniform(0.1, 0.6) for g in range(goods) for t in range(periods)}

    def bal(g: int, t: int) -> str:
        return f"BAL{g}_{t}"

    def cap(t: int) -> str:
        return f"CAP{t}"

    lines = ["NAME          SCALING", "ROWS", " N  COST"]
    for t in range(periods):
        for g in range(goods):
            lines.append(f" E  {bal(g, t)}")
        lines.append(f" L  {cap(t)}")

    lines.append("COLUMNS")
    nonzeros = 0
    for t in range(periods):
        for g in range(goods):
            # produce[g, t]: objective, its own balance row, and the period capacity row
            name = f"P{g}_{t}"
            lines.append(f"    {name:<9} COST      {produce_cost[(g, t)]:>12.6f}   "
                         f"{bal(g, t):<9} {1.0:>12.6f}")
            lines.append(f"    {name:<9} {cap(t):<9} {1.0:>12.6f}")
            nonzeros += 3

            # store[g, t]: leaves this period's balance, enters the next one's
            name = f"S{g}_{t}"
            lines.append(f"    {name:<9} COST      {store_cost[(g, t)]:>12.6f}   "
                         f"{bal(g, t):<9} {-1.0:>12.6f}")
            nonzeros += 2
            if t + 1 < periods:
                lines.append(f"    {name:<9} {bal(g, t + 1):<9} {1.0:>12.6f}")
                nonzeros += 1

    lines.append("RHS")
    for t in range(periods):
        for g in range(goods):
            lines.append(f"    RHS       {bal(g, t):<9} {demand[(g, t)]:>12.6f}")
        lines.append(f"    RHS       {cap(t):<9} {capacity[t]:>12.6f}")

    lines.append("ENDATA")

    rows = periods * (goods + 1)
    cols = periods * 2 * goods
    return lines, rows, cols, nonzeros


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--periods", type=int, default=200)
    parser.add_argument("--goods", type=int, default=3)
    parser.add_argument("--seed", type=int, default=26119)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lines, rows, cols, nonzeros = build(args.periods, args.goods, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    density = 100.0 * nonzeros / (rows * cols) if rows and cols else 0.0
    print(f"{args.out}: {rows} rows, {cols} columns, {nonzeros} nonzeros "
          f"({density:.4f}% dense)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
