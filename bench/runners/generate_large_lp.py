#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate a large, sparse LP with a KNOWN analytic optimum, as MPS.

Issue #18: the committed Netlib instances top out around 500 rows, which says nothing about
"thousands to millions of variables" - and generating a LARGE instance at random says nothing
either, because nobody knows what the right answer is. The fix used everywhere else in this
project (tests/oracles/lp_generator.cpp::kkt_lp) is to build the instance BACKWARDS from a
chosen primal-dual pair that already satisfies the KKT conditions, so the optimal objective is
known before the solver ever sees the file. This script is the same construction, at a size
kkt_lp is deliberately never run at (it fills an O(rows*cols) dense array; this is sparse from
the start, one column at a time).

Construction, for  min c^T x  s.t.  A x >= b,  x >= 0,  0 <= x <= upper:

    x*_j  the chosen primal point, >= 0, some coordinates forced to 0
    y*_i  the chosen dual point, >= 0, nonzero only on rows made ACTIVE (tight at x*)
    d_j   >= 0, forced to 0 wherever x*_j > 0            (primal complementary slackness)
    c_j   = d_j + sum_i A_ij * y*_i                       (dual feasibility, by construction)
    b_i   = A_i x*        on an active row (tight)
          < A_i x*        on an inactive row (y*_i = 0, so its slack is consistent)
    upper_j = x*_j + a small nonnegative slack             (x* stays feasible for the bound)

Every coefficient is an integer, so the analytic optimum c^T x* is exact - no floating-point
rounding to account for when comparing it against what the solver reports.

Sparsity is controlled per COLUMN (--nnz-per-col), sampling that many distinct rows per
column rather than a row x column density, so generation and memory stay O(cols * nnz_per_col)
at any size instead of O(rows * cols).

Usage:
    python bench/runners/generate_large_lp.py --rows 1000 --cols 1000 --nnz-per-col 5 \
        --seed 42 --out /tmp/large_1k.mps
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

COEFFICIENT_RANGE = 9  # A, y*, d entries drawn from [1, this], never 0
PRIMAL_RANGE = 9  # x*_j drawn from [0, this]
BOUND_SLACK = 3  # upper_j = x*_j + [0, this]
ACTIVE_PROBABILITY = 0.5  # fraction of rows made tight at x*


def build(rows: int, cols: int, nnz_per_col: int, seed: int, structure: str = "random",
          periods: int = 0) -> dict:
    """Build the instance. Returns MPS line list plus the numbers the caller needs to report
    and verify: the analytic optimum, and the actual (row, col) nonzero count achieved.

    WHERE THE NONZEROS GO IS THE ONLY CHOICE THIS CONSTRUCTION LEAVES OPEN, and it decides
    what the instance can and cannot say (#198).

    `random` places each column's entries in rows drawn uniformly. That is the worst possible
    shape for any method that factorizes: a random sparse graph is an expander, with no small
    separators, so every elimination ordering fills catastrophically. Measured on the
    interior point: 17 percent of n^2 nonzeros in L from a matrix with five per column, at
    two sizes, whatever the ordering. It is a fair stress test of a first-order method and an
    unfair one of a direct method, and it looks nothing like a refinery.

    `staircase` places each column's entries in its own period's rows plus one in the next
    period's, which is the shape of a multi-period planning model: today's production meets
    today's balance and carries stock into tomorrow's. That is the structure PS26119's own
    domain produces - refinery scheduling, production planning, unit commitment - and it is
    the structure a direct method can exploit, because the graph is a band of width one
    period. Same construction, same exact optimum, different sparsity pattern; the
    comparison between the two families is the point.
    """
    rng = random.Random(seed)
    nnz_per_col = min(nnz_per_col, rows)  # cannot sample more distinct rows than exist
    if structure not in ("random", "staircase"):
        raise ValueError(f"unknown structure {structure!r}")
    if structure == "staircase":
        periods = periods or max(2, min(rows, cols) // 200)
        periods = max(2, min(periods, rows, cols))
    rows_per_period = rows // periods if structure == "staircase" else rows
    cols_per_period = cols // periods if structure == "staircase" else cols

    def rows_for_column(j: int) -> list[int]:
        if structure == "random":
            return rng.sample(range(rows), nnz_per_col)
        # The column belongs to period t. Most of its entries sit in period t's rows; one
        # couples forward into period t+1, and the last period couples to itself.
        t = min(j // cols_per_period, periods - 1)
        own_start = t * rows_per_period
        own_end = rows if t == periods - 1 else (t + 1) * rows_per_period
        next_start = own_end if t < periods - 1 else own_start
        next_end = rows if t >= periods - 2 else (t + 2) * rows_per_period
        own = list(range(own_start, own_end))
        forward = list(range(next_start, next_end))
        want_own = min(max(nnz_per_col - 1, 1), len(own))
        chosen = rng.sample(own, want_own)
        if nnz_per_col > want_own and forward:
            extra = rng.sample([r for r in forward if r not in chosen],
                               min(nnz_per_col - want_own, len(forward)))
            chosen.extend(extra)
        return chosen

    # ---- the sparse matrix, one column at a time -----------------------------------------
    # entries[j] = list of (row, value) for column j.
    entries: list[list[tuple[int, int]]] = []
    row_has_entry = [False] * rows
    for j in range(cols):
        chosen_rows = rows_for_column(j)
        col_entries = []
        for i in chosen_rows:
            value = rng.randint(1, COEFFICIENT_RANGE)
            if rng.random() < 0.5:
                value = -value
            col_entries.append((i, value))
            row_has_entry[i] = True
        entries.append(col_entries)
    nonzeros = sum(len(col) for col in entries)

    # ---- the chosen primal point x*, some coordinates forced to zero ---------------------
    x_star = [rng.randint(0, PRIMAL_RANGE) if rng.random() < 0.6 else 0 for _ in range(cols)]

    # ---- the chosen dual point y*, nonzero only on rows made active (tight at x*) --------
    active = [rng.random() < ACTIVE_PROBABILITY for _ in range(rows)]
    y_star = [rng.randint(0, COEFFICIENT_RANGE) if active[i] else 0 for i in range(rows)]

    # ---- row activities A_i x*, needed for both b and the RHS below ----------------------
    row_activity = [0] * rows
    for j in range(cols):
        xj = x_star[j]
        if xj == 0:
            continue
        for i, value in entries[j]:
            row_activity[i] += value * xj

    # ---- d >= 0, zero wherever x*_j > 0: primal complementary slackness ------------------
    d = [0 if x_star[j] > 0 else rng.randint(0, COEFFICIENT_RANGE) for j in range(cols)]

    # ---- c = A^T y* + d: dual feasibility by construction ---------------------------------
    cost = list(d)
    for j in range(cols):
        acc = cost[j]
        for i, value in entries[j]:
            if y_star[i] != 0:
                acc += value * y_star[i]
        cost[j] = acc

    # ---- b: tight on active rows, strictly slack (Ax >= b) on inactive ones --------------
    rhs = [0] * rows
    for i in range(rows):
        rhs[i] = row_activity[i] if active[i] else row_activity[i] - 1 - rng.randint(0, 3)

    # ---- upper bounds, chosen so x* is never cut off --------------------------------------
    upper = [x_star[j] + rng.randint(0, BOUND_SLACK) for j in range(cols)]

    optimal_objective = sum(cost[j] * x_star[j] for j in range(cols))

    lines = [
        f"* SANKHYA synthetic large sparse LP - issue #18",
        f"* rows={rows} cols={cols} nnz_per_col={nnz_per_col} seed={seed}",
        f"* Built backwards from a known KKT-satisfying primal-dual pair (see this file's",
        f"* generate_large_lp.py docstring, same construction as tests/oracles/lp_generator",
        f"* .cpp's kkt_lp). Reproduce with the seed above; analytic optimum below is exact.",
        f"* analytic optimum = {optimal_objective}",
        # Only named when it is not the default, so the random family stays byte-identical to
        # every instance the committed CSVs record a sha256 for.
        *([f"* structure = staircase, {periods} periods"] if structure == "staircase" else []),
        "NAME          LARGELP",
        "ROWS",
        " N  COST",
    ]
    for i in range(rows):
        lines.append(f" G  R{i}")
    lines.append("COLUMNS")
    for j in range(cols):
        parts = []
        if cost[j] != 0:
            parts.append(("COST", cost[j]))
        for i, value in entries[j]:
            parts.append((f"R{i}", value))
        # A column with a zero cost and no nonzero rows would emit nothing at all, which is
        # legal MPS (an implicit zero column) but pointless for a stress test; give it a
        # zero-cost entry so it still appears in the file and the column count is exact.
        if not parts:
            parts.append(("COST", 0))
        name = f"X{j}"
        for k in range(0, len(parts), 2):
            pair = parts[k:k + 2]
            fields = "   ".join(f"{row:<10}{value:>12d}" for row, value in pair)
            lines.append(f"    {name:<10}{fields}")
    lines.append("RHS")
    for i in range(rows):
        lines.append(f"    RHS       R{i:<9}{rhs[i]:>12d}")
    lines.append("BOUNDS")
    for j in range(cols):
        lines.append(f" UP BND       X{j:<9}{upper[j]:>12d}")
    lines.append("ENDATA")

    return {
        "lines": lines,
        "rows": rows,
        "cols": cols,
        "nonzeros": nonzeros,
        "optimal_objective": optimal_objective,
        "seed": seed,
        "structure": structure,
        "periods": periods if structure == "staircase" else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--nnz-per-col", type=int, required=True,
                        help="distinct rows sampled per column")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--structure", choices=("random", "staircase"), default="random",
                        help="where the nonzeros go. random: rows drawn uniformly, an expander "
                             "graph, the worst case for a direct method. staircase: each "
                             "column in its own period plus one coupling into the next, the "
                             "shape of a multi-period planning model (#198)")
    parser.add_argument("--periods", type=int, default=0,
                        help="periods for --structure staircase; default about one per 200 "
                             "rows")
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.nnz_per_col <= 0:
        parser.error("--nnz-per-col must be positive")

    result = build(args.rows, args.cols, args.nnz_per_col, args.seed, args.structure,
                   args.periods)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(result["lines"]) + "\n", encoding="utf-8", newline="\n")

    density = 100.0 * result["nonzeros"] / (args.rows * args.cols)
    print(f"wrote {args.out}")
    print(f"  {args.rows} rows, {args.cols} columns, {result['nonzeros']} nonzeros "
          f"({density:.4f}% dense), seed {args.seed}")
    print(f"  analytic optimum: {result['optimal_objective']}")
    if result["structure"] == "staircase":
        print(f"  structure: staircase, {result['periods']} periods")
    return 0


if __name__ == "__main__":
    sys.exit(main())
