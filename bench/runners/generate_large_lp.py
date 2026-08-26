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


def build(rows: int, cols: int, nnz_per_col: int, seed: int) -> dict:
    """Build the instance. Returns MPS line list plus the numbers the caller needs to report
    and verify: the analytic optimum, and the actual (row, col) nonzero count achieved."""
    rng = random.Random(seed)
    nnz_per_col = min(nnz_per_col, rows)  # cannot sample more distinct rows than exist

    # ---- the sparse matrix, one column at a time -----------------------------------------
    # entries[j] = list of (row, value) for column j.
    entries: list[list[tuple[int, int]]] = []
    row_has_entry = [False] * rows
    for _ in range(cols):
        chosen_rows = rng.sample(range(rows), nnz_per_col)
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
    args = parser.parse_args()

    if args.rows <= 0 or args.cols <= 0:
        parser.error("--rows and --cols must be positive")
    if args.nnz_per_col <= 0:
        parser.error("--nnz-per-col must be positive")

    result = build(args.rows, args.cols, args.nnz_per_col, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(result["lines"]) + "\n", encoding="utf-8", newline="\n")

    density = 100.0 * result["nonzeros"] / (args.rows * args.cols)
    print(f"wrote {args.out}")
    print(f"  {args.rows} rows, {args.cols} columns, {result['nonzeros']} nonzeros "
          f"({density:.4f}% dense), seed {args.seed}")
    print(f"  analytic optimum: {result['optimal_objective']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
