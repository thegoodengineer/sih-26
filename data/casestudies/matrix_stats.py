#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report the numerical character of an instance's constraint matrix.

Used by demo/run_sih_demo.sh to show that the degeneracy and ill-conditioning claims are
properties the instances measurably HAVE, rather than adjectives attached to them.

The matrix is read with tools/verify_solution.py's MPS reader - the independent one, which
shares no code with the C++ solver - so none of this can flatter us.

RANK IS COMPUTED EXACTLY, in Fraction arithmetic, not by thresholding singular values. That
matters for the claim being made: "the constraint matrix is rank deficient" is a statement
about the model's structure, and answering it in floating point would mean choosing a
tolerance below which a singular value counts as zero. Since the entries here are exact
decimals, Gaussian elimination over the rationals settles it with no tolerance at all. It
also means the degeneracy claim needs no third-party library.

The condition number does need an SVD, so that one line uses numpy when it is importable and
says so plainly when it is not.

    python data/casestudies/matrix_stats.py data/casestudies/supply_chain.mps
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from fractions import Fraction

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

from verify_solution import parse_mps  # noqa: E402


def dense(path: pathlib.Path) -> tuple[list[list[Fraction]], int, int]:
    """The constraint matrix as exact rationals."""
    model = parse_mps(path)
    a = [[Fraction(0) for _ in range(model.num_cols)] for _ in range(model.num_rows)]
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            # Fraction(str(float)) round-trips the decimal the file actually contains.
            a[i][j] = Fraction(str(value))
    return a, model.num_rows, model.num_cols


def exact_rank(a: list[list[Fraction]]) -> int:
    """Rank by Gaussian elimination over the rationals. No tolerance is involved."""
    m = [row[:] for row in a]
    rows, cols = len(m), len(m[0]) if m else 0
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if m[r][col] != 0), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        inverse = m[rank][col]
        for r in range(rows):
            if r != rank and m[r][col] != 0:
                factor = m[r][col] / inverse
                for c in range(col, cols):
                    m[r][c] -= factor * m[rank][c]
        rank += 1
        if rank == rows:
            break
    return rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=pathlib.Path)
    parser.add_argument("--rank", action="store_true", help="report rank and redundancy")
    parser.add_argument("--conditioning", action="store_true",
                        help="report entry spread and condition number")
    args = parser.parse_args()
    if not args.rank and not args.conditioning:
        args.rank = args.conditioning = True

    a, rows, cols = dense(args.instance)

    if args.rank:
        rank = exact_rank(a)
        redundant = rows - rank
        print("{} rows x {} columns, exact rank {}  ->  {} redundant row(s)".format(
            rows, cols, rank, redundant))
        if redundant:
            print("every basis is therefore degenerate: {} basic variable(s) pinned at zero"
                  .format(redundant))

    if args.conditioning:
        nonzero = [abs(float(v)) for row in a for v in row if v != 0]
        print("entries span {:.3e} to {:.3e}, a ratio of {:.3e}".format(
            min(nonzero), max(nonzero), max(nonzero) / min(nonzero)))
        try:
            import numpy as np
        except ImportError:
            print("2-norm condition number: needs an SVD; install numpy to see it")
        else:
            dense_float = np.array([[float(v) for v in row] for row in a])
            # A rank-deficient matrix has a zero singular value and an infinite 2-norm
            # condition number. numpy reports a very large finite value instead, limited by
            # rounding - that is not an error here, it is the degeneracy showing up again.
            print("2-norm condition number: {:.3e}".format(np.linalg.cond(dense_float)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
