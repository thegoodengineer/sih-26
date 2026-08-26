#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Report the numerical character of an instance's constraint matrix.

Used by demo/run_sih_demo.sh to show that the degeneracy and ill-conditioning claims are
properties the instances measurably HAVE, rather than adjectives attached to them.

The matrix is read with tools/verify_solution.py's MPS reader - the independent one, which
shares no code with the C++ solver - and the rank and condition number come from numpy's
SVD. Neither is any part of SANKHYA, so this cannot flatter us.

    python data/casestudies/matrix_stats.py data/casestudies/supply_chain.mps
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))

import numpy as np  # noqa: E402

from verify_solution import parse_mps  # noqa: E402


def dense(path: pathlib.Path) -> np.ndarray:
    model = parse_mps(path)
    a = np.zeros((model.num_rows, model.num_cols))
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            a[i, j] = value
    return a


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", type=pathlib.Path)
    parser.add_argument("--rank", action="store_true", help="report rank and redundancy")
    parser.add_argument("--conditioning", action="store_true",
                        help="report entry spread and condition number")
    args = parser.parse_args()

    a = dense(args.instance)
    if not args.rank and not args.conditioning:
        args.rank = args.conditioning = True

    if args.rank:
        rank = int(np.linalg.matrix_rank(a))
        redundant = a.shape[0] - rank
        print("{} rows x {} columns, rank {}  ->  {} redundant row(s)".format(
            a.shape[0], a.shape[1], rank, redundant))
        if redundant:
            print("every basis is therefore degenerate: {} basic variable(s) pinned at zero"
                  .format(redundant))

    if args.conditioning:
        nonzero = np.abs(a[a != 0.0])
        print("entries span {:.3e} to {:.3e}, a ratio of {:.3e}".format(
            nonzero.min(), nonzero.max(), nonzero.max() / nonzero.min()))
        # Condition number of the full rectangular matrix. A rank-deficient matrix has a zero
        # singular value and an infinite 2-norm condition number, which numpy reports as inf -
        # that is not an error here, it is the degeneracy showing up in a second way.
        print("2-norm condition number: {:.3e}".format(np.linalg.cond(a)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
