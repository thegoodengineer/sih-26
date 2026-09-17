#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared low-level helpers for verify_solution.py's independent readers (issue #262).

Kept separate, tiny, and dependency-free: verify_solution_mps.py and verify_solution_sol.py
both need `open_text`, and only the MPS side additionally needs `normalize_infinity` - a
third module for the two lines they share is cheaper than either importing the other.
"""
from __future__ import annotations

import gzip
import math
from pathlib import Path

INF = math.inf
MPS_INFINITY = 1e30


def normalize_infinity(value: float) -> float:
    """MPS uses 1e30 for infinity. Normalise so only true infinities reach the checks."""
    if value >= MPS_INFINITY:
        return INF
    if value <= -MPS_INFINITY:
        return -INF
    return value


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")
