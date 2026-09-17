# SPDX-License-Identifier: Apache-2.0
"""Locating the SANKHYA CLI executable.

This mirrors _library.py but discovers the standalone binary rather than the shared
library, ensuring python tooling and benchmark runners use the exact same discovery rules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ._library import SankhyaError

__all__ = ["locate"]


def _repository_root() -> Path:
    # bindings/python/sankhya/_executable.py -> up three.
    return Path(__file__).resolve().parents[3]


def locate() -> Path:
    """Find the SANKHYA executable.

    Checks SANKHYA_EXECUTABLE, then SANKHYA_BIN, then searches standard build directories.
    Raises SankhyaError if no executable is found or if an explicit override is invalid.
    """
    override = os.environ.get("SANKHYA_EXECUTABLE") or os.environ.get("SANKHYA_BIN")
    if override:
        path = Path(override)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        # An explicit path wins outright and is not combined with the search.
        # Quietly falling back to a different one would be worse than failing.
        raise SankhyaError(
            f"The SANKHYA executable was not found at the explicitly configured path:\n"
            f"  {path}\n\n"
            f"Check the SANKHYA_EXECUTABLE or SANKHYA_BIN environment variables."
        )

    exe_name = "sankhya.exe" if sys.platform == "win32" else "sankhya"

    # Matches _library.py
    search_directories = (
        "build",
        "build-cuda",
        "build-release",
        "build-fresh",
        "build-dbg",
        "build-gate",
        "build-main",
        "cmake-build-release",
        ".",
    )

    root = _repository_root()
    tried: list[str] = []

    for directory in search_directories:
        candidate = root / directory / exe_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        tried.append(f"{candidate}")

    raise SankhyaError(
        f"The SANKHYA executable ({exe_name}) was not found.\n\n"
        "Build it first, for example with:\n"
        "    scripts/configure.sh build Release && cmake --build build -j\n\n"
        "Or set the SANKHYA_EXECUTABLE environment variable to its full path.\n\n"
        "Locations searched:\n  " + "\n  ".join(tried)
    )
