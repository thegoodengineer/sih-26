#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every repository path docs/PROVENANCE.md names must exist.

The algorithm table in section 3 is the document a judge opens to check that every method
is cited and to find where it lives. A row that points at a file that has moved is worse
than no row: it reads as a citation nobody checked. Two rows pointed at src/pdhg/scaling.cpp
for a week after the file moved to src/la/, which is how this check came to exist.

    python tools/check_provenance_paths.py            # exit 1 and list every missing path

Only paths under the repository's own top-level directories are checked; URLs, package
names and the toolchain paths in section 2b are not repository paths.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = REPO_ROOT / "docs" / "PROVENANCE.md"
TOP_LEVEL = ("src", "include", "tests", "tools", "bench", "apps", "bindings", "scripts",
             "docs", "demo", "data")


def named_paths(text: str) -> list[str]:
    pattern = re.compile(r"`((?:" + "|".join(TOP_LEVEL) + r")/[^`\s]+)`")
    found = set()
    for match in pattern.finditer(text):
        path = match.group(1).rstrip(".,;:")
        # `bench/results/netlib-{full,medium}-abc.csv` names a family, not one file.
        if "{" in path or "*" in path:
            continue
        found.add(path)
    return sorted(found)


def main() -> int:
    text = DOCUMENT.read_text(encoding="utf-8")
    paths = named_paths(text)
    missing = [p for p in paths if not (REPO_ROOT / p).exists()]
    for p in missing:
        print(f"MISSING: {p}")
    print(f"{len(paths) - len(missing)} of {len(paths)} paths named in docs/PROVENANCE.md exist")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
