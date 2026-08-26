#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the Netlib LP test set and its PUBLISHED optimal objective values.

Two things this script is careful about, both for the same reason: a benchmark table is
only evidence if a judge can regenerate it.

1.  The reference optima are PARSED from Netlib's own ``readme`` (the PROBLEM SUMMARY
    TABLE), never typed in from memory. If the published value and our value disagree, the
    disagreement has to be about our solver, not about somebody's recollection.

2.  Netlib distributes these instances in a custom compressed encoding, not as MPS. The
    canonical decoder is ``emps.c``, published by Netlib alongside the data. We download and
    compile it at fetch time rather than vendoring it, so this repository contains no
    third-party source; the sha256 of everything downloaded is recorded in the manifest.

``emps.c`` is a file-format converter, not a solver, so it is outside the CLAUDE.md red
line. Nothing it produces is linked into SANKHYA; it runs once, offline, to turn Netlib's
archive format into plain MPS.

Usage:
    python bench/runners/fetch_data.py                # the default small set
    python bench/runners/fetch_data.py afiro sc50a    # named instances
    python bench/runners/fetch_data.py --all          # everything in the summary table
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

NETLIB_BASE = "https://netlib.org/lp/data"
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"

# The Phase 2/3 working set: small, well conditioned, and every one of them has a published
# optimum. blend is a petroleum blending model, which is why it earns its place in a demo
# for refinery judges.
# Named instance sets. `small` is what CI runs; the other two are for manual evidence runs.
#
# The point of naming these is that "8 of 8" reads as full coverage when it is 9% of the set,
# and the largest instance carried by `small` is 118 rows - nothing that could exercise the
# degeneracy or ill-conditioning the problem statement asks about. `medium` is the first tier
# with instances big enough for the basis factorization to matter, and `full` is the number
# Phase 6's ">= 95% of Netlib" exit criterion is actually measured against.
#
# `medium` is defined by the PUBLISHED row count rather than by a hand-written list, so it
# does not silently drift as instances are added, and so nobody has to curate it.
MEDIUM_MAX_ROWS = 500

# Counts as of the current Netlib readme: small 8, medium 50, full 89.
SET_NAMES = ("small", "medium", "full")

DEFAULT_SET = [
    "afiro",
    "sc50a",
    "sc50b",
    "sc105",
    "adlittle",
    "share2b",
    "blend",
    "stocfor1",
]

# Name Rows Cols Nonzeros Bytes [BR flags] Optimal
# Name Rows Cols Nonzeros Bytes [BR flags] Optimal [footnote]
# The trailing footnote group is not decoration: DFL001 is published as "1.12664E+07 **",
# an APPROXIMATE optimum. A harness that silently drops the marker would later report a
# relative gap against a number Netlib itself does not claim to be exact.
SUMMARY_ROW = re.compile(
    r"^([A-Z0-9_\-]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([BR ]*?)\s*"
    r"(-?\d+\.\d+E[+-]\d+)\s*(\**)\s*$"
)


def download(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_summary_table(readme: str) -> dict[str, dict]:
    """Extract the published PROBLEM SUMMARY TABLE.

    Scans the whole file rather than trying to bracket the table. An earlier version stopped
    at the first left-margin line that failed to match, which quietly truncated the table at
    DFL001 - the one row carrying a footnote marker - and lost every instance after D.
    """
    entries: dict[str, dict] = {}
    for line in readme.splitlines():
        match = SUMMARY_ROW.match(line.strip())
        if not match:
            continue
        name, rows, cols, nonzeros, size, flags, optimal, footnote = match.groups()
        entries[name.lower()] = {
            "name": name.lower(),
            "published_rows": int(rows),
            "published_cols": int(cols),
            "published_nonzeros": int(nonzeros),
            "published_bytes": int(size),
            "has_bounds": "B" in flags,
            "has_ranges": "R" in flags,
            "published_optimal": float(optimal),
            # Netlib footnotes an optimum it does not claim to full precision.
            "optimal_is_approximate": bool(footnote),
        }
    return entries


def find_c_compiler() -> list[str] | None:
    """A C compiler for emps.c. Prefers a modern one over whatever PATH offers first."""
    candidates = [
        "C:/Strawberry/c/bin/gcc.exe",
        shutil.which("cc"),
        shutil.which("gcc"),
        shutil.which("clang"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return [candidate]
    return None


def build_emps(work_dir: Path) -> tuple[Path, str]:
    """Download and compile Netlib's emps decoder. Returns (binary path, source sha256)."""
    source_bytes = download(f"{NETLIB_BASE}/emps.c")
    digest = sha256(source_bytes)
    source = work_dir / "emps.c"
    source.write_bytes(source_bytes)

    compiler = find_c_compiler()
    if compiler is None:
        raise SystemExit(
            "no C compiler found to build Netlib's emps decoder; install gcc or clang"
        )

    binary = work_dir / ("emps.exe" if sys.platform == "win32" else "emps")
    # emps.c is 1990s K&R-flavoured C; modern compilers need to be told not to reject it.
    result = subprocess.run(
        compiler + ["-w", "-std=gnu89", "-O1", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"failed to compile emps.c:\n{result.stderr}")
    return binary, digest


def decompress(emps: Path, packed: Path, destination: Path) -> None:
    """Expand `packed` into `destination`, atomically.

    The obvious version opens the destination and points emps at it. That TRUNCATES the
    target before emps has produced a byte, so any failure - emps erroring, a partial
    download upstream, the process being interrupted - leaves a 0-byte file behind.

    That is worse than it sounds now that data/netlib/ is tracked: the instance shows up as
    MODIFIED rather than missing, `git status` looks like an ordinary edit, and the next
    solve fails with "file ends without an ENDATA record" on a file git is perfectly happy
    with. It is also a `git add -A` away from committing an empty benchmark instance.

    Writing beside the target and renaming only on success means the destination is always
    either its previous content or the complete new content, never a truncated middle.
    """
    scratch = destination.with_name(destination.name + ".partial")
    try:
        with scratch.open("wb") as out:
            result = subprocess.run([str(emps), str(packed)], stdout=out,
                                    stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise SystemExit(f"emps failed on {packed.name}: "
                             f"{result.stderr.decode(errors='replace')}")

        # An expander that exits 0 having written nothing is still a failure. Checking here
        # reports it against the instance being fetched; letting it through moves the
        # complaint to a solve hours later, far from the cause.
        size = scratch.stat().st_size
        if size == 0:
            raise SystemExit(f"emps produced an empty file for {packed.name}")
        with scratch.open("rb") as handle:
            handle.seek(max(0, size - 64))
            if b"ENDATA" not in handle.read():
                raise SystemExit(
                    f"emps output for {packed.name} has no ENDATA record; the expansion was "
                    f"truncated")

        os.replace(scratch, destination)
    finally:
        scratch.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances", nargs="*", help="instance names (default: a small set)")
    parser.add_argument("--set", dest="instance_set", choices=SET_NAMES, default=None,
                        help=f"named instance set: small (CI default), medium (published "
                             f"rows <= {MEDIUM_MAX_ROWS}), or full (everything listed)")
    parser.add_argument("--all", action="store_true",
                        help="deprecated alias for --set full")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"reading the published summary table from {NETLIB_BASE}/readme")
    readme_bytes = download(f"{NETLIB_BASE}/readme")
    published = parse_summary_table(readme_bytes.decode("latin-1"))
    print(f"  {len(published)} instances listed with a published optimal value")
    if not published:
        raise SystemExit("could not parse the summary table; the readme format may have changed")

    # Explicit names win, then --set, then the deprecated --all, then the small default.
    if args.instances:
        wanted = [name.lower() for name in args.instances]
        chosen_set = "explicit"
    else:
        chosen_set = args.instance_set or ("full" if args.all else "small")
        if chosen_set == "full":
            wanted = sorted(published)
        elif chosen_set == "medium":
            wanted = sorted(name for name, entry in published.items()
                            if entry["published_rows"] <= MEDIUM_MAX_ROWS)
        else:
            wanted = list(DEFAULT_SET)

    print(f"  set '{chosen_set}': {len(wanted)} of {len(published)} listed instances")

    unknown = [name for name in wanted if name not in published]
    if unknown:
        raise SystemExit(f"not in the Netlib summary table: {', '.join(unknown)}")

    manifest_path = DATA_DIR / "reference.json"
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("source", NETLIB_BASE)
    manifest.setdefault("instances", {})
    # The denominator, recorded so that make_benchmarks_doc.py can state coverage honestly
    # without re-fetching. Without it the generated table says "8 of 8", which reads as full
    # coverage of Netlib rather than of what was run.
    manifest["available_instances"] = len(published)
    manifest["instance_set"] = chosen_set

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print("building Netlib's emps decoder")
        emps, emps_digest = build_emps(work)
        manifest["emps_sha256"] = emps_digest
        manifest["readme_sha256"] = sha256(readme_bytes)
        print(f"  emps.c sha256 {emps_digest}")

        failures = []
        for name in wanted:
            entry = dict(published[name])
            try:
                packed_bytes = download(f"{NETLIB_BASE}/{name}")
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"  {name:<10} DOWNLOAD FAILED: {error}")
                failures.append(name)
                continue

            packed = work / name
            packed.write_bytes(packed_bytes)
            entry["packed_sha256"] = sha256(packed_bytes)

            target = DATA_DIR / f"{name}.mps"
            decompress(emps, packed, target)
            mps_bytes = target.read_bytes()
            entry["mps_sha256"] = sha256(mps_bytes)
            entry["mps_bytes"] = len(mps_bytes)
            manifest["instances"][name] = entry
            print(
                f"  {name:<10} {entry['published_rows']:>5} rows "
                f"{entry['published_cols']:>5} cols "
                f"{entry['published_nonzeros']:>7} nz   "
                f"optimal {entry['published_optimal']:>18.10E}"
            )

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {manifest_path.relative_to(REPO_ROOT)}")
    print(f"{len(wanted) - len(failures)} instance(s) in {DATA_DIR.relative_to(REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
