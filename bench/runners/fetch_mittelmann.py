#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch Mittelmann's LP benchmark instances (#60), with recorded provenance.

PS26119 names "MIPLIB, Netlib or Mittelmann benchmark sets". Netlib and MIPLIB have runners;
this is the third. The instances come from Hans Mittelmann's LP test set,

    https://plato.asu.edu/ftp/lptestset/          (the archive this script reads)
    https://plato.asu.edu/ftp/lpfeas.html          (the benchmark page and its time limits)

as bzip2-compressed MPS files. They are LARGE - the benchmark page runs them with a 15,000 s
limit and its smallest members are still an order of magnitude beyond Netlib's largest - so
this script fetches a NAMED SUBSET, `small`, of the smallest archives by size, which is the
honest place to start: a runner that reports failures by name is worth more than one that
never ran. `--set full` fetches every top-level archive in the directory.

Provenance recorded per instance in data/mittelmann/reference.json: the archive URL, the
sha256 of the archive and of the decompressed MPS, and both sizes. No published optimum is
recorded, because the benchmark page publishes solver TIMES, not objective values; the
runner therefore reports the verifier's verdict and cross-checks against HiGHS when it is
available, rather than a pass against a number nobody printed.

    python bench/runners/fetch_mittelmann.py              # the small set
    python bench/runners/fetch_mittelmann.py --set full   # everything at the top level
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "mittelmann"
BASE_URL = "https://plato.asu.edu/ftp/lptestset/"

# The smallest archives in the top-level directory, by compressed size as listed on
# 2026-09-06, smallest first. Compressed size is a rough proxy for instance size; it is the
# only figure the listing gives.
SMALL_SET = [
    "qap15.mps.bz2",
    "brazil3.mps.bz2",
    "irish-electricity.mps.bz2",
    "chromaticindex1024-7.mps.bz2",
    "Linf_520c.bz2",
    "supportcase10.mps.bz2",
    "bdry2.bz2",
    "rmine15.mps.bz2",
]

FULL_SET = SMALL_SET + [
    "physiciansched3-3.mps.bz2", "ex10.mps.bz2", "datt256_lp.mps.bz2", "s250r10.mps.bz2",
    "graph40-40.mps.bz2", "savsched1.mps.bz2", "s100.mps.bz2", "neos-5251015.mps.bz2",
    "woodlands09.mps.bz2", "neos-5052403-cygnet.mps.bz2", "L1_sixm250obs.bz2",
    "neos-3025225.mps.bz2", "bharat.mps.bz2", "supportcase19.mps.bz2", "square41.mps.bz2",
    "Primal2_1000.mps.bz2", "scpm1.mps.bz2", "s82.mps.bz2", "tpl-tub-ws1617.mps.bz2",
    "fhnw-binschedule1.mps.bz2", "L1_sixm1000obs.bz2", "dlr1.mps.bz2", "a2864.mps.bz2",
    "set-cover-model.mps.bz2", "thk_63.mps.bz2", "thk_48.mps.bz2", "L2CTA3D.mps.bz2",
    "dlr2.mps.bz2", "Dual2_5000.mps.bz2",
]


def instance_name(archive: str) -> str:
    name = archive
    for suffix in (".mps.bz2", ".bz2"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, attempts: int = 4) -> None:
    """Fetch with retries; the archive is written to a temp name and renamed at the end so a
    half-written file is never mistaken for a complete one."""
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
            partial.replace(target)
            return
        except (urllib.error.URLError, OSError) as error:
            if attempt == attempts:
                raise
            wait = 2 ** (attempt - 1)
            print(f"  {url}: {error}; retrying in {wait}s ({attempt + 1} of {attempts})")
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--set", dest="instance_set", choices=("small", "full"), default="small")
    parser.add_argument("--force", action="store_true", help="re-download present archives")
    args = parser.parse_args()

    archives = SMALL_SET if args.instance_set == "small" else FULL_SET
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATA_DIR / "reference.json"
    manifest = {"source": BASE_URL, "benchmark_page": "https://plato.asu.edu/ftp/lpfeas.html",
                "instance_set": args.instance_set, "instances": {}}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
            manifest["instances"] = previous.get("instances", {})
        except (OSError, ValueError):
            pass

    for archive in archives:
        name = instance_name(archive)
        archive_path = DATA_DIR / archive
        mps_path = DATA_DIR / f"{name}.mps"
        url = BASE_URL + archive
        if args.force or not archive_path.exists():
            print(f"fetching {url}")
            download(url, archive_path)
        if args.force or not mps_path.exists():
            print(f"  decompressing {archive}")
            with bz2.open(archive_path, "rb") as src, mps_path.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
        manifest["instances"][name] = {
            "name": name,
            "archive": archive,
            "url": url,
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_of(archive_path),
            "mps_bytes": mps_path.stat().st_size,
            "mps_sha256": sha256_of(mps_path),
            "in_set": args.instance_set,
        }
        print(f"  {name}: {archive_path.stat().st_size / 1e6:.1f} MB compressed, "
              f"{mps_path.stat().st_size / 1e6:.1f} MB MPS")

    manifest["instances"] = dict(sorted(manifest["instances"].items()))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"wrote {manifest_path} ({len(archives)} instances in set '{args.instance_set}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
