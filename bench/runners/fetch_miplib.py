#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch a MIPLIB 2017 subset and its PUBLISHED optimal objective values.

The MILP counterpart of fetch_data.py, and careful about the same things for the same
reasons: a benchmark table is only evidence if a judge can regenerate it, and a reference
value we typed in ourselves is not a reference value.

THREE THINGS THIS IS DELIBERATE ABOUT.

1.  ONLY INSTANCES MARKED `=opt=`. MIPLIB's solution file marks each instance `=opt=` (a
    PROVEN optimum), `=best=` (the best value anyone has found, not proven), `=unkn=`,
    `=inf=` or `=unbd=`. Only `=opt=` gives a value we can asserted equality against. Scoring
    ourselves against a `=best=` bound would mean a run could "beat" the reference and look
    like a triumph while actually being wrong, which is the failure mode this project exists
    to avoid.

2.  THE SUBSET IS CHOSEN BY SIZE, AND SIZE COMES FROM AN HTTP HEAD. MIPLIB tags instances
    easy/hard/open, but "easy" there means easy for a mature solver with cuts, presolve and
    a dual simplex. Ours has none of the first two, so the tag alone selects models we cannot
    finish. Compressed file size is a crude proxy for difficulty and an honest one: it needs
    no endpoint beyond the instance URL itself, it is identical for everyone, and the exact
    list ends up recorded in the manifest either way.

3.  NOTHING IS COMMITTED. `data/miplib/` is gitignored. The manifest records the sha256 of
    every instance downloaded, so a later run can prove it solved the same bytes.

    python bench/runners/fetch_miplib.py                 # 30 smallest easy instances
    python bench/runners/fetch_miplib.py --count 10
    python bench/runners/fetch_miplib.py flugpl gen-ip002
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import ssl
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "miplib"

MIPLIB = "https://miplib.zib.de"
SOLUTION_FILE = f"{MIPLIB}/downloads/miplib2017-v28.solu"
EASY_LIST = f"{MIPLIB}/downloads/easy-v15.test"
INSTANCE_BASE = f"{MIPLIB}/WebData/instances"


def ssl_context() -> ssl.SSLContext:
    """certifi's bundle when it is installed, the system default otherwise.

    This box has no default CA file at all (`ssl.get_default_verify_paths().cafile` is None),
    so miplib.zib.de fails certificate verification while netlib.org happens to succeed.
    Falling back rather than requiring certifi keeps the script working where the system
    store is fine.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def get(url: str, timeout: int = 120) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout, context=ssl_context()) as response:
        return response.read()


def content_length(url: str, timeout: int = 30) -> int | None:
    """Size without downloading the body. Returns None when the server will not say."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as r:
            value = r.headers.get("Content-Length")
            return int(value) if value else None
    except OSError:
        return None


def published_optima() -> dict[str, float]:
    """Parse MIPLIB's own solution file. `=opt=` lines only - see the module docstring."""
    text = get(SOLUTION_FILE).decode(errors="replace")
    optima: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "=opt=":
            try:
                optima[parts[1]] = float(parts[2])
            except ValueError:
                continue
    return optima


def easy_instances() -> list[str]:
    text = get(EASY_LIST).decode(errors="replace")
    return [entry.replace(".mps.gz", "") for entry in text.split() if entry.endswith(".mps.gz")]


def choose(count: int) -> list[tuple[str, int]]:
    """The `count` smallest easy instances that have a proven optimum, with their sizes."""
    optima = published_optima()
    candidates = [name for name in easy_instances() if name in optima]
    print(f"  {len(optima)} instances have a proven optimum", flush=True)
    print(f"  {len(candidates)} of the easy-tagged set are among them", flush=True)
    print(f"  sizing them by HTTP HEAD (no bodies downloaded)...", flush=True)

    # Concurrently, and flushed as it goes. Sequentially this is roughly one request per
    # instance per second - twelve minutes of a script printing nothing, because Python
    # buffers stdout when it is not a terminal. Both halves of that were worth fixing: a
    # fetch that looks hung is a fetch people interrupt.
    sized: list[tuple[str, int]] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(content_length, f"{INSTANCE_BASE}/{n}.mps.gz"): n
                   for n in candidates}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            size = future.result()
            if size is not None:
                sized.append((futures[future], size))
            if done % 100 == 0 or done == len(candidates):
                print(f"    {done}/{len(candidates)}", flush=True)
    sized.sort(key=lambda pair: (pair[1], pair[0]))
    return sized[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instances", nargs="*",
                        help="explicit instance names; omit to take the smallest easy ones")
    parser.add_argument("--count", type=int, default=30,
                        help="how many of the smallest easy instances to take (default 30)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    optima = published_optima()

    if args.instances:
        missing = [n for n in args.instances if n not in optima]
        if missing:
            print(f"no PROVEN optimum published for: {', '.join(missing)}", file=sys.stderr)
            print("MIPLIB marks those =best=, =unkn=, =inf= or =unbd=; see the module "
                  "docstring for why only =opt= is usable here", file=sys.stderr)
            return 1
        chosen = [(name, 0) for name in args.instances]
        selection = "explicit"
    else:
        chosen = choose(args.count)
        selection = f"smallest-{args.count}-easy"

    manifest = {
        "source": MIPLIB,
        "solution_file": SOLUTION_FILE,
        "easy_list": EASY_LIST,
        "selection": selection,
        "available_with_proven_optimum": len(optima),
        "instances": {},
    }

    print()
    print(f"  {'instance':<28}{'bytes':>10}  published optimum")
    for name, _size in chosen:
        url = f"{INSTANCE_BASE}/{name}.mps.gz"
        blob = get(url)
        destination = DATA_DIR / f"{name}.mps.gz"
        # Written to a scratch path and renamed, so an interrupted fetch cannot leave a
        # truncated instance behind that looks like a real one. Same reasoning as #87.
        scratch = destination.with_suffix(destination.suffix + ".partial")
        try:
            scratch.write_bytes(blob)
            scratch.replace(destination)
        finally:
            scratch.unlink(missing_ok=True)

        manifest["instances"][name] = {
            "name": name,
            "url": url,
            "gz_bytes": len(blob),
            "gz_sha256": hashlib.sha256(blob).hexdigest(),
            "published_optimal": optima[name],
        }
        print(f"  {name:<28}{len(blob):>10}  {optima[name]:.10g}", flush=True)

    reference = DATA_DIR / "reference.json"
    reference.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print()
    print(f"wrote {reference.relative_to(REPO_ROOT)}")
    print(f"{len(manifest['instances'])} instance(s) in {DATA_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
