#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run SANKHYA over the fetched Netlib instances and emit the evidence CSV.

Every column CLAUDE.md requires is here: instance, sha256 of the instance file, our
objective, the PUBLISHED reference objective, absolute and relative gap, status, wall time,
iterations, git commit and a machine tag. Without the CSV there is no claim.

Two things this runner does that a plain timing loop would not:

*   The reference values come from ``data/netlib/reference.json``, which ``fetch_data.py``
    parsed out of Netlib's own readme. No number here was typed from memory.

*   Every solution is handed to ``tools/verify_solution.py``, which re-parses the model with
    its own MPS reader and re-derives feasibility, the objective and strong duality without
    touching our C++. Matching the published optimum says the answer is right; the verifier
    says the answer is *self-consistent*, and the two failures look nothing alike.

Usage:
    python bench/runners/netlib.py
    python bench/runners/netlib.py --binary build/sankhya --time-limit 60
    python bench/runners/netlib.py --check        # fail if the pass rate dropped
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "netlib"
RESULTS_DIR = REPO_ROOT / "bench" / "results"

def display_path(path: Path) -> Path | str:
    """A path for printing: repo-relative when it is inside the repo, absolute otherwise.

    `relative_to` RAISES when the target is outside REPO_ROOT, and that turned a successful
    run into a traceback after every result had already been printed - taking the exit code
    with it, so a run where everything passed reported failure. Writing a CSV somewhere else
    on purpose is a legitimate thing to ask for, not an error.
    """
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path

VERIFIER = REPO_ROOT / "tools" / "verify_solution.py"

# A run counts as a pass when the status is optimal AND the objective matches the published
# value to this relative accuracy. Status alone is not enough: a solver that confidently
# reports "optimal" with the wrong number is the exact failure this project exists to catch.
PASS_RELATIVE_TOLERANCE = 1e-6

CSV_COLUMNS = [
    "instance",
    "instance_sha256",
    "rows",
    "columns",
    "nonzeros",
    "status",
    # The solver's own explanation. Without it every failure is just "numerical_error" and
    # docs/BENCHMARKS.md cannot say WHICH failure, which is most of what makes a named
    # failure useful to anyone deciding whether the tool fits their model.
    "message",
    "our_objective",
    "published_objective",
    "absolute_gap",
    "relative_gap",
    "matches_published",
    "independently_verified",
    # Why the verifier said no. Without it, "the verifier rejected this" cannot distinguish
    # a bad point from a model the verifier could not parse - and those want different
    # people looking at them.
    "verifier_message",
    "passed",
    "objective_offset",
    # 1 when the only disagreement with the published value IS the objective-row constant,
    # which Netlib's table excludes and we include. See the comment at the comparison.
    "differs_by_objective_constant",
    "wall_seconds",
    "solver_seconds",
    "iterations",
    "algorithm",
    "git_commit",
    "machine",
    "timestamp_utc",
]


def as_number(value) -> float | None:
    """Coerce a JSON numeric field to float.

    JSON has no literal for infinity, so the writer emits non-finite values as the strings
    "inf", "-inf" and "nan" rather than letting them collapse to null. Python's float()
    accepts all three, so this is the only special case a consumer needs."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def git_commit() -> str:
    """Short commit hash, with "-dirty" appended when tracked files are modified.

    The commit names the CSV and is recorded in every row, and until now it named the
    commit whether or not the tree matched it. Two benchmark runs from the same commit but
    different working trees - a baseline and a branch with uncommitted changes - wrote the
    SAME filename, the second overwrote the first, and the comparison that followed diffed
    a file against itself and found no change. A CSV that claims to be evidence for a commit
    while measuring something else is the exact failure CLAUDE.md's evidence rules exist to
    prevent, so the marker goes in the name and in the column.

    Untracked files are ignored: the fetched instances under data/netlib/ are untracked by
    design and would otherwise mark every run dirty.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=False)
        commit = result.stdout.strip() or "unknown"
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        if status.stdout.strip():
            commit += "-dirty"
        return commit
    except OSError:
        return "unknown"


def machine_tag() -> str:
    return f"{platform.system()}-{platform.machine()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_binary() -> Path:
    for candidate in ("build/sankhya.exe", "build/sankhya", "build-main/sankhya.exe",
                      "build-main/sankhya"):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    raise SystemExit("no solver binary found; build first, or pass --binary")


def run_one(binary: Path, mps: Path, time_limit: float, verify: bool) -> dict:
    """Solve one instance, then verify the solution independently."""
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "stats.json"
        sol_path = Path(tmp) / "solution.sol"
        command = [
            str(binary), "solve", str(mps),
            "--stats", str(stats_path),
            "--write-sol", str(sol_path),
            "--time-limit", str(time_limit),
            "--option", "log_to_console=false",
        ]
        started = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True)
        wall = time.perf_counter() - started

        if not stats_path.exists():
            return {
                "status": "crashed" if completed.returncode not in (0, 1) else "no_output",
                "wall_seconds": wall,
                "stderr": completed.stderr.strip()[:400],
                "verified": None,
            }

        blob = json.loads(stats_path.read_text())
        # The writer nests the blob; flatten the fields this runner reports on.
        result = blob.get("result", {})
        model = blob.get("model", {})
        effort = blob.get("effort", {})
        flat = {
            "status": result.get("status", "unknown"),
            "message": result.get("message", ""),
            "objective": as_number(result.get("objective")),
            "absolute_gap": as_number(result.get("absolute_gap")),
            "relative_gap": as_number(result.get("relative_gap")),
            "algorithm": result.get("algorithm", ""),
            "rows": model.get("rows", ""),
            "columns": model.get("columns", ""),
            "nonzeros": model.get("nonzeros", ""),
            "objective_offset": as_number(model.get("objective_offset")) or 0.0,
            "iterations": effort.get("iterations", ""),
            "solver_seconds": effort.get("solve_seconds", ""),
            "wall_seconds": wall,
            "stderr": completed.stderr.strip()[:400],
            "verified": None,
        }

        if verify and sol_path.exists() and flat["status"] in ("optimal", "feasible"):
            check = subprocess.run(
                [sys.executable, str(VERIFIER), str(mps), str(sol_path), "--quiet"],
                capture_output=True, text=True)
            flat["verified"] = check.returncode == 0
            if check.returncode != 0:
                flat["verifier_output"] = (check.stdout + check.stderr).strip()[:600]
        return flat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--instances", nargs="*")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the independent verifier (not recommended)")
    parser.add_argument("--check", action="store_true",
                        help="fail if the pass count dropped versus the newest committed CSV")
    parser.add_argument("--require-verified", action="store_true",
                        help="fail ONLY when the independent verifier rejects a solution, "
                             "not when an instance merely fails to reach the published "
                             "optimum. This is the gate for tiers with known failures.")
    parser.add_argument("--out", type=Path, default=None,
                        help="destination CSV; relative paths are resolved "
                             "against the repository root")
    args = parser.parse_args()

    binary = args.binary or default_binary()
    reference_path = DATA_DIR / "reference.json"
    if not reference_path.exists():
        raise SystemExit("no reference data; run bench/runners/fetch_data.py first")
    # READ ONCE. The tier tag used in the output filename used to be read from this file
    # again at the END of the run, hundreds of solves later, and anything that changed the
    # file in between silently mislabelled the result. That is not hypothetical: a stray
    # `git checkout -- data/netlib/reference.json` during a run produced an 89-instance CSV
    # named netlib-small-*.csv, and make_benchmarks_doc.py selects CSVs BY THAT NAME - so
    # docs/BENCHMARKS.md would have reported a full-set figure as the small tier's, in a
    # document whose whole purpose is that it cannot drift from the evidence.
    reference_blob = json.loads(reference_path.read_text())
    reference = reference_blob["instances"]
    tier = reference_blob.get("instance_set", "")

    names = sorted(args.instances or reference)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    commit, machine = git_commit(), machine_tag()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    rows: list[dict] = []
    print(f"solver   {binary}")
    print(f"commit   {commit}   machine {machine}")
    print()
    print(f"{'instance':<11}{'status':<9}{'our objective':>22}{'published':>22}"
          f"{'rel err':>10}{'iters':>7}{'time':>8}  verified  result")
    print("-" * 104)

    for name in names:
        entry = reference[name]
        mps = DATA_DIR / f"{name}.mps"
        if not mps.exists():
            print(f"{name:<11}{'MISSING':<9}")
            continue

        published = entry["published_optimal"]
        blob = run_one(binary, mps, args.time_limit, not args.no_verify)
        status = blob["status"]
        ours = blob.get("objective")
        verified = blob.get("verified")
        gap = None if ours is None else abs(ours - published) / max(1.0, abs(published))
        matches = bool(status == "optimal" and gap is not None
                       and gap <= PASS_RELATIVE_TOLERANCE)

        # NETLIB'S PUBLISHED TABLE EXCLUDES THE OBJECTIVE-ROW CONSTANT, and ours includes it.
        #
        # An RHS entry on the N row is the objective constant, negated - the convention
        # src/io/mps_reader.cpp implements and documents. Netlib's readme reports objective
        # values computed WITHOUT it. On e226, the only instance in the medium set that has
        # one, that is the entire disagreement: we report -11.638929, the table says
        # -18.751929, the constant is 7.113, and -11.638929 - 7.113 is the published value to
        # every digit printed. tools/verify_solution.py passes our answer 11/11 with strong
        # duality closing to 1.8e-15, so the point is genuinely optimal for the model as read.
        #
        # This is therefore a units mismatch in the COMPARISON, not a solver failure, and
        # counting it as one overstates how much is broken. It is reported as its own outcome
        # rather than silently forgiven: the run still does not match the table, and a reader
        # deserves to see why rather than find an instance quietly reclassified as a pass.
        offset = blob.get("objective_offset") or 0.0
        offset_gap = (None if ours is None or not offset else
                      abs((ours - offset) - published) / max(1.0, abs(published)))
        # `verified is not False` is part of the test and not an afterthought. The whole
        # justification for not calling this a failure is that our point is PROVABLY optimal
        # for the model as read - which is a claim tools/verify_solution.py makes, not one
        # the arithmetic above establishes. If the independent checker rejects the answer,
        # a gap that happens to equal the objective constant is a coincidence rather than an
        # explanation, and the row belongs in the failure list.
        explained_by_offset = bool(
            not matches and status == "optimal" and offset_gap is not None
            and offset_gap <= PASS_RELATIVE_TOLERANCE and verified is not False)
        # A pass needs BOTH: the right number, and a solution that survives independent
        # re-derivation. Either one alone can be satisfied by a solver that is wrong.
        passed = matches and (verified is not False)

        rows.append({
            "instance": name,
            "instance_sha256": sha256_file(mps),
            "rows": blob.get("rows", ""),
            "columns": blob.get("columns", ""),
            "nonzeros": blob.get("nonzeros", ""),
            "status": status,
            "message": blob.get("message", ""),
            "verifier_message": blob.get("verifier_output", ""),
            "our_objective": "" if ours is None else repr(ours),
            "published_objective": repr(published),
            "absolute_gap": "" if ours is None else repr(abs(ours - published)),
            "relative_gap": "" if gap is None else repr(gap),
            "matches_published": int(matches),
            "independently_verified": "" if verified is None else int(verified),
            "passed": int(passed),
            "objective_offset": repr(offset),
            "differs_by_objective_constant": int(explained_by_offset),
            "wall_seconds": round(blob.get("wall_seconds", 0.0), 6),
            "solver_seconds": blob.get("solver_seconds", ""),
            "iterations": blob.get("iterations", ""),
            "algorithm": blob.get("algorithm", ""),
            "git_commit": commit,
            "machine": machine,
            "timestamp_utc": timestamp,
        })

        ours_text = "-" if ours is None else f"{ours:>22.12e}"
        gap_text = "-" if gap is None else f"{gap:>10.1e}"
        verified_text = {True: "  yes   ", False: "  NO    ", None: "  -     "}[verified]
        print(f"{name:<11}{status:<9}{ours_text}{published:>22.12e}{gap_text}"
              f"{str(blob.get('iterations', '-')):>7}{blob.get('wall_seconds', 0.0):>7.2f}s"
              f"{verified_text}  "
              f"{'PASS' if passed else ('OFFSET' if explained_by_offset else 'FAIL')}")
        if explained_by_offset:
            print(f"             the gap IS the objective-row constant ({offset:g}); Netlib's "
                  f"table excludes it, we include it. Our point verifies as optimal.")
        if not passed:
            if blob.get("stderr"):
                print(f"             stderr: {blob['stderr']}")
            if blob.get("verifier_output"):
                print(f"             verifier: {blob['verifier_output']}")

    passes = sum(row["passed"] for row in rows)
    total = len(rows)
    print("-" * 104)
    print(f"{passes}/{total} matched the published optimum to a relative "
          f"{PASS_RELATIVE_TOLERANCE:g} AND passed independent verification")
    offset_rows = [row["instance"] for row in rows
                   if row.get("differs_by_objective_constant")]
    if offset_rows:
        # Reported on its own line and NOT folded into the pass count. The run genuinely does
        # not match the table, and quietly reclassifying it as a pass would be the same kind
        # of flattering arithmetic this harness exists to prevent - but calling it a solver
        # failure overstates what is broken, so it gets its own name.
        print(f"differs only by the objective-row constant: {', '.join(offset_rows)} "
              f"(Netlib's table excludes it; our answer verifies as optimal)")
    if total and passes < total:
        failed = [row["instance"] for row in rows
                  if not row["passed"] and not row.get("differs_by_objective_constant")]
        # Naming the failures is not optional. A pass rate without them is a claim.
        if failed:
            print(f"failed: {', '.join(failed)}")

    # Tier goes in the FILENAME. Both tiers at the same commit previously produced the same
    # path, so running medium after small silently overwrote it and docs/BENCHMARKS.md could
    # only ever describe whichever ran last.
    tier_tag = f"{tier}-" if tier and tier != "explicit" else ""
    out_path = args.out or (RESULTS_DIR / f"netlib-{tier_tag}{commit}.csv")
    # Resolve against the repository root BEFORE anything else touches it. Two separate
    # problems came from leaving a user-supplied relative path alone:
    #
    #   1. `relative_to(REPO_ROOT)` on the status line raised ValueError, after the results
    #      had been printed, taking the exit code with it.
    #
    #   2. Worse and quieter: RESULTS_DIR.glob() yields ABSOLUTE paths, so a relative
    #      out_path never compared equal to any of them. The CSV just written was therefore
    #      not excluded from the "previous runs" set, and being the newest by mtime it became
    #      the baseline - so the run was compared against ITSELF and the --check regression
    #      gate could never fire. That is a gate that silently passes, on the evidence
    #      CLAUDE.md says the project stands or falls by.
    #
    # Problem 2 was unreachable only because problem 1 crashed first. Fixing the traceback
    # alone would have exposed it.
    out_path = (REPO_ROOT / out_path).resolve()
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {display_path(out_path)}")

    if args.check:
        # Compare against the SAME TIER only. Now that the tier is in the filename, a bare
        # netlib-*.csv glob would happily take a 50-instance medium baseline for an 8-instance
        # small run and report a catastrophic regression, or the reverse and report a triumph.
        # Either way the gate would be measuring the size of the instance set rather than the
        # health of the solver - the same class of silently-wrong gate #31 fixed here.
        # Compare against a baseline covering the SAME INSTANCES, by row count rather than by
        # filename. Matching on the tier tag alone would silently discard every CSV written
        # before the tag existed - all of them small-set runs, and the only history there is.
        # Matching on size keeps them and still refuses to weigh an 8-instance run against a
        # 50-instance one, which would measure the size of the set rather than the health of
        # the solver: the same class of silently-wrong gate #31 fixed here.
        def comparable(candidate: Path) -> bool:
            if candidate == out_path:
                return False
            try:
                with candidate.open(newline="") as handle:
                    return len(list(csv.DictReader(handle))) == len(rows)
            except OSError:
                return False

        previous = sorted((p for p in RESULTS_DIR.glob("netlib-*.csv") if comparable(p)),
                          key=lambda p: p.stat().st_mtime)
        if not previous:
            print("no earlier CSV to compare against; this run is the baseline")
            return 0
        with previous[-1].open(newline="") as handle:
            baseline = list(csv.DictReader(handle))
        baseline_passes = sum(int(row["passed"]) for row in baseline)
        print(f"baseline {previous[-1].name}: {baseline_passes}/{len(baseline)}")
        if passes < baseline_passes:
            print(f"REGRESSION: pass count fell from {baseline_passes} to {passes}")
            return 1

    # --require-verified: the gate for a tier we do not pass completely.
    #
    # The default rule is `passes == total`, which is right for the small set and useless
    # anywhere else: the medium tier is 43/50 and the full set 71/89, so pointing CI at
    # either would paint the job permanently red, and a job that is always red is a job
    # everyone learns to ignore - the exact outcome the cache comment above worries about.
    #
    # A VERIFIER REJECTION IS A DIFFERENT KIND OF EVENT FROM A MISSED OPTIMUM. Failing to
    # reach the published value can be a known limitation: too slow, not accurate enough,
    # honestly recorded in docs/BENCHMARKS.md and named in #34. But the verifier re-derives
    # the answer from the original file and shares no code with the solver, so its rejection
    # says the solution is internally inconsistent - infeasible, or its duals do not price
    # its primal. That is never acceptable and never a known limitation.
    #
    # This distinction is not hypothetical. #149 (presolve free-column-singleton and
    # doubleton-equation) passed all five CI checks and regressed the DUAL on seven of the
    # first thirty Netlib instances: the objective matched the published optimum to ten
    # significant figures while the reduced costs were out by 6.9e-01. Nothing except the
    # verifier could see it, and the job that runs the verifier was pointed at nine
    # instances none of which trigger those reductions.
    if args.require_verified:
        rejected = [row["instance"] for row in rows if row["independently_verified"] == 0]

        # A DOWNGRADED OPTIMALITY CLAIM COUNTS AS A REJECTION TOO (#157).
        #
        # The verifier only checks the dual conditions when the solver claims optimality,
        # and that is right: kFeasible is a solver declining to make the claim. But there is
        # a second way to arrive at kFeasible - the engine claimed optimal and our own status
        # check in solve.cpp caught the duals violating tolerance and overrode it. That is
        # not declining a claim; it is making one and being caught. And it was invisible
        # here: the status is no longer "optimal", so the verifier skipped the duals, so
        # nothing was rejected, and recipe dropped from a pass to a silent non-pass on the
        # #149 merge with the gate green. The stricter our own check, the less this gate
        # saw. Every override writes a message beginning with the same words, so it is
        # matched on those - a string this project owns, not a heuristic.
        self_rejected = [row["instance"] for row in rows
                         if str(row.get("message", "")).startswith("engine reported optimal but")
                         and row["instance"] not in rejected]

        if rejected or self_rejected:
            if rejected:
                print(f"REJECTED BY THE VERIFIER: {len(rejected)} instance(s): "
                      f"{', '.join(rejected)}")
            if self_rejected:
                print(f"OPTIMALITY CLAIM REJECTED BY OUR OWN CHECK: {len(self_rejected)} "
                      f"instance(s): {', '.join(self_rejected)}")
            print("An answer that fails an independent check - or that claimed optimality "
                  "and failed our own - is internally inconsistent, which is a bug whatever "
                  "the objective says.")
            return 1
        print(f"no rejections across {total} instance(s); "
              f"{passes} also matched the published optimum")
        return 0

    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
