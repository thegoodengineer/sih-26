#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate docs/BENCHMARKS.md from the CSVs in bench/results/.

docs/BENCHMARKS.md is NOT hand-written. It is regenerated from the evidence files so it
cannot drift from them: if a number appears in the document, a CSV row produced it, and the
CSV records the instance sha256, the git commit and the machine that produced it.

Reporting follows Mittelmann's conventions (plato.asu.edu):

*   the SHIFTED GEOMETRIC MEAN of solve times, shift 1 second. An arithmetic mean is
    dominated by the slowest instance and a plain geometric mean is dominated by the
    fastest; the shift damps both ends, which is why the benchmark community uses it.
*   the time limit is stated explicitly, because a mean over a censored sample is
    meaningless without it.
*   failures are COUNTED AND NAMED. An instance we cannot solve stays in the table.

Usage:
    python bench/runners/make_benchmarks_doc.py
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latest_result

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"
DATA_DIR = REPO_ROOT / "data" / "netlib"


def coverage_note(run_count: int, set_name: str | None = None) -> str:
    """State the DENOMINATOR, not just the pass rate.

    "8 of 8" is true and reads as full coverage of Netlib. It is 9% of the set, and the
    largest instance in the small tier is 118 rows - nothing that could exercise the
    degeneracy or ill-conditioning the problem statement asks about. A judge seeing a 100%
    pass rate will assume the set is representative unless told otherwise, so the generated
    file says so itself rather than relying on anyone opening fetch_data.py.
    """
    manifest_path = DATA_DIR / "reference.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return ""

    available = manifest.get("available_instances")
    # The tier comes from the CSV BEING RENDERED, not from whatever reference.json holds at
    # generation time. reference.json describes the last fetch, so reading it here labelled
    # the 50-instance medium table as "set small" whenever the small set had been fetched
    # more recently - a caption contradicting the table directly above it.
    if set_name is None:
        set_name = manifest.get("instance_set")
    if not available:
        return ""

    note = (f"Coverage: this run used **{run_count} of the {available} instances** Netlib "
            f"publishes an optimal value for")
    if set_name and set_name != "explicit":
        note += f" (set `{set_name}`, selected by `fetch_data.py --set {set_name}`)"
    # The exit-criterion sentence has to change on the full set, where "measured against the
    # full set, not against this one" would be talking about the table it is printed under.
    if set_name == "full":
        note += (". Phase 6's \"full Netlib >= 95%\" exit criterion is measured against this "
                 "set.")
    else:
        note += (". Phase 6's \"full Netlib >= 95%\" exit criterion is measured against the "
                 "full set, not against this one.")
    return note
OUTPUT = REPO_ROOT / "docs" / "BENCHMARKS.md"

SHIFT_SECONDS = 1.0


def shifted_geometric_mean(values: list[float], shift: float = SHIFT_SECONDS) -> float:
    """exp(mean(log(v + s))) - s. Mittelmann's convention, shift 1 second."""
    if not values:
        return float("nan")
    total = sum(math.log(max(value, 0.0) + shift) for value in values)
    return math.exp(total / len(values)) - shift


def newest(pattern: str) -> Path | None:
    """The most recent matching CSV, ordered by GIT HISTORY rather than by mtime.

    mtime is right on the machine that produced the files and wrong everywhere else: git
    does not record it, so a fresh clone stamps every file with the checkout time and the
    order becomes arbitrary. That is exactly the situation a judge regenerating this document
    is in, and the failure is silent - a plausible number from a superseded run. See
    bench/runners/latest_result.py.
    """
    return latest_result.latest(pattern)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict, key: str) -> float | None:
    raw = row.get(key, "")
    if raw in ("", None):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


FAILURE_CLASSES = [
    # (substring of the solver's own message, short label, tracking issue)
    ("basis became singular", "basis went singular", "#49"),
    ("consecutive degenerate", "degenerate stall", "#51"),
    ("violates primal feasibility", "point misses feasibility", "#72"),
    ("violate dual feasibility", "duals miss feasibility", "#52"),
    ("iteration limit", "hit the iteration limit", None),
    ("time limit", "hit the time limit", None),
]


def classify_failure(row: dict) -> str:
    """Name WHY an instance failed, from the solver's own message.

    CLAUDE.md requires failures to be named rather than dropped. A list of names is only
    half of it - "24 failed: bandm, boeing1, ..." tells a reader nothing about whether the
    tool fits their model. Eighteen instances failing for one reason is a very different
    thing from eighteen failing for eighteen reasons, and only the second is alarming.
    """
    message = (row.get("message") or "").lower()
    for needle, label, issue in FAILURE_CLASSES:
        if needle in message:
            return f"{label} ({issue})" if issue else label

    status = row.get("status", "")
    # An instance can be `optimal`, match nothing, and still be a failure - either the
    # objective disagrees with the published value or the independent verifier rejected the
    # point. Those are different problems and are not collapsed together here.
    # The verifier saying no covers two very different situations, and collapsing them
    # would point the reader at the wrong problem. forplan is the case in point: the
    # objective matches the published optimum exactly, and the verifier rejected it only
    # because its own MPS reader cannot parse names containing spaces. Nothing is wrong with
    # the answer there - what is wrong is that nothing independently checked it.
    verifier = (row.get("verifier_message") or "").lower()
    if "cannot read the model" in verifier or "could not convert" in verifier:
        return "verifier cannot parse the model (#48)"
    if row.get("independently_verified") == "0":
        return "verifier rejected the point (#75)"
    if status == "optimal" and row.get("matches_published") != "1":
        return "disagrees with the published optimum (#75)"
    return status or "unknown"


def failure_breakdown(failed: list[dict]) -> list[str]:
    """The failures, grouped by cause, most common first."""
    if not failed:
        return ["Every instance in this set passed.", ""]

    grouped: dict[str, list[str]] = {}
    for row in failed:
        grouped.setdefault(classify_failure(row), []).append(row["instance"])

    lines = [
        f"**{len(failed)} failed**, grouped by the reason the solver itself gave. They are "
        f"named here because a pass rate without its failures is a claim, not evidence:",
        "",
        "| why it failed | count | instances |",
        "|---|---:|---|",
    ]
    for label, names in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        lines.append(f"| {label} | {len(names)} | {', '.join(sorted(names))} |")
    lines.append("")
    return lines


def tier_of(path: Path) -> str | None:
    """The instance set a results CSV came from, read off its filename."""
    stem = path.stem
    parts = stem.split("-")
    return parts[1] if len(parts) > 2 else None


def netlib_section(path: Path) -> str:
    set_name = tier_of(path)
    rows = read_csv(path)
    if not rows:
        return "No Netlib results recorded yet.\n"

    commit = rows[0].get("git_commit", "unknown")
    machine = rows[0].get("machine", "unknown")
    timestamp = rows[0].get("timestamp_utc", "unknown")

    passed = [r for r in rows if r.get("passed") == "1"]
    failed = [r for r in rows if r.get("passed") != "1"]
    times = [t for t in (as_float(r, "wall_seconds") for r in passed) if t is not None]
    errors = [e for e in (as_float(r, "relative_gap") for r in passed) if e is not None]

    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}` · generated {timestamp}",
        "",
        f"**{len(passed)} of {len(rows)} instances in this working set** matched their "
        f"published optimum to a relative 1e-6 **and** passed independent verification by "
        f"`tools/verify_solution.py`.",
        "",
        coverage_note(len(rows), set_name),
        "",
        *failure_breakdown(failed),
        "| instance | rows | cols | status | our objective | published optimum | rel. error |"
        " iters | time (s) | verified |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|",
    ]


    for row in sorted(rows, key=lambda r: r["instance"]):
        ours = as_float(row, "our_objective")
        published = as_float(row, "published_objective")
        error = as_float(row, "relative_gap")
        seconds = as_float(row, "wall_seconds")
        verified = row.get("independently_verified", "")
        mark = {"1": "yes", "0": "**NO**", "": "-"}.get(verified, "-")
        out.append(
            f"| `{row['instance']}` | {row.get('rows', '')} | {row.get('columns', '')} "
            f"| {row.get('status', '')} "
            f"| {'-' if ours is None else f'{ours:.10e}'} "
            f"| {'-' if published is None else f'{published:.10e}'} "
            f"| {'-' if error is None else f'{error:.1e}'} "
            f"| {row.get('iterations', '')} "
            f"| {'-' if seconds is None else f'{seconds:.3f}'} | {mark} |")

    out += ["", "**Summary**", ""]
    if times:
        out.append(f"- shifted geometric mean solve time (shift {SHIFT_SECONDS:g}s): "
                   f"**{shifted_geometric_mean(times):.3f}s**")
        out.append(f"- slowest solved instance: {max(times):.3f}s")
    if errors:
        out.append(f"- worst relative error against a published optimum: "
                   f"**{max(errors):.2e}**")
    if failed:
        names = ", ".join(f"`{r['instance']}`" for r in failed)
        out.append(f"- **failed: {names}** — kept in the table on purpose")
    else:
        out.append("- no failures on this set")
    out.append("")
    return "\n".join(out)


def comparison_verdict(ratio):
    """Say what the measured ratio shows, rather than a fixed sentence that can go stale."""
    standing = (
        "HiGHS is a decade of specialist work with presolve, a dual simplex and a mature "
        "pricing scheme, and this solver still has neither of the first two. The part that "
        "has to be right first is that **the answers agree** - the problem statement asks us "
        "to compare, not to win.")
    if ratio is None:
        return standing
    if ratio > 1.15:
        return (f"We are **{ratio:.2f}x slower** than HiGHS by this measure, and publish that "
                f"rather than bury it. ") + standing
    if ratio < 0.87:
        return (f"We come out **{1.0 / ratio:.2f}x faster** than HiGHS by this measure on "
                f"this set. That is a real measurement and a narrow one: these are small, "
                f"well conditioned instances, and a shifted geometric mean over eight of them "
                f"settles nothing about large models. ") + standing
    return (f"The two are **within noise of each other** here, at {ratio:.2f}x. A narrow "
            f"claim: eight small instances settle nothing about large models. ") + standing


def comparison_section(path: Path | None) -> str:
    if path is None:
        return (
            "No comparison has been run yet, so **this section states no numbers**.\n"
            "\n"
            "`bench/runners/compare.py` is written and ready; it needs a HiGHS binary on the\n"
            "machine, which is invoked purely as an external subprocess and is never linked\n"
            "into SANKHYA (see the red line in `CLAUDE.md`).\n"
            "\n"
            "```bash\n"
            "apt-get install highs      # or conda install -c conda-forge highs\n"
            "python bench/runners/compare.py --time-limit 60\n"
            "```\n"
            "\n"
            "Once run, this section regenerates itself from the emitted CSV.\n")

    rows = read_csv(path)
    if not rows:
        return "The comparison CSV is empty.\n"

    ours = [t for t in (as_float(r, "sankhya_seconds") for r in rows) if t is not None]
    theirs = [t for t in (as_float(r, "highs_seconds") for r in rows) if t is not None]
    agreed = sum(1 for r in rows if r.get("objectives_agree") == "1")

    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{rows[0].get('git_commit', '?')}` · machine "
        f"`{rows[0].get('machine', '?')}`",
        "",
        f"**{agreed} of {len(rows)}** instances where the two solvers agree on the objective.",
        "",
        "Times are **solver-internal on both sides** - HiGHS's own `getRunTime()` against our "
        "`effort.solve_seconds` - so process start-up is excluded for both. At this instance "
        "size start-up would otherwise dominate and the comparison would measure the wrong "
        "thing entirely.",
        "",
        "| instance | SANKHYA obj | HiGHS obj | agree | SANKHYA (s) | HiGHS (s) | ratio |",
        "|---|---:|---:|:--:|---:|---:|---:|",
    ]
    ratios: list[float] = []
    for row in sorted(rows, key=lambda r: r["instance"]):
        a = as_float(row, "sankhya_objective")
        b = as_float(row, "highs_objective")
        sa = as_float(row, "sankhya_seconds")
        sb = as_float(row, "highs_seconds")
        ratio = as_float(row, "speed_ratio_sankhya_over_highs")
        if ratio is not None:
            ratios.append(ratio)
        out.append(
            f"| `{row['instance']}` "
            f"| {'-' if a is None else f'{a:.8e}'} | {'-' if b is None else f'{b:.8e}'} "
            f"| {'yes' if row.get('objectives_agree') == '1' else '**NO**'} "
            f"| {'-' if sa is None else f'{sa:.3f}'} | {'-' if sb is None else f'{sb:.3f}'} "
            f"| {'-' if ratio is None else f'{ratio:.2f}x'} |")

    out += ["", "**Summary**", ""]
    if ours:
        out.append(f"- SANKHYA shifted geometric mean: "
                   f"**{shifted_geometric_mean(ours):.3f}s**")
    if theirs:
        out.append(f"- HiGHS shifted geometric mean: "
                   f"**{shifted_geometric_mean(theirs):.3f}s**")
    if ours and theirs:
        ours_mean = shifted_geometric_mean(ours)
        theirs_mean = shifted_geometric_mean(theirs)
        if theirs_mean > 0:
            out.append(f"- SANKHYA is **{ours_mean / theirs_mean:.1f}x** the HiGHS time by "
                       f"that measure")
    out.append("")
    # Derived, not asserted. This paragraph used to state flatly that we lose on time.
    # That was true when written and stopped being true when the product-form basis
    # update landed, at which point the file argued against its own table two lines up.
    ours_mean = shifted_geometric_mean(ours) if ours else 0.0
    theirs_mean = shifted_geometric_mean(theirs) if theirs else 0.0
    ratio = ours_mean / theirs_mean if (ours and theirs and theirs_mean > 0) else None

    # The MEDIAN alongside the geometric mean, because on the medium tier they say different
    # things - 1.4x against 3.3x - and the gap between them is the finding. A uniform 3.3x
    # would mean the solver is broadly slow; a median near 1 with a mean of 3.3 means it is
    # competitive on most instances and pathological on a few, which points at specific
    # instances to fix rather than at the whole engine. Reporting only the mean would hide
    # that, and reporting only the median would flatter us.
    per_instance = sorted(r for r in ratios if r is not None)
    if len(per_instance) >= 3:
        median = statistics.median(per_instance)
        slowest = per_instance[-1]
        out.append(f"- per-instance ratio: median **{median:.2f}x**, worst **{slowest:.2f}x**, "
                   f"faster than HiGHS on **{sum(1 for r in per_instance if r < 1.0)} of "
                   f"{len(per_instance)}** instances")
    out.append("")
    out.append(comparison_verdict(ratio))
    out.append("")
    return "\n".join(out)


def medium_section(path: Path | None) -> str:
    """The 50-instance tier, which is the number that should be quoted.

    Kept separate from the small set rather than merged into one table, because the two
    answer different questions. The small set shows the pipeline works end to end and that
    a judge can pick an instance safely. The medium tier says how far the solver actually
    goes, and it is the one with failures in it.
    """
    if path is None:
        return chr(10).join([
            "Not yet run at this commit. Reproduce with:",
            "",
            "```",
            "python bench/runners/fetch_data.py --set medium",
            "python bench/runners/netlib.py --time-limit 60",
            "```",
            "",
        ])
    return netlib_section(path)


def full_section(path: Path | None) -> str:
    """The whole of Netlib, which is the number the exit criterion is measured against.

    Added for the same reason issue #53 split the small set from the medium tier, one level
    further out. `medium` is defined by a published row count of 500 or fewer, so quoting it
    as the headline reports the easier half of the library and calls it the library. The full
    set is lower and it is the one Phase 6's ">= 95% of Netlib" target is actually about.
    """
    if path is None:
        return chr(10).join([
            "Not yet run at this commit. Reproduce with:",
            "",
            "```",
            "python bench/runners/fetch_data.py --set full",
            "python bench/runners/netlib.py --time-limit 120",
            "```",
            "",
        ])
    return netlib_section(path)


def mittelmann_section(path: Path | None) -> str:
    """Mittelmann's LP set (#60): the scale evidence, which is a table of named failures.

    No published optimum exists for these instances - Mittelmann's page publishes solver
    TIMES - so a row cannot be a pass against a number. What the runner records instead is
    the status inside the limit, the verifier's verdict on any solution written, and HiGHS's
    objective as a separate process under the same limit. The point of the section is that
    every instance beyond Netlib's size is named with the outcome it had, rather than the
    page stopping where the solver does.
    """
    if path is None:
        return chr(10).join([
            "Not yet run at this commit. Reproduce with:",
            "",
            "```",
            "python bench/runners/fetch_mittelmann.py",
            "python bench/runners/mittelmann.py --time-limit 300",
            "```",
            "",
        ])
    rows = read_csv(path)
    if not rows:
        return "No Mittelmann results recorded yet." + chr(10)
    commit = rows[0].get("git_commit", "unknown")
    machine = rows[0].get("machine", "unknown")
    limit = as_float(rows[0], "time_limit")
    solved = [r for r in rows if r.get("status") == "optimal"]
    passed = [r for r in rows if r.get("passed") == "1"]
    highs_solved = [r for r in rows if as_float(r, "highs_objective") is not None]
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}` · time limit "
        f"{'-' if limit is None else f'{limit:g}'} s per instance, both solvers",
        "",
        f"**{len(solved)} of {len(rows)}** instances reached `optimal` inside the limit; "
        f"**{len(passed)} of {len(rows)}** also passed the independent verifier and agree "
        f"with HiGHS. HiGHS, run as a separate process under the same limit, finished "
        f"**{len(highs_solved)} of {len(rows)}**.",
        "",
        "These are the smallest archives in Mittelmann's LP directory; against Netlib's largest "
        "instance (dfl001, 6,071 rows, 35,632 nonzeros) they range from the same row count with "
        "2.7x the nonzeros (qap15) to 62x the rows and 42x the nonzeros (bdry2). No published optimum "
        "exists for them, so there is no pass-against-a-number column: the outcome is the "
        "status, the verifier's verdict where a solution was written, and HiGHS's objective "
        "where HiGHS finished. `our objective` on a `time_limit` row is the last iterate's "
        "value, not a bound, and is printed only so that a later run can be compared with it.",
        "",
        "| instance | rows | cols | nonzeros | status | our objective | HiGHS objective | "
        "rel. diff | iters | solver time (s) | verified |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in sorted(rows, key=lambda r: r["instance"]):
        ours = as_float(row, "our_objective")
        highs = as_float(row, "highs_objective")
        # A relative difference to HiGHS is only meaningful for an answer; the objective
        # on a time_limit row is the last iterate's and comparing it would print a
        # distance nobody should read.
        diff = as_float(row, "relative_difference") if row.get("status") == "optimal" else None
        seconds = as_float(row, "solver_seconds")
        verified = str(row.get("independently_verified", "")).strip()
        mark = {"1": "yes", "0": "**NO**", "true": "yes", "false": "**NO**"}.get(verified, "-")
        highs_cell = (f"{highs:.10g}" if highs is not None
                      else (row.get("highs_objective") or "-").replace("|", "/"))
        out.append(
            f"| `{row['instance']}` | {row.get('rows', '')} | {row.get('columns', '')} "
            f"| {row.get('nonzeros', '')} | {row.get('status', '')} "
            f"| {'-' if ours is None else f'{ours:.10g}'} | {highs_cell} "
            f"| {'-' if diff is None or not math.isfinite(diff) else f'{diff:.1e}'} "
            f"| {row.get('iterations', '')} "
            f"| {'-' if seconds is None else f'{seconds:.1f}'} | {mark} |")
    unsolved = sorted(r["instance"] for r in rows if r.get("status") != "optimal")
    if unsolved:
        out += ["", "**Not solved inside the limit**, named rather than dropped: "
                + ", ".join(f"`{n}`" for n in unsolved) + ".", ""]
    else:
        out += ["", "Every instance in the set finished inside the limit.", ""]
    return chr(10).join(out)


def pdhg_section(path: Path | None) -> str:
    """The first-order engine, at two tolerances, with restarts on and off (#28, #179).

    PDHG is not the default engine and this section is not a pass rate: the point of a
    first-order method is what it costs to reach a given accuracy, so the same instances are
    run at 1e-4 and at 1e-8 and reported separately. A single blended number would hide the
    only thing worth knowing about it.
    """
    if path is None:
        return chr(10).join([
            "Not yet run at this commit. Reproduce with:",
            "",
            "```",
            "python bench/runners/pdhg_report.py --time-limit 60 --instances \\",
            "    adlittle afiro blend israel sc105 sc50a sc50b share2b stocfor1",
            "```",
            "",
        ])
    rows = read_csv(path)
    if not rows:
        return "No PDHG results recorded yet." + chr(10)

    commit = rows[0].get("git_commit", "unknown")
    machine = rows[0].get("machine", "unknown")
    simplex = {r["instance"]: r for r in rows if r.get("algorithm") == "simplex"}
    names = sorted(simplex)

    def cell(instance: str, tolerance: str, restarts: str, field: str) -> str:
        for r in rows:
            if (r.get("algorithm") == "pdhg" and r["instance"] == instance
                    and r.get("tolerance") == tolerance and r.get("restarts_enabled") == restarts):
                return r.get(field, "")
        return ""

    def tally(tolerance: str, restarts: str) -> tuple[int, int]:
        # BY INSTANCE, not by row. pdhg_report.py runs the tightest tolerance with restarts on
        # twice - once in its tolerance sweep and once as the baseline of its restart
        # comparison - so counting rows reports 18 of 18 for nine instances.
        seen: dict[str, str] = {}
        for r in rows:
            if (r.get("algorithm") == "pdhg" and r.get("tolerance") == tolerance
                    and r.get("restarts_enabled") == restarts):
                seen[r["instance"]] = r["status"]
        return sum(status == "optimal" for status in seen.values()), len(seen)

    loose = next((r["tolerance"] for r in rows
                  if r.get("algorithm") == "pdhg" and r.get("tolerance") not in ("", None)), "")
    tolerances = sorted({r["tolerance"] for r in rows
                         if r.get("algorithm") == "pdhg" and r.get("tolerance")},
                        key=lambda t: -float(t))
    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}` · {len(names)} instances, "
        f"the ones committed to the repository",
        "",
    ]
    for tolerance in tolerances:
        on_opt, on_n = tally(tolerance, "1")
        off_opt, off_n = tally(tolerance, "0")
        parts = [f"**{on_opt} of {on_n}** reach `optimal` at a requested {tolerance} with "
                 f"restarts on"]
        if off_n:
            parts.append(f"**{off_opt} of {off_n}** with restarts off")
        out.append("- " + ", ".join(parts) + ".")
    out += [
        "",
        "`optimal` here means what it means everywhere else in this document: the point also "
        "survives the project's absolute tolerances, not merely the relative ones the "
        "first-order loop converges on. That distinction is the whole of #179 - the loop used "
        "to stop on the relative measure and the report then downgraded the point it stopped "
        "on, so the engine gave up early and handed back the weaker answer.",
        "",
        "**Read the two tolerance columns together, because they are the same run.** Since "
        "#179 the loop stops only where the absolute standard is met, so a request looser "
        "than that standard no longer stops the solve any earlier - ask for 1e-4 and you get "
        "the 1e-8 point, at the 1e-8 cost. That is the honest reading of the identical "
        "columns below, and it is a real trade: the old behaviour honoured a loose request "
        "and returned a point it then had to label `feasible`. #180 made that the opt-in: "
        "`--option pdhg_stop_at_request=true` waives the dual, gap and complementarity halves "
        "of the standard - absolute primal feasibility is kept, so `feasible` still means a "
        "feasible point - and reports the point as `feasible` unless it meets the full "
        "standard anyway. Measured on these instances at 1e-4 it costs 0.85x the iterations "
        "(`bench/results/pdhg-stop-at-request-f18d4b0.csv`) and turns `share2b` from an "
        "iteration limit into a usable point at 807,760. The two tolerance columns stay "
        "identical on `adlittle`, `israel` and `sc50b` even with the switch on, because on "
        "those the kept primal clause is what binds.",
        "",
        "| instance | simplex | " + " | ".join(
            f"PDHG {t}: objective / iterations" for t in tolerances) + " |",
        "|---|---:|" + "---:|" * len(tolerances),
    ]
    for name in names:
        row = [f"`{name}`", f"{as_float(simplex[name], 'objective') or float('nan'):.10g}"]
        for tolerance in tolerances:
            objective = cell(name, tolerance, "1", "objective")
            iterations = cell(name, tolerance, "1", "iterations")
            status = cell(name, tolerance, "1", "status")
            mark = "" if status == "optimal" else f" ({status})"
            value = "-" if not objective else f"{float(objective):.10g}"
            row.append(f"{value} / {iterations}{mark}")
        out.append("| " + " | ".join(row) + " |")

    # Restarts, the claim #28 asks to be measured rather than asserted.
    tightest = tolerances[-1] if tolerances else ""
    if tightest:
        out += ["", f"**Restarts, measured at {tightest}.** The claim that restarting the "
                    f"averaging helps is checked rather than repeated:", "",
                "| instance | restarts on | restarts off | ratio |", "|---|---:|---:|---:|"]
        for name in names:
            on = cell(name, tightest, "1", "iterations")
            off = cell(name, tightest, "0", "iterations")
            if not on or not off:
                continue
            ratio = "-" if float(on) == 0 else f"{float(off) / float(on):.2f}x"
            out.append(f"| `{name}` | {on} | {off} | {ratio} |")
        out.append("")
        out.append("A ratio above 1 means restarts saved iterations on that instance.")
    out.append("")
    return chr(10).join(out)


def milp_section(path: Path | None) -> str:
    """MIPLIB, where TWO questions have to be answered separately.

    On an LP there is one: is the objective right. On a MILP there are two, and they come
    apart constantly - reaching the published optimum is not the same as proving it is the
    optimum. `flugpl` returns exactly 1201500, which IS the published value, while the search
    stopped on a relative gap target rather than closing the bound. Reporting one number for
    both would either discard a correct answer or launder a tolerance stop into a proof.
    """
    if path is None:
        return chr(10).join([
            "Not yet run at this commit. Reproduce with:",
            "",
            "```",
            "python bench/runners/fetch_miplib.py --count 30",
            "python bench/runners/miplib.py --time-limit 600",
            "```",
            "",
        ])

    rows = read_csv(path)
    if not rows:
        return "No MIPLIB results recorded yet." + chr(10)

    matched = [r for r in rows if r.get("matched_published") == "1"]
    proved = [r for r in rows if r.get("proved_optimal") == "1"]
    commit = rows[0].get("git_commit", "unknown")
    machine = rows[0].get("machine", "unknown")

    out = [
        f"Source CSV: `bench/results/{path.name}`  ",
        f"Commit `{commit}` · machine `{machine}`",
        "",
        f"**{len(matched)} of {len(rows)}** instances reached the published optimum. "
        f"**{len(proved)} of {len(rows)}** also PROVED it - closed the bound rather than "
        f"stopping at a gap target or a limit.",
        "",
        "Those are different claims and are kept apart deliberately. Branch and bound here has "
        "no cutting planes - a rounding heuristic and a root dive, but nothing that tightens "
        "the relaxation - so it finds good incumbents far more often than it finishes the "
        "proof. Collapsing the two columns would hide exactly the thing #23 is meant to "
        "improve.",
        "",
        "**The time limit decides some of these, not the solver.** A row that stops at the limit "
        "with a small gap says \"needs more time than we gave it\", not \"cannot\"; which side of "
        "the limit such a row lands on moves with the machine's speed rather than with anything "
        "about the search. The remedy is a longer limit, and the reason this table does not "
        f"already use one is that the set already adds up to {sum(as_float(r, 'wall_seconds') or 0.0 for r in rows) / 60:.0f} minutes "
        "of solve time per run at this one.",
        "",
        "Instances are the smallest MIPLIB 2017 instances tagged easy that carry a **proven** "
        "optimum (`=opt=` in MIPLIB's own solution file). A `=best=` value is the best anyone "
        "has found, not a proof, and scoring against one would let a wrong answer look like a "
        "record.",
        "",
        "| instance | rows | cols | int | status | our objective | published | rel. gap | "
        "nodes | time (s) | matched | proved | verified |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|",
    ]

    def mark(value: str) -> str:
        return "yes" if value == "1" else ("**NO**" if value in ("0", "") else "-")

    for row in sorted(rows, key=lambda r: r["instance"]):
        ours = as_float(row, "our_objective")
        published = as_float(row, "published_objective")
        gap = as_float(row, "relative_gap")
        seconds = as_float(row, "wall_seconds")
        out.append(
            f"| `{row['instance']}` | {row.get('rows', '')} | {row.get('columns', '')} "
            f"| {row.get('integer_columns', '')} | {row.get('status', '')} "
            f"| {'-' if ours is None else f'{ours:.10g}'} "
            f"| {'-' if published is None else f'{published:.10g}'} "
            f"| {'-' if gap is None or not math.isfinite(gap) else f'{gap:.2e}'} "
            f"| {row.get('nodes', '')} "
            f"| {'-' if seconds is None else f'{seconds:.1f}'} "
            f"| {mark(row.get('matched_published', ''))} "
            f"| {mark(row.get('proved_optimal', ''))} "
            f"| {mark(row.get('independently_verified', ''))} |")

    unproved = sorted(r["instance"] for r in rows if r.get("proved_optimal") != "1")
    out += ["", "**Not proved optimal**, named rather than dropped: "
            + ", ".join(f"`{n}`" for n in unproved) + ".", ""]
    return chr(10).join(out)


def robustness_section(path: Path | None) -> str:
    """Where each numerical hazard breaks the solver (#71), from bench/runners/robustness.py.

    Every instance in the sweep has an optimum known by construction, so a pass means the
    status was optimal, the objective matched to 1e-6 relative AND the independent verifier
    accepted the certificate. The table names, per family, the last parameter that passed
    on every instance and the first that failed on any, with the failing check. Families
    that never fail are said to pass the whole sweep, with its extent; that is a statement
    about the sweep, not a claim that nothing beyond it can fail.
    """
    if path is None:
        return ("_No `robustness-*.csv` in `bench/results/`. Run "
                "`python bench/runners/robustness.py`._\n")
    rows = read_csv(path)
    families: dict[str, dict] = {}
    for row in rows:
        family = row["family"]
        entry = families.setdefault(family, {"parameter": row["parameter"], "points": {}})
        k = int(float(row["value"]))
        try:
            relative_error = float(row["relative_error"])
        except ValueError:
            relative_error = float("inf")
        passed = (row["status"] == "optimal" and relative_error <= 1e-6
                  and str(row.get("verified", "")).strip() == "1")
        point = entry["points"].setdefault(k, {"passed": True, "reason": ""})
        if not passed and point["passed"]:
            point["passed"] = False
            point["reason"] = (f"{row['status']}, relative error {relative_error:.1e}, "
                               f"verified {row.get('verified', '') or 'no'}"
                               + (f": {row['message'][:110]}" if row.get("message") else ""))
    out = [f"Measured on commit `{rows[0]['git_commit']}` ({rows[0]['machine']}), "
           f"{len(rows)} solves, {len(families)} families. Source: `{path.name}`.", "",
           "| family | parameter | last k that passed on every instance | first k that failed | what failed |",
           "|---|---|---|---|---|"]
    for family, entry in families.items():
        ks = sorted(entry["points"])
        last_pass, first_fail, reason = None, None, ""
        for k in ks:
            if entry["points"][k]["passed"]:
                if first_fail is None:
                    last_pass = k
            elif first_fail is None:
                first_fail, reason = k, entry["points"][k]["reason"]
        if first_fail is None:
            verdict = f"passes the whole sweep (k up to {ks[-1]})"
            out.append(f"| `{family}` | {entry['parameter']} | {ks[-1]} | {verdict} | - |")
        else:
            out.append(f"| `{family}` | {entry['parameter']} | "
                       f"{last_pass if last_pass is not None else 'none'} | {first_fail} | "
                       f"{reason.replace('|', '/')} |")
    out += ["",
            "Reading the table: the `conditioning` cliff is `kZeroDrop` (`tolerances.hpp`), "
            "the threshold below which a coefficient is treated as zero everywhere in the "
            "solver. At an entry spread of 1e12 the smallest coefficients fall under 1e-11, "
            "the model that gets solved is not the model that was written, and presolve then "
            "reports - correctly, about the truncated model - that a row cannot reach its "
            "bound. A model whose answer depends on a coefficient below 1e-11 is outside this "
            "solver's range; lowering the threshold would move the cliff, not remove it. The "
            "other limits are limits of the CERTIFICATE, not the answer: where the objective is "
            "right to 1e-15 and the verifier still rejects, the reduced costs or multipliers "
            "carry more rounding than its tolerances allow, which is worth knowing exactly "
            "because those tolerances are what a downstream consumer of the duals gets.", ""]
    return chr(10).join(out)


def main() -> int:
    # Both tiers, separately. Reporting only one was the whole of issue #53: the small set
    # is 8/8, which reads as a solved problem, and the medium tier is the number that says
    # what the solver can actually do. Publishing the first without the second is true and
    # misleading, which CLAUDE.md's evidence rules treat as the same thing as false.
    small_csv = newest("netlib-small-*.csv")
    medium_csv = newest("netlib-medium-*.csv")
    full_csv = newest("netlib-full-*.csv")
    milp_csv = newest("miplib-*.csv")
    pdhg_csv = newest("pdhg-*.csv")
    mittelmann_csv = newest("mittelmann-*.csv")
    compare_small_csv = newest("compare-highs-small-*.csv")
    compare_medium_csv = newest("compare-highs-medium-*.csv")
    compare_csv = compare_medium_csv or compare_small_csv or newest("compare-highs-*.csv")
    robustness_csv = newest("robustness-*.csv")

    # Legacy untagged CSVs predate the tier tag; fall back so an old results directory still
    # generates something rather than failing.
    netlib_csv = small_csv or newest("netlib-*.csv")

    if netlib_csv is None:
        print("no netlib-*.csv in bench/results/; run bench/runners/netlib.py first",
              file=sys.stderr)
        return 1

    document = f"""# SANKHYA — benchmarks

<!-- GENERATED FILE. Do not edit by hand. -->
<!-- Regenerate with: python bench/runners/make_benchmarks_doc.py -->

This file is generated from the CSVs in `bench/results/`, so it cannot drift from the
evidence. Every number below came out of a run that recorded the instance sha256, the git
commit and the machine tag alongside it.

Times in sections 1a-1c and 2 are wall-clock, measured around the whole process, so they
include reading the model and writing the outputs. That makes them slightly pessimistic and
honest; it is not the figure to quote for algorithmic speed, and no attempt is made to
dress it up. Section 1d prints solver-internal seconds (its instances take minutes, and the
read is not what is being measured) and section 4 is solver-internal on both sides, as it
says.

Reporting follows Mittelmann's conventions: shifted geometric means with a
{SHIFT_SECONDS:g}-second shift, an explicit time limit, and failures counted and named
rather than dropped.

---

## 1. Netlib LP — accuracy against published optima

The reference optimum for each instance is parsed by `bench/runners/fetch_data.py` from
Netlib's own `readme`. None of these values was typed from memory.

### 1a. The small set — what the demo runs

Nine instances, committed to the repository so a fresh clone can reproduce this with no
network. **This is the set `demo/run_demo.sh` lets a judge pick from, and it is the easy end
of Netlib.** Its pass rate is not the headline; section 1c is.

{netlib_section(netlib_csv)}
### 1b. The medium tier — instances up to 500 rows

{medium_section(medium_csv)}
### 1c. The full set — the honest headline

Every instance in Netlib's summary table. Both tiers above are defined by a row cap, which
makes them the easier half of the library by construction; this is the number Phase 6's
">= 95% of Netlib" exit criterion is measured against, and the one the README quotes.

{full_section(full_csv)}
### 1d. Beyond Netlib — Mittelmann's LP set

Netlib's largest instance has about 6,000 rows. PS26119 asks about "thousands to millions
of variables", and the only honest way to say where this solver stands on that is to run
instances of that size and name what happens. These are the eight smallest archives in
Mittelmann's LP test set (`bench/runners/fetch_mittelmann.py`, provenance in
`data/mittelmann/reference.json`).

{mittelmann_section(mittelmann_csv)}
### 1e. The first-order engine — PDHG

The simplex is not the only continuous engine. Restarted PDHG (`--option algorithm=pdhg`) is
a first-order method: no basis, no factorization, and a cost that depends enormously on the
accuracy asked of it - which is why this section reports two tolerances separately rather
than one blended number. It is also the engine the GPU work targets, so its CPU behaviour is
the baseline every GPU claim will be measured against.

{pdhg_section(pdhg_csv)}
---

## 2. MIPLIB — the mixed-integer side

The LP tiers above say nothing about the branch and bound. This is the MILP evidence, and it
is a harder library: MIPLIB instances are chosen to be difficult for mature solvers.

{milp_section(milp_csv)}
---

## 3. Correctness beyond the objective value

An objective that matches a published number is necessary, not sufficient — it says nothing
about whether the reported solution is internally consistent. Two independent checks cover
that, and both run in CI:

- **`tools/verify_solution.py`** re-parses the model with its own MPS reader, recomputes the
  row activities, the objective, the reduced costs and the dual objective, and checks primal
  feasibility, dual feasibility, complementary slackness and strong duality. It shares no
  code with the solver, so a reader bug shows up as a disagreement rather than as agreement.
  The `verified` column above is its verdict.

- **The exact rational oracle** (`tests/oracles/`) solves generated instances in exact
  arithmetic with no rounding error anywhere, and the floating-point simplex is compared
  against it. See `docs/PROVENANCE.md` for the citations.

---

## 4. Comparison against an established solver

HiGHS is the reference. It runs as a SEPARATE PROCESS over the same MPS files; no HiGHS code
is linked into, or read by, SANKHYA - see `docs/PROVENANCE.md`. Both sides are timed on
solver-internal time only.

The comparison below is run on **the same tier as section 1b**, not on the nine-instance
demo set. Comparing only where we pass would be the easy version of this table and would say
nothing: the instances we fail are exactly the ones a reader should want to see against a
mature solver.

{comparison_section(compare_csv)}
---

## 5. Robustness — where the solver stops working

PS26119 asks for "a clear demonstration of numerical robustness ... involving degeneracy,
weak LP relaxations or ill-conditioned constraint matrices". `data/casestudies/` demonstrates
each hazard on one chosen instance; this section is the sweep that finds the case we do not
handle. Every instance is built from a chosen primal-dual pair, so its optimum is known
before it is solved (the construction is `tests/oracles/lp_generator.hpp`'s, in
`bench/runners/robustness.py`), and each family pushes one hazard until the answer, or the
certificate, moves. The reduced version runs in CI (`tests/robustness/`), together with the
adversarial families judged by the exact rational oracle and the classic cycling examples
of Beale and Kuhn.

{robustness_section(robustness_csv)}
---

## 6. What these numbers do not say

- **Nothing here supports a claim about large models.** Section 1d is the evidence at
  the scale PS26119's "thousands to millions of variables" means, and it is a table of
  named time limits: the solver reaches Netlib's largest instances and stops there. No
  pass rate above substitutes for that table. Tracked as part of #54.
- Wall-clock times at this size are dominated by process start-up and file reading, so
  ratios between solvers are not meaningful until the instances get big enough to matter.
  The comparison in section 4 uses solver-internal time on both sides for that reason.
- The failures in section 1b are real and are not going to be quietly dropped from a later
  edition of this file. Each one carries the issue tracking it.
- Two engines named in PS26119 are not measured on this page. The interior-point method
  (`algorithm=ipm`, #56) is opt-in and produces no basis, so it is not the engine behind any
  table above; its own Netlib run is committed as `netlib-full-*-ipm.csv` and quoted in
  `docs/PS26119_COVERAGE.md`, not here, because a run made with a non-default option is a
  measurement of that option rather than the tier's evidence. There is no GPU backend on
  `main` (#16-#19). `docs/PROVENANCE.md` and issue #54 carry the full accounting.
"""

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: Python defaults to the locale encoding on Windows, which mangles
    # every em-dash in the document into a replacement character.
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} from {netlib_csv.name}"
          + (f" and {compare_csv.name}" if compare_csv else " (no comparison CSV yet)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
