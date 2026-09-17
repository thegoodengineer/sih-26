#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The certificate, IIS and pool checks verify_solution.py runs against a solve (issue
#262 split this out of one 1,529-line file). Each function appends PASS/FAIL lines to
the `report` it is given rather than raising - see verify_solution.py's verify() for how
these compose with the primal/dual checks that stay there.
"""
from __future__ import annotations

import math

def bound_contribution(multiplier: float, lower: float, upper: float) -> float:
    """The Lagrangian term a bound contributes to the dual objective, in minimize space.

    A positive multiplier prices the lower bound and a negative one the upper. A bound that
    does not exist contributes nothing: the multiplier is then a sign violation, judged in
    its own check, and its whole product with the primal value lands in the duality gap,
    where it is accounted for. An earlier version zeroed every multiplier below the dual
    tolerance, which put |multiplier| * bound into the gap for each column sitting exactly
    at a bound with a rounding-sized reduced cost - a gap manufactured by the test.
    """
    if multiplier > 0.0 and math.isfinite(lower):
        return multiplier * lower
    if multiplier < 0.0 and math.isfinite(upper):
        return multiplier * upper
    return 0.0


def transpose_times(model: Model, y: list[float]) -> list[float]:
    """A' y, from the column-wise entries. One line of arithmetic, written out."""
    d = [0.0] * model.num_cols
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            d[j] += value * y[i]
    return d


def times(model: Model, x: list[float]) -> list[float]:
    """A x, from the same column-wise entries."""
    out = [0.0] * model.num_rows
    for j in range(model.num_cols):
        if x[j] == 0.0:
            continue
        for i, value in model.entries[j]:
            out[i] += value * x[j]
    return out


def verify_iis(model: Model, solution: Solution, report: Report, primal_tol: float) -> None:
    """Check the two properties that make a set of constraints an IIS (#217).

    1. INFEASIBLE ON ITS OWN. The solver writes, as the farkas section, the certificate of
       the subsystem rather than of the whole model, so verify_farkas() has already proved
       that *some* aggregate of rows contradicts the column box. What remains is to check
       that the aggregate uses nothing outside the IIS: every row with a nonzero multiplier
       is an IIS row, and every column bound the aggregate leans on is an IIS bound.

    2. IRREDUCIBLE. For every element the solver writes a witness: a point that satisfies
       every other element and violates that one. Checking a witness is arithmetic - row
       activities against row bounds, values against column bounds - and needs no solver.

    The solver says `iis_irreducible not-claimed` in the header when one of its trial solves
    ended in neither verdict; then the set may be a superset of an IIS, no witnesses are
    written, and this function notes that rather than failing a claim that was never made.
    """
    if not solution.iis:
        return

    row_iis = [name for kind, name in solution.iis if kind == "row"]
    col_lo_iis = [name for kind, name in solution.iis if kind == "col_lo"]
    col_hi_iis = [name for kind, name in solution.iis if kind == "col_hi"]

    report.note("IIS",
                f"{len(solution.iis)} element(s): "
                f"{len(row_iis)} row(s), {len(col_lo_iis)} col_lo bound(s), "
                f"{len(col_hi_iis)} col_hi bound(s)")

    unknown_rows = [n for n in row_iis if n not in model.row_index]
    if not report.check(not unknown_rows, "IIS rows exist in model",
                        f"unknown row name(s): {unknown_rows}" if unknown_rows
                        else "every IIS row names a row of the model"):
        return
    unknown_cols = [n for n in col_lo_iis + col_hi_iis if n not in model.col_index]
    if not report.check(not unknown_cols, "IIS columns exist in model",
                        f"unknown column name(s): {unknown_cols}" if unknown_cols
                        else "every IIS bound names a column of the model"):
        return

    claimed = solution.header.get("iis_irreducible", "not-claimed")
    if claimed != "yes":
        report.note("IIS properties",
                    "not claimed by the solver (a trial solve was inconclusive), so the set "
                    "may be a superset of an IIS; neither property is checked")
        return

    rows = set(row_iis)
    lo = set(col_lo_iis)
    hi = set(col_hi_iis)

    # ---- 1. the certificate lives inside the IIS ------------------------------------------
    y = [solution.farkas.get(name, 0.0) for name in model.row_names]
    if any(y):
        outside_rows = [model.row_names[i] for i, m in enumerate(y)
                        if m != 0.0 and model.row_names[i] not in rows]
        d = transpose_times(model, y)
        term_scale = 1.0
        for j in range(model.num_cols):
            for i, value in model.entries[j]:
                term_scale = max(term_scale, abs(value * y[i]))
        zero = 1e-11 * term_scale  # the same zero as verify_farkas and the C++ checker
        outside_bounds = []
        for j in range(model.num_cols):
            if d[j] > zero and model.col_names[j] not in hi:
                outside_bounds.append(model.col_names[j] + " (upper)")
            elif d[j] < -zero and model.col_names[j] not in lo:
                outside_bounds.append(model.col_names[j] + " (lower)")
        report.check(not outside_rows and not outside_bounds, "IIS is infeasible on its own",
                     "the certificate's multipliers and the bounds its aggregate leans on "
                     "all belong to the IIS, so the proof above proves the subsystem alone"
                     if not outside_rows and not outside_bounds
                     else "the certificate reaches outside the IIS: rows "
                          + ", ".join(outside_rows[:5]) + "; bounds "
                          + ", ".join(outside_bounds[:5]))
    else:
        report.check(False, "IIS is infeasible on its own",
                     "no certificate in the file, so the subsystem's infeasibility is unproved")

    # ---- 2. every element is necessary: its witness satisfies all the others ---------------
    witnesses = {(kind, name): point for kind, name, point in solution.iis_witnesses}
    missing = [f"{kind} {name}" for kind, name in solution.iis if (kind, name) not in witnesses]
    if not report.check(not missing, "IIS witnesses present",
                        f"no witness for {len(missing)} element(s): " + ", ".join(missing[:5])
                        if missing else f"one witness for each of the {len(solution.iis)} elements"):
        return

    def violation(value: float, lower: float, upper: float) -> float:
        """Relative distance outside [lower, upper], zero inside."""
        below = lower - value if math.isfinite(lower) else 0.0
        above = value - upper if math.isfinite(upper) else 0.0
        worst = max(below, above, 0.0)
        return worst / max(1.0, abs(value),
                           abs(lower) if math.isfinite(lower) else 0.0,
                           abs(upper) if math.isfinite(upper) else 0.0)

    failures = []
    for kind, name in solution.iis:
        point = witnesses[(kind, name)]
        x = [point.get(col, 0.0) for col in model.col_names]
        activity = times(model, x)
        for other in row_iis:
            i = model.row_index[other]
            v = violation(activity[i], model.row_lower[i], model.row_upper[i])
            is_self = kind == "row" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates row {other} by {v:.3e}")
        for other in col_lo_iis:
            j = model.col_index[other]
            v = violation(x[j], model.col_lower[j], math.inf)
            is_self = kind == "col_lo" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates lower bound of {other} by {v:.3e}")
        for other in col_hi_iis:
            j = model.col_index[other]
            v = violation(x[j], -math.inf, model.col_upper[j])
            is_self = kind == "col_hi" and other == name
            if is_self and v <= primal_tol:
                failures.append(f"{kind} {name}: its witness satisfies it, so it is not needed")
            elif not is_self and v > primal_tol:
                failures.append(f"{kind} {name}: witness violates upper bound of {other} by {v:.3e}")
    report.check(not failures, "IIS is irreducible",
                 f"each of the {len(solution.iis)} witnesses satisfies the other "
                 f"{len(solution.iis) - 1} element(s) and violates its own"
                 if not failures else "; ".join(failures[:4]))


def verify_farkas(model: Model, solution: Solution, report: Report) -> Report:
    """Check a claim of INFEASIBILITY, in the only way a claim of infeasibility can be checked.

    There is no point to test - that is the whole content of the verdict - so a checker that
    asks for one is asking the wrong question. What CAN be handed over is a Farkas
    certificate: one multiplier per row. Aggregating the rows with those weights produces a
    single inequality that every feasible point would have to satisfy, and the certificate is
    good exactly when no point in the column box satisfies it.

    Written out, with rows `row_lower <= a_i.x <= row_upper` and columns in `[l, u]`:

        a multiplier y_i > 0 uses the row's LOWER bound   (a_i.x >= row_lower[i])
        a multiplier y_i < 0 uses the row's UPPER bound   (a_i.x <= row_upper[i])

    so every feasible x satisfies  d.x >= S  where  d = A'y  and  S = sum_i y_i * (that
    bound). The largest d.x can be over the box is M, taking each column to whichever of its
    own bounds the sign of d_j prefers. If M < S the system has no feasible point at all.

    Two ways a certificate can be bogus, both checked rather than assumed: a multiplier that
    leans on a bound the row does not have (infinite), and a column free in the direction d
    prefers, which makes M infinite and proves nothing.
    """
    y = [solution.farkas.get(name, 0.0) for name in model.row_names]
    if not any(y):
        # NOT a failure. Presolve proves infeasibility from bound arithmetic and does not
        # keep the chain of tightenings that would make a Farkas vector, so it says
        # `certificate none` and puts its reason in the message. Failing here would be the
        # very thing #191 exists to stop: this script calling a correct verdict wrong.
        report.note("infeasibility proof",
                    "no certificate offered, so nothing is claimed and nothing is checked; "
                    + (solution.header.get("message", "the solver gave no reason")))
        return report

    # A multiplier may only use a bound the row actually has.
    borrowed = [model.row_names[i] for i, m in enumerate(y)
                if (m > 0.0 and not math.isfinite(model.row_lower[i]))
                or (m < 0.0 and not math.isfinite(model.row_upper[i]))]
    if not report.check(not borrowed, "certificate uses only real bounds",
                        "every multiplier leans on a finite row bound" if not borrowed
                        else f"{len(borrowed)} lean on an infinite bound: "
                             + ", ".join(borrowed[:5])):
        return report

    required = sum(bound_contribution(y[i], model.row_lower[i], model.row_upper[i])
                   for i in range(model.num_rows))

    d = transpose_times(model, y)
    # A coefficient of the aggregate that is zero up to rounding is zero. The rows of the
    # crude-blend demo aggregate to exactly 0 on one column - two terms of 0.577 that
    # cancel - and floating point leaves 1e-17 behind; read as a sign, that "uses" a bound
    # the column does not have and rejects a correct certificate. The same rule the C++
    # checker applies (src/core/certificate.cpp): below 1e-11 of the largest term is zero.
    term_scale = 1.0
    for j in range(model.num_cols):
        for i, value in model.entries[j]:
            term_scale = max(term_scale, abs(value * y[i]))
    zero = 1e-11 * term_scale
    reachable = 0.0
    free = []
    for j in range(model.num_cols):
        if d[j] > zero:
            if not math.isfinite(model.col_upper[j]):
                free.append(model.col_names[j])
            else:
                reachable += d[j] * model.col_upper[j]
        elif d[j] < -zero:
            if not math.isfinite(model.col_lower[j]):
                free.append(model.col_names[j])
            else:
                reachable += d[j] * model.col_lower[j]
    if not report.check(not free, "aggregate is bounded above",
                        "every column the aggregate uses is bounded in that direction"
                        if not free
                        else f"{len(free)} unbounded in the direction used, so the aggregate "
                             f"proves nothing: " + ", ".join(free[:5])):
        return report

    # Strictly, and by more than the arithmetic could have invented.
    scale = max(1.0, abs(required), abs(reachable))
    report.check(reachable < required - 1e-9 * scale, "infeasibility proof",
                 f"the rows aggregate to at least {required:.12e}, the column bounds allow at "
                 f"most {reachable:.12e}, a contradiction of {required - reachable:.3e}")
    report.note("certificate size",
                f"{sum(1 for m in y if m != 0.0)} of {model.num_rows} rows carry a multiplier")
    return report


def verify_ray(model: Model, solution: Solution, report: Report, primal_tol: float) -> Report:
    """Check a claim of UNBOUNDEDNESS: a feasible point, and a direction that never stops.

    Unbounded is two claims, and a checker that tests one of them tests nothing. The point in
    the columns section must be feasible - checked by the ordinary primal checks, which run
    first - and the ray must satisfy, for every t >= 0, that x + t*d stays inside every bound
    while the objective improves without limit. That holds exactly when moving along d is
    blocked by nothing:

        (A d)_i > 0 needs the row to have NO upper bound, and < 0 no lower bound
        d_j     > 0 needs the column to have NO upper bound, and < 0 no lower bound

    and the objective strictly improves, c.d < 0 in minimize space. For a quadratic objective
    the ray must also not curve back up, d'Qd <= 0, or the improvement is only local.
    """
    if not solution.ray:
        report.note("unboundedness proof",
                    "no ray offered, so nothing is claimed and nothing is checked; "
                    + (solution.header.get("message", "the solver gave no reason")))
        return report
    d = [solution.ray.get(name, 0.0) for name in model.col_names]
    if not report.check(any(d), "ray is a direction",
                        f"{sum(1 for v in d if v != 0.0)} of {model.num_cols} columns move"
                        if any(d) else "the ray is all zeros, which is not a direction"):
        return report

    scale = max(abs(v) for v in d)
    moving = primal_tol * scale

    blocked_cols = [model.col_names[j] for j in range(model.num_cols)
                    if (d[j] > moving and math.isfinite(model.col_upper[j]))
                    or (d[j] < -moving and math.isfinite(model.col_lower[j]))]
    report.check(not blocked_cols, "ray respects the column bounds",
                 "no column bound blocks the ray" if not blocked_cols
                 else f"{len(blocked_cols)} would be crossed: " + ", ".join(blocked_cols[:5]))

    activity = times(model, d)
    blocked_rows = [model.row_names[i] for i in range(model.num_rows)
                    if (activity[i] > moving and math.isfinite(model.row_upper[i]))
                    or (activity[i] < -moving and math.isfinite(model.row_lower[i]))]
    report.check(not blocked_rows, "ray respects the row bounds",
                 "no row bound blocks the ray" if not blocked_rows
                 else f"{len(blocked_rows)} would be crossed: " + ", ".join(blocked_rows[:5]))

    sigma = -1.0 if model.maximize else 1.0
    improvement = sigma * sum(model.col_cost[j] * d[j] for j in range(model.num_cols))
    report.check(improvement < -1e-9 * max(1.0, abs(improvement)), "ray improves the objective",
                 f"objective changes by {improvement:.12e} per unit step, in minimize space")

    if model.hessian:
        qd = model.hessian_times(d)
        curvature = sum(d[j] * qd[j] for j in range(model.num_cols))
        report.check(curvature <= 1e-9 * max(1.0, abs(curvature)), "ray does not curve back",
                     f"d'Qd = {curvature:.12e}; a positive value means the improvement is "
                     f"only local")
    return report


# The verdicts that hand back a point, mirroring claims_a_point() in include/sankhya/model.hpp.
# The two lists are the .sol file's contract and have to agree; this script deliberately shares
# no code with the solver, so they are kept in step by saying so in both places rather than by
# a header. `unbounded` is here because since #191 it carries the feasible point its ray starts
# from - a ray from outside the feasible region proves nothing.
STATUSES_WITH_A_POINT = ("optimal", "feasible", "unbounded", "iteration_limit", "time_limit",
                         "node_limit", "interrupted")

# Of those, the ones that assert the point is FEASIBLE. The distinction is the whole of what
# a limit means: `optimal` and `feasible` say "here is a point inside the model", and a limit
# says only "here is where I stopped". An interior-point iterate stopped by the clock is not
# feasible and was never claimed to be - it approaches feasibility from outside - so holding
# it to a feasibility standard measures something nobody asserted. `unbounded` is here
# because its ray is only a proof if it starts somewhere the model allows.
#
# A limit is still checked, on the claim it DOES make: the solver reports its own
# primal_infeasibility in the header, and that number has to be true. Understating it is the
# failure worth catching, and it is the one a solver has an incentive to make.
STATUSES_ASSERTING_FEASIBILITY = ("optimal", "feasible", "unbounded")


def verify_pool(model: Model, solution: Solution, report: Report, x: list[float],
                objective: float, primal_tol: float, integer_tol: float) -> None:
    """The solution pool (#225), checked from what the file says and nothing else.

    By default the file carries only the INTEGER columns of each member, so what can be
    proved depends on the model. When every member is a full point - a pure-integer model, or
    a file written with pool_write_all_columns - each is checked the way the main solution
    is: bounds, integrality, every row, and the objective recomputed. With continuous columns
    missing, the integer part is checked exactly and each row is checked for whether the
    continuous columns' own bounds can still close it - a necessary condition, not a proof
    that one continuous completion satisfies every row at once, and the check says so rather
    than claiming more.
    """
    sigma = -1.0 if model.maximize else 1.0
    integer_names = [n for j, n in enumerate(model.col_names) if model.col_integer[j]]
    members = solution.pool
    ranks = [rank for rank, _, _ in members]
    all_names = set(model.col_names)
    full = bool(members) and all(set(values) == all_names for _, _, values in members)
    complete = full or all(set(values) == set(integer_names) for _, _, values in members)
    written = model.col_names if full else integer_names
    report.check(ranks == list(range(1, len(members) + 1)) and complete, "pool: structure",
                 (f"{len(members)} member(s), each listing all {len(written)} "
                  + ("columns" if full else "integer columns")) if complete
                 else f"{len(members)} member(s); ranks {ranks[:5]}, or a member listing "
                      "neither every integer column nor every column")
    if not complete:
        return

    first = members[0]
    worst_first = max((abs(first[2][n] - x[model.col_index[n]])
                       / max(1.0, abs(x[model.col_index[n]])) for n in written), default=0.0)
    scale = max(1.0, abs(objective))
    report.check(worst_first <= integer_tol and abs(first[1] - objective) <= 1e-9 * scale,
                 "pool: first member is the reported solution",
                 f"values differ by at most {worst_first:.3e} (relative), objective "
                 f"{first[1]:.12e} against {objective:.12e}")

    order_ok = all(sigma * members[k + 1][1] >= sigma * members[k][1]
                   - 1e-9 * max(1.0, abs(members[k][1])) for k in range(len(members) - 1))
    report.check(order_ok, "pool: best first",
                 "objectives " + ", ".join(f"{obj:.10g}" for _, obj, _ in members[:10]))

    keys = [tuple(round(values[n]) for n in integer_names) for _, _, values in members]
    report.check(len(set(keys)) == len(keys), "pool: distinct integer assignments",
                 f"{len(set(keys))} distinct of {len(keys)}")

    worst_integrality, worst_bound = 0.0, 0.0
    for _, _, values in members:
        for n in written:
            j = model.col_index[n]
            v = values[n]
            if model.col_integer[j]:
                worst_integrality = max(worst_integrality, abs(v - round(v)))
            worst_bound = max(worst_bound,
                              (model.col_lower[j] - v) / max(1.0, abs(v)),
                              (v - model.col_upper[j]) / max(1.0, abs(v)))
    report.check(worst_integrality <= integer_tol and worst_bound <= primal_tol,
                 "pool: integrality and column bounds",
                 f"worst integrality {worst_integrality:.3e}, worst bound violation "
                 f"{max(worst_bound, 0.0):.3e}")

    # Rows: the written part is fixed; each unwritten continuous column contributes an interval.
    continuous = [] if full else [j for j in range(model.num_cols) if not model.col_integer[j]]
    known = [full or model.col_integer[j] for j in range(model.num_cols)]
    by_row: list[list[tuple[int, float]]] = [[] for _ in range(model.num_rows)]
    for j in range(model.num_cols):
        for i, a in model.entries[j]:
            by_row[i].append((j, a))
    mixed_rows = sum(1 for i in range(model.num_rows)
                     if any(not known[j] for j, _ in by_row[i]))
    worst_row, where = 0.0, ""
    for rank, _, values in members:
        for i in range(model.num_rows):
            low = high = 0.0
            magnitude = 1.0
            for j, a in by_row[i]:
                if known[j]:
                    term = a * values[model.col_names[j]]
                    low += term
                    high += term
                    magnitude = max(magnitude, abs(term))
                else:
                    lo, hi = model.col_lower[j], model.col_upper[j]
                    low += a * lo if a > 0 else a * hi
                    high += a * hi if a > 0 else a * lo
            violation = max(model.row_lower[i] - high, low - model.row_upper[i], 0.0)
            if math.isnan(violation):
                continue
            scaled = violation / magnitude
            if scaled > worst_row:
                worst_row, where = scaled, f"{model.row_names[i]} in member {rank}"
    report.check(worst_row <= primal_tol, "pool: rows",
                 (f"exact on all {model.num_rows} rows ("
                  + ("every column written" if full else "no continuous columns") + ")"
                  if not continuous
                  else f"{model.num_rows - mixed_rows} row(s) exact; on {mixed_rows} row(s) with "
                       "continuous columns, only that their bounds can still close the row")
                 + f"; worst {worst_row:.3e}" + (f" on {where}" if where else ""))

    if continuous:
        report.note("pool: objectives",
                    "not recomputed: the continuous values are not written, by design "
                    "(pool_write_all_columns writes them)")
        return
    worst_objective = 0.0
    for _, claimed, values in members:
        point = [values[n] for n in model.col_names]
        recomputed = (model.objective_offset
                      + sum(model.col_cost[j] * point[j] for j in range(model.num_cols))
                      + model.quadratic_objective(point))
        worst_objective = max(worst_objective,
                              abs(recomputed - claimed) / max(1.0, abs(recomputed)))
    report.check(worst_objective <= 1e-9, "pool: objectives recomputed",
                 f"worst relative difference {worst_objective:.3e}")


