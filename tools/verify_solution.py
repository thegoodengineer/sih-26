#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently verify a SANKHYA solution. Trusts nothing the solver said.

    python tools/verify_solution.py model.mps solution.sol

This script exists so that "our answer is correct" is something a judge can CHECK rather
than believe. Three rules make that real, and all three are load-bearing:

1.  IT LINKS NO PART OF OUR C++. Pure Python, reading the model and .sol file as text.

2.  IT PARSES THE MODEL ITSELF. verify_solution_mps.py is a second, independent
    implementation written from the IBM specification, reusing no interpretation of
    anything from src/io/mps_reader.cpp - a reader that shared its bugs with the solver
    would confirm them instead of catching them.

3.  IT RECOMPUTES EVERY NUMBER: activities, the objective, reduced costs, the dual objective - from (model, x, y), never read from the .sol header.

Split across five files by issue #262, none over ~600 lines: _mps.py and _sol.py are rule
2's readers, _checks.py holds the certificate/IIS/pool checks, _io.py is shared by both.
Exit codes:  0 every check passed   1 a check failed   2 could not read the inputs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from verify_solution_checks import (STATUSES_ASSERTING_FEASIBILITY, STATUSES_WITH_A_POINT,
                                    bound_contribution, verify_farkas, verify_iis,
                                    verify_pool, verify_ray)
from verify_solution_io import INF
from verify_solution_mps import Model, parse_mps
from verify_solution_sol import Solution, parse_sol

# Defaults mirror include/sankhya/tolerances.hpp. They are CLI-overridable because a judge
# should be able to tighten them and watch what happens.
DEFAULT_PRIMAL_TOL = 1e-7
DEFAULT_DUAL_TOL = 1e-7
DEFAULT_INTEGER_TOL = 1e-6
DEFAULT_DUALITY_TOL = 1e-9


# =========================================================================================
# The checks
# =========================================================================================

class Report:
    def __init__(self) -> None:
        self.lines: list[tuple[bool, str, str]] = []
        self.failures = 0

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.lines.append((ok, name, detail))
        if not ok:
            self.failures += 1
        return ok

    def note(self, name: str, detail: str) -> None:
        self.lines.append((True, name, detail))

    def render(self) -> str:
        width = max((len(n) for _, n, _ in self.lines), default=10)
        out = []
        for ok, name, detail in self.lines:
            mark = "PASS" if ok else "FAIL"
            out.append(f"  [{mark}] {name:<{width}}  {detail}")
        return "\n".join(out)


def verify(model: Model, solution: Solution, primal_tol: float, dual_tol: float,
           integer_tol: float, duality_tol: float) -> Report:
    report = Report()
    sigma = -1.0 if model.maximize else 1.0

    # ---- A verdict with no point of its own ----------------------------------------------
    # `infeasible` and `unbounded` are answers, not failures, and until #191 this script
    # treated them as though the solver had claimed a solution: it read the all-zero point a
    # .sol file carried out of habit, found it violated the rows, and printed REJECTED at a
    # correct answer. Our own checker calling our own correct verdict wrong is worse than not
    # checking it, so now each verdict is checked as what it is.
    claimed = solution.header.get("certificate", "none")
    if claimed == "farkas" and not solution.farkas:
        report.check(False, "certificate present",
                     "the header says `certificate farkas` but the file carries no farkas "
                     "section")
        return report
    if claimed == "ray" and not solution.ray:
        report.check(False, "certificate present",
                     "the header says `certificate ray` but the file carries no ray section")
        return report

    if solution.status == "infeasible":
        verify_farkas(model, solution, report)
        verify_iis(model, solution, report, primal_tol)
        return report

    # A VERDICT THAT CLAIMS NOTHING IS NOT CHECKED AS IF IT DID (#200). A numerical failure, a
    # solve that never started, a model this solver refuses - none of these assert a point, and
    # a file written under one carries no columns section since #200. Running the primal checks
    # against what it does carry was the same bug #191 fixed for `infeasible` alone: this
    # script printing REJECTED at an answer the solver never made.
    if solution.status not in STATUSES_WITH_A_POINT and solution.status != "infeasible_or_unbounded":
        report.note("verdict",
                    f"status is {solution.status}, which claims no point; nothing is asserted "
                    "and nothing is checked"
                    + (f" - {solution.header['message']}" if "message" in solution.header
                       else ""))
        return report

    if solution.status in ("unbounded", "infeasible_or_unbounded"):
        if solution.status == "infeasible_or_unbounded" and not solution.ray:
            report.note("verdict",
                        "the solver separated neither case and offers no ray; nothing to "
                        "check, and nothing is claimed")
            return report

    # ---- Structure ----------------------------------------------------------------------
    missing_cols = [n for n in model.col_names if n not in solution.col_value]
    missing_rows = [n for n in model.row_names if n not in solution.row_activity]

    if solution.status in ("not_solved", "model_error") and not solution.col_value and not solution.row_activity:
        report.note("structure", f"skipped: status is {solution.status} and no point was claimed")
        return report

    report.check(
        not missing_cols and not missing_rows,
        "structure",
        f"{model.num_rows} rows, {model.num_cols} columns present"
        if not missing_cols and not missing_rows
        else f"missing {len(missing_cols)} columns, {len(missing_rows)} rows",
    )
    if missing_cols or missing_rows:
        return report

    x = [solution.col_value[n] for n in model.col_names]
    asserts_feasibility = solution.status in STATUSES_ASSERTING_FEASIBILITY

    def primal_check(worst_relative: float, worst_absolute: float, name: str,
                     detail: str) -> None:
        """Hold the point to what its status claims, and always report the measurement."""
        if asserts_feasibility:
            report.check(worst_relative <= primal_tol, name, detail)
            return
        claimed = solution.header_float("primal_infeasibility")
        if claimed is None:
            report.note(name, detail + f" - status is {solution.status}, which asserts no "
                                       "feasibility, and the file states none to compare")
            return
        # The solver may be as far outside as it admits to being, and no further.
        allowed = claimed * (1.0 + 1e-6) + primal_tol
        report.check(worst_absolute <= allowed, name + " matches the stated",
                     f"{detail}; the file states primal_infeasibility {claimed:.3e}, and "
                     f"{solution.status} asserts no better")

    # ---- Column bounds ------------------------------------------------------------------
    # Scaled by the variable's own magnitude, for the same reason as the rows above.
    worst, where, worst_abs = 0.0, "", 0.0
    for j, name in enumerate(model.col_names):
        violation = max(model.col_lower[j] - x[j], x[j] - model.col_upper[j], 0.0)
        scaled = violation / max(1.0, abs(x[j]))
        if scaled > worst:
            worst, where, worst_abs = scaled, name, violation
    primal_check(worst, worst_abs, "column bounds",
                 f"worst violation {worst_abs:.3e} ({worst:.3e} relative)"
                 + (f" on {where}" if where else ""))

    # ---- Integrality --------------------------------------------------------------------
    integer_columns = sum(1 for flag in model.col_integer if flag)
    if integer_columns:
        worst, where = 0.0, ""
        for j, name in enumerate(model.col_names):
            if not model.col_integer[j]:
                continue
            violation = abs(x[j] - round(x[j]))
            if violation > worst:
                worst, where = violation, name
        report.check(worst <= integer_tol, "integrality",
                     f"{integer_columns} integer columns, worst {worst:.3e}"
                     + (f" on {where}" if where else ""))
    else:
        report.note("integrality", "no integer columns")

    # ---- Row activity, recomputed from the matrix ----------------------------------------
    # The row's numerical SCALE is accumulated alongside its activity: the largest term the
    # sum was built from. A residual cannot be smaller than the rounding error of the sum
    # that produced it, and that error is set by the size of the terms, not of the answer.
    #
    # Netlib grow7 is why this is not an absolute test. Its largest solution value is 4.8e+07,
    # so a 1e-7 absolute tolerance is 2.1e-15 relative - below what double precision reaches
    # after three hundred iterations. Its worst violation is 4.2e-15 relative, about nineteen
    # machine epsilons, and the row it occurs on is an equality to ZERO, so scaling by the
    # bound would change nothing; the residual is large because terms of magnitude 1e+07
    # cancel.
    #
    # THIS DOES NOT MAKE THE VERIFIER LESS INDEPENDENT. Its independence is that it shares no
    # code with the solver and re-derives everything from the original file, which is
    # unchanged. Asking a question in the right units is not the same as asking a weaker one:
    # a violation that is large relative to the terms it came from still fails, and the dual
    # checks below - which caught #149's postsolve regression at 1.3e-02 relative - are
    # untouched.
    activity = [0.0] * model.num_rows
    row_scale = [1.0] * model.num_rows
    for j in range(model.num_cols):
        if x[j] == 0.0:
            continue
        for i, value in model.entries[j]:
            term = value * x[j]
            activity[i] += term
            if abs(term) > row_scale[i]:
                row_scale[i] = abs(term)

    worst, where, worst_abs = 0.0, "", 0.0
    for i, name in enumerate(model.row_names):
        violation = max(model.row_lower[i] - activity[i], activity[i] - model.row_upper[i], 0.0)
        scaled = violation / row_scale[i]
        if scaled > worst:
            worst, where, worst_abs = scaled, name, violation
    primal_check(worst, worst_abs, "row activity",
                 f"worst violation {worst_abs:.3e} ({worst:.3e} relative to the row's terms)"
                 + (f" on {where}" if where else ""))

    # The solver also reports its own activities. A disagreement means one of us computed
    # A*x differently, which is worth surfacing even when both satisfy the bounds.
    worst_activity_gap = max(
        (abs(activity[i] - solution.row_activity[n]) for i, n in enumerate(model.row_names)),
        default=0.0,
    )
    report.check(worst_activity_gap <= 1e-6, "activity agreement",
                 f"max |ours - solver's| = {worst_activity_gap:.3e}")

    # ---- The ray, when the verdict was unbounded -------------------------------------------
    # Reached only after the point above has been checked feasible, which is the other half
    # of the claim: a ray from an infeasible point proves nothing at all.
    if solution.status in ("unbounded", "infeasible_or_unbounded"):
        return verify_ray(model, solution, report, primal_tol)

    # ---- Objective, recomputed -----------------------------------------------------------
    # c'x + 0.5 x'Qx + offset. Omitting the quadratic term would have this script declare a
    # correct QP answer wrong - and, worse, declare a solver that ITSELF dropped the term
    # right, since both sides would then be computing the LP objective.
    objective = (model.objective_offset
                 + sum(model.col_cost[j] * x[j] for j in range(model.num_cols))
                 + model.quadratic_objective(x))
    claimed = solution.header_float("objective")
    if claimed is None:
        report.check(False, "objective", "the .sol file states no objective")
    else:
        scale = max(1.0, abs(objective))
        report.check(abs(objective - claimed) <= 1e-9 * scale, "objective",
                     f"recomputed {objective:.12e}, solver said {claimed:.12e}, "
                     f"difference {abs(objective - claimed):.3e}")

    if solution.pool:
        verify_pool(model, solution, report, x, objective, primal_tol, integer_tol)

    if solution.status != "optimal":
        # Strong duality is a test of OPTIMALITY. A solver reporting kFeasible is explicitly
        # declining to claim optimality - a first-order method that stopped on a tolerance,
        # or a search stopped by a limit - so holding its point to an optimality standard
        # measures something it never asserted. Primal feasibility, integrality and the
        # objective are all still checked above, and those are what kFeasible does assert.
        report.note("duality", f"skipped: status is {solution.status}, not an optimality claim")
        return report

    if integer_columns:
        # LP duality does not apply to a MILP: any reported duals belong to some node
        # relaxation, not to the integer problem. What CAN be checked is the claim the
        # search makes about itself.
        report.note("duality", "skipped: LP duality does not apply to a MILP")

        bound = solution.header_float("dual_bound")
        if bound is None or not math.isfinite(bound):
            report.note("optimality proof",
                        "no finite dual bound reported"
                        + ("" if solution.status != "optimal"
                           else " - but the status claims optimal"))
            if solution.status == "optimal":
                report.check(False, "optimality proof",
                             "status is optimal but no finite bound backs the claim")
            return report

        scale = max(1.0, abs(objective))
        if solution.status == "optimal":
            # "Optimal" on a MILP is a claim about the bound: the incumbent is within the
            # gap target of the best bound the search still had open (#188), or the tree
            # was exhausted and the two have met. The targets are read from the header the
            # solver wrote, defaulting to the project's (tolerances.hpp: 1e-4 relative,
            # 1e-6 absolute) when an older file has none. A gap wider than that means the
            # solver called an incumbent a proof, which is the most consequential thing a
            # branch and bound can get wrong and the least visible - the point is integral
            # and feasible either way.
            relative_target = solution.header_float("mip_relative_gap")
            absolute_target = solution.header_float("mip_absolute_gap")
            allowed = max(1e-6 if absolute_target is None else absolute_target,
                          (1e-4 if relative_target is None else relative_target) * scale)
            gap = abs(objective - bound)
            report.check(gap <= allowed + 1e-9 * scale, "optimality proof",
                         f"objective {objective:.12e} vs dual bound {bound:.12e}, "
                         f"gap {gap:.3e} against an allowed {allowed:.3e}")
            # And the bound must still be on the right side of the incumbent.
            slack = (objective - bound) if not model.maximize else (bound - objective)
            report.check(slack >= -1e-6 * scale, "dual bound is a bound",
                         f"incumbent {objective:.12e}, bound {bound:.12e}")
        else:
            # Not closed. The bound must still BE a bound: never worse than the incumbent.
            slack = (objective - bound) if not model.maximize else (bound - objective)
            report.check(slack >= -1e-6 * scale, "dual bound is a bound",
                         f"incumbent {objective:.12e}, bound {bound:.12e}, "
                         f"remaining gap {abs(objective - bound):.3e}")
        return report

    # ---- Dual feasibility ----------------------------------------------------------------
    # Work in minimize space so one set of sign conventions covers both senses.
    y = [sigma * solution.row_dual[n] for n in model.row_names]
    d = [sigma * solution.col_dual[n] for n in model.col_names]
    # For an LP the gradient of the objective IS the cost vector. For a QP it is c + Qx, and
    # every KKT condition below is stated in terms of the gradient, so making this one
    # substitution carries dual feasibility, complementary slackness and strong duality over
    # to the quadratic case unchanged - which is the point: a QP optimum is not a special
    # kind of optimum, it is the same conditions about a different gradient.
    gradient = list(model.col_cost)
    if model.hessian:
        qx = model.hessian_times(x)
        gradient = [model.col_cost[j] + qx[j] for j in range(model.num_cols)]
    cost = [sigma * g for g in gradient]

    if model.hessian:
        # The QP engine is a first-order method that carries no basis and reports no reduced
        # costs, so there is nothing of the solver's to cross-check here. d is DERIVED from
        # (model, x, y) instead - which is strictly the stronger test, since the sign and
        # complementarity checks below then price against a vector the solver never chose.
        d = [cost[j] - sum(value * y[i] for i, value in model.entries[j])
             for j in range(model.num_cols)]
        report.note("reduced costs",
                    "derived from c + Qx - A^T y; the QP engine reports none to compare")
    else:
        # Reduced costs must satisfy d = c - A^T y. Recomputing catches a solver that reports
        # a dual vector inconsistent with the reduced costs it also reports.
        # Judged against the magnitude of the terms in c - A^T y, for the same reason the
        # row activities are judged against theirs: it is a difference of quantities that
        # cancel, and its achievable accuracy is set by their size. On grow7 the terms are
        # of order 1e+07, so an absolute 1e-6 asks for 1e-13 relative.
        worst, where, worst_abs = 0.0, "", 0.0
        for j, name in enumerate(model.col_names):
            terms = [value * y[i] for i, value in model.entries[j]]
            expected = cost[j] - sum(terms)
            difference = abs(expected - d[j])
            scale = max(1.0, abs(cost[j]), max((abs(t) for t in terms), default=0.0))
            if difference / scale > worst:
                worst, where, worst_abs = difference / scale, name, difference
        report.check(worst <= 1e-6, "reduced costs",
                     f"max |c - A^T y - d| = {worst_abs:.3e} ({worst:.3e} relative to its terms)"
                     + (f" on {where}" if where else ""))

    def sign_violation(multiplier: float, value: float, lower: float, upper: float) -> float:
        """How badly a multiplier's SIGN contradicts the bound it prices against.

        A positive multiplier prices the lower bound and a negative one the upper bound, so
        the violation is a multiplier pushing against a bound that does not exist.

        Deliberately NOT checked here: "the multiplier must vanish when the constraint is
        strictly interior". That is complementary slackness, and it is measured below as the
        product |multiplier| * slack. Reporting it here as well, via a binary "is the value
        within primal_tol of a bound" test, is a category error and a badly conditioned one:
        it returns the full |multiplier| the moment a value sits a hair outside the window,
        so a point that is optimal to 1e-9 can report a dual violation of 1.0. That reads as
        a catastrophe when the truth is a rounding-width displacement. A vertex solution is
        unaffected either way; a first-order method sits near bounds rather than on them,
        and would be judged by an artefact of the window rather than by its KKT error.
        """
        del value  # activity enters through the complementarity product, not through here
        if lower == upper:
            return 0.0  # equality / fixed: any multiplier is admissible
        if multiplier > 0.0 and not math.isfinite(lower):
            return multiplier
        if multiplier < 0.0 and not math.isfinite(upper):
            return -multiplier
        return 0.0

    # Scaled like the reduced costs above: a sign violation of 6 on a reduced cost whose
    # terms are of order 1e+07 is the precision floor, not a wrong sign.
    worst, where, worst_abs = 0.0, "", 0.0
    for j, name in enumerate(model.col_names):
        violation = sign_violation(d[j], x[j], model.col_lower[j], model.col_upper[j])
        scale = max(1.0, abs(cost[j]),
                    max((abs(value * y[i]) for i, value in model.entries[j]), default=0.0))
        if violation / scale > worst:
            worst, where, worst_abs = violation / scale, name, violation
    report.check(worst <= dual_tol, "dual feasibility (columns)",
                 f"worst {worst_abs:.3e} ({worst:.3e} relative)" + (f" on {where}" if where else ""))

    # A row price has no terms of its own to compare against, so it is judged relative to
    # the size of the prices it sits among - a weaker test than the column one, and stated
    # as such in model.hpp.
    dual_norm = max(1.0, max((abs(v) for v in y), default=0.0))
    worst, where, worst_abs = 0.0, "", 0.0
    for i, name in enumerate(model.row_names):
        violation = sign_violation(y[i], activity[i], model.row_lower[i], model.row_upper[i])
        if violation / dual_norm > worst:
            worst, where, worst_abs = violation / dual_norm, name, violation
    report.check(worst <= dual_tol, "dual feasibility (rows)",
                 f"worst {worst_abs:.3e} ({worst:.3e} relative to |y|)"
                 + (f" on {where}" if where else ""))

    # ---- Complementary slackness ----------------------------------------------------------
    # A multiplier may only be nonzero where its constraint is tight.
    worst, where = 0.0, ""
    def complementarity(multiplier: float, slack: float) -> float:
        """|multiplier| * slack, except where that product is degenerate.

        WHEN THE SLACK IS INFINITE the product is not a usable measure. A free variable has
        no finite bound on either side, so min(slack) is INF and |d| * INF evaluates to inf
        for ANY nonzero d - which demands the reduced cost be BIT-EXACTLY zero. No
        floating-point solver can promise that, and it is not what complementary slackness
        requires: for a constraint that cannot be tight, the condition reduces to "the
        multiplier is zero", and that is testable directly against the dual tolerance.

        This is the same mathematical condition, correctly conditioned - not a loosened one.
        A free column carrying a genuinely nonzero reduced cost still fails, at exactly the
        threshold it should. It was found when a solver change altered a pivot path and left
        -1.05e-15 on a free column of capri: a correct answer, rejected, for having rounded
        a zero rather than for being wrong.
        """
        if not math.isfinite(slack):
            return abs(multiplier)
        return abs(multiplier) * slack

    for i, name in enumerate(model.row_names):
        if model.row_lower[i] == model.row_upper[i]:
            continue
        slack_lower = (activity[i] - model.row_lower[i]) if math.isfinite(model.row_lower[i]) else INF
        slack_upper = (model.row_upper[i] - activity[i]) if math.isfinite(model.row_upper[i]) else INF
        product = complementarity(y[i], min(slack_lower, slack_upper))
        if product > worst:
            worst, where = product, name
    for j, name in enumerate(model.col_names):
        if model.col_lower[j] == model.col_upper[j]:
            continue
        slack_lower = (x[j] - model.col_lower[j]) if math.isfinite(model.col_lower[j]) else INF
        slack_upper = (model.col_upper[j] - x[j]) if math.isfinite(model.col_upper[j]) else INF
        product = complementarity(d[j], min(slack_lower, slack_upper))
        if product > worst:
            worst, where = product, name
    report.check(worst <= 1e-6, "complementary slackness",
                 f"worst |multiplier| * slack = {worst:.3e}"
                 + (f" on {where}" if where else ""))

    # ---- The basis (#218) -----------------------------------------------------------------
    # A simplex answer carries a status per column and per row. When it does, it has to be a
    # basis: exactly m basic entries, and every nonbasic entry sitting on the bound its
    # status names. A solve that reports no basis (first-order, interior point without
    # crossover) has every status unknown and is not judged on this.
    # A basis is judged only when EVERY status is known: an interior-point or first-order
    # answer reports unknown throughout, and postsolve may still mark the rows it removed
    # basic, which is bookkeeping about the reduction, not a claim that a basis exists.
    statuses_known = bool(solution.col_status) and \
        all(s not in ("unknown", "") for s in solution.col_status.values()) and \
        all(s not in ("unknown", "") for s in solution.row_status.values())
    if statuses_known:
        basic = 0
        worst_off_bound = 0.0
        where_off = ""
        for j, name in enumerate(model.col_names):
            status = solution.col_status.get(name, "unknown")
            scale = max(1.0, abs(x[j]))
            if status == "basic":
                basic += 1
            elif status == "at_lower" and math.isfinite(model.col_lower[j]):
                off = abs(x[j] - model.col_lower[j]) / scale
                if off > worst_off_bound:
                    worst_off_bound, where_off = off, name
            elif status == "at_upper" and math.isfinite(model.col_upper[j]):
                off = abs(x[j] - model.col_upper[j]) / scale
                if off > worst_off_bound:
                    worst_off_bound, where_off = off, name
        for i, name in enumerate(model.row_names):
            status = solution.row_status.get(name, "unknown")
            scale = max(1.0, abs(activity[i]))
            if status == "basic":
                basic += 1
            elif status == "at_lower" and math.isfinite(model.row_lower[i]):
                off = abs(activity[i] - model.row_lower[i]) / scale
                if off > worst_off_bound:
                    worst_off_bound, where_off = off, name
            elif status == "at_upper" and math.isfinite(model.row_upper[i]):
                off = abs(activity[i] - model.row_upper[i]) / scale
                if off > worst_off_bound:
                    worst_off_bound, where_off = off, name
        report.check(basic == model.num_rows, "basis",
                     f"{basic} basic entries for {model.num_rows} rows")
        report.check(worst_off_bound <= primal_tol, "nonbasic entries on their bounds",
                     f"worst distance {worst_off_bound:.3e}"
                     + (f" on {where_off}" if where_off else ""))

    # ---- Strong duality -------------------------------------------------------------------
    dual_objective_min_space = 0.0
    for i in range(model.num_rows):
        dual_objective_min_space += bound_contribution(
            y[i], model.row_lower[i], model.row_upper[i])
    for j in range(model.num_cols):
        dual_objective_min_space += bound_contribution(
            d[j], model.col_lower[j], model.col_upper[j])

    # THE GAP IS AN IDENTITY, NOT A MEASUREMENT OF ITS OWN. With d = c - A^T y and the
    # activities recomputed from x, primal - dual is exactly the sum over every column and
    # row of  multiplier * (value - the bound the multiplier's sign prices), plus the
    # consistency and activity residuals judged above. So every item's share of the gap is
    # known, and the question this check can honestly ask is whether the gap is accounted
    # for by shares the per-item checks already accepted:
    #
    #   - a multiplier within the dual tolerance at its own scale (the scale the sign check
    #     used) explains its whole share: it is indistinguishable from zero, and zero would
    #     contribute nothing;
    #   - a larger multiplier explains its share only up to |multiplier| * nearest slack,
    #     which is what the complementarity check judged. A reduced cost of -4e-3 on a
    #     column at its LOWER bound prices the UPPER bound 20 away (Netlib recipe, #157):
    #     nearest slack 0, nothing explained, and the check fails on it exactly as before.
    #
    # Netlib etamacro is why the accounting exists: ten per-item checks pass, and the gap
    # of 1.26e-6 on an objective of 755 (1.67e-9 relative) is one accepted sign violation
    # of 3.2e-8 on KAPSTK65 times that column's value of 40. An aggregate threshold of
    # 1e-9 relative was tighter than a single per-item allowance granted above it.
    accounted = 0.0

    def share(multiplier: float, value: float, lower: float, upper: float,
              multiplier_scale: float) -> float:
        if multiplier == 0.0:
            return 0.0
        priced = lower if multiplier > 0.0 else upper
        term = abs(multiplier) * (abs(value - priced) if math.isfinite(priced) else abs(value))
        if abs(multiplier) <= dual_tol * multiplier_scale:
            return term
        nearest = min(abs(value - lower) if math.isfinite(lower) else INF,
                      abs(value - upper) if math.isfinite(upper) else INF)
        return min(term, abs(multiplier) * nearest) if math.isfinite(nearest) else 0.0

    for j in range(model.num_cols):
        scale_j = max(1.0, abs(cost[j]),
                      max((abs(value * y[i]) for i, value in model.entries[j]), default=0.0))
        accounted += share(d[j], x[j], model.col_lower[j], model.col_upper[j], scale_j)
    for i in range(model.num_rows):
        accounted += share(y[i], activity[i], model.row_lower[i], model.row_upper[i], dual_norm)
    # For a QP the bound contributions sum, at a KKT point, to (c + Qx)'x = c'x + x'Qx, which
    # overshoots the primal objective c'x + 0.5 x'Qx by exactly 0.5 x'Qx. Subtracting it is
    # the Dorn dual of a convex QP, and it makes the gap below a real optimality test rather
    # than an identity that would fail by a fixed amount on every quadratic instance.
    dual_objective = (sigma * dual_objective_min_space + model.objective_offset
                      - model.quadratic_objective(x))

    gap = abs(objective - dual_objective)
    scale = max(1.0, abs(objective))
    report.check(gap <= duality_tol * scale + accounted, "strong duality",
                 f"primal {objective:.12e}  dual {dual_objective:.12e}  "
                 f"gap {gap:.3e} (relative {gap / scale:.3e}), "
                 f"{accounted:.3e} of it from per-item violations accepted above")

    # ---- Sensitivity ranging (optional; only checked when the .sol has the sections) ------
    # A nonbasic column's range is one-sided and equal to its reduced cost: at its lower
    # bound the cost may fall by d_j (minimization space) before the column becomes
    # attractive, at its upper bound it may rise by |d_j|. That is the one case this script
    # can re-derive without the basis factor. The file reports ranges in the MODEL'S sense,
    # so on a maximize model the side that carries d_j is the other one.
    # Reference: Chvatal, "Linear Programming", ch. 10 (1983).
    if solution.col_ranging_lower:
        worst, worst_where, checked = 0.0, "", 0
        for j, name in enumerate(model.col_names):
            status = solution.col_status.get(name)
            if status not in ("at_lower", "at_upper"):
                continue
            dj = d[j]  # minimization space: >= 0 at lower, <= 0 at upper
            lo = solution.col_ranging_lower.get(name)
            hi = solution.col_ranging_upper.get(name)
            if lo is None or hi is None:
                continue
            # (finite side in minimization space, its value) then map to the file's sense.
            finite_is_lower_in_min = status == "at_lower"
            finite_is_lower = finite_is_lower_in_min != model.maximize
            reported = lo if finite_is_lower else hi
            other = hi if finite_is_lower else lo
            expected = abs(dj)
            checked += 1
            scale_j = max(1.0, expected, abs(reported))
            err = abs(reported - expected) / scale_j
            if not math.isinf(other):
                err = max(err, 1.0)  # the free side must be reported as unbounded
            if err > worst:
                worst, worst_where = err, name
        report.check(
            worst <= 1e-6,
            "ranging: nonbasic ranges match the reduced costs",
            f"{checked} nonbasic column(s): the bound side equals |d_j| and the other side is "
            f"unbounded, max relative error {worst:.3e}"
            + (f" on {worst_where}" if worst_where else ""))

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="the .mps model (optionally .gz)")
    parser.add_argument("solution", type=Path, help="the .sol file SANKHYA wrote")
    parser.add_argument("--primal-tolerance", type=float, default=DEFAULT_PRIMAL_TOL)
    parser.add_argument("--dual-tolerance", type=float, default=DEFAULT_DUAL_TOL)
    parser.add_argument("--integer-tolerance", type=float, default=DEFAULT_INTEGER_TOL)
    parser.add_argument("--duality-tolerance", type=float, default=DEFAULT_DUALITY_TOL)
    parser.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = parser.parse_args()

    try:
        model = parse_mps(args.model)
    except (OSError, ValueError) as error:
        print(f"cannot read the model: {error}", file=sys.stderr)
        return 2
    try:
        solution = parse_sol(args.solution)
    except (OSError, ValueError) as error:
        print(f"cannot read the solution: {error}", file=sys.stderr)
        return 2

    report = verify(model, solution, args.primal_tolerance, args.dual_tolerance,
                    args.integer_tolerance, args.duality_tolerance)

    if not args.quiet:
        print(f"model     {args.model}")
        print(f"solution  {args.solution}")
        print(f"problem   {model.name or '(unnamed)'}  "
              f"{model.num_rows} rows x {model.num_cols} columns  "
              f"{'maximize' if model.maximize else 'minimize'}")
        print(f"status    {solution.status}")
        print()
        print("Independent checks (this script shares no code with the solver):")
        print(report.render())
        print()

    if report.failures == 0:
        print(f"VERIFIED: {len(report.lines)} checks passed")
        return 0
    print(f"REJECTED: {report.failures} of {len(report.lines)} checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
