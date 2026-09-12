// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solve() dispatcher.
//
// THIS FILE IS THE SEAM. Every engine registers here and nowhere else:
//   Phase 2  primal revised simplex  -> LP            [registered]
//   Phase 4  restarted PDHG          -> LP, large and sparse
//   Phase 5  branch and cut          -> MILP
//   Phase 8  Mehrotra IPM, convex QP -> LP and QP
// A class with no engine returns kNotSolved and says so. Per CLAUDE.md an unimplemented
// path reports the truth rather than a plausible zero - and in particular a MILP is NOT
// quietly handed to the simplex and its fractional relaxation reported as optimal, which
// is the single most damaging thing this dispatcher could do.

#include <algorithm>
#include <cmath>
#include <string>

#include <fmt/format.h>

#include "core/status_guard.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/certificate.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/timer.hpp"
#include "util/threads.hpp"

#include "../simplex/primal_simplex.hpp"

namespace sankhya {
namespace {

/// The problem class, decided from the model rather than from a user assertion.
enum class ProblemClass { kLp, kMilp, kQp, kMiqp };

ProblemClass classify(const Model& model) {
  const bool integral = model.has_integrality();
  const bool quadratic = model.has_quadratic_objective();
  if (integral && quadratic) return ProblemClass::kMiqp;
  if (integral) return ProblemClass::kMilp;
  if (quadratic) return ProblemClass::kQp;
  return ProblemClass::kLp;
}

const char* class_name(ProblemClass c) {
  switch (c) {
    case ProblemClass::kLp: return "LP";
    case ProblemClass::kMilp: return "MILP";
    case ProblemClass::kQp: return "QP";
    case ProblemClass::kMiqp: return "MIQP";
  }
  return "unknown";
}

}  // namespace

/// Force the reported status to agree with the measured quality of the point.
///
/// Every engine calls Solution::recompute_quality() before returning, so primal_infeasibility
/// and dual_infeasibility are MEASURED from the returned vectors rather than asserted by the
/// engine about itself. Nothing was checking that the status agreed with them, and the two
/// drifted apart: PDHG terminates on a RELATIVE KKT criterion at pdhg_tolerance (1e-4 by
/// default), and that was being translated straight into kOptimal. A relative KKT residual of
/// 1e-4 is not the same claim as "primal feasible to 1e-7", and on all eight Netlib instances
/// the gap between those two statements was three to five orders of magnitude. sc50b, whose
/// published optimum is exactly -70, was returned as -70.0139 and labelled optimal - a value
/// better than the optimum, which is only reachable from outside the feasible region.
///
/// The check lives here, in the dispatcher, rather than inside any one engine. It is a
/// property of the Solution contract, not of an algorithm, so the interior-point and QP
/// engines inherit it in Phase 8 instead of having to re-derive it. An engine remains free to
/// report kFeasible, kIterationLimit or anything else; what it cannot do is claim a proof
/// whose evidence is on the same object and disagrees.
///
/// Statuses that make no claim about the point are left alone.
///
/// `check_dual` is false for a MILP. A branch-and-bound incumbent is produced by a NODE LP
/// whose bounds were tightened by branching, so its reduced costs are dual feasible for that
/// node and generally are not for the original model. Optimality of a MILP is proved by the
/// bound closing against the incumbent, not by the reduced costs of the last LP solved, so
/// applying the dual test there would reject correct answers. Integrality is checked instead:
/// it is the condition that actually distinguishes a MILP solution from its relaxation.
/// Never publish an answer whose numbers are not numbers (#194).
///
/// This is the belt to the interior-point method's braces, and it is deliberately engine
/// agnostic: any solve that claims a point and then reports a non-finite objective is
/// reporting something no consumer can use and no reader can check. The benchmark runners
/// write whatever comes back into a CSV, so a NaN here does not stay here - it becomes an
/// entry in the project's evidence, in the column that exists to say whether the answer was
/// right. Downgrading is the honest outcome: the solve failed numerically, and saying so is
/// worth more than a plausible-looking row.
void refuse_a_non_finite_answer(Solution* solution, Logger& logger) {
  const bool claims_a_point = solution->status == SolveStatus::kOptimal ||
                              solution->status == SolveStatus::kFeasible ||
                              solution->status == SolveStatus::kIterationLimit ||
                              solution->status == SolveStatus::kTimeLimit;
  if (!claims_a_point) return;

  const bool finite = std::isfinite(solution->objective) &&
                      std::all_of(solution->col_value.begin(), solution->col_value.end(),
                                  [](double v) { return std::isfinite(v); });
  if (finite) return;

  logger.warning(
      "the solve returned a non-finite answer under status {}; reporting it as a numerical "
      "failure rather than as a point",
      to_string(solution->status));
  solution->message +=
      "; the answer contained a value that is not a number, so it is reported as a numerical "
      "failure rather than as a point (#194)";
  solution->status = SolveStatus::kNumericalError;
  solution->col_value.clear();
  solution->row_activity.clear();
  solution->objective = 0.0;
  solution->dual_bound = 0.0;
}

/// Keep a certificate only if it proves what the status claims, against the ORIGINAL model.
///
/// The engines compute these on a scaled model, under perturbed bounds, from factors that may
/// have drifted, and a Farkas vector's SIGN depends on which bound the leaving variable
/// crossed. Rather than derive the convention and hope, both signs are tried and the proof is
/// checked here; a candidate that does not prove the claim is dropped and the message says
/// so. An unproven certificate published as a proof would be worse than the empty field this
/// project already uses to mean "no proof was produced" (#191).
void keep_only_a_proved_certificate(Solution* solution, const Model& model, Logger& logger) {
  std::string why;
  if (solution->status == SolveStatus::kInfeasible && !solution->farkas_dual.empty()) {
    if (farkas_proves_infeasible(model, solution->farkas_dual, &why)) {
      solution->message += fmt::format("; proof: {}", why);
      return;
    }
    std::vector<double> flipped = solution->farkas_dual;
    for (double& value : flipped) value = -value;
    if (farkas_proves_infeasible(model, flipped, &why)) {
      solution->farkas_dual = std::move(flipped);
      solution->message += fmt::format("; proof: {}", why);
      return;
    }
    logger.verbose("the infeasibility certificate did not check out and was dropped: {}", why);
    solution->farkas_dual.clear();
    solution->message += "; no machine-checkable certificate accompanies this verdict";
    return;
  }
  if (solution->status == SolveStatus::kUnbounded && !solution->primal_ray.empty()) {
    if (ray_proves_unbounded(model, solution->primal_ray, &why)) {
      solution->message += fmt::format("; proof: {}", why);
      return;
    }
    logger.verbose("the unboundedness ray did not check out and was dropped: {}", why);
    solution->primal_ray.clear();
    solution->message += "; no machine-checkable certificate accompanies this verdict";
  }
}

/// PDHG's answer, finished by the interior point (#229).
///
/// A first-order method converges linearly, with a rate that flattens as the iterate nears
/// the optimum and a floor set by floating-point noise in the step: on the scale families it
/// stops between 1e-5 and 1e-7 relative, and a million iterations do not move it (#198). A
/// second-order method started from that point converges quadratically. So when PDHG stops
/// short of the standard - at a limit, or at its own tolerance without meeting the absolute
/// one - its point, row duals and reduced costs are handed to the interior point as a
/// starting point, with a small iteration budget and whatever time the caller has left.
///
/// The polish is not free and does not pretend to be: the merged answer's iteration count is
/// the SUM of both phases, its algorithm reads "pdhg+ipm", and the polish's own count is kept
/// in polish_iterations so a benchmark row can say which phase did what. The factor is the
/// cost that can be prohibitive - the random scale family fills 17% of n^2 (#193) - and the
/// interior point measures it from the ordering before building it: above
/// polish_max_factor_nonzeros it declines, PDHG's answer stands, and the message says why.
///
/// A polished answer replaces PDHG's only when it is better - optimal, or feasible with
/// smaller scaled violations - so the polish cannot make the answer worse.
void polish_with_the_interior_point(Solution* first, const Model& model, const Options& options,
                                    Logger& logger, const Timer& timer) {
  if (!options.get_bool("pdhg_polish")) return;
  if (first->status == SolveStatus::kOptimal || !claims_a_point(first->status)) return;
  const auto n = static_cast<std::size_t>(model.num_cols());
  const auto m = static_cast<std::size_t>(model.num_rows());
  if (first->col_value.size() != n || first->row_dual.size() != m ||
      first->col_dual.size() != n) {
    return;
  }

  Options polish = options;
  polish.set_int("iteration_limit", options.get_int("polish_iteration_limit"));
  // THE POLISH HAS A CLOCK OF ITS OWN. The factor cap above catches a factor the ordering
  // has already sized, but on the random scale family at 20,000 rows the ORDERING is the
  // cost: it ran for the whole 1,200 s limit before it could report a factor too large to
  // build, and PDHG's 1,000 iterations had taken 1.8 s. A polish that costs a thousand
  // times the solve it finishes is not a polish. So the interior point gets the smaller of
  // the time the limit has left and polish_max_seconds, and #197's deadline - which reaches
  // inside the ordering - is what enforces it. A model whose factor is affordable (the
  // staircase family) finishes in seconds; one whose factor is not is declined in
  // polish_max_seconds and the first-order answer stands, which is the honest outcome.
  double budget = options.get_double("polish_max_seconds");
  const double time_limit = options.get_double("time_limit");
  if (time_limit > 0.0 && std::isfinite(time_limit)) {
    const double remaining = time_limit - timer.elapsed_seconds();
    if (remaining <= 0.0) {
      first->message += "; no time left for the interior-point polish";
      return;
    }
    budget = std::min(budget, remaining);
  }
  polish.set_double("time_limit", budget);
  logger.info("Polish: handing PDHG's point to the interior point, {} iterations at most",
              polish.get_int("iteration_limit"));
  const ipm::WarmStart warm{first->col_value, first->row_dual, first->col_dual};
  Solution polished = ipm::solve_ipm(model, polish, logger, &warm);

  const auto worst = [](const Solution& s) {
    return std::max(s.primal_infeasibility_scaled, s.dual_infeasibility_scaled);
  };
  const bool better =
      polished.status == SolveStatus::kOptimal ||
      (polished.status == SolveStatus::kFeasible && worst(polished) < worst(*first));
  if (!better) {
    first->polish_iterations = polished.iterations;
    first->message += fmt::format(
        "; the interior-point polish did not improve it ({} after {} iterations: {})",
        to_string(polished.status), polished.iterations, polished.message);
    return;
  }
  polished.polish_iterations = polished.iterations;
  polished.iterations += first->iterations;
  polished.algorithm = first->algorithm + "+ipm";  // "pdhg-cpu+ipm"
  polished.message =
      fmt::format("{}; polished by the interior point in {} iterations{}", first->message,
                  polished.polish_iterations,
                  polished.message.empty() ? std::string() : " (" + polished.message + ")");
  *first = std::move(polished);
}

/// PDHG's share of a finite time limit when a polish is to follow; the rest is the polish's.
constexpr double kPdhgShareOfTheTimeLimit = 0.7;

void reconcile_status_with_measurement(Solution* solution, const Options& options,
                                       Logger& logger, bool check_dual) {
  const bool claims_a_point =
      solution->status == SolveStatus::kOptimal || solution->status == SolveStatus::kFeasible;
  if (!claims_a_point) return;

  const double primal_tolerance = options.get_double("primal_feasibility_tolerance");
  const double dual_tolerance = options.get_double("dual_feasibility_tolerance");
  const double integrality_tolerance = options.get_double("integrality_tolerance");

  // A point that violates its own constraints is not feasible, so neither kOptimal nor
  // kFeasible is available. The engine stopped believing it had converged, so this is a
  // numerical failure and is reported as one, with the number that contradicts it.
  //
  // THE TEST IS ON THE SCALED VIOLATION, and the absolute one is still what gets printed.
  // An absolute tolerance asks a badly scaled model for accuracy it cannot have: on Netlib
  // grow7, whose largest solution value is 4.8e+07, 1e-7 absolute is 2.1e-15 relative, which
  // is below double precision's reach after three hundred iterations of arithmetic. Judging
  // that point infeasible says nothing about the point and everything about the units the
  // question was asked in. See Solution::primal_infeasibility_scaled for the derivation.
  if (solution->primal_infeasibility_scaled > primal_tolerance) {
    const std::string detail = fmt::format(
        "engine reported {} but the returned point violates primal feasibility by {:.3e} "
        "({:.3e} relative to the scale it was measured on), above the {:.1e} tolerance; it is "
        "not a feasible point",
        to_string(solution->status), solution->primal_infeasibility,
        solution->primal_infeasibility_scaled, primal_tolerance);
    solution->status = SolveStatus::kNumericalError;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
    return;
  }

  // Integrality, for the same reason and with the same force. A branch-and-bound run that
  // reports optimal while holding a fractional integer variable has reported the relaxation,
  // which CLAUDE.md names as the single most damaging thing this dispatcher could do.
  if (solution->integrality_violation > integrality_tolerance) {
    const std::string detail = fmt::format(
        "engine reported {} but an integer column is fractional by {:.3e}, above the {:.1e} "
        "tolerance; this is a relaxation, not an integer solution",
        to_string(solution->status), solution->integrality_violation, integrality_tolerance);
    solution->status = SolveStatus::kNumericalError;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
    return;
  }

  // Primal feasible but dual infeasible: the point is usable, the optimality claim is not
  // supported. kFeasible says exactly that and already exists for the purpose.
  // The SCALED violation decides, the absolute one is still printed - the same split as
  // the primal check above and for the same reason (#152). On grow7 the absolute dual
  // infeasibility can be 6.1 against prices of order 1e+07; judged absolutely that is a
  // failed optimality claim, judged against its own terms it is 6e-07 and the claim stands.
  if (check_dual && solution->status == SolveStatus::kOptimal &&
      solution->dual_infeasibility_scaled > dual_tolerance) {
    const std::string detail = fmt::format(
        "engine reported optimal but the reduced costs violate dual feasibility by {:.3e} "
        "({:.3e} relative to the terms they are computed from), above the {:.1e} tolerance; "
        "reporting a feasible point rather than a proof",
        solution->dual_infeasibility, solution->dual_infeasibility_scaled, dual_tolerance);
    solution->status = SolveStatus::kFeasible;
    solution->message = solution->message.empty() ? detail : solution->message + "; " + detail;
    logger.warning("{}", detail);
  }
}

Solution solve(const Model& model, const Options& options) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);

  const std::string problem = model.validate();
  if (!problem.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem;
    solution.solve_seconds = timer.elapsed_seconds();
    return solution;
  }

  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  apply_thread_option(options, logger);
  LogLevel level = LogLevel::kInfo;
  if (parse_log_level(options.get_string("log_level"), &level)) logger.set_level(level);
  const std::string progress_out = options.get_string("progress_out");
  if (!progress_out.empty()) logger.enable_progress_output(progress_out);

  const ProblemClass problem_class = classify(model);
  logger.info("Model {}: {} rows, {} columns, {} nonzeros, {} integer columns",
              model.name.empty() ? std::string("(unnamed)") : model.name, model.num_rows(),
              model.num_cols(), model.num_nonzeros(), model.num_integer_columns());
  logger.info("Problem class: {}", class_name(problem_class));

  if (problem_class == ProblemClass::kLp) {
    const std::string requested = options.get_string("algorithm");

    // "auto" means the DUAL simplex (#65), measured rather than assumed: on the Netlib full
    // set at 120 s it passes 78/89 against the primal's 74/89, solves d6cube, modszk1 and
    // fit2p where the primal hits the limit, leaves no verifier rejection, and takes 0.37x
    // the primal's time on the 83 instances both solve (bench/results/netlib-full-dual-
    // a947a1e.csv against netlib-full-default-a947a1e.csv). The primal stays selectable as
    // "simplex". PDHG is a first-order method: it converges to a tolerance rather than to a
    // vertex, produces no basis, and on the small instances we benchmark today the simplex
    // is both faster and exact. It is selected explicitly, and it becomes the automatic
    // choice only once there is evidence for a crossover point to switch on.
    const bool want_pdhg = requested == "pdhg";
    const bool want_ipm = requested == "ipm";
    const bool want_dual = requested == "dual-simplex" || requested == "auto";
    // One place runs the engine on whichever model - reduced or original - is being solved,
    // so the polish of a PDHG answer happens before postsolve in both cases.
    const auto run_lp_engine = [&](const Model& target) -> Solution {
      if (want_pdhg) {
        Options first_pass = options;
        const double time_limit = options.get_double("time_limit");
        if (options.get_bool("pdhg_polish") && time_limit > 0.0 && std::isfinite(time_limit)) {
          first_pass.set_double("time_limit", time_limit * kPdhgShareOfTheTimeLimit);
        }
        Solution first = pdhg::solve_pdhg(target, first_pass, logger);
        polish_with_the_interior_point(&first, target, options, logger, timer);
        return first;
      }
      return want_ipm    ? ipm::solve_ipm(target, options, logger)
             : want_dual ? solve_dual_simplex(target, options, logger)
                         : solve_primal_simplex(target, options, logger);
    };
    if (requested != "auto" && requested != "simplex" && !want_pdhg && !want_dual &&
        !want_ipm) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = fmt::format(
          "algorithm '{}' is not an engine; auto, simplex, dual-simplex, pdhg and ipm "
          "are available",
          requested);
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }

    if (options.get_bool("gpu")) {
      // Honest fallback, per CLAUDE.md: the CPU build must work with zero CUDA installed,
      // and --gpu must never crash. No CUDA backend is compiled in yet, so say so once.
      logger.warning(
          "--gpu requested but this build has no CUDA backend compiled in; running on CPU");
    }

    // PRESOLVE RUNS HERE, not inside an engine. The reductions are properties of the model,
    // so both engines get them, and - more importantly - postsolve then re-measures the
    // recovered point against the ORIGINAL model before the status guard below sees it. A
    // reduction or postsolve bug therefore surfaces as a feasibility violation on a model no
    // engine ever touched, and the guard downgrades the status rather than letting a
    // confident answer to a different problem out of the door.
    if (options.get_bool("presolve")) {
      const presolve::Result reduced = presolve::presolve(model, options, logger);
      if (reduced.proved_infeasible) {
        solution.status = SolveStatus::kInfeasible;
        solution.algorithm = "presolve";
        // Presolve proves infeasibility from bound arithmetic, and the chain of tightenings
        // that led there is not kept, so there is no Farkas vector to hand over. The reason
        // is in the message, which names the row and the two quantities that collide, and
        // the .sol file says `certificate none` rather than pretending otherwise. Turning
        // presolve off makes the simplex prove the same conclusion with a certificate (#191).
        solution.message =
            reduced.message +
            "; proved by presolve, which carries no Farkas certificate - re-run with "
            "--option presolve=false for one";
        solution.solve_seconds = timer.elapsed_seconds();
        logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                    solution.solve_seconds);
        return solution;
      }
      Solution inner = run_lp_engine(reduced.model);
      solution = presolve::postsolve(reduced, model, inner);
      solution.solve_seconds = timer.elapsed_seconds();
    } else {
      solution = run_lp_engine(model);
    }
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/true);
    refuse_a_non_finite_answer(&solution, logger);
    keep_only_a_proved_certificate(&solution, model, logger);
    logger.info("Result: {}  objective {:.10g}  {} iterations  {:.3f}s",
                to_string(solution.status), solution.objective, solution.iterations,
                solution.solve_seconds);
    logger.info("Measured primal infeasibility {:.3e}, dual infeasibility {:.3e}",
                solution.primal_infeasibility, solution.dual_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kMilp) {
    solution = mip::solve_branch_and_bound(model, options, logger);
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    // NOT for the branch and bound's "nothing found" convention, which deliberately reports
    // the worst representable objective - an infinity there is a considered statement that
    // no point exists, not a broken number. Only a claimed POINT is checked, and that
    // convention comes with kInfeasible or a limit and no values.
    refuse_a_non_finite_answer(&solution, logger);
    logger.info("Result: {}  objective {:.10g}  bound {:.10g}  {} nodes  {:.3f}s",
                to_string(solution.status), solution.objective, solution.dual_bound,
                solution.nodes, solution.solve_seconds);
    logger.info("Measured integrality violation {:.3e}, primal infeasibility {:.3e}",
                solution.integrality_violation, solution.primal_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kQp) {
    solution = qp::solve_convex_qp(model, options, logger);
    // check_dual is false: the QP's reduced costs are c + Qx - A'y, which is not the
    // quantity Solution::recompute_quality() tests, and applying the LP dual rule here
    // would reject correct answers. Primal feasibility and the status still have to agree.
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    logger.info("Result: {}  objective {:.10g}  {} iterations  {:.3f}s",
                to_string(solution.status), solution.objective, solution.iterations,
                solution.solve_seconds);
    logger.info("Measured primal infeasibility {:.3e}", solution.primal_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kMiqp) {
    // MIQP is branch and bound over QP node relaxations - the two engines joined, which is
    // exactly what the message this replaces said was missing. The QP engine refuses a
    // non-convex Hessian before any arithmetic starts, so a non-convex MIQP is still refused
    // rather than solved to a local point; that check now happens at the first node.
    //
    // check_dual stays false for the same reason it is false for a MILP: the reduced costs
    // belong to a node whose bounds branching tightened, and for a QP they are c + Qx - A'y
    // rather than the quantity recompute_quality() measures. Integrality and primal
    // feasibility are what distinguish an MIQP answer from its relaxation, and both are
    // checked.
    solution = mip::solve_branch_and_bound(model, options, logger);
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/false);
    logger.info("Result: {}  objective {:.10g}  bound {:.10g}  {} nodes  {:.3f}s",
                to_string(solution.status), solution.objective, solution.dual_bound,
                solution.nodes, solution.solve_seconds);
    logger.info("Measured integrality violation {:.3e}, primal infeasibility {:.3e}",
                solution.integrality_violation, solution.primal_infeasibility);
    return solution;
  }

  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message =
      fmt::format("no engine is implemented for {} yet", class_name(problem_class));
  logger.warning("{}", solution.message);

  solution.solve_seconds = timer.elapsed_seconds();
  return solution;
}

}  // namespace sankhya
