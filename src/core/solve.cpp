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

#include <string>

#include <fmt/format.h>

#ifdef SANKHYA_ENABLE_CUDA
#include "gpu/device.hpp"
#endif

#include "core/status_guard.hpp"
#include "presolve/presolve.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/pdhg.hpp"
#include "sankhya/qp.hpp"
#include "sankhya/timer.hpp"

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
  if (solution->primal_infeasibility > primal_tolerance) {
    const std::string detail = fmt::format(
        "engine reported {} but the returned point violates primal feasibility by {:.3e}, "
        "above the {:.1e} tolerance; it is not a feasible point",
        to_string(solution->status), solution->primal_infeasibility, primal_tolerance);
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
  if (check_dual && solution->status == SolveStatus::kOptimal &&
      solution->dual_infeasibility > dual_tolerance) {
    const std::string detail = fmt::format(
        "engine reported optimal but the reduced costs violate dual feasibility by {:.3e}, "
        "above the {:.1e} tolerance; reporting a feasible point rather than a proof",
        solution->dual_infeasibility, dual_tolerance);
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
    // Explicit engine selection.
    // "auto" keeps the existing simplex behaviour for ordinary LPs. Large sparse LPs
    // are dispatched to PDHG after presolve, where PDHG performs the CPU/CUDA decision.
    const bool want_simplex = requested == "simplex";
    const bool want_pdhg = requested == "pdhg";
    const bool want_auto = requested == "auto";

    if (!want_simplex && !want_pdhg && !want_auto) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = fmt::format(
          "algorithm '{}' is not implemented yet; simplex and pdhg are available", requested);
      logger.warning("{}", solution.message);
      solution.solve_seconds = timer.elapsed_seconds();
      return solution;
    }

    if (options.get_bool("gpu")) {
#ifdef SANKHYA_ENABLE_CUDA
      std::string device_description;
      if (gpu::device_available(&device_description)) {
        logger.info("--gpu requested; CUDA device available: {}", device_description);
      } else {
        logger.warning("--gpu requested but CUDA device unavailable: {}; running on CPU",
                       device_description);
      }
#else
      logger.warning(
          "--gpu requested but this build has no CUDA backend compiled in; running on CPU");
#endif
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
        solution.message = reduced.message;
        solution.solve_seconds = timer.elapsed_seconds();
        logger.info("Result: {} (proved during presolve)  {:.3f}s", to_string(solution.status),
                    solution.solve_seconds);
        return solution;
      }
      Solution inner = want_pdhg ? pdhg::solve_pdhg(reduced.model, options, logger)
                                 : solve_primal_simplex(reduced.model, options, logger);
      solution = presolve::postsolve(reduced, model, inner);
      solution.solve_seconds = timer.elapsed_seconds();
    } else {
      solution = want_pdhg ? pdhg::solve_pdhg(model, options, logger)
                           : solve_primal_simplex(model, options, logger);
    }
    reconcile_status_with_measurement(&solution, options, logger, /*check_dual=*/true);
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
