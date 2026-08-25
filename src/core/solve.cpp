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

#include "sankhya/logging.hpp"
#include "sankhya/mip.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/pdhg.hpp"
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

  const ProblemClass problem_class = classify(model);
  logger.info("Model {}: {} rows, {} columns, {} nonzeros, {} integer columns",
              model.name.empty() ? std::string("(unnamed)") : model.name, model.num_rows(),
              model.num_cols(), model.num_nonzeros(), model.num_integer_columns());
  logger.info("Problem class: {}", class_name(problem_class));

  if (problem_class == ProblemClass::kLp) {
    const std::string requested = options.get_string("algorithm");

    // "auto" means the simplex. PDHG is a first-order method: it converges to a tolerance
    // rather than to a vertex, produces no basis, and on the small instances we benchmark
    // today the simplex is both faster and exact. It is selected explicitly, and it becomes
    // the automatic choice only once there is evidence for a crossover point to switch on.
    const bool want_pdhg = requested == "pdhg";
    if (requested != "auto" && requested != "simplex" && !want_pdhg) {
      solution.status = SolveStatus::kNotSolved;
      solution.algorithm = "none";
      solution.message = fmt::format(
          "algorithm '{}' is not implemented yet; simplex and pdhg are available", requested);
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

    solution = want_pdhg ? pdhg::solve_pdhg(model, options, logger)
                         : solve_primal_simplex(model, options, logger);
    logger.info("Result: {}  objective {:.10g}  {} iterations  {:.3f}s",
                to_string(solution.status), solution.objective, solution.iterations,
                solution.solve_seconds);
    logger.info("Measured primal infeasibility {:.3e}, dual infeasibility {:.3e}",
                solution.primal_infeasibility, solution.dual_infeasibility);
    return solution;
  }

  if (problem_class == ProblemClass::kMilp) {
    solution = mip::solve_branch_and_bound(model, options, logger);
    logger.info("Result: {}  objective {:.10g}  bound {:.10g}  {} nodes  {:.3f}s",
                to_string(solution.status), solution.objective, solution.dual_bound,
                solution.nodes, solution.solve_seconds);
    logger.info("Measured integrality violation {:.3e}, primal infeasibility {:.3e}",
                solution.integrality_violation, solution.primal_infeasibility);
    return solution;
  }

  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message = fmt::format(
      "no engine is implemented for {} yet; the QP and MIQP engines land in Phase 8. The "
      "relaxation is deliberately NOT reported as a solution",
      class_name(problem_class));
  logger.warning("{}", solution.message);

  solution.solve_seconds = timer.elapsed_seconds();
  return solution;
}

}  // namespace sankhya
