// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solve() dispatcher.
//
// THIS FILE IS THE SEAM. Every engine registers here and nowhere else:
//   Phase 2  primal revised simplex  -> LP
//   Phase 4  restarted PDHG          -> LP, large and sparse
//   Phase 5  branch and cut          -> MILP
//   Phase 8  Mehrotra IPM, convex QP -> LP and QP
// Phase 1 has no engine yet, so the dispatcher classifies the model, reports what it would
// have dispatched to, and returns kNotSolved. That is deliberately not a stub that lies:
// per CLAUDE.md an unimplemented path reports the truth rather than a plausible zero.

#include <string>

#include <fmt/format.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/timer.hpp"

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

  // No engine is registered yet. Phase 2 replaces this block with the simplex call.
  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "none";
  solution.message = fmt::format(
      "no engine is implemented for {} yet (Phase 1 provides the model, the "
      "option table and the linear algebra only)",
      class_name(problem_class));
  logger.warning("{}", solution.message);

  solution.solve_seconds = timer.elapsed_seconds();
  return solution;
}

}  // namespace sankhya
