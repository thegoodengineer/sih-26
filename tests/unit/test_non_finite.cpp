// SPDX-License-Identifier: Apache-2.0
// SANKHYA - an answer whose numbers are not numbers is not an answer (#194).
//
// WHERE THIS CAME FROM. The first time this solver was run at size, the interior-point
// method hit a time limit on a generated 5000x5000 instance with its iterate full of NaN,
// reported that iterate as its point, and the objective computed from it was written into
// bench/results/scale-eac6f75.csv as the string `nan` - in the column that exists to say
// whether the answer was right.
//
// That is worse than a wrong number. A NaN does not announce itself: it survives every
// arithmetic operation downstream looking like data, and the runners write whatever comes
// back. The guard under test refuses to publish one, from any engine, and reports the
// numerical failure that actually happened.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

/// A small, ordinary LP: min -x - y subject to x + y <= 4, both in [0, 3].
Model easy_lp() {
  Model model;
  model.col_cost = {-1.0, -1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {3.0, 3.0};
  model.col_type.assign(2, VarType::kContinuous);
  model.row_lower = {-kInfinity};
  model.row_upper = {4.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();
  return model;
}

TEST(NonFiniteAnswer, AnOrdinaryAnswerIsUntouched) {
  // The guard must not be reachable on a healthy solve. If this ever fails, the check is
  // rejecting answers rather than protecting them, which would be far worse than the bug.
  const Solution solution = solve(easy_lp(), quiet());
  ASSERT_EQ(solution.status, SolveStatus::kOptimal) << solution.message;
  EXPECT_TRUE(std::isfinite(solution.objective)) << solution.message;
  EXPECT_NEAR(solution.objective, -4.0, 1e-9);
  EXPECT_EQ(solution.message.find("not a number"), std::string::npos) << solution.message;
}

TEST(NonFiniteAnswer, EveryEngineReturnsFiniteNumbersOrSaysItFailed) {
  // Across all four engines, on a model each of them can solve. The property is the point:
  // whatever comes back, if it claims a point then every number in it is a number. A status
  // that does not claim a point is free to say infinity, which is how "nothing found" and
  // "no bound" are spelled elsewhere and is a considered statement rather than a broken one.
  const Model model = easy_lp();
  for (const char* engine : {"auto", "simplex", "dual-simplex", "pdhg", "ipm"}) {
    Options options = quiet();
    options.set_string("algorithm", engine);
    const Solution solution = solve(model, options);
    const bool claims_a_point = solution.status == SolveStatus::kOptimal ||
                                solution.status == SolveStatus::kFeasible ||
                                solution.status == SolveStatus::kIterationLimit ||
                                solution.status == SolveStatus::kTimeLimit;
    if (!claims_a_point) continue;
    EXPECT_TRUE(std::isfinite(solution.objective))
        << engine << " reported a non-finite objective: " << solution.message;
    for (std::size_t j = 0; j < solution.col_value.size(); ++j) {
      ASSERT_TRUE(std::isfinite(solution.col_value[j]))
          << engine << " reported a non-finite value in column " << j << ": "
          << solution.message;
    }
  }
}

TEST(NonFiniteAnswer, NothingFoundMayStillBeReportedAsInfinite) {
  // The control in the other direction. Branch and bound reports the worst representable
  // objective when it has found no integer point, with the reasoning written out in
  // branch_and_bound.cpp: a gap of zero would read as CLOSED, which is the opposite of what
  // happened. That infinity is a statement, not a broken number, and the guard must leave it
  // alone - which it does by only examining statuses that claim a point.
  Model model = easy_lp();
  model.col_type.assign(2, VarType::kInteger);
  // 2x = 3 has no integer solution; added as an equality row so the search closes empty.
  model.row_lower = {3.0};
  model.row_upper = {3.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 2.0);
  model.matrix.finalize();
  model.col_upper = {3.0, 0.0};

  const Solution solution = solve(model, quiet());
  ASSERT_EQ(solution.status, SolveStatus::kInfeasible) << solution.message;
  EXPECT_EQ(solution.message.find("not a number"), std::string::npos) << solution.message;
}

}  // namespace
}  // namespace sankhya
