// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the interior-point method (#56) against the simplex and the exact oracle.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/ipm.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

Options with_algorithm(const char* algorithm) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", algorithm);
  return options;
}

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper, ObjSense sense = ObjSense::kMinimize) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = sense;
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

TEST(InteriorPoint, SolvesATextbookLpToTheSimplexAnswer) {
  // max 3x + 5y s.t. x <= 4, 2y <= 12, 3x + 2y <= 18: optimum 36 at (2, 6).
  const Model model = make_lp(
      {{1.0, 0.0}, {0.0, 2.0}, {3.0, 2.0}}, {-kInfinity, -kInfinity, -kInfinity},
      {4.0, 12.0, 18.0}, {3.0, 5.0}, {0.0, 0.0}, {kInfinity, kInfinity}, ObjSense::kMaximize);
  const Solution ipm = solve(model, with_algorithm("ipm"));
  ASSERT_EQ(ipm.status, SolveStatus::kOptimal) << ipm.message;
  EXPECT_EQ(ipm.algorithm, "ipm");
  EXPECT_NEAR(ipm.objective, 36.0, 1e-6);
  EXPECT_NEAR(ipm.col_value[0], 2.0, 1e-5);
  EXPECT_NEAR(ipm.col_value[1], 6.0, 1e-5);
  EXPECT_LE(ipm.primal_infeasibility, tol::kPrimalFeasibility);
  EXPECT_LE(ipm.dual_infeasibility_scaled, tol::kDualFeasibility) << ipm.message;
  // No basis is produced, by design and by the header's statement. The status vectors
  // still exist - the dispatcher allocates them and postsolve annotates the columns it
  // removed - so what is asserted is the engine's name, not an empty vector.
}

TEST(InteriorPoint, HandlesRangedRowsBoxedAndFreeColumns) {
  // min x - y + z  s.t.  1 <= x + y <= 4,  x - z = 0.5,  x in [0, 3], y in [0, 2], z free.
  // z = x - 0.5, so min 2x - y - 0.5 with x + y in [1, 4]: y = 2, x = 0 -> objective -2.5.
  const Model model = make_lp({{1.0, 1.0, 0.0}, {1.0, 0.0, -1.0}}, {1.0, 0.5}, {4.0, 0.5},
                              {1.0, -1.0, 1.0}, {0.0, 0.0, -kInfinity}, {3.0, 2.0, kInfinity});
  const Solution ipm = solve(model, with_algorithm("ipm"));
  const Solution simplex = solve(model, with_algorithm("dual-simplex"));
  ASSERT_EQ(ipm.status, SolveStatus::kOptimal) << ipm.message;
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << simplex.message;
  EXPECT_NEAR(ipm.objective, simplex.objective,
              1e-6 * std::max(1.0, std::fabs(simplex.objective)));
  EXPECT_NEAR(ipm.objective, -2.5, 1e-6);
  EXPECT_LE(ipm.primal_infeasibility, tol::kPrimalFeasibility);
}

TEST(InteriorPoint, AgreesWithTheExactOracleOnKktInstances) {
  // The KKT generator knows its optimum analytically, so this needs no second solver and
  // cannot be fooled by two engines sharing a mistake. A first-order-free, tolerance-based
  // method is judged at 1e-6 relative, the same bar the fuzz sets the simplex.
  std::mt19937_64 rng(56001);
  oracle::GeneratorConfig config;
  config.min_rows = 3;
  config.max_rows = 12;
  config.min_cols = 3;
  config.max_cols = 12;
  config.magnitude = 6;
  int solved = 0;
  int failed = 0;
  std::string first_failure;
  for (int trial = 0; trial < 150; ++trial) {
    const oracle::KktInstance instance = oracle::kkt_lp(rng, config);
    const Model model = oracle::to_model(instance.lp);
    const Solution ipm = solve(model, with_algorithm("ipm"));
    const double expected = static_cast<double>(instance.optimal_objective);
    const bool ok =
        ipm.status == SolveStatus::kOptimal &&
        std::fabs(ipm.objective - expected) <= 1e-6 * std::max(1.0, std::fabs(expected));
    if (ok) {
      ++solved;
    } else {
      ++failed;
      if (first_failure.empty()) {
        first_failure = "trial " + std::to_string(trial) + ": " + to_string(ipm.status) +
                        " objective " + std::to_string(ipm.objective) + " expected " +
                        std::to_string(expected) + " (" + ipm.message + ")\n" +
                        instance.lp.to_text();
      }
    }
  }
  std::cout << "ipm: " << solved << " KKT instances solved, " << failed << " failed\n";
  EXPECT_EQ(failed, 0) << first_failure;
}

TEST(InteriorPoint, SolvesEqualityRowsAndFixedColumns) {
  // Every Netlib instance has equality rows, and the KKT generator produces none: an
  // equality row's logical is FIXED (lower == upper), which the bounded form handled as two
  // slacks that both had to vanish - no interior - and afiro (28 x 32) hit the iteration
  // limit. Beale's example is three equalities; the fixed column x2 = 1 pins a value the
  // simplex agrees on.
  //   min -3/4 x3 + 20 x4 - 1/2 x5 + 6 x6
  //   s.t. x0 + 1/4 x3 - 8 x4 - x5 + 9 x6 = 0;  x1 + 1/2 x3 - 12 x4 - 1/2 x5 + 3 x6 = 0;
  //        x2 + x5 = 1;  x >= 0;  optimum -5/4.
  const Model model =
      make_lp({{1.0, 0.0, 0.0, 0.25, -8.0, -1.0, 9.0},
               {0.0, 1.0, 0.0, 0.5, -12.0, -0.5, 3.0},
               {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0}},
              {0.0, 0.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0, -0.75, 20.0, -0.5, 6.0},
              {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
              {kInfinity, kInfinity, kInfinity, kInfinity, kInfinity, kInfinity, kInfinity});
  for (const bool presolve : {true, false}) {
    Options options = with_algorithm("ipm");
    options.set_bool("presolve", presolve);
    const Solution ipm = solve(model, options);
    ASSERT_EQ(ipm.status, SolveStatus::kOptimal)
        << "presolve " << presolve << ": " << ipm.message;
    EXPECT_NEAR(ipm.objective, -1.25, 1e-6) << "presolve " << presolve;
    EXPECT_LE(ipm.primal_infeasibility, tol::kPrimalFeasibility);
    EXPECT_LE(ipm.dual_infeasibility_scaled, tol::kDualFeasibility) << ipm.message;
  }
  // A fixed structural column, both engines.
  Model fixed = model;
  fixed.col_lower[5] = 1.0;
  fixed.col_upper[5] = 1.0;
  const Solution ipm = solve(fixed, with_algorithm("ipm"));
  const Solution simplex = solve(fixed, with_algorithm("dual-simplex"));
  ASSERT_EQ(ipm.status, SolveStatus::kOptimal) << ipm.message;
  ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << simplex.message;
  EXPECT_NEAR(ipm.objective, simplex.objective,
              1e-6 * std::max(1.0, std::fabs(simplex.objective)));
}

TEST(InteriorPoint, SolvesTheCommittedNetlibInstancesToTheSimplexAnswer) {
  // The eight committed Netlib instances, every one with equality rows: afiro is the one
  // that hit the iteration limit before fixed variables were handled. The simplex's answer
  // is the reference; the published optimum is what the benchmark runner checks.
  const char* names[] = {"afiro", "adlittle", "sc50a",    "sc50b",
                         "blend", "share2b",  "stocfor1", "israel"};
  for (const char* name : names) {
    Model model;
    // From this file's own location, not the working directory: gtest_discover_tests runs
    // the binary from build/tests, where the relative path did not exist, and this test had
    // been skipping every instance in CI while reporting a pass.
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(name) + ".mps"))
            .string();
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;
    const Solution ipm = solve(model, with_algorithm("ipm"));
    const Solution simplex = solve(model, with_algorithm("dual-simplex"));
    ASSERT_EQ(simplex.status, SolveStatus::kOptimal) << name << ": " << simplex.message;
    EXPECT_EQ(ipm.status, SolveStatus::kOptimal) << name << ": " << ipm.message;
    if (ipm.status == SolveStatus::kOptimal) {
      EXPECT_NEAR(ipm.objective, simplex.objective,
                  1e-6 * std::max(1.0, std::fabs(simplex.objective)))
          << name;
    }
  }
}

TEST(InteriorPoint, ReportsRatherThanClaimsOnAnInfeasibleModel) {
  // x >= 5 and x <= 1: the method cannot certify infeasibility and must not say optimal.
  const Model model =
      make_lp({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 1.0}, {1.0}, {0.0}, {kInfinity});
  const Solution ipm = solve(model, with_algorithm("ipm"));
  EXPECT_NE(ipm.status, SolveStatus::kOptimal) << ipm.message;
}

}  // namespace
}  // namespace sankhya
