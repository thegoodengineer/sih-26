// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what presolve says it did, and whether that is what it did (#286).
//
// Presolve is the one stage that changes the model the user handed over, and the only account
// of it used to be a single log line. The report structures that account: sizes before and
// after, which reductions fired, what presolve DECLINED to do, how many passes it took and
// why it stopped.
//
// A statistics layer has one interesting failure mode - numbers that drift away from the
// thing they describe - so every count below is checked against a model whose reductions are
// known by construction, and the sizes are checked against the reduced model itself rather
// than against each other.

#include <cmath>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sankhya/io.hpp"
#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "presolve/presolve.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using Report = Solution::PresolveReport;
using testing::TempFile;

Options quiet(bool presolve = true) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_bool("presolve", presolve);
  return options;
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integral,
            const std::vector<std::tuple<Index, Index, double>>& hessian_lower = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  for (Index j = 0; j < n; ++j) {
    if (integral[static_cast<std::size_t>(j)]) {
      model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
    }
  }
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
  for (const auto& [i, j, v] : hessian_lower) model.hessian.add_entry(i, j, v);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

presolve::Result reduce(const Model& model) {
  Options options = quiet();
  Logger logger(nullptr);
  return presolve::presolve(model, options, logger);
}

// =========================================================================================

TEST(PresolveReport, TheSizesMatchTheModelPresolveActuallyProduced) {
  // A redundant row (x2 <= 50 against x2 <= 1) and a fixed column, so something moves.
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {3.0, 50.0},
            {-1.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {false, false, false});
  const presolve::Result result = reduce(model);
  const Report& report = result.report;

  EXPECT_TRUE(report.ran);
  EXPECT_EQ(report.termination, Report::Termination::kFixedPoint);
  EXPECT_GE(report.passes, 1);

  // Against the model itself, not against other fields of the report.
  EXPECT_EQ(report.original_rows, model.num_rows());
  EXPECT_EQ(report.original_cols, model.num_cols());
  EXPECT_EQ(report.original_nonzeros, model.num_nonzeros());
  EXPECT_EQ(report.reduced_rows, result.model.num_rows());
  EXPECT_EQ(report.reduced_cols, result.model.num_cols());
  EXPECT_EQ(report.reduced_nonzeros, result.model.num_nonzeros());

  EXPECT_EQ(report.rows_removed(), result.rows_removed());
  EXPECT_EQ(report.columns_removed(), result.cols_removed());
  EXPECT_GT(report.row_reduction_percent(), 0.0);
  EXPECT_LE(report.row_reduction_percent(), 100.0);
}

TEST(PresolveReport, TheCountsNameTheReductionsThatFired) {
  // Built so each count is known: row 1 is redundant, row 2 is a singleton bounding x1,
  // column 2 is fixed, and column 3 appears in no row at all.
  const Model model =
      build({{1.0, 1.0, 0.0, 0.0}, {0.0, 2.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}},
            {-kInfinity, -kInfinity, -kInfinity}, {1000.0, 6.0, 40.0}, {-1.0, -1.0, 0.0, 1.0},
            {0.0, 0.0, 2.0, 0.0}, {1.0, 10.0, 2.0, 5.0}, {false, false, false, false});
  const Report& report = reduce(model).report;

  EXPECT_GE(report.redundant_rows, 1) << "row 1 cannot bind at these bounds";
  EXPECT_GE(report.singleton_rows, 1) << "row 2 is one entry, so it is a bound on x1";
  EXPECT_GE(report.fixed_columns, 1) << "column 2 has equal bounds";
  EXPECT_GE(report.empty_columns, 1) << "column 3 is in no row";
  // The total of what fired cannot exceed the records postsolve will replay, because the
  // counts come from those records.
  const Count named = report.empty_rows + report.redundant_rows + report.singleton_rows +
                      report.fixed_columns + report.empty_columns +
                      report.free_column_singletons + report.doubleton_equations;
  EXPECT_EQ(named, static_cast<Count>(reduce(model).records.size()));
}

TEST(PresolveReport, IntegerRoundingAndTheDeclinesAreCountedToo) {
  // x1 is an integer column bounded [0.5, 2.5]: one rounding. x0 carries curvature and is
  // fixed, so the fixed-column reduction is declined and says so.
  const Model model = build({{1.0, 0.0}}, {-kInfinity}, {4.0}, {0.0, 1.0}, {2.0, 0.5},
                            {2.0, 2.5}, {false, true}, {{0, 0, 2.0}});
  const Report& report = reduce(model).report;

  EXPECT_GE(report.integer_bounds_rounded, 1);
  EXPECT_GE(report.bounds_tightened, report.integer_bounds_rounded);
  EXPECT_GE(report.quadratic_columns_protected, 1)
      << "a fixed column carrying curvature is kept, and the report says why the model is "
         "not smaller";
  EXPECT_EQ(report.fixed_columns, 0);
}

TEST(PresolveReport, AnInfeasibilityProvedByPresolveIsReportedAsSuch) {
  const Model model =
      build({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 2.0}, {1.0}, {0.0}, {10.0}, {false});
  const presolve::Result result = reduce(model);
  ASSERT_TRUE(result.proved_infeasible);
  EXPECT_TRUE(result.report.ran);
  EXPECT_EQ(result.report.termination, Report::Termination::kProvedInfeasible);
  // No reduced model was built, so the "after" figures stay at the original rather than
  // reading as a model reduced to nothing.
  EXPECT_EQ(result.report.reduced_rows, result.report.original_rows);
  EXPECT_EQ(result.report.rows_removed(), 0);
}

TEST(PresolveReport, PercentagesAreSafeOnAModelWithNothingInIt) {
  Report empty;
  EXPECT_DOUBLE_EQ(empty.row_reduction_percent(), 0.0);
  EXPECT_DOUBLE_EQ(empty.column_reduction_percent(), 0.0);
  EXPECT_DOUBLE_EQ(empty.nonzero_reduction_percent(), 0.0);
  EXPECT_FALSE(std::isnan(empty.row_reduction_percent()));
}

TEST(PresolveReport, TheSolutionCarriesTheReportAndPresolveOffSaysSo) {
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {3.0, 50.0},
            {-1.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {false, false, false});

  const Solution on = solve(model, quiet(true));
  EXPECT_TRUE(on.presolve_report.ran);
  EXPECT_GT(on.presolve_report.rows_removed() + on.presolve_report.columns_removed(), 0);
  EXPECT_GE(on.presolve_report.seconds, 0.0);

  const Solution off = solve(model, quiet(false));
  EXPECT_FALSE(off.presolve_report.ran);
  EXPECT_EQ(off.presolve_report.termination, Report::Termination::kNotRun);
  EXPECT_TRUE(off.presolve_report.skipped_because.empty())
      << "the option decided it, so there is no other reason to give";

  // The report is an account of the solve, never an input to it.
  EXPECT_EQ(on.status, off.status);
  EXPECT_NEAR(on.objective, off.objective, 1e-9);
}

TEST(PresolveReport, TheStatsFileCarriesTheSameNumbers) {
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {3.0, 50.0},
            {-1.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {false, false, false});
  const Solution solution = solve(model, quiet(true));

  const TempFile file("", ".json");
  std::string error;
  ASSERT_TRUE(io::write_stats_json(file.path(), model, solution, &error)) << error;

  std::ifstream in(file.path());
  ASSERT_TRUE(in.good());
  nlohmann::json blob;
  in >> blob;
  ASSERT_TRUE(blob.contains("presolve")) << blob.dump(1);
  const nlohmann::json& presolve = blob["presolve"];
  EXPECT_TRUE(presolve["ran"].get<bool>());
  EXPECT_EQ(presolve["termination"].get<std::string>(), "fixed_point");
  EXPECT_EQ(presolve["rows"]["before"].get<Index>(), solution.presolve_report.original_rows);
  EXPECT_EQ(presolve["rows"]["after"].get<Index>(), solution.presolve_report.reduced_rows);
  EXPECT_EQ(presolve["columns"]["after"].get<Index>(), solution.presolve_report.reduced_cols);
  EXPECT_EQ(presolve["passes"].get<Count>(), solution.presolve_report.passes);
  EXPECT_TRUE(presolve["reductions"].contains("redundant_rows"));
  EXPECT_TRUE(presolve["declined"].contains("quadratic_columns_protected"));
}

}  // namespace
}  // namespace sankhya
