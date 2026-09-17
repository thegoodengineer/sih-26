// SPDX-License-Identifier: Apache-2.0
// SANKHYA - what `sankhya diagnose` says about a model (#294).
//
// A diagnostic is read before anything has been solved, which is exactly when a reader has no
// way to check it. So the tests below build models whose every count is known by construction
// and hold the report to them, and where the report repeats a judgement another part of the
// project owns - the problem class, the convexity verdict, what presolve would remove - they
// check it against that part rather than against a copy of its answer.

#include <cmath>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "diagnose/diagnose.hpp"
#include "presolve/presolve.hpp"

namespace sankhya::diagnose {
namespace {

Options quiet() {
  Options options;
  options.set_bool("log_to_console", false);
  return options;
}

Diagnosis run(const Model& model) {
  Logger logger(nullptr);
  return analyse(model, quiet(), logger);
}

Model build(const std::vector<std::vector<double>>& rows, const std::vector<double>& row_lower,
            const std::vector<double>& row_upper, const std::vector<double>& cost,
            const std::vector<double>& col_lower, const std::vector<double>& col_upper,
            const std::vector<bool>& integral,
            const std::vector<std::tuple<Index, Index, double>>& hessian_lower = {}) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.name = "TESTMODEL";
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

// =========================================================================================

TEST(Diagnose, CountsEveryRowAndColumnKindAsTheModelDefinesThem) {
  // 4 columns: continuous boxed, binary, integer boxed, fixed. 4 rows: equality, <=, >=,
  // range. One column appears in no row.
  Model model =
      build({{1.0, 1.0, 0.0, 0.0},   // equality
             {1.0, 0.0, 1.0, 0.0},   // <=
             {0.0, 1.0, 1.0, 0.0},   // >=
             {1.0, 0.0, 0.0, 0.0}},  // range
            {5.0, -kInfinity, 2.0, 0.0}, {5.0, 10.0, kInfinity, 4.0}, {1.0, 1.0, 1.0, 0.0},
            {0.0, 0.0, 0.0, 7.0}, {10.0, 1.0, 5.0, 7.0}, {false, true, true, false});

  const Diagnosis d = run(model);
  EXPECT_EQ(d.rows, 4);
  EXPECT_EQ(d.columns, 4);
  EXPECT_EQ(d.nonzeros, model.num_nonzeros());
  EXPECT_EQ(d.problem_class, "MILP");

  EXPECT_EQ(d.equality_rows, 1);
  EXPECT_EQ(d.less_than_rows, 1);
  EXPECT_EQ(d.greater_than_rows, 1);
  EXPECT_EQ(d.range_rows, 1);
  EXPECT_EQ(d.free_rows, 0);

  EXPECT_EQ(d.integer_columns, 2);
  EXPECT_EQ(d.binary_columns, 1) << "only the column bounded [0, 1] is binary";
  EXPECT_EQ(d.continuous_columns, 2);
  EXPECT_EQ(d.fixed_columns, 1) << "the column with equal bounds";
  EXPECT_EQ(d.columns_without_entries, 1);
  EXPECT_EQ(d.integer_columns + d.continuous_columns, d.columns);

  // Sparsity, against the same nonzero count.
  EXPECT_NEAR(d.average_nonzeros_per_row, static_cast<double>(d.nonzeros) / 4.0, 1e-12);
  EXPECT_NEAR(d.average_nonzeros_per_column, static_cast<double>(d.nonzeros) / 4.0, 1e-12);
  EXPECT_GT(d.densest_row_nonzeros, 0);
}

TEST(Diagnose, TheCoefficientRatioDecidesTheScalingRisk) {
  const auto ratio_model = [](double big) {
    return build({{1e-3, big}}, {-kInfinity}, {1.0}, {1.0, 1.0}, {0.0, 0.0}, {1.0, 1.0},
                 {false, false});
  };

  const Diagnosis low = run(ratio_model(1e-2));
  EXPECT_EQ(low.scaling_risk, Risk::kLow);
  EXPECT_NEAR(low.coefficient_ratio, 10.0, 1e-9);

  const Diagnosis medium = run(ratio_model(1e6));  // ratio 1e9
  EXPECT_EQ(medium.scaling_risk, Risk::kMedium);

  const Diagnosis high = run(ratio_model(1e11));  // ratio 1e14
  EXPECT_EQ(high.scaling_risk, Risk::kHigh);
  EXPECT_FALSE(high.scaling_note.empty());
}

TEST(Diagnose, TheClassAndConvexityComeFromTheSolversOwnTests) {
  const Model lp = build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {0.0}, {1.0}, {false});
  EXPECT_EQ(run(lp).problem_class, "LP");
  EXPECT_TRUE(run(lp).convexity.empty()) << "no quadratic term, nothing to say";

  const Model milp = build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {0.0}, {1.0}, {true});
  EXPECT_EQ(run(milp).problem_class, "MILP");

  const Model qp =
      build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {0.0}, {1.0}, {false}, {{0, 0, 2.0}});
  const Diagnosis qp_diagnosis = run(qp);
  EXPECT_EQ(qp_diagnosis.problem_class, "QP");
  EXPECT_EQ(qp_diagnosis.convexity, "convex");
  EXPECT_TRUE(qp_diagnosis.warnings.empty());

  const Model miqp =
      build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {0.0}, {1.0}, {true}, {{0, 0, 2.0}});
  EXPECT_EQ(run(miqp).problem_class, "MIQP");

  // A model the QP engine would refuse must be flagged before the solve, not after it.
  const Model nonconvex =
      build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {-10.0}, {10.0}, {false}, {{0, 0, -2.0}});
  const Diagnosis refused = run(nonconvex);
  EXPECT_EQ(refused.convexity, "not convex");
  ASSERT_FALSE(refused.warnings.empty());
  EXPECT_NE(refused.warnings.front().find("not convex"), std::string::npos);
}

TEST(Diagnose, ThePresolvePreviewAgreesWithPresolveItself) {
  const Model model =
      build({{1.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}, {-kInfinity, -kInfinity}, {3.0, 50.0},
            {-1.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {1.0, 1.0, 1.0}, {false, false, false});

  Logger logger(nullptr);
  const presolve::Result truth = presolve::presolve(model, quiet(), logger);
  const Diagnosis d = run(model);

  EXPECT_TRUE(d.presolve_ran);
  EXPECT_EQ(d.presolve_rows_removed, truth.rows_removed());
  EXPECT_EQ(d.presolve_columns_removed, truth.cols_removed());
  EXPECT_GT(d.presolve_rows_removed + d.presolve_columns_removed, 0)
      << "a preview that removes nothing on this model is not testing anything";
}

TEST(Diagnose, AnInfeasibilityPresolveCanProveIsSaidBeforeTheSolve) {
  const Model model =
      build({{1.0}, {1.0}}, {5.0, -kInfinity}, {kInfinity, 2.0}, {1.0}, {0.0}, {10.0}, {false});
  const Diagnosis d = run(model);
  EXPECT_TRUE(d.presolve_proved_infeasible);
  ASSERT_FALSE(d.warnings.empty());
  EXPECT_NE(d.warnings.back().find("no feasible point"), std::string::npos)
      << d.warnings.back();
}

TEST(Diagnose, PresolveIsNotRunWhenTheOptionIsOff) {
  const Model model = build({{1.0, 1.0}}, {-kInfinity}, {3.0}, {1.0, 1.0}, {0.0, 0.0},
                            {1.0, 1.0}, {false, false});
  Options options = quiet();
  options.set_bool("presolve", false);
  Logger logger(nullptr);
  const Diagnosis d = analyse(model, options, logger);
  EXPECT_FALSE(d.presolve_ran);
  EXPECT_EQ(d.presolve_rows_removed, 0);
}

TEST(Diagnose, AModelWithNoRowsOrEntriesProducesNumbersRatherThanNaN) {
  Model model;
  model.col_cost = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {1.0};
  model.col_type = {VarType::kContinuous};
  model.matrix.reset(0, 1);
  model.matrix.finalize();
  model.hessian.reset(1, 1);
  model.hessian.finalize();

  const Diagnosis d = run(model);
  EXPECT_EQ(d.rows, 0);
  EXPECT_DOUBLE_EQ(d.density_percent, 0.0);
  EXPECT_DOUBLE_EQ(d.average_nonzeros_per_row, 0.0);
  EXPECT_DOUBLE_EQ(d.coefficient_ratio, 0.0);
  EXPECT_FALSE(std::isnan(d.density_percent));
  // And both renderings survive it.
  EXPECT_FALSE(format_text(d).empty());
  EXPECT_NO_THROW(nlohmann::json::parse(format_json(d)));
}

TEST(Diagnose, TheJsonCarriesTheSameNumbersAsTheText) {
  const Model model = build({{1.0, 2.0}, {3.0, 0.0}}, {-kInfinity, 1.0}, {4.0, 1.0},
                            {1.0, -1.0}, {0.0, 0.0}, {10.0, 1.0}, {false, true});
  const Diagnosis d = run(model);

  const nlohmann::json blob = nlohmann::json::parse(format_json(d));
  EXPECT_EQ(blob["model"]["rows"].get<Index>(), d.rows);
  EXPECT_EQ(blob["model"]["columns"].get<Index>(), d.columns);
  EXPECT_EQ(blob["model"]["nonzeros"].get<Index>(), d.nonzeros);
  EXPECT_EQ(blob["model"]["class"].get<std::string>(), d.problem_class);
  EXPECT_EQ(blob["columns"]["integer"].get<Index>(), d.integer_columns);
  EXPECT_EQ(blob["rows"]["equality"].get<Index>(), d.equality_rows);
  EXPECT_NEAR(blob["numerics"]["coefficient_ratio"].get<double>(), d.coefficient_ratio, 1e-12);
  EXPECT_EQ(blob["numerics"]["scaling_risk"].get<std::string>(), to_string(d.scaling_risk));
  EXPECT_EQ(blob["presolve"]["ran"].get<bool>(), d.presolve_ran);
  EXPECT_FALSE(blob["guidance"]["engine"].get<std::string>().empty());

  // The text rendering names the model and the class; a report that omitted either would be
  // useless to the reader it is written for.
  const std::string text = format_text(d);
  EXPECT_NE(text.find("TESTMODEL"), std::string::npos);
  EXPECT_NE(text.find(d.problem_class), std::string::npos);
  EXPECT_NE(text.find("Guidance"), std::string::npos);
}

TEST(Diagnose, NothingHereClaimsToPredictTheSolve) {
  // The report deliberately carries no solve-time or node-count estimate. This test exists so
  // that adding one is a deliberate act with a number behind it, rather than a helpful-looking
  // string somebody guessed.
  const Model model = build({{1.0}}, {-kInfinity}, {1.0}, {1.0}, {0.0}, {1.0}, {true});
  const std::string text = format_text(run(model));
  for (const char* forbidden :
       {"estimated time", "expected nodes", "will take", "seconds to solve"}) {
    EXPECT_EQ(text.find(forbidden), std::string::npos) << forbidden;
  }
}

}  // namespace
}  // namespace sankhya::diagnose
