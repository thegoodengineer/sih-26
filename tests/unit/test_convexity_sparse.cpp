// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the convexity decision at scale, and the two implementations of it (#303).
//
// Two things are being checked here, and the second is the one that matters.
//
// SCALE. The dense test needs an n x n working set: 20 GB at 50,000 columns, which is why it
// refused anything past a couple of thousand columns and a large sparse convex QP could not
// be solved at all. The sparse test decides the same question through the LDL^T the interior
// point already uses, in memory proportional to the factor.
//
// SOUNDNESS. A convexity test that says "convex" about an indefinite matrix is worse than one
// that refuses: the QP engine then returns a local point, labelled optimal, for a model with
// no global minimum. Q = [[0, 1], [1, 0]] is the smallest witness - a pure saddle, x'Qx =
// 2*x0*x1 - and the dense test used to pass it, because a zero pivot was skipped without
// checking that the column beside it vanished too. Both implementations are held to it here,
// and to each other on matrices nobody chose.

#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "qp/convexity.hpp"

namespace sankhya::qp {
namespace {

/// A model with nothing but a Hessian: bounds wide, no rows, no linear cost.
Model hessian_only(Index n, const std::vector<std::tuple<Index, Index, double>>& lower,
                   ObjSense sense = ObjSense::kMinimize) {
  Model model;
  model.sense = sense;
  model.col_cost.assign(static_cast<std::size_t>(n), 0.0);
  model.col_lower.assign(static_cast<std::size_t>(n), -10.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 10.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.matrix.reset(0, n);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  for (const auto& [i, j, v] : lower) model.hessian.add_entry(i, j, v);
  model.hessian.finalize();
  return model;
}

/// x^T Q x for the model's Hessian, in minimization sense. The certificate a test needs when
/// it claims a matrix is indefinite: the arithmetic below shares no code with the solver's.
double quadratic_form(const Model& model, const std::vector<double>& x) {
  double total = 0.0;
  const double sense = model.sense_multiplier();
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      const double v = sense * column.values[k];
      const double term = v * x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(j)];
      total += (i == j) ? term : 2.0 * term;  // the stored entry stands for both halves
    }
  }
  return total;
}

// =========================================================================================
// Soundness: the saddle that used to pass
// =========================================================================================

TEST(SparseConvexity, AZeroDiagonalBesideANonzeroOffDiagonalIsIndefinite) {
  // Q = [[0, 1], [1, 0]]: eigenvalues +1 and -1.
  const Model model = hessian_only(2, {{1, 0, 1.0}});

  // The certificate, computed here rather than trusted: (1, -1) makes the form negative.
  EXPECT_LT(quadratic_form(model, {1.0, -1.0}), 0.0);
  EXPECT_GT(quadratic_form(model, {1.0, 1.0}), 0.0);

  const ConvexityResult sparse = check_convexity(model);
  const ConvexityResult dense = check_convexity_dense(model);
  EXPECT_EQ(sparse.verdict, Convexity::kIndefinite) << sparse.detail;
  EXPECT_EQ(dense.verdict, Convexity::kIndefinite) << dense.detail;
}

TEST(SparseConvexity, AnIndefiniteQpIsRefusedRatherThanSolvedToALocalPoint) {
  Model model = hessian_only(2, {{1, 0, 1.0}});
  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kModelError);
  EXPECT_NE(solution.message.find("not convex"), std::string::npos) << solution.message;
}

TEST(SparseConvexity, ARankDeficientButSemidefiniteHessianStaysConvex) {
  // Q = [[1, 1], [1, 1]] = v v^T with v = (1, 1). Singular, and positive semidefinite:
  // x'Qx = (x0 + x1)^2 >= 0. The zero pivot here is legitimate and must not be refused.
  const Model model = hessian_only(2, {{0, 0, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});
  EXPECT_DOUBLE_EQ(quadratic_form(model, {1.0, -1.0}), 0.0);

  EXPECT_EQ(check_convexity(model).verdict, Convexity::kConvex);
  EXPECT_EQ(check_convexity_dense(model).verdict, Convexity::kConvex);
}

TEST(SparseConvexity, AConcaveMaximizationIsConvexInMinimizationSense) {
  // maximize -x^2 is concave, so minimizing +x^2 - the sense multiplier decides it.
  const Model concave = hessian_only(1, {{0, 0, -2.0}}, ObjSense::kMaximize);
  EXPECT_EQ(check_convexity(concave).verdict, Convexity::kConvex);

  // maximize +x^2 is unbounded above and must be refused.
  const Model convex_but_maximized = hessian_only(1, {{0, 0, 2.0}}, ObjSense::kMaximize);
  EXPECT_EQ(check_convexity(convex_but_maximized).verdict, Convexity::kIndefinite);
}

// =========================================================================================
// The two implementations against each other
// =========================================================================================

TEST(SparseConvexity, DenseAndSparseAgreeOnMatricesNobodyChose) {
  // Half the instances are positive semidefinite by construction (B^T B), half have a
  // negative direction planted in them. The dense reference and the sparse decision must
  // return the same verdict on every one; disagreement is the only interesting outcome and
  // there is no tolerance to tune it away with, because both read the same slack rule.
  std::mt19937 rng(20260918);
  std::uniform_int_distribution<int> size(1, 12);
  std::uniform_int_distribution<int> entry(-3, 3);
  int psd_cases = 0;
  int indefinite_cases = 0;

  for (int instance = 0; instance < 300; ++instance) {
    const auto n = static_cast<Index>(size(rng));
    const auto un = static_cast<std::size_t>(n);

    // B is n x n with small integer entries; Q = B^T B is positive semidefinite exactly.
    std::vector<std::vector<double>> b(un, std::vector<double>(un, 0.0));
    for (std::size_t r = 0; r < un; ++r) {
      for (std::size_t c = 0; c < un; ++c) {
        if (entry(rng) > 1) b[r][c] = static_cast<double>(entry(rng));
      }
    }
    std::vector<std::vector<double>> q(un, std::vector<double>(un, 0.0));
    for (std::size_t i = 0; i < un; ++i) {
      for (std::size_t j = 0; j < un; ++j) {
        double sum = 0.0;
        for (std::size_t k = 0; k < un; ++k) sum += b[k][i] * b[k][j];
        q[i][j] = sum;
      }
    }

    const bool plant_negative = (instance % 2) == 1;
    if (plant_negative) {
      // Subtract enough from one diagonal entry to make e_i^T Q e_i negative, or - half the
      // time - plant the subtler case: a zero diagonal beside a nonzero off-diagonal.
      const auto i = static_cast<std::size_t>(rng() % un);
      if (n > 1 && (instance % 4) == 1) {
        const std::size_t j = (i + 1) % un;
        for (std::size_t k = 0; k < un; ++k) {
          q[i][k] = 0.0;
          q[k][i] = 0.0;
        }
        q[i][j] = 1.0;
        q[j][i] = 1.0;
        q[j][j] = 0.0;
      } else {
        q[i][i] -= q[i][i] + 1.0;
      }
    }

    std::vector<std::tuple<Index, Index, double>> lower;
    for (Index i = 0; i < n; ++i) {
      for (Index j = 0; j <= i; ++j) {
        const double v = q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        if (v != 0.0) lower.emplace_back(i, j, v);
      }
    }
    const Model model = hessian_only(n, lower);

    const ConvexityResult sparse = check_convexity(model);
    const ConvexityResult dense = check_convexity_dense(model);
    ASSERT_EQ(sparse.verdict, dense.verdict)
        << "instance " << instance << " n=" << n << "\n  sparse: " << sparse.detail
        << "\n  dense:  " << dense.detail;
    if (sparse.verdict == Convexity::kConvex) ++psd_cases;
    if (sparse.verdict == Convexity::kIndefinite) ++indefinite_cases;
  }

  // A run where everything landed on one verdict would agree trivially and prove nothing.
  EXPECT_GT(psd_cases, 50);
  EXPECT_GT(indefinite_cases, 50);
}

// =========================================================================================
// Scale: the case the dense test refused
// =========================================================================================

TEST(SparseConvexity, ALargeSparseHessianIsDecidedRatherThanRefused) {
  // 50,000 columns. The dense working set for this is 50,000^2 * 8 bytes = 20 GB, which is
  // the allocation #303 is about; the sparse test holds one entry per column.
  constexpr Index n = 50000;
  std::vector<std::tuple<Index, Index, double>> lower;
  lower.reserve(static_cast<std::size_t>(n) * 2);
  for (Index i = 0; i < n; ++i) {
    lower.emplace_back(i, i, 2.0);
    if (i > 0) lower.emplace_back(i, i - 1, 1.0);  // tridiagonal, diagonally dominant
  }
  const Model model = hessian_only(n, lower);

  const ConvexityResult sparse = check_convexity(model);
  EXPECT_EQ(sparse.verdict, Convexity::kConvex) << sparse.detail;

  // The dense reference declines at this size, which is the behaviour the production path
  // used to inherit.
  EXPECT_EQ(check_convexity_dense(model).verdict, Convexity::kUnverified);
}

TEST(SparseConvexity, ALargeSparseIndefiniteHessianIsStillCaught) {
  constexpr Index n = 20000;
  std::vector<std::tuple<Index, Index, double>> lower;
  for (Index i = 0; i < n; ++i) lower.emplace_back(i, i, 2.0);
  lower.emplace_back(n / 2, n / 2, -6.0);  // sums with the diagonal entry above it: -4
  const Model model = hessian_only(n, lower);

  const ConvexityResult sparse = check_convexity(model);
  EXPECT_EQ(sparse.verdict, Convexity::kIndefinite) << sparse.detail;
}

TEST(SparseConvexity, ALargeSparseConvexQpReachesTheEngineInsteadOfBeingRefused) {
  // End to end, at a size the old limit refused: minimize sum (x_j - 1)^2 over 4,000 columns
  // with one coupling row. What is being checked is that the model is ACCEPTED - that the
  // convexity gate is no longer the thing that stops it - not how fast Condat-Vu converges.
  constexpr Index n = 4000;
  Model model;
  model.col_cost.assign(static_cast<std::size_t>(n), -2.0);
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 10.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = {-kInfinity};
  model.row_upper = {static_cast<double>(n)};
  model.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) model.matrix.add_entry(0, j, 1.0);
  model.matrix.finalize();
  model.hessian.reset(n, n);
  for (Index j = 0; j < n; ++j) model.hessian.add_entry(j, j, 2.0);
  model.hessian.finalize();
  ASSERT_EQ(model.validate(), "");

  Options options;
  options.set_bool("log_to_console", false);
  options.set_double("time_limit", 20.0);
  const Solution solution = solve(model, options);

  EXPECT_NE(solution.status, SolveStatus::kModelError) << solution.message;
  EXPECT_EQ(solution.message.find("convexity could not be established"), std::string::npos)
      << solution.message;
}

}  // namespace
}  // namespace sankhya::qp
