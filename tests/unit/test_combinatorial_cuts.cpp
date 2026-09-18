// SPDX-License-Identifier: Apache-2.0
// SANKHYA - clique and {0,1/2}-Chvatal-Gomory cuts (#358).
//
// The obligation cuts.hpp sets for every family: prove no cut removes an integer-feasible
// point, and show the harness can fail. Here the proof is ENUMERATION - every integer point
// of every small random model is listed and checked against every cut generated at a random
// fractional point - which needs no oracle and misses nothing within the models it covers.
// The negative control is a deliberately invalid cut the same checker must reject.

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "mip/combinatorial_cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Model integer_model(const std::vector<std::vector<double>>& rows,
                    const std::vector<double>& row_lower, const std::vector<double>& row_upper,
                    const std::vector<double>& lower, const std::vector<double>& upper) {
  Model m;
  const auto n = lower.size();
  m.col_cost.assign(n, 0.0);
  m.col_lower = lower;
  m.col_upper = upper;
  m.col_type.assign(n, VarType::kInteger);
  m.matrix.reset(static_cast<Index>(rows.size()), static_cast<Index>(n));
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (rows[i][j] != 0.0) {
        m.matrix.add_entry(static_cast<Index>(i), static_cast<Index>(j), rows[i][j]);
      }
    }
  }
  m.matrix.finalize();
  m.row_lower = row_lower;
  m.row_upper = row_upper;
  m.hessian.reset(static_cast<Index>(n), static_cast<Index>(n));
  m.hessian.finalize();
  return m;
}

/// Every integer point of the model's box that satisfies every row.
std::vector<std::vector<double>> integer_points(const Model& m) {
  const auto n = static_cast<std::size_t>(m.num_cols());
  std::vector<std::int64_t> lo(n);
  std::vector<std::int64_t> hi(n);
  for (std::size_t j = 0; j < n; ++j) {
    lo[j] = static_cast<std::int64_t>(std::ceil(m.col_lower[j]));
    hi[j] = static_cast<std::int64_t>(std::floor(m.col_upper[j]));
    if (lo[j] > hi[j]) return {};
  }
  std::vector<std::vector<double>> points;
  std::vector<std::int64_t> x = lo;
  for (;;) {
    bool ok = true;
    for (Index i = 0; i < m.num_rows() && ok; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        activity += m.matrix.at(i, static_cast<Index>(j)) * static_cast<double>(x[j]);
      }
      const auto u = static_cast<std::size_t>(i);
      ok = activity >= m.row_lower[u] - 1e-9 && activity <= m.row_upper[u] + 1e-9;
    }
    if (ok) points.emplace_back(x.begin(), x.end());
    std::size_t k = 0;
    while (k < n && x[k] == hi[k]) {
      x[k] = lo[k];
      ++k;
    }
    if (k == n) break;
    ++x[k];
  }
  return points;
}

/// The first integer point the cut removes, or empty.
std::vector<double> removed_point(const Cut& cut,
                                  const std::vector<std::vector<double>>& points) {
  for (const std::vector<double>& p : points) {
    double activity = 0.0;
    for (std::size_t j = 0; j < p.size(); ++j) activity += cut.coeff[j] * p[j];
    if (activity > cut.rhs + 1e-9) return p;
  }
  return {};
}

Solution at(const std::vector<double>& x) {
  Solution s;
  s.col_value = x;
  return s;
}

// ---- Closed-form cases ---------------------------------------------------------------------

TEST(CliqueCuts, APackingTriangleGivesTheCliqueInequality) {
  // x0 + x1 <= 1, x1 + x2 <= 1, x0 + x2 <= 1: pairwise, so x0 + x1 + x2 <= 1. The LP point
  // (1/2, 1/2, 1/2) satisfies every row and violates the clique.
  const Model m = integer_model({{1, 1, 0}, {0, 1, 1}, {1, 0, 1}}, {-kInf, -kInf, -kInf},
                                {1, 1, 1}, {0, 0, 0}, {1, 1, 1});
  CombinatorialCutStats stats;
  const std::vector<Cut> cuts =
      generate_clique_cuts(m, at({0.5, 0.5, 0.5}), m.col_lower, m.col_upper, &stats);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_EQ(cuts[0].coeff, (std::vector<double>{1, 1, 1}));
  EXPECT_EQ(cuts[0].rhs, 1.0);
  EXPECT_EQ(stats.conflict_edges, 3);
}

TEST(CliqueCuts, AKnapsackRowGivesConflictsOnlyWhereTheyAreReal) {
  // 3 x0 + 3 x1 + 1 x2 <= 4: x0 and x1 cannot both be 1; x2 fits with either.
  const Model m = integer_model({{3, 3, 1}}, {-kInf}, {4}, {0, 0, 0}, {1, 1, 1});
  CombinatorialCutStats stats;
  (void)generate_clique_cuts(m, at({0.6, 0.6, 0.0}), m.col_lower, m.col_upper, &stats);
  EXPECT_EQ(stats.conflict_edges, 1);
}

TEST(ZeroHalfCuts, PairsAloneCannotReachTheTriangleAndClaimNothingFalse) {
  // The triangle x0 + x1 <= 1, x1 + x2 <= 1, x0 + x2 <= 1 has the {0,1/2} cut x0 + x1 + x2 <= 1
  // only from ALL THREE rows at 1/2. This generator tries single rows and pairs, so it cannot
  // reach that cut - the bounded neighbourhood the PR states - and the test holds it to the
  // weaker promise: whatever it does return removes no integer point.
  const Model m = integer_model({{1, 1, 0}, {0, 1, 1}, {1, 0, 1}}, {-kInf, -kInf, -kInf},
                                {1, 1, 1}, {0, 0, 0}, {1, 1, 1});
  const std::vector<Cut> cuts =
      generate_zero_half_cuts(m, at({0.5, 0.5, 0.5}), m.col_lower, m.col_upper);
  for (const Cut& cut : cuts) EXPECT_TRUE(removed_point(cut, integer_points(m)).empty());
}

TEST(ZeroHalfCuts, AnOddRightHandSideRoundsDown) {
  // 2 x0 + 2 x1 <= 3 over integers: halving gives x0 + x1 <= 1.5 -> 1. At (0.75, 0.75) the
  // row holds (3 <= 3) and the cut is violated (1.5 > 1).
  const Model m = integer_model({{2, 2}}, {-kInf}, {3}, {0, 0}, {5, 5});
  const std::vector<Cut> cuts =
      generate_zero_half_cuts(m, at({0.75, 0.75}), m.col_lower, m.col_upper);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_EQ(cuts[0].coeff, (std::vector<double>{1, 1}));
  EXPECT_EQ(cuts[0].rhs, 1.0);
}

TEST(ZeroHalfCuts, AShiftedLowerBoundIsCarriedBackIntoTheCut) {
  // x in [2, 9]: 2 x0 + 2 x1 <= 11 -> x' = x - 2: 2 x0' + 2 x1' <= 3 -> x0' + x1' <= 1
  // -> x0 + x1 <= 5. At (2.75, 2.75) the row holds and the cut does not.
  const Model m = integer_model({{2, 2}}, {-kInf}, {11}, {2, 2}, {9, 9});
  const std::vector<Cut> cuts =
      generate_zero_half_cuts(m, at({2.75, 2.75}), m.col_lower, m.col_upper);
  ASSERT_EQ(cuts.size(), 1u);
  EXPECT_EQ(cuts[0].rhs, 5.0);
  EXPECT_TRUE(removed_point(cuts[0], integer_points(m)).empty());
}

// ---- The harness can fail ------------------------------------------------------------------

TEST(CombinatorialCuts, TheCheckerRejectsAnInvalidCut) {
  const Model m = integer_model({{1, 1}}, {-kInf}, {1}, {0, 0}, {1, 1});
  Cut invalid;
  invalid.coeff = {1.0, 1.0};
  invalid.rhs = 0.0;  // removes (1, 0), which is feasible
  EXPECT_FALSE(removed_point(invalid, integer_points(m)).empty());
}

// ---- Validity, by enumeration --------------------------------------------------------------

TEST(CombinatorialCuts, NoCutRemovesAnyIntegerPointOfRandomModels) {
  std::mt19937 rng(358);
  std::uniform_int_distribution<int> coefficient(-2, 3);
  std::uniform_int_distribution<int> small(0, 2);
  std::uniform_real_distribution<double> point(0.0, 1.0);
  int clique_cuts = 0;
  int zero_half_cuts = 0;
  int models = 0;
  for (int trial = 0; trial < 1500; ++trial) {
    const std::size_t n = 3 + static_cast<std::size_t>(trial % 4);
    const std::size_t rows = 2 + static_cast<std::size_t>(trial % 3);
    const bool binary = trial % 2 == 0;
    std::vector<double> lower(n);
    std::vector<double> upper(n);
    for (std::size_t j = 0; j < n; ++j) {
      lower[j] = binary ? 0.0 : static_cast<double>(small(rng)) - 1.0;
      upper[j] = binary ? 1.0 : lower[j] + 1.0 + static_cast<double>(small(rng));
    }
    std::vector<std::vector<double>> a(rows, std::vector<double>(n));
    std::vector<double> rl(rows, -kInf);
    std::vector<double> ru(rows, kInf);
    for (std::size_t i = 0; i < rows; ++i) {
      double reach = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        a[i][j] = coefficient(rng);
        reach += std::fabs(a[i][j]);
      }
      // A mix of packing, covering and equality rows with odd right-hand sides.
      const double b = std::floor(reach / 2.0) + (trial % 3 == 0 ? 0.5 : 0.0);
      if (i % 3 == 0) {
        ru[i] = b;
      } else if (i % 3 == 1) {
        rl[i] = -b;
      } else {
        rl[i] = std::floor(b / 2.0);
        ru[i] = std::floor(b / 2.0);
      }
    }
    const Model m = integer_model(a, rl, ru, lower, upper);
    const std::vector<std::vector<double>> points = integer_points(m);
    if (points.empty()) continue;
    ++models;
    for (int probe = 0; probe < 3; ++probe) {
      std::vector<double> x(n);
      for (std::size_t j = 0; j < n; ++j) x[j] = lower[j] + point(rng) * (upper[j] - lower[j]);
      for (const Cut& cut : generate_clique_cuts(m, at(x), lower, upper)) {
        ++clique_cuts;
        const std::vector<double> removed = removed_point(cut, points);
        ASSERT_TRUE(removed.empty())
            << "trial " << trial << ": a clique cut removes a feasible point";
      }
      for (const Cut& cut : generate_zero_half_cuts(m, at(x), lower, upper)) {
        ++zero_half_cuts;
        const std::vector<double> removed = removed_point(cut, points);
        ASSERT_TRUE(removed.empty())
            << "trial " << trial << ": a {0,1/2} cut removes a feasible point";
      }
    }
  }
  EXPECT_GT(models, 500);
  EXPECT_GT(clique_cuts, 50) << "the sweep should exercise the clique generator";
  EXPECT_GT(zero_half_cuts, 50) << "and the {0,1/2} generator";
}

}  // namespace
}  // namespace sankhya::mip
