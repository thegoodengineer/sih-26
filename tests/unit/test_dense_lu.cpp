// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense LU tests.
//
// The simplex reduces to two calls on this class, FTRAN and BTRAN, and it makes several
// hundred of them per solve. A transposed solve that is subtly wrong does not crash: it
// yields duals that look plausible, reduced costs that price the wrong column, and a
// simplex that terminates confidently at a non-optimal vertex. So the transpose is checked
// against an INDEPENDENT route to the same answer - factorize A^T and solve normally - and
// not against another call into the same code path.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

#include "simplex/dense_lu.hpp"

namespace sankhya {
namespace {

/// Column-major m x m matrix, the layout DenseLu consumes.
struct Dense {
  Index m = 0;
  std::vector<double> a;

  explicit Dense(Index dimension)
      : m(dimension),
        a(static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension), 0.0) {}

  double& operator()(Index row, Index col) {
    return a[static_cast<std::size_t>(col) * static_cast<std::size_t>(m) +
             static_cast<std::size_t>(row)];
  }
  double operator()(Index row, Index col) const {
    return a[static_cast<std::size_t>(col) * static_cast<std::size_t>(m) +
             static_cast<std::size_t>(row)];
  }

  [[nodiscard]] Dense transposed() const {
    Dense t(m);
    for (Index i = 0; i < m; ++i) {
      for (Index j = 0; j < m; ++j) t(j, i) = (*this)(i, j);
    }
    return t;
  }

  /// y = A x, computed directly from the definition. This is the reference the factorized
  /// solve is measured against.
  [[nodiscard]] std::vector<double> multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    for (Index j = 0; j < m; ++j) {
      for (Index i = 0; i < m; ++i) {
        y[static_cast<std::size_t>(i)] += (*this)(i, j) * x[static_cast<std::size_t>(j)];
      }
    }
    return y;
  }
};

[[nodiscard]] double max_difference(const std::vector<double>& a,
                                    const std::vector<double>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
  return worst;
}

TEST(DenseLu, SolvesATwoByTwoSystem) {
  Dense a(2);
  a(0, 0) = 4.0;
  a(0, 1) = 3.0;
  a(1, 0) = 6.0;
  a(1, 1) = 3.0;

  DenseLu lu;
  ASSERT_TRUE(lu.factorize(a.a, 2, tol::kPivotTolerance));

  // A [1, 2]^T = [10, 12]^T
  std::vector<double> b{10.0, 12.0};
  lu.solve(b.data());
  EXPECT_NEAR(b[0], 1.0, 1e-12);
  EXPECT_NEAR(b[1], 2.0, 1e-12);
}

TEST(DenseLu, RequiresPivotingWhenTheLeadingEntryIsZero) {
  // Without row interchange the very first pivot is zero and the factorization dies. This
  // configuration is not exotic: the simplex reaches it whenever a logical column with a
  // structural zero on the diagonal enters the basis.
  Dense a(2);
  a(0, 0) = 0.0;
  a(0, 1) = 1.0;
  a(1, 0) = 1.0;
  a(1, 1) = 0.0;

  DenseLu lu;
  ASSERT_TRUE(lu.factorize(a.a, 2, tol::kPivotTolerance));
  std::vector<double> b{3.0, 5.0};
  lu.solve(b.data());
  EXPECT_NEAR(b[0], 5.0, 1e-12);
  EXPECT_NEAR(b[1], 3.0, 1e-12);
}

TEST(DenseLu, ReportsASingularMatrix) {
  Dense a(3);
  a(0, 0) = 1.0;
  a(0, 1) = 2.0;
  a(0, 2) = 3.0;
  a(1, 0) = 2.0;
  a(1, 1) = 4.0;
  a(1, 2) = 6.0;  // exactly twice row 0
  a(2, 0) = 1.0;
  a(2, 1) = 1.0;
  a(2, 2) = 1.0;

  DenseLu lu;
  EXPECT_FALSE(lu.factorize(a.a, 3, tol::kPivotTolerance));
}

TEST(DenseLu, HandlesTheEmptyBasis) {
  // A model with no rows produces a 0 x 0 basis. Every routine must degrade to a no-op
  // rather than indexing into nothing.
  DenseLu lu;
  EXPECT_TRUE(lu.factorize({}, 0, tol::kPivotTolerance));
  EXPECT_EQ(lu.dimension(), 0);
  lu.solve(nullptr);
  lu.solve_transpose(nullptr);
}

TEST(DenseLu, IdentityBasisIsExact) {
  // The slack basis every solve starts from is -I. Round-tripping it must be bit-exact,
  // not merely close, or the very first iterate already carries error.
  constexpr Index m = 12;
  Dense a(m);
  for (Index i = 0; i < m; ++i) a(i, i) = -1.0;

  DenseLu lu;
  ASSERT_TRUE(lu.factorize(a.a, m, tol::kPivotTolerance));
  std::vector<double> b(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) b[static_cast<std::size_t>(i)] = static_cast<double>(i) + 1.0;
  lu.solve(b.data());
  for (Index i = 0; i < m; ++i) {
    EXPECT_DOUBLE_EQ(b[static_cast<std::size_t>(i)], -(static_cast<double>(i) + 1.0));
  }
}

TEST(DenseLu, FuzzSolveAgainstTheDefinition) {
  std::mt19937 rng(20260825);
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  double worst_residual = 0.0;
  int factorized = 0;

  for (int trial = 0; trial < 300; ++trial) {
    const Index m = 1 + static_cast<Index>(trial % 18);
    Dense a(m);
    for (Index j = 0; j < m; ++j) {
      for (Index i = 0; i < m; ++i) {
        // 60% sparse, plus a strong diagonal so most draws are comfortably nonsingular.
        a(i, j) = (unit(rng) < 0.4) ? value(rng) : 0.0;
      }
      a(j, j) += 3.0 + unit(rng);
    }

    std::vector<double> x(static_cast<std::size_t>(m));
    for (double& v : x) v = value(rng);
    std::vector<double> b = a.multiply(x);

    DenseLu lu;
    if (!lu.factorize(a.a, m, tol::kPivotTolerance)) continue;
    ++factorized;
    lu.solve(b.data());
    worst_residual = std::max(worst_residual, max_difference(b, x));
  }

  EXPECT_GT(factorized, 250) << "the generator produced too many singular matrices to be a "
                                "meaningful test";
  EXPECT_LT(worst_residual, 1e-9) << "worst reconstruction error " << worst_residual;
}

TEST(DenseLu, FuzzTransposeSolveAgainstAnIndependentFactorization) {
  // solve_transpose(A, b) must agree with solve(A^T, b) where A^T is formed explicitly and
  // factorized on its own. The two share no code beyond factorize() itself, so a sign or a
  // permutation-order error in the transposed triangular solves cannot cancel out.
  std::mt19937 rng(987654321);
  std::uniform_real_distribution<double> value(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  double worst = 0.0;
  int compared = 0;

  for (int trial = 0; trial < 300; ++trial) {
    const Index m = 1 + static_cast<Index>(trial % 15);
    Dense a(m);
    for (Index j = 0; j < m; ++j) {
      for (Index i = 0; i < m; ++i) a(i, j) = (unit(rng) < 0.5) ? value(rng) : 0.0;
      a(j, j) += 2.5 + unit(rng);
    }

    std::vector<double> b(static_cast<std::size_t>(m));
    for (double& v : b) v = value(rng);

    DenseLu lu;
    if (!lu.factorize(a.a, m, tol::kPivotTolerance)) continue;
    std::vector<double> via_transpose_solve = b;
    lu.solve_transpose(via_transpose_solve.data());

    const Dense at = a.transposed();
    DenseLu lu_transposed;
    if (!lu_transposed.factorize(at.a, m, tol::kPivotTolerance)) continue;
    std::vector<double> via_explicit_transpose = b;
    lu_transposed.solve(via_explicit_transpose.data());

    ++compared;
    worst = std::max(worst, max_difference(via_transpose_solve, via_explicit_transpose));
  }

  EXPECT_GT(compared, 250);
  EXPECT_LT(worst, 1e-9) << "BTRAN disagrees with an explicit transposed factorization by "
                         << worst;
}

TEST(DenseLu, TransposeSolveSatisfiesItsOwnEquation) {
  // A^T z = b, checked by multiplying back through A^T. Independent of the previous test:
  // this one would still catch an error that happened to be symmetric.
  std::mt19937 rng(13579);
  std::uniform_real_distribution<double> value(-3.0, 3.0);

  constexpr Index m = 9;
  Dense a(m);
  for (Index j = 0; j < m; ++j) {
    for (Index i = 0; i < m; ++i) a(i, j) = value(rng);
    a(j, j) += 5.0;
  }

  DenseLu lu;
  ASSERT_TRUE(lu.factorize(a.a, m, tol::kPivotTolerance));

  std::vector<double> b(static_cast<std::size_t>(m));
  for (double& v : b) v = value(rng);
  std::vector<double> z = b;
  lu.solve_transpose(z.data());

  const std::vector<double> reconstructed = a.transposed().multiply(z);
  EXPECT_LT(max_difference(reconstructed, b), 1e-10);
}

}  // namespace
}  // namespace sankhya
