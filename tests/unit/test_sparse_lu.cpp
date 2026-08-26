// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse LU tests.
//
// A permutation or ordering error in a sparse LU is invisible. It does not crash, does not
// warn, and does not even produce obviously wrong numbers: FTRAN returns a vector of
// plausible magnitude, the simplex prices some column with it, and the solve terminates at a
// non-optimal vertex reporting "optimal". Two independent checks are therefore applied to
// every random instance:
//
//   1. The RESIDUAL. B x must reproduce b, recomputed from the original matrix. This catches
//      any factorization that is not actually a factorization of B.
//   2. The DENSE ORACLE. DenseLu factorizes the same matrix by a completely different route
//      (right-looking, partial pivoting, dense storage) and must return the same answer.
//      This is what catches a transposed solve that happens to be self-consistent.
//
// Check 2 is the reason lu.hpp forbids deleting DenseLu when the sparse version ships.

#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

#include "la/lu.hpp"
#include "simplex/dense_lu.hpp"

namespace sankhya {
namespace {

constexpr double kThreshold = tol::kMarkowitzThreshold;

/// A square matrix held both ways: dense column-major for DenseLu, and as sparse columns for
/// SparseLu. Built once so the two factorizations cannot disagree about the input.
class TestMatrix {
 public:
  explicit TestMatrix(Index dimension) : m_(dimension) {
    dense_.assign(static_cast<std::size_t>(dimension) * static_cast<std::size_t>(dimension),
                  0.0);
    rows_.resize(static_cast<std::size_t>(dimension));
    values_.resize(static_cast<std::size_t>(dimension));
  }

  void set(Index row, Index col, double value) {
    dense_[static_cast<std::size_t>(col) * static_cast<std::size_t>(m_) +
           static_cast<std::size_t>(row)] = value;
    rows_[static_cast<std::size_t>(col)].push_back(row);
    values_[static_cast<std::size_t>(col)].push_back(value);
  }

  [[nodiscard]] std::vector<LuColumn> columns() const {
    std::vector<LuColumn> out(static_cast<std::size_t>(m_));
    for (Index j = 0; j < m_; ++j) {
      const auto uj = static_cast<std::size_t>(j);
      out[uj].size = static_cast<Index>(rows_[uj].size());
      if (out[uj].size > 0) {
        out[uj].rows = rows_[uj].data();
        out[uj].values = values_[uj].data();
      }
    }
    return out;
  }

  [[nodiscard]] const std::vector<double>& dense() const { return dense_; }
  [[nodiscard]] Index dimension() const { return m_; }

  /// y = A x, straight from the definition.
  [[nodiscard]] std::vector<double> multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(m_), 0.0);
    for (Index j = 0; j < m_; ++j) {
      const double xj = x[static_cast<std::size_t>(j)];
      if (xj == 0.0) continue;
      const auto uj = static_cast<std::size_t>(j);
      for (std::size_t k = 0; k < rows_[uj].size(); ++k) {
        y[static_cast<std::size_t>(rows_[uj][k])] += values_[uj][k] * xj;
      }
    }
    return y;
  }

  /// y = A^T x.
  [[nodiscard]] std::vector<double> transpose_multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(m_), 0.0);
    for (Index j = 0; j < m_; ++j) {
      const auto uj = static_cast<std::size_t>(j);
      double dot = 0.0;
      for (std::size_t k = 0; k < rows_[uj].size(); ++k) {
        dot += values_[uj][k] * x[static_cast<std::size_t>(rows_[uj][k])];
      }
      y[uj] = dot;
    }
    return y;
  }

 private:
  Index m_;
  std::vector<double> dense_;
  std::vector<std::vector<Index>> rows_;
  std::vector<std::vector<double>> values_;
};

[[nodiscard]] double max_difference(const std::vector<double>& a,
                                    const std::vector<double>& b) {
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
  return worst;
}

/// Relative residual of a claimed solution to A x = b.
[[nodiscard]] double residual(const TestMatrix& matrix, const std::vector<double>& x,
                              const std::vector<double>& b) {
  const std::vector<double> product = matrix.multiply(x);
  double scale = 1.0;
  for (const double v : b) scale = std::max(scale, std::fabs(v));
  return max_difference(product, b) / scale;
}

[[nodiscard]] double transpose_residual(const TestMatrix& matrix, const std::vector<double>& x,
                                        const std::vector<double>& b) {
  const std::vector<double> product = matrix.transpose_multiply(x);
  double scale = 1.0;
  for (const double v : b) scale = std::max(scale, std::fabs(v));
  return max_difference(product, b) / scale;
}

// =========================================================================================
// Hand-checkable cases
// =========================================================================================

TEST(SparseLu, SolvesATwoByTwoSystem) {
  TestMatrix matrix(2);
  matrix.set(0, 0, 4.0);
  matrix.set(0, 1, 3.0);
  matrix.set(1, 0, 6.0);
  matrix.set(1, 1, 3.0);

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), 2, tol::kPivotTolerance, kThreshold));

  std::vector<double> b{10.0, 12.0};  // A [1, 2]^T
  lu.solve(b.data());
  EXPECT_NEAR(b[0], 1.0, 1e-12);
  EXPECT_NEAR(b[1], 2.0, 1e-12);
}

TEST(SparseLu, IdentityBasisIsExact) {
  // The slack basis every solve starts from is -I. It must round-trip bit-exactly, not
  // merely closely, or the very first iterate already carries error.
  constexpr Index m = 12;
  TestMatrix matrix(m);
  for (Index i = 0; i < m; ++i) matrix.set(i, i, -1.0);

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold));
  std::vector<double> b(static_cast<std::size_t>(m));
  for (Index i = 0; i < m; ++i) b[static_cast<std::size_t>(i)] = static_cast<double>(i) + 1.0;
  lu.solve(b.data());
  for (Index i = 0; i < m; ++i) {
    EXPECT_DOUBLE_EQ(b[static_cast<std::size_t>(i)], -(static_cast<double>(i) + 1.0));
  }
}

TEST(SparseLu, PermutedIdentityExercisesTheOrdering) {
  // A pure permutation matrix has no arithmetic at all, so anything wrong with the answer is
  // purely an indexing error - which is exactly the failure mode this class is prone to.
  constexpr Index m = 6;
  const Index target[m] = {3, 5, 0, 4, 1, 2};
  TestMatrix matrix(m);
  for (Index j = 0; j < m; ++j) matrix.set(target[j], j, 2.0);

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold));

  std::vector<double> b(static_cast<std::size_t>(m), 0.0);
  for (Index i = 0; i < m; ++i) b[static_cast<std::size_t>(i)] = static_cast<double>(i) + 1.0;
  const std::vector<double> original = b;
  lu.solve(b.data());
  // A x = b with A e_j = 2 e_{target[j]} means x[j] = b[target[j]] / 2.
  for (Index j = 0; j < m; ++j) {
    EXPECT_DOUBLE_EQ(b[static_cast<std::size_t>(j)],
                     original[static_cast<std::size_t>(target[j])] / 2.0);
  }
}

TEST(SparseLu, ReportsASingularMatrix) {
  TestMatrix matrix(3);
  matrix.set(0, 0, 1.0);
  matrix.set(0, 1, 2.0);
  matrix.set(0, 2, 3.0);
  matrix.set(1, 0, 2.0);
  matrix.set(1, 1, 4.0);
  matrix.set(1, 2, 6.0);  // exactly twice row 0
  matrix.set(2, 0, 1.0);
  matrix.set(2, 1, 1.0);
  matrix.set(2, 2, 1.0);

  SparseLu lu;
  EXPECT_FALSE(lu.factorize(matrix.columns(), 3, tol::kPivotTolerance, kThreshold));
}

TEST(SparseLu, ReportsAStructurallyEmptyColumn) {
  TestMatrix matrix(3);
  matrix.set(0, 0, 1.0);
  matrix.set(1, 1, 1.0);
  // column 2 has no entries at all
  SparseLu lu;
  EXPECT_FALSE(lu.factorize(matrix.columns(), 3, tol::kPivotTolerance, kThreshold));
}

TEST(SparseLu, HandlesTheEmptyBasis) {
  SparseLu lu;
  EXPECT_TRUE(lu.factorize({}, 0, tol::kPivotTolerance, kThreshold));
  EXPECT_EQ(lu.dimension(), 0);
  lu.solve(nullptr);
  lu.solve_transpose(nullptr);
}

TEST(SparseLu, ATriangularMatrixProducesNoFill) {
  // Markowitz picks singletons first, so a triangular basis - which is most of what the
  // simplex actually sees - should factorize with the entries it started with and nothing
  // more. If this regresses, the pivot search has stopped finding singletons.
  constexpr Index m = 30;
  TestMatrix matrix(m);
  Index entries = 0;
  for (Index j = 0; j < m; ++j) {
    matrix.set(j, j, 2.0 + static_cast<double>(j));
    ++entries;
    if (j + 1 < m) {
      matrix.set(j + 1, j, 1.0);
      ++entries;
    }
  }

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold));
  EXPECT_LE(lu.factor_nonzeros(), entries)
      << "a lower-bidiagonal matrix should factorize without fill";
}

// =========================================================================================
// Fuzz against the dense oracle
// =========================================================================================

TEST(SparseLu, FuzzAgainstTheDenseOracle) {
  std::mt19937 rng(26119);
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int compared = 0;
  double worst_ftran = 0.0;
  double worst_btran = 0.0;
  double worst_residual = 0.0;
  double worst_transpose_residual = 0.0;

  for (int trial = 0; trial < 400; ++trial) {
    const Index m = 1 + static_cast<Index>(trial % 22);
    TestMatrix matrix(m);
    for (Index j = 0; j < m; ++j) {
      for (Index i = 0; i < m; ++i) {
        // Sparse off the diagonal, with a strong diagonal so most draws are nonsingular.
        if (i != j && unit(rng) < 0.25) matrix.set(i, j, value(rng));
      }
      matrix.set(j, j, 3.0 + unit(rng));
    }

    std::vector<double> x(static_cast<std::size_t>(m));
    for (double& v : x) v = value(rng);
    const std::vector<double> b = matrix.multiply(x);
    const std::vector<double> bt = matrix.transpose_multiply(x);

    SparseLu sparse;
    if (!sparse.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold)) continue;
    DenseLu dense;
    if (!dense.factorize(matrix.dense(), m, tol::kPivotTolerance)) continue;
    ++compared;

    // ---- FTRAN --------------------------------------------------------------------------
    std::vector<double> sparse_x = b;
    sparse.solve(sparse_x.data());
    std::vector<double> dense_x = b;
    dense.solve(dense_x.data());

    worst_ftran = std::max(worst_ftran, max_difference(sparse_x, dense_x));
    worst_residual = std::max(worst_residual, residual(matrix, sparse_x, b));

    // ---- BTRAN --------------------------------------------------------------------------
    std::vector<double> sparse_y = bt;
    sparse.solve_transpose(sparse_y.data());
    std::vector<double> dense_y = bt;
    dense.solve_transpose(dense_y.data());

    worst_btran = std::max(worst_btran, max_difference(sparse_y, dense_y));
    worst_transpose_residual =
        std::max(worst_transpose_residual, transpose_residual(matrix, sparse_y, bt));
  }

  EXPECT_GT(compared, 350) << "the generator produced too many singular matrices to be a "
                              "meaningful test";
  EXPECT_LT(worst_ftran, 1e-8) << "FTRAN disagrees with the dense oracle by " << worst_ftran;
  EXPECT_LT(worst_btran, 1e-8) << "BTRAN disagrees with the dense oracle by " << worst_btran;
  EXPECT_LT(worst_residual, 1e-9) << "worst FTRAN residual " << worst_residual;
  EXPECT_LT(worst_transpose_residual, 1e-9)
      << "worst BTRAN residual " << worst_transpose_residual;
}

TEST(SparseLu, FuzzOnVerySparseMatricesWhereFillMatters) {
  // Closer to a real basis: mostly triangular with a scattering of off-triangular entries,
  // which is where the Markowitz ordering earns its keep and where a fill bug would show.
  std::mt19937 rng(987654321);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  int compared = 0;
  double worst = 0.0;
  double worst_residual = 0.0;

  for (int trial = 0; trial < 250; ++trial) {
    const Index m = 5 + static_cast<Index>(trial % 40);
    TestMatrix matrix(m);
    for (Index j = 0; j < m; ++j) {
      matrix.set(j, j, 2.0 + 2.0 * unit(rng));
      for (Index i = j + 1; i < m; ++i) {
        if (unit(rng) < 3.0 / static_cast<double>(m)) matrix.set(i, j, value(rng));
      }
      for (Index i = 0; i < j; ++i) {
        if (unit(rng) < 1.0 / static_cast<double>(m)) matrix.set(i, j, value(rng));
      }
    }

    std::vector<double> x(static_cast<std::size_t>(m));
    for (double& v : x) v = value(rng);
    const std::vector<double> b = matrix.multiply(x);

    SparseLu sparse;
    if (!sparse.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold)) continue;
    DenseLu dense;
    if (!dense.factorize(matrix.dense(), m, tol::kPivotTolerance)) continue;
    ++compared;

    std::vector<double> sparse_x = b;
    sparse.solve(sparse_x.data());
    std::vector<double> dense_x = b;
    dense.solve(dense_x.data());
    worst = std::max(worst, max_difference(sparse_x, dense_x));
    worst_residual = std::max(worst_residual, residual(matrix, sparse_x, b));
  }

  EXPECT_GT(compared, 200);
  EXPECT_LT(worst, 1e-8) << "sparse and dense disagree by " << worst;
  EXPECT_LT(worst_residual, 1e-9) << "worst residual " << worst_residual;
}

// =========================================================================================
// Basis update (product form)
//
// The update is checked against a FROM-SCRATCH factorization of the updated basis - never
// against itself, never against the pre-update state. An update rule with the eta order
// reversed, or with the transpose applied in the wrong direction, still produces a vector of
// entirely plausible magnitude and still passes any self-consistency check. Only an
// independent factorization of what the basis actually became can tell the difference.
// =========================================================================================

/// A random, diagonally dominant sparse matrix - nonsingular with high probability.
TestMatrix random_basis(std::mt19937& rng, Index m, double density) {
  std::uniform_real_distribution<double> value(-4.0, 4.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  TestMatrix matrix(m);
  for (Index j = 0; j < m; ++j) {
    for (Index i = 0; i < m; ++i) {
      if (i != j && unit(rng) < density) matrix.set(i, j, value(rng));
    }
    matrix.set(j, j, 3.0 + unit(rng));
  }
  return matrix;
}

/// Rebuild `basis` with column `leaving` replaced by `entering`, as a fresh TestMatrix.
TestMatrix with_column_replaced(const TestMatrix& basis, Index m, Index leaving,
                                const std::vector<double>& entering) {
  TestMatrix next(m);
  const std::vector<LuColumn> cols = basis.columns();
  for (Index j = 0; j < m; ++j) {
    if (j == leaving) {
      for (Index i = 0; i < m; ++i) {
        const double v = entering[static_cast<std::size_t>(i)];
        if (v != 0.0) next.set(i, j, v);
      }
      continue;
    }
    const LuColumn& c = cols[static_cast<std::size_t>(j)];
    for (Index k = 0; k < c.size; ++k) next.set(c.rows[k], j, c.values[k]);
  }
  return next;
}

TEST(SparseLuUpdate, OneUpdateMatchesAFreshFactorization) {
  std::mt19937 rng(4242);
  std::uniform_real_distribution<double> value(-4.0, 4.0);

  int compared = 0;
  double worst_ftran = 0.0;
  double worst_btran = 0.0;

  for (int trial = 0; trial < 200; ++trial) {
    const Index m = 2 + static_cast<Index>(trial % 18);
    const TestMatrix basis = random_basis(rng, m, 0.3);

    SparseLu lu;
    if (!lu.factorize(basis.columns(), m, tol::kPivotTolerance, kThreshold)) continue;

    const Index leaving = static_cast<Index>(trial) % m;
    std::vector<double> entering(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) {
      entering[static_cast<std::size_t>(i)] = value(rng) + (i == leaving ? 5.0 : 0.0);
    }

    // alpha = B^-1 a, which is exactly what the simplex already has to hand at a pivot.
    std::vector<double> alpha = entering;
    lu.solve(alpha.data());
    if (!lu.update(leaving, alpha.data())) continue;
    ++compared;

    const TestMatrix updated = with_column_replaced(basis, m, leaving, entering);
    SparseLu reference;
    ASSERT_TRUE(reference.factorize(updated.columns(), m, tol::kPivotTolerance, kThreshold));

    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (double& v : rhs) v = value(rng);

    std::vector<double> a = rhs;
    lu.solve(a.data());
    std::vector<double> b = rhs;
    reference.solve(b.data());
    worst_ftran = std::max(worst_ftran, max_difference(a, b));

    std::vector<double> at = rhs;
    lu.solve_transpose(at.data());
    std::vector<double> bt = rhs;
    reference.solve_transpose(bt.data());
    worst_btran = std::max(worst_btran, max_difference(at, bt));
  }

  EXPECT_GT(compared, 150) << "too few usable updates for this test to mean anything";
  EXPECT_LT(worst_ftran, 1e-8) << "updated FTRAN disagrees with a fresh factorization by "
                               << worst_ftran;
  EXPECT_LT(worst_btran, 1e-8) << "updated BTRAN disagrees with a fresh factorization by "
                               << worst_btran;
}

TEST(SparseLuUpdate, ManyUpdatesInSequenceStayCorrect) {
  // A single update can be right while the ORDER the etas are applied in is wrong; that only
  // shows once more than one is stacked. Both directions are re-checked after every update
  // against a fresh factorization of the basis as it now stands.
  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> value(-3.0, 3.0);

  constexpr Index m = 14;
  TestMatrix current = random_basis(rng, m, 0.35);
  SparseLu lu;
  ASSERT_TRUE(lu.factorize(current.columns(), m, tol::kPivotTolerance, kThreshold));

  double worst = 0.0;
  int applied = 0;

  for (int step = 0; step < 25; ++step) {
    const Index leaving = static_cast<Index>(step) % m;
    std::vector<double> entering(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) {
      entering[static_cast<std::size_t>(i)] = value(rng) + (i == leaving ? 6.0 : 0.0);
    }

    std::vector<double> alpha = entering;
    lu.solve(alpha.data());
    if (!lu.update(leaving, alpha.data())) break;
    ++applied;

    current = with_column_replaced(current, m, leaving, entering);

    SparseLu reference;
    ASSERT_TRUE(reference.factorize(current.columns(), m, tol::kPivotTolerance, kThreshold));

    std::vector<double> rhs(static_cast<std::size_t>(m));
    for (double& v : rhs) v = value(rng);

    std::vector<double> a = rhs;
    lu.solve(a.data());
    std::vector<double> b = rhs;
    reference.solve(b.data());
    worst = std::max(worst, max_difference(a, b));

    std::vector<double> at = rhs;
    lu.solve_transpose(at.data());
    std::vector<double> bt = rhs;
    reference.solve_transpose(bt.data());
    worst = std::max(worst, max_difference(at, bt));
  }

  EXPECT_GE(applied, 10) << "the update was rejected too early to test stacking";
  EXPECT_EQ(lu.eta_count(), applied);
  EXPECT_LT(worst, 1e-7) << "stacked updates drift from a fresh factorization by " << worst;
}

TEST(SparseLuUpdate, RejectsAnUnsafePivotInsteadOfDividingByIt) {
  // alpha[leaving] near zero means the entering column barely moves the basis in the
  // direction being replaced. Dividing by it is how a product form silently loses accuracy,
  // so the update refuses and leaves the factorization usable for a caller that refactorizes.
  constexpr Index m = 4;
  TestMatrix matrix(m);
  for (Index i = 0; i < m; ++i) matrix.set(i, i, 1.0);

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold));

  std::vector<double> alpha(static_cast<std::size_t>(m), 1.0);
  alpha[2] = 1e-14;
  EXPECT_FALSE(lu.update(2, alpha.data()));
  EXPECT_EQ(lu.eta_count(), 0);

  // The factorization is untouched and still solves as it did.
  std::vector<double> b{1.0, 2.0, 3.0, 4.0};
  lu.solve(b.data());
  EXPECT_DOUBLE_EQ(b[0], 1.0);
  EXPECT_DOUBLE_EQ(b[3], 4.0);
}

TEST(SparseLuUpdate, AsksToRefactorizeOnceTheEtaFileGrows) {
  constexpr Index m = 10;
  std::mt19937 rng(99);
  const TestMatrix matrix = random_basis(rng, m, 0.3);

  SparseLu lu;
  ASSERT_TRUE(lu.factorize(matrix.columns(), m, tol::kPivotTolerance, kThreshold));
  EXPECT_FALSE(lu.should_refactorize()) << "a fresh factorization should not ask immediately";

  std::uniform_real_distribution<double> value(-2.0, 2.0);
  for (int step = 0; step < 500 && !lu.should_refactorize(); ++step) {
    const Index leaving = static_cast<Index>(step) % m;
    std::vector<double> alpha(static_cast<std::size_t>(m));
    for (Index i = 0; i < m; ++i) {
      alpha[static_cast<std::size_t>(i)] = value(rng) + (i == leaving ? 8.0 : 0.0);
    }
    if (!lu.update(leaving, alpha.data())) break;
  }
  EXPECT_TRUE(lu.should_refactorize())
      << "the eta file grew without bound; the refactorization trigger never fired";
}

}  // namespace
}  // namespace sankhya
