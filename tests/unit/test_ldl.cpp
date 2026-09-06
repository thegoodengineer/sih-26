// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the sparse LDL^T (#70) against the dense LU, and its limits measured.

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "la/ldl.hpp"
#include "sankhya/sparse.hpp"
#include "simplex/dense_lu.hpp"

namespace sankhya {
namespace {

/// A random sparse SPD matrix M = B B^T + shift I with B n x n of the given density, as
/// its lower triangle in CSC, plus the dense column-major copy the reference needs.
struct SpdSystem {
  SparseMatrix lower;
  std::vector<double> dense;  // column-major n x n
  Index n = 0;
};

SpdSystem random_spd(std::mt19937_64& rng, Index n, double density, double shift) {
  std::uniform_real_distribution<double> value(-2.0, 2.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::vector<double> b(static_cast<std::size_t>(n * n), 0.0);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      if (unit(rng) < density) b[static_cast<std::size_t>(i + j * n)] = value(rng);
    }
  }
  SpdSystem system;
  system.n = n;
  system.dense.assign(static_cast<std::size_t>(n * n), 0.0);
  for (Index i = 0; i < n; ++i) {
    for (Index j = 0; j < n; ++j) {
      double sum = i == j ? shift : 0.0;
      for (Index k = 0; k < n; ++k) {
        sum += b[static_cast<std::size_t>(i + k * n)] * b[static_cast<std::size_t>(j + k * n)];
      }
      system.dense[static_cast<std::size_t>(i + j * n)] = sum;
    }
  }
  system.lower.reset(n, n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = j; i < n; ++i) {
      const double v = system.dense[static_cast<std::size_t>(i + j * n)];
      if (v != 0.0) system.lower.add_entry(i, j, v);
    }
  }
  system.lower.finalize(0.0);
  return system;
}

double residual_norm(const SpdSystem& system, const std::vector<double>& x,
                     const std::vector<double>& rhs) {
  double worst = 0.0;
  for (Index i = 0; i < system.n; ++i) {
    double sum = 0.0;
    for (Index j = 0; j < system.n; ++j) {
      sum += system.dense[static_cast<std::size_t>(i + j * system.n)] *
             x[static_cast<std::size_t>(j)];
    }
    worst = std::max(worst, std::fabs(sum - rhs[static_cast<std::size_t>(i)]));
  }
  return worst;
}

TEST(SparseLdl, AgreesWithTheDenseReferenceOnRandomSpdSystems) {
  // The dense LU is the oracle here, as it is for the sparse LU: same system, two
  // factorizations that share no code, and the solutions must agree to rounding.
  std::mt19937_64 rng(70001);
  int solved = 0;
  double worst_disagreement = 0.0;
  double worst_residual = 0.0;
  for (int trial = 0; trial < 60; ++trial) {
    const Index n = 3 + static_cast<Index>(trial % 40);
    const SpdSystem system = random_spd(rng, n, 0.3, 1.0);
    SparseLdl ldl;
    ASSERT_TRUE(ldl.analyze(system.lower));
    ASSERT_TRUE(ldl.factorize(system.lower, 1e-12));
    EXPECT_EQ(ldl.regularized_pivots(), 0)
        << "an SPD matrix with shift 1 needs no regularization";

    DenseLu dense;
    ASSERT_TRUE(dense.factorize(system.dense, n, 1e-14));

    std::uniform_real_distribution<double> value(-5.0, 5.0);
    std::vector<double> rhs(static_cast<std::size_t>(n));
    for (auto& v : rhs) v = value(rng);
    std::vector<double> x_sparse = rhs;
    std::vector<double> x_dense = rhs;
    ldl.solve(x_sparse.data());
    dense.solve(x_dense.data());
    for (Index i = 0; i < n; ++i) {
      worst_disagreement =
          std::max(worst_disagreement,
                   std::fabs(x_sparse[static_cast<std::size_t>(i)] -
                             x_dense[static_cast<std::size_t>(i)]) /
                       std::max(1.0, std::fabs(x_dense[static_cast<std::size_t>(i)])));
    }
    worst_residual = std::max(worst_residual, residual_norm(system, x_sparse, rhs));
    ++solved;
  }
  std::cout << "ldl: " << solved << " SPD systems, worst disagreement with the dense LU "
            << worst_disagreement << ", worst residual " << worst_residual << "\n";
  EXPECT_LT(worst_disagreement, 1e-9);
  EXPECT_LT(worst_residual, 1e-8);
}

TEST(SparseLdl, ThePatternIsReusedAcrossRefactorizations) {
  // analyze() once, factorize() many times with new values on the same pattern - the IPM's
  // use. The second factorization must be as accurate as a fresh one.
  std::mt19937_64 rng(70002);
  const Index n = 25;
  SpdSystem first = random_spd(rng, n, 0.35, 2.0);
  SparseLdl ldl;
  ASSERT_TRUE(ldl.analyze(first.lower));
  ASSERT_TRUE(ldl.factorize(first.lower, 1e-12));
  // Same pattern, scaled values: D M D with a random positive diagonal.
  std::uniform_real_distribution<double> scale(0.5, 3.0);
  std::vector<double> d(static_cast<std::size_t>(n));
  for (auto& v : d) v = scale(rng);
  SpdSystem second = first;
  second.lower.reset(n, n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = j; i < n; ++i) {
      const double v = first.dense[static_cast<std::size_t>(i + j * n)] *
                       d[static_cast<std::size_t>(i)] * d[static_cast<std::size_t>(j)];
      second.dense[static_cast<std::size_t>(i + j * n)] = v;
      second.dense[static_cast<std::size_t>(j + i * n)] = v;
      if (v != 0.0) second.lower.add_entry(i, j, v);
    }
  }
  second.lower.finalize(0.0);
  ASSERT_TRUE(ldl.factorize(second.lower, 1e-12));
  std::vector<double> rhs(static_cast<std::size_t>(n), 1.0);
  std::vector<double> x = rhs;
  ldl.solve(x.data());
  EXPECT_LT(residual_norm(second, x, rhs), 1e-8);
}

TEST(SparseLdl, ConditioningSweepFindsWhereRegularizationTakesOver) {
  // M = B B^T + I scaled by a diagonal spanning 10^k: the condition number grows as 10^(2k)
  // and at some k the regularization threshold starts replacing genuine pivots. That k is
  // the factorization's documented limit; the test asserts the floor that holds and
  // reports the point where it stops holding, rather than hiding either.
  std::mt19937_64 rng(70003);
  const Index n = 30;
  const SpdSystem base = random_spd(rng, n, 0.3, 1.0);
  int first_regularized = -1;
  int first_inaccurate = -1;
  for (int k = 0; k <= 16; k += 2) {
    SpdSystem scaled = base;
    scaled.lower.reset(n, n);
    for (Index j = 0; j < n; ++j) {
      const double dj = std::pow(10.0, (j % 2 == 0 ? 1.0 : -1.0) * k / 2.0);
      for (Index i = j; i < n; ++i) {
        const double di = std::pow(10.0, (i % 2 == 0 ? 1.0 : -1.0) * k / 2.0);
        const double v = base.dense[static_cast<std::size_t>(i + j * n)] * di * dj;
        scaled.dense[static_cast<std::size_t>(i + j * n)] = v;
        scaled.dense[static_cast<std::size_t>(j + i * n)] = v;
        if (v != 0.0) scaled.lower.add_entry(i, j, v);
      }
    }
    scaled.lower.finalize(0.0);
    SparseLdl ldl;
    ASSERT_TRUE(ldl.analyze(scaled.lower));
    ASSERT_TRUE(ldl.factorize(scaled.lower, 1e-10));
    std::vector<double> rhs(static_cast<std::size_t>(n), 1.0);
    std::vector<double> x = rhs;
    ldl.solve(x.data());
    const double residual = residual_norm(scaled, x, rhs);
    if (ldl.regularized_pivots() > 0 && first_regularized < 0) first_regularized = k;
    if (residual > 1e-6 && first_inaccurate < 0) first_inaccurate = k;
    std::cout << "ldl conditioning: diagonal spread 1e" << k << ": residual " << residual
              << ", regularized pivots " << ldl.regularized_pivots() << "\n";
    // A diagonal scaling is a change of units; to a spread of 1e8 the factorization must
    // neither regularize a genuine pivot nor lose accuracy.
    if (k <= 8) {
      EXPECT_EQ(ldl.regularized_pivots(), 0) << "at spread 1e" << k;
      EXPECT_LT(residual, 1e-6) << "at spread 1e" << k;
    }
  }
  std::cout << "ldl conditioning: first regularized pivot at spread 1e" << first_regularized
            << ", first residual above 1e-6 at spread 1e" << first_inaccurate << "\n";
}

TEST(SparseLdl, NormalEquationsMatchTheDenseProduct) {
  // A Theta A^T + delta I, lower triangle, against the same thing computed densely.
  std::mt19937_64 rng(70004);
  std::uniform_real_distribution<double> value(-3.0, 3.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const Index m = 12;
  const Index n = 20;
  SparseMatrix a(m, n);
  std::vector<double> dense_a(static_cast<std::size_t>(m * n), 0.0);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < m; ++i) {
      if (unit(rng) < 0.3) {
        const double v = value(rng);
        a.add_entry(i, j, v);
        dense_a[static_cast<std::size_t>(i + j * m)] = v;
      }
    }
  }
  a.finalize(0.0);
  std::vector<double> theta(static_cast<std::size_t>(n));
  for (auto& t : theta) t = 0.1 + unit(rng);
  SparseMatrix lower;
  normal_equations_lower(a, theta, {}, 0.5, &lower);
  ASSERT_EQ(lower.num_rows(), m);
  ASSERT_EQ(lower.num_cols(), m);
  double worst = 0.0;
  for (Index j = 0; j < m; ++j) {
    for (Index i = j; i < m; ++i) {
      double expected = i == j ? 0.5 : 0.0;
      for (Index k = 0; k < n; ++k) {
        expected += theta[static_cast<std::size_t>(k)] *
                    dense_a[static_cast<std::size_t>(i + k * m)] *
                    dense_a[static_cast<std::size_t>(j + k * m)];
      }
      worst = std::max(worst, std::fabs(lower.at(i, j) - expected));
    }
    // Nothing above the diagonal.
    for (Index i = 0; i < j; ++i) EXPECT_EQ(lower.at(i, j), 0.0);
  }
  EXPECT_LT(worst, 1e-12);
}

}  // namespace
}  // namespace sankhya
