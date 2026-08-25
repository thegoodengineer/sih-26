// SPDX-License-Identifier: Apache-2.0
// SANKHYA - SparseMatrix tests, fuzzed against a naive dense reference.
//
// The reference implementation lives in this file on purpose. CLAUDE.md forbids linking a
// third-party matrix library into src/, and an oracle that shares code with the thing it
// checks is worthless. DenseReference is written the obvious O(m*n) way, from the textbook
// definition of a matrix product, so that a disagreement means the compressed code is wrong.

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

/// Row-major dense matrix with the textbook definitions of the two products.
class DenseReference {
 public:
  DenseReference(Index rows, Index cols)
      : rows_(rows),
        cols_(cols),
        a_(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0.0) {}

  double& at(Index i, Index j) {
    return a_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) +
              static_cast<std::size_t>(j)];
  }
  double at(Index i, Index j) const {
    return a_[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols_) +
              static_cast<std::size_t>(j)];
  }

  std::vector<double> multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(rows_), 0.0);
    for (Index i = 0; i < rows_; ++i) {
      double sum = 0.0;
      for (Index j = 0; j < cols_; ++j) sum += at(i, j) * x[static_cast<std::size_t>(j)];
      y[static_cast<std::size_t>(i)] = sum;
    }
    return y;
  }

  std::vector<double> transpose_multiply(const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(cols_), 0.0);
    for (Index j = 0; j < cols_; ++j) {
      double sum = 0.0;
      for (Index i = 0; i < rows_; ++i) sum += at(i, j) * x[static_cast<std::size_t>(i)];
      y[static_cast<std::size_t>(j)] = sum;
    }
    return y;
  }

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }

 private:
  Index rows_;
  Index cols_;
  std::vector<double> a_;
};

/// Build a matching (SparseMatrix, DenseReference) pair from random triplets, deliberately
/// including duplicate positions so that the duplicate-summing path in finalize() is
/// exercised on every single fuzz case rather than in one hand-written test.
struct FuzzCase {
  SparseMatrix sparse;
  DenseReference dense;
};

FuzzCase make_case(std::mt19937_64& rng, Index rows, Index cols, double density) {
  FuzzCase c{SparseMatrix(rows, cols), DenseReference(rows, cols)};
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  std::uniform_real_distribution<double> value(-10.0, 10.0);
  std::uniform_int_distribution<int> duplicates(1, 3);

  for (Index j = 0; j < cols; ++j) {
    for (Index i = 0; i < rows; ++i) {
      if (unit(rng) >= density) continue;
      const int repeats = duplicates(rng);
      for (int r = 0; r < repeats; ++r) {
        const double v = value(rng);
        c.sparse.add_entry(i, j, v);
        c.dense.at(i, j) += v;
      }
    }
  }
  c.sparse.finalize();
  return c;
}

std::vector<double> random_vector(std::mt19937_64& rng, Index n) {
  std::uniform_real_distribution<double> value(-5.0, 5.0);
  std::vector<double> v(static_cast<std::size_t>(n));
  for (double& x : v) x = value(rng);
  return v;
}

// =========================================================================================

TEST(SparseMatrix, EmptyMatrixIsWellFormed) {
  SparseMatrix a(0, 0);
  a.finalize();
  EXPECT_TRUE(a.frozen());
  EXPECT_EQ(a.num_rows(), 0);
  EXPECT_EQ(a.num_cols(), 0);
  EXPECT_EQ(a.num_nonzeros(), 0);
  EXPECT_EQ(a.column_starts().size(), 1u);
}

TEST(SparseMatrix, EmptyColumnsKeepValidStarts) {
  SparseMatrix a(3, 4);
  a.add_entry(1, 2, 5.0);
  a.finalize();
  EXPECT_EQ(a.num_nonzeros(), 1);
  for (Index j = 0; j < 4; ++j) {
    EXPECT_LE(a.column_starts()[static_cast<std::size_t>(j)],
              a.column_starts()[static_cast<std::size_t>(j) + 1]);
  }
  EXPECT_TRUE(a.column(0).empty());
  EXPECT_TRUE(a.column(1).empty());
  EXPECT_EQ(a.column(2).size, 1);
  EXPECT_TRUE(a.column(3).empty());
  EXPECT_DOUBLE_EQ(a.at(1, 2), 5.0);
  EXPECT_DOUBLE_EQ(a.at(0, 2), 0.0);
}

TEST(SparseMatrix, FinalizeSumsDuplicatesAndOrdersRows) {
  SparseMatrix a(4, 2);
  a.add_entry(3, 0, 1.0);
  a.add_entry(1, 0, 2.0);
  a.add_entry(3, 0, 4.0);
  a.add_entry(0, 1, -1.0);
  a.finalize();

  const ColumnView c0 = a.column(0);
  ASSERT_EQ(c0.size, 2);
  EXPECT_EQ(c0.rows[0], 1);
  EXPECT_EQ(c0.rows[1], 3);
  EXPECT_DOUBLE_EQ(c0.values[0], 2.0);
  EXPECT_DOUBLE_EQ(c0.values[1], 5.0);
}

TEST(SparseMatrix, FinalizeDropsCancellingDuplicates) {
  // Two entries that cancel exactly must leave no structural nonzero behind: a stored
  // explicit zero would later be accepted as a pivot candidate.
  SparseMatrix a(2, 1);
  a.add_entry(0, 0, 3.0);
  a.add_entry(0, 0, -3.0);
  a.add_entry(1, 0, 7.0);
  a.finalize();
  EXPECT_EQ(a.num_nonzeros(), 1);
  EXPECT_DOUBLE_EQ(a.at(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(a.at(1, 0), 7.0);
}

TEST(SparseMatrix, FinalizeDropsBelowTolerance) {
  SparseMatrix a(3, 1);
  a.add_entry(0, 0, 1e-15);
  a.add_entry(1, 0, 1.0);
  a.add_entry(2, 0, 1e-8);
  a.finalize(tol::kZeroDrop);
  EXPECT_EQ(a.num_nonzeros(), 2);
  EXPECT_DOUBLE_EQ(a.at(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(a.at(2, 0), 1e-8);
}

TEST(SparseMatrix, UnfreezeRoundTripsThroughBuildState) {
  std::mt19937_64 rng(12345);
  FuzzCase c = make_case(rng, 12, 9, 0.3);
  const std::vector<Index> starts = c.sparse.column_starts();
  const std::vector<Index> rows = c.sparse.row_indices();
  const std::vector<double> values = c.sparse.values();

  c.sparse.unfreeze();
  EXPECT_FALSE(c.sparse.frozen());
  c.sparse.finalize();

  EXPECT_EQ(c.sparse.column_starts(), starts);
  EXPECT_EQ(c.sparse.row_indices(), rows);
  EXPECT_EQ(c.sparse.values(), values);
}

TEST(SparseMatrix, FuzzMatVecAgainstDenseReference) {
  std::mt19937_64 rng(20260825);
  std::uniform_int_distribution<Index> dim(1, 40);
  const std::array<double, 4> densities{0.02, 0.1, 0.4, 0.95};

  int cases = 0;
  double worst_error = 0.0;
  for (int trial = 0; trial < 500; ++trial) {
    const Index m = dim(rng);
    const Index n = dim(rng);
    const double density = densities[static_cast<std::size_t>(trial) % densities.size()];
    FuzzCase c = make_case(rng, m, n, density);

    // A * x
    const std::vector<double> x = random_vector(rng, n);
    std::vector<double> y(static_cast<std::size_t>(m), 0.0);
    c.sparse.multiply(x.data(), y.data());
    const std::vector<double> y_ref = c.dense.multiply(x);
    for (std::size_t i = 0; i < y.size(); ++i) {
      worst_error = std::max(worst_error, std::fabs(y[i] - y_ref[i]));
      ASSERT_NEAR(y[i], y_ref[i], 1e-9) << "A*x mismatch, trial " << trial << " row " << i;
    }

    // A^T * z
    const std::vector<double> z = random_vector(rng, m);
    std::vector<double> w(static_cast<std::size_t>(n), 0.0);
    c.sparse.transpose_multiply(z.data(), w.data());
    const std::vector<double> w_ref = c.dense.transpose_multiply(z);
    for (std::size_t j = 0; j < w.size(); ++j) {
      worst_error = std::max(worst_error, std::fabs(w[j] - w_ref[j]));
      ASSERT_NEAR(w[j], w_ref[j], 1e-9) << "A^T*z mismatch, trial " << trial << " col " << j;
    }
    ++cases;
  }
  EXPECT_EQ(cases, 500);
  RecordProperty("worst_absolute_error", std::to_string(worst_error));
}

TEST(SparseMatrix, MultiplyAddAccumulatesAndScales) {
  std::mt19937_64 rng(777);
  FuzzCase c = make_case(rng, 15, 11, 0.35);
  const std::vector<double> x = random_vector(rng, 11);

  std::vector<double> y(15, 1.0);
  c.sparse.multiply_add(x.data(), y.data(), 2.0);
  const std::vector<double> ref = c.dense.multiply(x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    EXPECT_NEAR(y[i], 1.0 + 2.0 * ref[i], 1e-9);
  }
}

TEST(SparseMatrix, FuzzTransposeAgainstDenseReference) {
  std::mt19937_64 rng(99991);
  std::uniform_int_distribution<Index> dim(1, 30);
  for (int trial = 0; trial < 200; ++trial) {
    const Index m = dim(rng);
    const Index n = dim(rng);
    FuzzCase c = make_case(rng, m, n, 0.25);
    const SparseMatrix t = c.sparse.transpose();

    ASSERT_EQ(t.num_rows(), n);
    ASSERT_EQ(t.num_cols(), m);
    ASSERT_EQ(t.num_nonzeros(), c.sparse.num_nonzeros());
    for (Index i = 0; i < m; ++i) {
      for (Index j = 0; j < n; ++j) {
        ASSERT_DOUBLE_EQ(t.at(j, i), c.sparse.at(i, j))
            << "transpose mismatch at " << i << "," << j;
      }
    }
    // Row indices within each column of the transpose must come out ascending without a
    // sort; the counting-sort argument in sparse.cpp depends on it.
    for (Index j = 0; j < t.num_cols(); ++j) {
      const ColumnView col = t.column(j);
      for (Index k = 1; k < col.size; ++k) ASSERT_LT(col.rows[k - 1], col.rows[k]);
    }
  }
}

TEST(SparseMatrix, DoubleTransposeIsIdentity) {
  std::mt19937_64 rng(31337);
  FuzzCase c = make_case(rng, 17, 23, 0.2);
  const SparseMatrix tt = c.sparse.transpose().transpose();
  EXPECT_EQ(tt.column_starts(), c.sparse.column_starts());
  EXPECT_EQ(tt.row_indices(), c.sparse.row_indices());
  EXPECT_EQ(tt.values(), c.sparse.values());
}

TEST(SparseMatrix, MaxAbsValue) {
  SparseMatrix a(3, 3);
  a.add_entry(0, 0, -4.5);
  a.add_entry(2, 1, 2.0);
  a.finalize();
  EXPECT_DOUBLE_EQ(a.max_abs_value(), 4.5);

  SparseMatrix empty(2, 2);
  empty.finalize();
  EXPECT_DOUBLE_EQ(empty.max_abs_value(), 0.0);
}

// =========================================================================================
// CsrView
// =========================================================================================

TEST(CsrView, FuzzRowsAgainstDenseReference) {
  std::mt19937_64 rng(4242);
  std::uniform_int_distribution<Index> dim(1, 25);
  for (int trial = 0; trial < 200; ++trial) {
    const Index m = dim(rng);
    const Index n = dim(rng);
    FuzzCase c = make_case(rng, m, n, 0.3);
    const CsrView csr(c.sparse);

    ASSERT_EQ(csr.num_rows(), m);
    ASSERT_EQ(csr.num_cols(), n);

    for (Index i = 0; i < m; ++i) {
      const ColumnView r = csr.row(i);
      // Every stored entry matches the dense reference...
      for (Index k = 0; k < r.size; ++k) {
        ASSERT_DOUBLE_EQ(r.values[k], c.sparse.at(i, r.rows[k]));
        if (k > 0) {
          ASSERT_LT(r.rows[k - 1], r.rows[k]);
        }
      }
      // ...and no dense nonzero is missing from the row.
      Index counted = 0;
      for (Index j = 0; j < n; ++j) {
        if (c.sparse.at(i, j) != 0.0) ++counted;
      }
      ASSERT_EQ(counted, r.size) << "row " << i << " lost an entry";
    }
  }
}

}  // namespace
}  // namespace sankhya
