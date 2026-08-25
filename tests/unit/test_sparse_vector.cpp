// SPDX-License-Identifier: Apache-2.0
// SANKHYA - SparseVector tests.
//
// SparseVector is the accumulator FTRAN and BTRAN will write into, so its invariant matters
// more than its interface: the pattern must list exactly the touched indices, clear() must
// leave the dense array genuinely zero (a stale value there becomes a phantom nonzero in
// the next solve), and the cost of clear() must be proportional to the pattern rather than
// to the dimension. The fuzz test below checks the first two against a dense mirror.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/sparse.hpp"

namespace sankhya {
namespace {

TEST(SparseVector, StartsZero) {
  SparseVector v(5);
  EXPECT_EQ(v.dimension(), 5);
  EXPECT_EQ(v.num_nonzeros(), 0);
  for (Index i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(v[i], 0.0);
}

TEST(SparseVector, AddRegistersOnFirstTouchOnly) {
  SparseVector v(4);
  v.add(2, 1.5);
  v.add(2, 2.5);
  v.add(0, -1.0);
  EXPECT_EQ(v.num_nonzeros(), 2);
  EXPECT_DOUBLE_EQ(v[2], 4.0);
  EXPECT_DOUBLE_EQ(v[0], -1.0);
  EXPECT_DOUBLE_EQ(v[1], 0.0);
  // Insertion order, not sorted order: FTRAN consumes the pattern in the order it was
  // produced and must not pay for a sort it does not need.
  ASSERT_EQ(v.pattern().size(), 2u);
  EXPECT_EQ(v.pattern()[0], 2);
  EXPECT_EQ(v.pattern()[1], 0);
}

TEST(SparseVector, SetOverwrites) {
  SparseVector v(3);
  v.add(1, 5.0);
  v.set(1, -2.0);
  EXPECT_EQ(v.num_nonzeros(), 1);
  EXPECT_DOUBLE_EQ(v[1], -2.0);
}

TEST(SparseVector, ClearLeavesNoStaleValues) {
  SparseVector v(6);
  v.add(0, 1.0);
  v.add(4, 2.0);
  v.clear();
  EXPECT_EQ(v.num_nonzeros(), 0);
  for (Index i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(v[i], 0.0);

  // Reuse after clear must behave exactly like a fresh vector.
  v.add(4, 7.0);
  EXPECT_EQ(v.num_nonzeros(), 1);
  EXPECT_DOUBLE_EQ(v[4], 7.0);
}

TEST(SparseVector, ExactCancellationSurvivesUntilCompress) {
  // Documented behaviour: a value that cancels to zero stays in the pattern until
  // compress() is called. Dropping it eagerly would cost a scan on every accumulate.
  SparseVector v(3);
  v.add(1, 4.0);
  v.add(1, -4.0);
  EXPECT_EQ(v.num_nonzeros(), 1);
  EXPECT_DOUBLE_EQ(v[1], 0.0);

  v.compress();
  EXPECT_EQ(v.num_nonzeros(), 0);
  EXPECT_DOUBLE_EQ(v[1], 0.0);
}

TEST(SparseVector, CompressKeepsValuesAtOrAboveTolerance) {
  SparseVector v(4);
  v.set(0, 1e-14);
  v.set(1, 1.0);
  v.set(2, 1e-11);
  v.compress(tol::kZeroDrop);
  EXPECT_EQ(v.num_nonzeros(), 2);
  EXPECT_DOUBLE_EQ(v[0], 0.0);
  EXPECT_DOUBLE_EQ(v[1], 1.0);
  EXPECT_DOUBLE_EQ(v[2], 1e-11);
}

TEST(SparseVector, ScatterAndGatherRoundTrip) {
  SparseVector v(5);
  v.set(1, 3.0);
  v.set(3, -4.0);

  std::vector<double> dense(5, 99.0);
  v.scatter_to_dense(dense.data());
  EXPECT_DOUBLE_EQ(dense[0], 0.0);
  EXPECT_DOUBLE_EQ(dense[1], 3.0);
  EXPECT_DOUBLE_EQ(dense[2], 0.0);
  EXPECT_DOUBLE_EQ(dense[3], -4.0);
  EXPECT_DOUBLE_EQ(dense[4], 0.0);

  SparseVector w(5);
  w.gather_from_dense(dense.data());
  EXPECT_EQ(w.num_nonzeros(), 2);
  for (Index i = 0; i < 5; ++i) EXPECT_DOUBLE_EQ(w[i], v[i]);
}

TEST(SparseVector, FuzzAgainstDenseMirror) {
  std::mt19937_64 rng(20260826);
  std::uniform_int_distribution<Index> dim(1, 60);
  std::uniform_real_distribution<double> value(-5.0, 5.0);

  for (int trial = 0; trial < 400; ++trial) {
    const Index n = dim(rng);
    std::uniform_int_distribution<Index> pick(0, n - 1);
    SparseVector v(n);
    std::vector<double> mirror(static_cast<std::size_t>(n), 0.0);

    const int operations = 3 * static_cast<int>(n);
    for (int op = 0; op < operations; ++op) {
      const Index i = pick(rng);
      const double x = value(rng);
      if ((op % 5) == 0) {
        v.set(i, x);
        mirror[static_cast<std::size_t>(i)] = x;
      } else {
        v.add(i, x);
        mirror[static_cast<std::size_t>(i)] += x;
      }
    }

    for (Index i = 0; i < n; ++i) {
      ASSERT_NEAR(v[i], mirror[static_cast<std::size_t>(i)], 1e-12)
          << "trial " << trial << " index " << i;
    }

    // The pattern must be a superset of the true support and must contain no duplicates.
    std::vector<Index> pattern = v.pattern();
    std::sort(pattern.begin(), pattern.end());
    ASSERT_EQ(std::adjacent_find(pattern.begin(), pattern.end()), pattern.end())
        << "duplicate index in the pattern, trial " << trial;
    for (Index i = 0; i < n; ++i) {
      if (mirror[static_cast<std::size_t>(i)] != 0.0) {
        ASSERT_TRUE(std::binary_search(pattern.begin(), pattern.end(), i))
            << "nonzero " << i << " missing from the pattern, trial " << trial;
      }
    }

    // After compress() the pattern is exactly the support above the drop tolerance.
    v.compress();
    Index expected = 0;
    for (Index i = 0; i < n; ++i) {
      if (std::fabs(mirror[static_cast<std::size_t>(i)]) >= tol::kZeroDrop) ++expected;
    }
    ASSERT_EQ(v.num_nonzeros(), expected) << "trial " << trial;
  }
}

TEST(SparseVector, ResizeClearsContents) {
  SparseVector v(3);
  v.set(0, 1.0);
  v.resize(7);
  EXPECT_EQ(v.dimension(), 7);
  EXPECT_EQ(v.num_nonzeros(), 0);
  for (Index i = 0; i < 7; ++i) EXPECT_DOUBLE_EQ(v[i], 0.0);
}

}  // namespace
}  // namespace sankhya
