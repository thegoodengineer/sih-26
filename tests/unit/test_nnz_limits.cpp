// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the nonzero ceiling and the arithmetic that guards it (#305).
//
// Index is 32-bit by design (include/sankhya/types.hpp says why), so every sparse offset,
// loop bound and allocation size in this project is safe exactly as long as the ENTRY COUNT
// is. The guard that keeps it so triggers at 2^31 entries, which no test can allocate: the
// triplet arrays alone would be 24 GB. An untested guard is not evidence that the guard
// works, so SparseMatrix::set_nonzero_limit() lowers the ceiling to a handful of entries and
// the same code path is exercised at a size a test can hold.
//
// What is being checked is not "the matrix refuses to grow". It is that the refusal is
// VISIBLE: an overflowed matrix must never freeze into a pattern whose offsets wrapped, and
// validate() must turn it into a diagnostic before any engine reads it.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

/// A one-row model whose matrix is handed in ready-made.
Model model_around(SparseMatrix matrix, Index rows, Index cols) {
  Model model;
  model.col_cost.assign(static_cast<std::size_t>(cols), 1.0);
  model.col_lower.assign(static_cast<std::size_t>(cols), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(cols), 1.0);
  model.col_type.assign(static_cast<std::size_t>(cols), VarType::kContinuous);
  model.row_lower.assign(static_cast<std::size_t>(rows), -kInfinity);
  model.row_upper.assign(static_cast<std::size_t>(rows), 1.0);
  model.matrix = std::move(matrix);
  model.hessian.reset(cols, cols);
  model.hessian.finalize();
  return model;
}

// =========================================================================================
// The arithmetic itself
// =========================================================================================

TEST(NonzeroLimits, TheOverflowPredicatesAgreeWithTheirDefinitionsAtTheBoundary) {
  constexpr Index kMax = std::numeric_limits<Index>::max();
  EXPECT_EQ(kMaxNonzeros, kMax);

  EXPECT_TRUE(nonzero_count_fits(0));
  EXPECT_TRUE(nonzero_count_fits(static_cast<std::size_t>(kMax)));
  EXPECT_FALSE(nonzero_count_fits(static_cast<std::size_t>(kMax) + 1));

  // a + b, at and one past the edge. The sum is never formed, which is the point: forming it
  // to test it would be the overflow.
  EXPECT_FALSE(index_sum_overflows(kMax - 1, 1));
  EXPECT_TRUE(index_sum_overflows(kMax - 1, 2));
  EXPECT_TRUE(index_sum_overflows(kMax, 1));
  EXPECT_FALSE(index_sum_overflows(0, kMax));
  EXPECT_TRUE(index_sum_overflows(-1, 0)) << "a negative operand is already a broken count";

  // a * b. 46341^2 = 2147488281 > 2^31 - 1, and 46340^2 = 2147395600 < it: the classic pair.
  EXPECT_FALSE(index_product_overflows(46340, 46340));
  EXPECT_TRUE(index_product_overflows(46341, 46341));
  EXPECT_FALSE(index_product_overflows(0, kMax));
  EXPECT_FALSE(index_product_overflows(1, kMax));
  EXPECT_TRUE(index_product_overflows(2, kMax));

  // The dense-square budget, which is about bytes rather than Index range: 50,000 columns of
  // doubles is 2e10 bytes, and that is the allocation #303 refuses to make.
  EXPECT_FALSE(dense_square_exceeds(1000, sizeof(double), 1u << 30));
  EXPECT_TRUE(dense_square_exceeds(50000, sizeof(double), 1u << 30));
  EXPECT_FALSE(dense_square_exceeds(0, sizeof(double), 0));
}

// =========================================================================================
// The matrix
// =========================================================================================

TEST(NonzeroLimits, AMatrixAtExactlyItsLimitIsBuiltNormally) {
  SparseMatrix a(2, 2);
  a.set_nonzero_limit(4);
  a.add_entry(0, 0, 1.0);
  a.add_entry(1, 0, 2.0);
  a.add_entry(0, 1, 3.0);
  a.add_entry(1, 1, 4.0);
  a.finalize();

  EXPECT_FALSE(a.overflowed());
  EXPECT_EQ(a.num_nonzeros(), 4);
  EXPECT_DOUBLE_EQ(a.at(1, 1), 4.0);
}

TEST(NonzeroLimits, AMatrixRefusesEntriesPastItsLimitInsteadOfWrapping) {
  SparseMatrix a(3, 3);
  a.set_nonzero_limit(4);
  for (Index j = 0; j < 3; ++j) {
    for (Index i = 0; i < 3; ++i) a.add_entry(i, j, 1.0 + static_cast<double>(i + j));
  }
  EXPECT_TRUE(a.overflowed()) << "nine entries offered against a limit of four";

  a.finalize();
  EXPECT_TRUE(a.frozen());
  EXPECT_TRUE(a.overflowed()) << "the flag must survive finalize()";
  EXPECT_EQ(a.num_nonzeros(), 0) << "an overflowed matrix freezes empty, never half-built";

  // The property that matters: no offset ran backwards. On the unguarded code this is where
  // a wrapped prefix sum showed up.
  const std::vector<Index>& starts = a.column_starts();
  ASSERT_EQ(starts.size(), 4u);
  for (std::size_t k = 1; k < starts.size(); ++k) {
    EXPECT_GE(starts[k], starts[k - 1]) << "column_starts must be non-decreasing";
    EXPECT_GE(starts[k], 0);
  }
}

TEST(NonzeroLimits, ResetClearsTheFlagAndKeepsTheLimit) {
  SparseMatrix a(2, 2);
  a.set_nonzero_limit(1);
  a.add_entry(0, 0, 1.0);
  a.add_entry(1, 1, 1.0);
  ASSERT_TRUE(a.overflowed());

  a.reset(2, 2);
  EXPECT_FALSE(a.overflowed()) << "a matrix reused for another model starts clean";
  EXPECT_EQ(a.nonzero_limit(), 1) << "the configured ceiling is a property of the container";
  a.add_entry(0, 0, 5.0);
  a.finalize();
  EXPECT_FALSE(a.overflowed());
  EXPECT_EQ(a.num_nonzeros(), 1);
}

TEST(NonzeroLimits, TheDefaultLimitIsTheIndexRangeAndOrdinaryMatricesAreUntouched) {
  SparseMatrix a(3, 3);
  EXPECT_EQ(a.nonzero_limit(), kMaxNonzeros);
  a.add_entry(0, 0, 1.0);
  a.add_entry(2, 1, 2.0);
  a.add_entry(1, 2, 3.0);
  a.add_entry(1, 2, 4.0);  // duplicate, summed by finalize
  a.finalize();
  EXPECT_FALSE(a.overflowed());
  EXPECT_EQ(a.num_nonzeros(), 3);
  EXPECT_DOUBLE_EQ(a.at(1, 2), 7.0);
}

// =========================================================================================
// The diagnostic a caller actually meets
// =========================================================================================

TEST(NonzeroLimits, ValidateRefusesAnOverflowedMatrixRatherThanReadingIt) {
  SparseMatrix a(2, 2);
  a.set_nonzero_limit(2);
  for (Index j = 0; j < 2; ++j) {
    for (Index i = 0; i < 2; ++i) a.add_entry(i, j, 1.0);
  }
  ASSERT_TRUE(a.overflowed());
  a.finalize();

  const Model model = model_around(std::move(a), 2, 2);
  const std::string problem = model.validate();
  EXPECT_NE(problem.find("nonzero limit"), std::string::npos) << problem;
  EXPECT_NE(problem.find("constraint matrix"), std::string::npos) << problem;
}

TEST(NonzeroLimits, ValidateRefusesAnOverflowedHessianToo) {
  SparseMatrix a(1, 2);
  a.add_entry(0, 0, 1.0);
  a.add_entry(0, 1, 1.0);
  a.finalize();
  Model model = model_around(std::move(a), 1, 2);

  model.hessian.reset(2, 2);
  model.hessian.set_nonzero_limit(1);
  model.hessian.add_entry(0, 0, 2.0);
  model.hessian.add_entry(1, 1, 2.0);
  ASSERT_TRUE(model.hessian.overflowed());
  model.hessian.finalize();

  const std::string problem = model.validate();
  EXPECT_NE(problem.find("quadratic objective"), std::string::npos) << problem;
  EXPECT_NE(problem.find("nonzero limit"), std::string::npos) << problem;
}

TEST(NonzeroLimits, AnOversizedModelIsAModelErrorNotAWrongAnswer) {
  // End to end: the solver must decline. Before the guard this model reached the simplex
  // with a pattern whose offsets had wrapped, which is the failure class the project's rules
  // put first - a confident answer to a problem nobody posed.
  SparseMatrix a(2, 2);
  a.set_nonzero_limit(2);
  for (Index j = 0; j < 2; ++j) {
    for (Index i = 0; i < 2; ++i) a.add_entry(i, j, 1.0);
  }
  a.finalize();
  const Model model = model_around(std::move(a), 2, 2);

  Options options;
  options.set_bool("log_to_console", false);
  const Solution solution = solve(model, options);
  EXPECT_EQ(solution.status, SolveStatus::kModelError);
  EXPECT_NE(solution.message.find("nonzero limit"), std::string::npos) << solution.message;
}

}  // namespace
}  // namespace sankhya
