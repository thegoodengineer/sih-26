// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense LU factorization of the simplex basis.
//
// Reference: Golub & Van Loan, "Matrix Computations" (4th ed.), sections 3.2 and 3.4, for
// right-looking LU with partial pivoting and for the transposed triangular solves.
//
// THIS IS DELIBERATELY THE SLOW IMPLEMENTATION. Phase 2's job is a simplex whose answers
// can be trusted; Phase 6 replaces this entire class with a sparse Markowitz LU plus
// Forrest-Tomlin updates. Refactorizing the whole basis from scratch on every iteration
// costs O(m^3) per pivot, which is indefensible at scale and invaluable now: there is no
// eta file, no update formula and no accumulated round-off, so any wrong answer the
// simplex produces is the simplex's fault and not the linear algebra's. Keeping those two
// error sources separated is worth more at this stage than any amount of speed.
//
// Storage is column-major so that the inner loop of the elimination walks contiguous
// memory, and L is unit-lower with its multipliers stored in place beneath the diagonal.
#pragma once

#include <vector>

#include "sankhya/types.hpp"

namespace sankhya {

class DenseLu {
 public:
  /// Factorize the m x m column-major matrix in `columns` (which is consumed, not aliased).
  /// Returns false when a pivot falls below `pivot_tolerance`, i.e. the basis is singular
  /// to working precision. `columns` must hold exactly m*m entries.
  [[nodiscard]] bool factorize(std::vector<double> columns, Index m, double pivot_tolerance);

  /// Solve B z = b in place. This is FTRAN.
  void solve(double* b) const;

  /// Solve B^T z = b in place. This is BTRAN.
  void solve_transpose(double* b) const;

  [[nodiscard]] Index dimension() const noexcept { return m_; }

  /// Smallest pivot magnitude seen during the factorization. A basis whose smallest pivot
  /// collapses is the first observable sign of numerical trouble, and Phase 9's numerics
  /// report is built on this.
  [[nodiscard]] double smallest_pivot() const noexcept { return smallest_pivot_; }

  /// Largest pivot magnitude. The ratio against smallest_pivot() is a cheap, crude
  /// condition estimate - crude enough that it is reported, never used to make decisions.
  [[nodiscard]] double largest_pivot() const noexcept { return largest_pivot_; }

 private:
  [[nodiscard]] double& at(Index row, Index col) noexcept {
    return lu_[static_cast<std::size_t>(col) * static_cast<std::size_t>(m_) +
                static_cast<std::size_t>(row)];
  }
  [[nodiscard]] double at(Index row, Index col) const noexcept {
    return lu_[static_cast<std::size_t>(col) * static_cast<std::size_t>(m_) +
                static_cast<std::size_t>(row)];
  }

  Index m_ = 0;
  std::vector<double> lu_;
  /// pivot_[k] is the row swapped with row k at elimination step k.
  std::vector<Index> pivot_;
  double smallest_pivot_ = 0.0;
  double largest_pivot_ = 0.0;
};

}  // namespace sankhya
