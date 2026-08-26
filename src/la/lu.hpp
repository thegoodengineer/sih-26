// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse LU factorization of the simplex basis, with Markowitz pivoting.
//
// References:
//   Markowitz, "The elimination form of the inverse and its application to linear
//     programming", Management Science 3 (1957) - the (r-1)(c-1) fill estimate.
//   Suhl & Suhl, "Computing sparse LU factorizations for large-scale linear programming
//     bases", ORSA Journal on Computing 2 (1990) - the threshold-stability compromise and
//     the bounded candidate search used here.
//   Duff, Erisman & Reid, "Direct Methods for Sparse Matrices" (2nd ed., 2017), ch. 7-8.
//
// WHY THIS EXISTS. Phase 2's DenseLu rebuilds and refactorizes an m x m dense array on
// EVERY pivot: O(m^2) memory traffic and O(m^3) flops per iteration. At m = 800 that is
// 5 MB and 1.7e8 flops per iteration, and a real basis of 800 columns holds perhaps 3000
// nonzeros - so better than 99% of that work is spent on structural zeros. This class
// factorizes the sparsity pattern instead.
//
// WHAT THE FACTORIZATION IS. Gaussian elimination with an arbitrary pivot ORDER rather
// than an arbitrary pivot SEQUENCE: at step k a pivot (r_k, c_k) is chosen from the active
// submatrix, and every remaining active row with a nonzero in column c_k is updated by
//
//     row_i  :=  row_i  -  mult_{i,k} * row_{r_k},     mult_{i,k} = a[i][c_k] / a[r_k][c_k]
//
// Writing M_k for that transformation, the elimination gives  M_{m-1} ... M_0 A = U, with
// U upper triangular under the permutations r and c. So  A = M_0^-1 ... M_{m-1}^-1 U, and
// the two solves fall out of that identity directly - see the derivations in lu.cpp, which
// are written out in full because a sign or an ordering error here does not crash. It
// produces duals that look plausible, prices the wrong column, and stops the simplex at a
// non-optimal vertex with a confident "optimal".
//
// STABILITY. Markowitz alone will happily choose a pivot of 1e-14 because it produces no
// fill. Every candidate must therefore also satisfy
//
//     |a[r][c]|  >=  tau * max_i |a[i][c]|          tau = tol::kMarkowitzThreshold
//
// which is the classic trade of a little fill for a bound on element growth.
//
// VERIFICATION. DenseLu is NOT deleted. It stays in the tree as the reference oracle: the
// fuzz tests factorize the same matrix both ways and require the FTRAN and BTRAN results to
// agree. That is the only cheap defence against a permutation bug, which is otherwise
// invisible.
#pragma once

#include <vector>

#include "sankhya/sparse.hpp"
#include "sankhya/types.hpp"

namespace sankhya {

/// One basis column, as a list of (row, value) pairs. The simplex holds its columns in the
/// model's CSC matrix and in the logical slack columns, so it hands them over rather than
/// materialising a matrix.
struct LuColumn {
  const Index* rows = nullptr;
  const double* values = nullptr;
  Index size = 0;
};

class SparseLu {
 public:
  /// Factorize the m x m matrix whose columns are `columns[0..m-1]`. Returns false when the
  /// basis is singular to working precision, i.e. when some step finds no candidate pivot
  /// above `pivot_tolerance` in absolute value.
  ///
  /// `markowitz_threshold` is tau above; 0.01 is the long-standing default and lives in
  /// tolerances.hpp. Passing 1.0 degenerates to partial pivoting (maximum stability, worst
  /// fill), passing 0.0 to pure Markowitz (best fill, no stability guarantee at all).
  [[nodiscard]] bool factorize(const std::vector<LuColumn>& columns, Index m,
                               double pivot_tolerance, double markowitz_threshold);

  /// Solve B z = b in place. FTRAN.
  void solve(double* b) const;

  /// Solve B^T z = b in place. BTRAN.
  void solve_transpose(double* b) const;

  // -------------------------------------------------------------------------------------
  // Basis update (product form of the inverse)
  //
  // Reference: Dantzig & Orchard-Hays, "The product form for the inverse in the simplex
  // method", Mathematical Tables and Other Aids to Computation 8 (1954).
  //
  // A simplex pivot replaces ONE column of the basis. Refactorizing all of it to absorb a
  // rank-one change is the dominant per-iteration cost once the factorization itself is
  // sparse. Replacing column p of B with a, and writing alpha = B^-1 a,
  //
  //     B_new = B (I + (alpha - e_p) e_p^T) = B E
  //
  // so the factorization of B is kept and E is recorded. After k updates
  // B_k = B_0 E_1 ... E_k, and the two solves follow directly:
  //
  //     FTRAN   x = E_k^-1 ... E_1^-1 (B_0^-1 b)     base solve first, etas OLDEST first
  //     BTRAN   x = B_0^-T (E_1^-T ... E_k^-T b)     etas NEWEST first, then the base solve
  //
  // The orders are opposite and neither is symmetric with the other. Getting one backwards
  // produces a vector of entirely plausible magnitude - see the tests, which check both
  // against a from-scratch factorization of the updated basis rather than against each
  // other.
  //
  // WHAT THIS COSTS. Refactorizing every iteration had one real virtue: no update error
  // could accumulate, so any wrong answer was the simplex's fault. Updates reintroduce
  // drift, which is why update() refuses a numerically unsafe pivot and why
  // should_refactorize() exists. Both are part of the feature, not optional extras.
  // -------------------------------------------------------------------------------------

  /// Record that column `leaving_position` of the basis has been replaced, given
  /// `alpha` = B^-1 a for the entering column a. `alpha` must have `dimension()` entries.
  ///
  /// Returns false when the pivot element alpha[leaving_position] is too small relative to
  /// the rest of the vector for the update to be numerically safe. The caller must then
  /// refactorize from scratch; the factorization is left untouched and usable.
  [[nodiscard]] bool update(Index leaving_position, const double* alpha);

  /// Number of updates applied since the last factorize().
  [[nodiscard]] Index eta_count() const noexcept {
    return static_cast<Index>(eta_start_.size()) - 1;
  }

  /// True when the accumulated updates have grown enough that refactorizing is cheaper, or
  /// enough that drift is a concern. Checked by the simplex once per iteration.
  [[nodiscard]] bool should_refactorize() const noexcept;

  [[nodiscard]] Index dimension() const noexcept { return m_; }

  /// Nonzeros in the computed factors, excluding the unit diagonal of L. Compared against
  /// the nonzero count of the basis itself this is the fill ratio, which is what decides
  /// when a Forrest-Tomlin update should give up and refactorize.
  [[nodiscard]] Index factor_nonzeros() const noexcept {
    return static_cast<Index>(l_rows_.size() + u_steps_.size());
  }

  /// Smallest and largest pivot magnitude. Their ratio is a crude condition estimate -
  /// crude enough that it is reported and never used to make a decision.
  [[nodiscard]] double smallest_pivot() const noexcept { return smallest_pivot_; }
  [[nodiscard]] double largest_pivot() const noexcept { return largest_pivot_; }

 private:
  /// Scratch state for the elimination, discarded once the factors are built. Kept out of
  /// the class proper so that a factorized SparseLu carries only what the solves need.
  struct Workspace;

  [[nodiscard]] bool eliminate(Workspace& w, double pivot_tolerance, double threshold);

  Index m_ = 0;

  // ---- the factors --------------------------------------------------------------------
  // Indexed by elimination step k, not by row or column. pivot_row_[k] and pivot_col_[k]
  // are the permutations; everything else is stored in step order so both solves walk it
  // linearly.
  std::vector<Index> pivot_row_;
  std::vector<Index> pivot_col_;
  std::vector<double> pivot_value_;

  /// L, as the multipliers produced at each step. l_rows_[p] is a ROW index; the pivot row
  /// it refers to is implied by the step.
  std::vector<Index> l_start_;  ///< m_ + 1 entries
  std::vector<Index> l_rows_;
  std::vector<double> l_values_;

  /// U, stored by pivot row. u_steps_[p] is an elimination STEP index l > k, meaning the
  /// entry lives in column pivot_col_[l]. Storing the step rather than the column is what
  /// lets the back substitution below run without a second permutation lookup.
  std::vector<Index> u_start_;  ///< m_ + 1 entries
  std::vector<Index> u_steps_;
  std::vector<double> u_values_;

  /// Eta file: one entry per update, stored sparsely. eta_pivot_position_[k] is the basis
  /// position that changed, and the (row, value) pairs are the nonzeros of alpha.
  std::vector<Index> eta_start_;  ///< eta_count() + 1 entries
  std::vector<Index> eta_rows_;
  std::vector<double> eta_values_;
  std::vector<Index> eta_pivot_position_;
  std::vector<double> eta_pivot_value_;

  /// Nonzeros in the factors at the last factorize(), so growth can be judged against it.
  Index base_nonzeros_ = 0;

  /// Scratch for the solves. Mutable because solve() is logically const: it must not
  /// allocate on a path the simplex takes several hundred times per second.
  mutable std::vector<double> work_;

  double smallest_pivot_ = 0.0;
  double largest_pivot_ = 0.0;
};

}  // namespace sankhya
