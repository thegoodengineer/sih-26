// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse symmetric LDL^T factorization (#70), the linear algebra an interior-point
// method needs and the simplex's unsymmetric LU cannot provide.
//
// References:
//   Davis, T.A., "Direct Methods for Sparse Linear Systems", SIAM (2006), ch. 4 - the
//     elimination tree, the symbolic column counts, and the up-looking numeric
//     factorization, implemented here from the book's description.
//   Liu, J.W.H., "The role of elimination trees in sparse factorization", SIAM J. Matrix
//     Anal. Appl. 11 (1990) - the tree and why the symbolic step is done once.
//   Tinney & Walker, "Direct solutions of sparse network equations by optimally ordered
//     triangular factorization", Proc. IEEE 55 (1967) - minimum degree ordering.
//   Altman & Gondzio, "Regularized symmetric indefinite systems in interior point methods
//     for linear and quadratic optimization", Optim. Methods Softw. 11 (1999) - the
//     diagonal regularization that keeps a near-singular pivot from destroying the factors.
//
// WHAT IT IS FOR. The normal equations A Theta A^T of an interior-point method are symmetric
// positive definite (semi-definite at the limit), and their sparsity pattern never changes
// across iterations while their values change every one. So the work splits in two:
// analyze() computes a fill-reducing ordering, the elimination tree and the exact pattern of
// L once; factorize() fills in the numbers of that pattern, as often as the IPM likes.
//
// WHAT IT IS NOT. Not a general indefinite factorization: there is no Bunch-Kaufman
// pivoting. A pivot below the regularization threshold is replaced by the threshold, which
// is the standard IPM remedy (Altman & Gondzio) and turns an exactly singular system into a
// slightly perturbed nonsingular one; the count of such pivots is reported so the caller can
// see how much the factors deviate from the matrix. Iterative refinement (the solve is cheap)
// recovers most of what the perturbation costs.
#pragma once

#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

class SparseLdl {
 public:
  /// Symbolic analysis of a symmetric matrix given by its LOWER triangle (entries with
  /// row >= col) in CSC form; entries above the diagonal are ignored. Computes a minimum
  /// degree ordering, the elimination tree of the permuted matrix and the pattern of L.
  /// Returns false on an empty or non-square input. O(n^2) worst case in the ordering.
  [[nodiscard]] bool analyze(const SparseMatrix& lower);

  /// Numeric factorization of a matrix with the SAME pattern (or a subset of it) as the
  /// one analyzed. Pivots below `regularization` are set to `regularization`; the number
  /// of pivots so treated is available afterwards. Returns false if analyze() was not
  /// called or the pattern does not fit.
  [[nodiscard]] bool factorize(const SparseMatrix& lower, double regularization);

  /// Solve (P^T L D L^T P) x = b in place.
  void solve(double* b) const;

  [[nodiscard]] Index dimension() const noexcept { return n_; }
  [[nodiscard]] Index factor_nonzeros() const noexcept {
    return static_cast<Index>(l_values_.size());
  }
  [[nodiscard]] Index regularized_pivots() const noexcept { return regularized_; }
  [[nodiscard]] double smallest_pivot() const noexcept { return smallest_pivot_; }
  [[nodiscard]] double largest_pivot() const noexcept { return largest_pivot_; }
  [[nodiscard]] const std::vector<Index>& permutation() const noexcept { return perm_; }

 private:
  void minimum_degree(const SparseMatrix& lower);
  void build_permuted_pattern(const SparseMatrix& lower);
  void elimination_tree();
  void symbolic_pattern();

  Index n_ = 0;
  bool analyzed_ = false;

  /// perm_[k] = original index of the k-th pivot; inverse_[i] = position of original i.
  std::vector<Index> perm_;
  std::vector<Index> inverse_;

  /// The permuted matrix's UPPER triangle by column (so column k holds the entries of row
  /// k of the lower triangle - what the up-looking factorization consumes), values refreshed
  /// per factorize().
  std::vector<Index> a_starts_;
  std::vector<Index> a_rows_;
  std::vector<double> a_values_;

  std::vector<Index> parent_;    ///< elimination tree
  std::vector<Index> l_starts_;  ///< CSC pattern of L (strictly lower), fixed by analyze()
  std::vector<Index> l_rows_;
  std::vector<double> l_values_;
  std::vector<double> d_;  ///< the diagonal of D
  Index regularized_ = 0;
  double smallest_pivot_ = 0.0;
  double largest_pivot_ = 0.0;
};

/// Build the lower triangle of A Theta A^T + diag(row_shift) + delta I from A in CSC form,
/// for an IPM's normal equations. Theta is a diagonal over the columns of A; row_shift a
/// diagonal over the rows (the logical block of [A | -I] Theta [A | -I]^T, or empty for
/// none); delta a uniform shift. Rebuilt in full on every call; the pattern is the same
/// each time, which is what lets the factorization's analysis be reused.
void normal_equations_lower(const SparseMatrix& a, const std::vector<double>& theta,
                            const std::vector<double>& row_shift, double delta,
                            SparseMatrix* out);

}  // namespace sankhya
