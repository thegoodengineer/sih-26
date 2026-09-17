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
//     triangular factorization", Proc. IEEE 55 (1967) - the minimum degree rule.
//   Amestoy, Davis & Duff, "An approximate minimum degree ordering algorithm", SIAM J.
//     Matrix Anal. Appl. 17 (1996) - the quotient graph the ordering runs on (#193).
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

#include <functional>
#include <vector>

#include "sankhya/sparse.hpp"

namespace sankhya {

/// What a semidefiniteness probe concluded about a symmetric matrix (#303).
///
/// `column` is an index into the ORIGINAL matrix, not the permuted one: the caller asked
/// about its own matrix and the AMD ordering is an implementation detail.
struct SemidefiniteReport {
  enum class Verdict {
    kPositiveSemidefinite,  ///< every pivot non-negative; x^T A x >= 0 for all x
    kIndefinite,            ///< a direction with x^T A x < 0 was exhibited
    kUndecided,             ///< the probe was abandoned (deadline, or a factor that cannot fit)
  };

  Verdict verdict = Verdict::kUndecided;
  /// The original column that decided an kIndefinite verdict, or -1.
  Index column = -1;
  /// The pivot at that column, or the residual that contradicted a zero pivot.
  double pivot = 0.0;
};

class SparseLdl {
 public:
  /// Symbolic analysis of a symmetric matrix given by its LOWER triangle (entries with
  /// A caller's deadline, asked between elimination steps and between columns.
  ///
  /// WHY THIS EXISTS (#193). The solver checks the clock between iterations, which is the
  /// right place for a method whose iterations are alike. An interior-point iteration is not:
  /// it factorizes, and the FIRST one also pays for the ordering. Measured on a generated
  /// 20,000-row model, one iteration took 813 s against a 120 s limit, most of it inside
  /// analyze(), which the loop had not returned from to look at the clock.
  ///
  /// Returning true here abandons the work. That is safe in the sense ENGINEERING_RULES.md
  /// means when it says wall-clock must never decide anything inside the solver: a solve that
  /// COMPLETES does exactly the arithmetic it always did, in the same order, and gets the same
  /// answer. The clock only decides whether an unfinished solve keeps running, which is what a
  /// time limit has always decided.
  using ShouldStop = std::function<bool()>;

  /// Symbolic analysis of the lower triangle of a symmetric matrix (entries with
  /// row >= col) in CSC form; entries above the diagonal are ignored. Computes an
  /// approximate minimum degree ordering on the quotient graph (AMD), the elimination tree
  /// of the permuted matrix and the pattern of L. Returns false on an empty or non-square
  /// input, or if `should_stop` asked it to give up.
  [[nodiscard]] bool analyze(const SparseMatrix& lower, const ShouldStop& should_stop = {});

  /// Cap the quotient graph's live storage during the ordering, in list entries (#246).
  ///
  /// On an expander-like matrix the minimum-degree ordering's fill is catastrophic and the
  /// element lists grow with it until the allocation fails: on the 100,000-row random scale
  /// model that was a std::bad_alloc 170 s past the time limit, on an 8 GB machine. The
  /// ordering counts the entries it holds live and gives up past this many, reporting
  /// ordering_too_large() rather than a deadline. One entry is one Index (4 bytes).
  /// SIZE_MAX (the default) means no cap.
  void set_ordering_budget(std::size_t entries) noexcept { ordering_budget_ = entries; }
  [[nodiscard]] std::size_t ordering_budget() const noexcept { return ordering_budget_; }

  /// Cap the factor's pattern, in strictly-lower nonzeros (#246). analyze() counts the
  /// pattern before storing it and gives up past this many, reporting factor_too_large();
  /// -1 (the default) means no cap. The interior point passes its ipm_max_factor_nonzeros
  /// or polish_max_factor_nonzeros here so a factor that would not fit is refused after a
  /// fraction of the counting, not after all of it and an allocation.
  void set_factor_budget(std::int64_t nonzeros) noexcept {
    factor_budget_ =
        nonzeros < 0 ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(nonzeros);
  }
  [[nodiscard]] bool factor_too_large() const noexcept { return factor_too_large_; }

  /// True when analyze() returned false because the ordering's storage passed the budget
  /// set by set_ordering_budget() (#246). Not a deadline and not a malformed matrix: the
  /// matrix fills in faster than this machine can afford to follow.
  [[nodiscard]] bool ordering_too_large() const noexcept { return ordering_too_large_; }

  /// True when analyze() returned false because the factor's pattern would hold more
  /// nonzeros than an Index offset can name (#305). Distinct from a deadline and from a
  /// malformed matrix: the input was well formed and the ordering finished, and the factor
  /// it implies is simply larger than this build addresses.
  [[nodiscard]] bool pattern_too_large() const noexcept { return pattern_too_large_; }

  /// True when the last analyze() or factorize() returned false because the deadline was
  /// reached rather than because the matrix was wrong. The caller reports a time limit in
  /// that case, not a numerical failure.
  [[nodiscard]] bool stopped_early() const noexcept { return stopped_early_; }

  /// Numeric factorization of a matrix with the SAME pattern (or a subset of it) as the
  /// one analyzed. Pivots below `regularization` are set to `regularization`; the number
  /// of pivots so treated is available afterwards. Returns false if analyze() was not
  /// called or the pattern does not fit.
  [[nodiscard]] bool factorize(const SparseMatrix& lower, double regularization,
                               const ShouldStop& should_stop = {});

  /// Is `lower` positive semidefinite? (#303)
  ///
  /// The same LDL^T that factorize() runs, with the IPM's regularization REMOVED and the
  /// semidefinite rule put in its place, because the two answer different questions.
  /// factorize() wants usable factors for a matrix it already knows is positive definite, so
  /// it lifts a small pivot to the regularization floor and carries on. A convexity test must
  /// not: lifting a NEGATIVE pivot to a positive floor would turn the one piece of evidence
  /// that matters - a direction of negative curvature - into a clean factorization, and the
  /// caller would solve a non-convex model and report a local point as optimal.
  ///
  /// Three outcomes, on a pivot measured against `slack_factor * max(1, largest |diagonal|)`:
  ///   pivot < -slack        indefinite, and the column is the certificate
  ///   |pivot| <= slack      a legitimately singular direction of a semidefinite matrix. The
  ///                         column is skipped rather than divided through - but only after
  ///                         checking that the entries that would have been divided are
  ///                         themselves negligible. For a semidefinite matrix they must be
  ///                         (Higham 1990); when they are not, the zero pivot sits beside a
  ///                         nonzero off-diagonal and the matrix is indefinite. Skipping
  ///                         without that check is how [[0, 1], [1, 0]] passed for convex.
  ///   otherwise             an ordinary positive pivot.
  ///
  /// Calls analyze() itself. Leaves no usable factors behind: this is a decision procedure,
  /// not a factorization, and the D it computes has deliberate zeros in it.
  [[nodiscard]] SemidefiniteReport check_semidefinite(const SparseMatrix& lower,
                                                      double slack_factor,
                                                      const ShouldStop& should_stop = {});

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
  [[nodiscard]] bool minimum_degree(const SparseMatrix& lower, const ShouldStop& should_stop);
  bool pattern_too_large_ = false;
  bool ordering_too_large_ = false;
  std::size_t ordering_budget_ = static_cast<std::size_t>(-1);
  bool factor_too_large_ = false;
  std::size_t factor_budget_ = static_cast<std::size_t>(-1);
  void build_permuted_pattern(const SparseMatrix& lower);
  void elimination_tree();
  [[nodiscard]] bool symbolic_pattern(const ShouldStop& should_stop);

  Index n_ = 0;
  bool analyzed_ = false;
  bool stopped_early_ = false;  ///< the last failure was a deadline, not a bad matrix

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
///
/// Takes the same deadline as analyze() and factorize(), checked every 256 rows, and returns
/// false when it fires, leaving `out` unusable (#232). The assembly is the one step of an
/// interior-point iteration that ran BEFORE the ordering's deadline could be consulted: on
/// the random scale family at 500,000 rows it took 90 s against a polish budget of 30, and
/// at 1,000,000 rows 40 s more than the limit, because forming A Theta A^T for five
/// nonzeros per column on a random pattern is itself most of a minute. Without a deadline
/// the check never returns false and the output is identical.
[[nodiscard]] bool normal_equations_lower(const SparseMatrix& a,
                                          const std::vector<double>& theta,
                                          const std::vector<double>& row_shift, double delta,
                                          SparseMatrix* out,
                                          const SparseLdl::ShouldStop& should_stop = {});

}  // namespace sankhya
