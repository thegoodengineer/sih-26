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

#include <functional>
#include <utility>
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
  /// A deadline the factorization consults before every pivot step, the same shape as
  /// SparseLdl's (#197). When it fires the factorization returns false with the factors
  /// unusable and stopped_early() true, which is how a caller tells a time limit from a
  /// singular basis. Never asked, it changes nothing: the same matrix gives the same
  /// factors with and without one (#208).
  using ShouldStop = std::function<bool()>;

  [[nodiscard]] bool factorize(const std::vector<LuColumn>& columns, Index m,
                               double pivot_tolerance, double markowitz_threshold,
                               const ShouldStop& should_stop = {});

  /// True when the last factorize() returned false because the deadline fired, not because
  /// the basis is singular. The factors are unusable either way.
  [[nodiscard]] bool stopped_early() const noexcept { return stopped_early_; }

  /// When factorize() returns false, the columns of the ORIGINAL `columns` array (identified
  /// by position, 0..m-1) that no pivot ever reached - the genuine rank defect, not merely
  /// "whatever was left when the search gave up" (issue #143). Empty after a successful
  /// factorize(). Basis repair (Maros 9.4; Suhl & Suhl 1990) evicts exactly these and
  /// substitutes the logical of each row in uncovered_rows() below, which restores a
  /// nonsingular basis by construction because a logical column is a unit vector.
  [[nodiscard]] const std::vector<Index>& dependent_positions() const noexcept {
    return dependent_positions_;
  }

  /// The rows left uncovered by any pivot when factorize() returns false - always the same
  /// count as dependent_positions(), since eliminating k fewer columns leaves exactly k rows
  /// with no pivot row assigned to them.
  [[nodiscard]] const std::vector<Index>& uncovered_rows() const noexcept {
    return uncovered_rows_;
  }

  /// Solve B z = b in place. FTRAN.
  void solve(double* b) const;
  /// The same solve with the back-substitution as a row-wise gather over every entry of
  /// U, kept as the reference the hyper-sparse push form is tested against (#68). Tests only.
  void solve_reference(double* b) const;

  /// Solve B^T z = b in place. BTRAN.
  void solve_transpose(double* b) const;
  /// The same solve with the transposed elimination factors applied as a gather over every
  /// entry of L, kept as the reference the hyper-sparse push form is tested against (#243).
  /// Tests only.
  void solve_transpose_reference(double* b) const;

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
  /// the rest of the vector for the update to be numerically safe, OR when
  /// update_forrest_tomlin() has already been used on this factorization - the two schemes
  /// are mutually exclusive (see below), and an eta appended here would never be read by
  /// either solve path once Forrest-Tomlin mode is active, which is silently wrong rather
  /// than merely inefficient. Either way the caller must refactorize from scratch; the
  /// factorization itself is left untouched and usable.
  [[nodiscard]] bool update(Index leaving_position, const double* alpha);

  // -------------------------------------------------------------------------------------
  // Basis update (Forrest-Tomlin)
  //
  // Reference: Forrest & Tomlin, "Updated triangular factors of the basis to maintain
  // sparsity in the product form simplex method", Mathematical Programming 2 (1972), via
  // the derivation in Huangfu & Hall, "Novel update techniques for the revised simplex
  // method" (ERGO 13-001, 2012), section 2.1 - eq. (11) onward is what this follows.
  //
  // update() above pays for the pivot on every SOLVE: the eta file is read in full, one
  // eta per update, whether or not the right-hand side touches most of it (issue #279).
  // This path pays for the pivot mostly AT THE UPDATE instead, by folding the new column
  // into U directly and leaving only a much smaller correction (the row eta below) for
  // solves to apply.
  //
  //   1. alpha already IS B^-1 a (the caller's FTRAN), so the column L^-1 P a - which is
  //      what replaces the leaving position's column of U - is recovered by multiplying the
  //      CURRENT U into alpha reindexed by step (U * alpha_by_step = that column, read the
  //      solve the other way), rather than resolving a again.
  //   2. That column, installed as every OTHER step's reference to the leaving step, is
  //      always valid regardless of where the leaving step's own diagonal ends up - so it is
  //      moved to the LAST position, and everything between its old and new position shifts
  //      down by one to keep the permutation a bijection (a pure relabelling: no entry any
  //      OTHER step owns is read or written by this).
  //   3. The leaving step's OWN old row, now invalid at the last position (nothing can
  //      validly reference it from there), is eliminated - not by rewriting other steps'
  //      rows, but by a single row transformation R computed via a partial BTRAN through U
  //      alone (Forrest & Tomlin's r = e_p^T - u_pp * (e_p^T U^-1), the "row spike" the issue
  //      names). R is applied between L and U on every later solve, exactly where PFI's
  //      product form is applied outside both - see ft_apply_retas().
  //
  // L is never touched. R grows by one small vector per update, same shape as update()'s
  // eta file but built from U's own sparsity rather than the dense-ish alpha, which is
  // where the win comes from; should_refactorize() caps it the same way.
  //
  // Only one of update() / update_forrest_tomlin() may be used on a given factorization;
  // calling the other afterwards fails rather than silently mixing the two schemes (both
  // directions refuse - see update() above and the guard at the top of this one).
  //
  // ACCEPTANCE THRESHOLD. update() rejects a pivot too small relative to max|alpha|, in the
  // basis's own coordinates. This rejects a pivot too small relative to max|spike|, where
  // spike = L^-1 P a: the same relative test, but against a vector on a different scale
  // (already partially reduced through U). The two modes' "declined as unsafe" counts are
  // therefore not directly comparable.
  //
  // KNOWN COST. ft_row_/ft_col_ are vector<vector<pair<Index,double>>>: correct and simple,
  // but two heap allocations per step at every ft_init() and pointer-chasing on every solve,
  // rather than the one contiguous array u_start_/u_steps_ gives the product form. A CSR
  // layout with per-row slack (Koberstein 2005, ch. 5) would keep this O(fill) instead of
  // O(m) allocations and let solves walk one array; not done here.
  // -------------------------------------------------------------------------------------

  /// The Forrest-Tomlin counterpart to update(): same contract, same `alpha`, but the
  /// replacement is folded into U (plus one row eta) instead of appended to an eta file.
  ///
  /// Returns false, WITHOUT MODIFYING ANYTHING, in exactly two cases: the recovered spike
  /// column is non-finite, or its entry at the leaving step is too small relative to the
  /// rest of it to divide by (see ACCEPTANCE THRESHOLD above). Both checks happen before the
  /// first write; a caller that reads false may keep using this instance exactly as
  /// update()'s caller does. Every caller today refactorizes anyway once either update
  /// returns false, but this instance itself is not the reason to.
  [[nodiscard]] bool update_forrest_tomlin(Index leaving_position, const double* alpha);

  /// True once update_forrest_tomlin() has been used at least once since factorize().
  [[nodiscard]] bool using_forrest_tomlin() const noexcept { return ft_active_; }

  /// Number of Forrest-Tomlin updates applied since factorize().
  [[nodiscard]] Index ft_update_count() const noexcept {
    return static_cast<Index>(ft_reta_pivot_step_.size());
  }

  /// Nonzeros folded into U beyond what factorize() produced: the Forrest-Tomlin analogue of
  /// eta_nonzeros() above, i.e. the fill the updates themselves introduced.
  [[nodiscard]] Index ft_extra_nonzeros() const noexcept;

  /// Choose the update scheme update() dispatches to: the product form (default) or the
  /// Forrest-Tomlin fold above (`--option basis_update=forrest-tomlin`, #279). Sticky
  /// across factorize(); the simplex sets it once from the options.
  void use_forrest_tomlin(bool on) noexcept { forrest_tomlin_ = on; }
  [[nodiscard]] bool forrest_tomlin() const noexcept { return forrest_tomlin_; }

  /// Number of updates applied since the last factorize(), whichever scheme applied them.
  /// Every "is the basis still on fresh factors?" decision in the simplex reads this.
  [[nodiscard]] Index eta_count() const noexcept {
    return ft_active_ ? ft_update_count() : static_cast<Index>(eta_start_.size()) - 1;
  }

  /// The extra work every solve does on top of the base factors: the eta file's nonzeros
  /// under the product form, the fill folded into U plus the row-eta file under
  /// Forrest-Tomlin. The simplex's break-even refactorization rule (#68) reads this, so the
  /// same rule governs both schemes.
  [[nodiscard]] Index eta_nonzeros() const noexcept {
    return ft_active_ ? ft_extra_nonzeros() : static_cast<Index>(eta_rows_.size());
  }

  /// True when the accumulated updates have grown enough that refactorizing is cheaper, or
  /// enough that drift is a concern. Checked by the simplex once per iteration.
  [[nodiscard]] bool should_refactorize() const noexcept;

  [[nodiscard]] Index dimension() const noexcept { return m_; }

  /// Nonzeros in the computed factors, excluding the unit diagonal of L. Compared against
  /// the nonzero count of the basis itself this is the fill ratio, which is what decides
  /// when the product-form update should give up and refactorize.
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

  [[nodiscard]] bool eliminate(Workspace& w, double pivot_tolerance, double threshold,
                               const ShouldStop& should_stop);

  Index m_ = 0;
  bool stopped_early_ = false;  ///< the last failure was a deadline, not a singular basis

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
  /// U BY COLUMN (#68): for step k, the steps i < k with U(i, k) != 0 and their values, so
  /// the back-substitution in solve() can push each nonzero result upward and skip the
  /// zeros. Built by build_column_u() at the end of factorize(); the product-form update
  /// never touches U, so it stays valid until the next factorization.
  std::vector<Index> uc_start_;  ///< m_ + 1 entries
  std::vector<Index> uc_steps_;
  std::vector<double> uc_values_;
  void build_column_u();
  /// L BY ROW (#243): for step k, the steps j < k whose multiplier vector has a nonzero on
  /// row pivot_row_[k], and the values. This is what lets solve_transpose() apply the
  /// transposed elimination factors in push form - each nonzero of the result pushes into
  /// the earlier steps it feeds and a zero pushes nothing - instead of gathering over the
  /// whole of L for every one of the m steps. Built beside the column-wise U; the
  /// product-form update never touches L, so it stays valid until the next factorization.
  std::vector<Index> lr_start_;  ///< m_ + 1 entries
  std::vector<Index> lr_steps_;
  std::vector<double> lr_values_;
  void build_row_l();
  /// The three passes of solve(), so the reference and the hyper-sparse form share two.
  void forward_l(double* b) const;
  void apply_etas(double* b) const;
  /// The two passes of solve_transpose() that the reference form shares with it.
  void apply_etas_transposed(double* b) const;
  void forward_u_transposed() const;

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

  /// The rank defect located by eliminate() (issue #143), reported only when factorize()
  /// returns false. Cleared at the start of every factorize() call.
  std::vector<Index> dependent_positions_;
  std::vector<Index> uncovered_rows_;

  // ---- Forrest-Tomlin update state (issue #279) ----------------------------------------
  // Everything below is indexed by STEP, exactly like u_start_/uc_start_ before any update:
  // a step's identity (which basis position it represents, via pivot_col_) never changes.
  // What an update changes is (a) a step's DIAGONAL and off-diagonal content, and (b) which
  // POSITION - purely a processing-order label, used to decide which step's diagonal a
  // solve may already treat as resolved - that step currently occupies. Position k is
  // "ready" only once every step it can validly reference (position >= k) has been visited,
  // exactly as it was when position and step coincided before the first update.
  bool forrest_tomlin_ = false;  ///< what update() dispatches to; sticky across factorize()
  bool ft_active_ = false;
  Index ft_base_row_nonzeros_ = 0;  ///< off-diagonal entry count at the moment FT mode began
  /// Running total of ft_row_'s entries, maintained by ft_set()/ft_erase() so
  /// ft_extra_nonzeros() is O(1) rather than a sum over every row on every call - the
  /// simplex's refactorization policy needs to read it every iteration (issue #68).
  Index ft_row_fill_ = 0;

  /// Fixed forever once built: the step that has represented basis position p since
  /// factorize(), i.e. the inverse of pivot_col_.
  std::vector<Index> ft_step_of_position_;
  std::vector<Index> ft_position_;  ///< step -> its current position
  std::vector<Index> ft_step_at_;   ///< position -> the step currently occupying it
  std::vector<double> ft_diag_;     ///< step -> its current diagonal value

  /// U's off-diagonal entries, both STEP-keyed and mirrored, kept in sync by ft_set() and
  /// ft_erase(): ft_row_[s] is step s's own row, i.e. the OTHER steps its row references
  /// (matching u_steps_ before any update); ft_col_[s] is the steps that reference s (their
  /// ROW has an entry AT s), the mirror image, matching uc_steps_. Both exclude the diagonal.
  std::vector<std::vector<std::pair<Index, double>>> ft_row_;
  std::vector<std::vector<std::pair<Index, double>>> ft_col_;

  /// The row-eta file (Forrest & Tomlin 1972, eq. for R): one entry per update, applied
  /// between L and U rather than appended outside both as update()'s eta file is. Update k
  /// replaces the basis position whose permanent step identity is ft_reta_pivot_step_[k];
  /// the entries are the sparse vector r (excluding that step itself, which is always 0 by
  /// construction). FTRAN applies these oldest first (like apply_etas), BTRAN newest first
  /// (like apply_etas_transposed) - see ft_apply_retas()/ft_apply_retas_transposed().
  std::vector<Index> ft_reta_pivot_step_;
  std::vector<Index> ft_reta_start_;  ///< ft_reta_pivot_step_.size() + 1 entries
  std::vector<Index> ft_reta_steps_;
  std::vector<double> ft_reta_values_;

  /// Step-indexed scratch for solve()'s own Forrest-Tomlin back-substitution
  /// (ft_back_substitute()): every position is written exactly once per call, so it needs no
  /// separate touched list - see the comment at its one call site.
  mutable std::vector<double> ft_scratch_;
  /// e~ = e_step^T U_current^-1, step-indexed, built by ft_btran_unit() for
  /// update_forrest_tomlin() alone. Deliberately NOT ft_scratch_ above, despite the similar
  /// name and shape: solve() runs constantly BETWEEN two update_forrest_tomlin() calls (every
  /// FTRAN while Forrest-Tomlin is active) and would silently invalidate a touched list kept
  /// on that buffer by overwriting it for an unrelated reason. ft_btran_touched_ is the
  /// sparse pattern THIS buffer's last ft_btran_unit() call actually set - dense entries
  /// outside it are exactly 0, since every read this file does of ft_btran_scratch_ goes
  /// through this list (issue #279's follow-up: "a sparse r").
  mutable std::vector<double> ft_btran_scratch_;
  mutable std::vector<Index> ft_btran_touched_;
  /// Per-update scratch for update_forrest_tomlin(), sized to m by ft_init() and reused
  /// across calls for the same reason work_ is: several hundred updates a second must not
  /// each allocate a fresh vector. ft_spike_marked_/ft_spike_touched_ are the same
  /// dense-accumulator-plus-pattern pair sparse.hpp's own SparseVector documents - kept as
  /// plain members here rather than that class so update_forrest_tomlin's existing
  /// dense-array field names (spike, ft_diag_, ...) did not all need to change shape.
  std::vector<double> ft_spike_;
  mutable std::vector<bool> ft_spike_marked_;
  mutable std::vector<Index> ft_spike_touched_;
  /// Nonzero steps of the CURRENT call's alpha-by-step vector (work_), collected for free
  /// while work_ is gathered from the caller's dense `alpha` (that gather is the one place
  /// this update genuinely must touch all m - alpha itself carries no sparsity pattern of
  /// its own). Everything downstream of the gather walks this list instead of 0..m (issue
  /// #279's follow-up: "a sparse spike... walk ft_row_/ft_col_ from the nonzeros of alpha").
  mutable std::vector<Index> ft_work_nz_;

  void ft_init();
  void ft_set(Index owner_step, Index referenced_step, double value);
  void ft_erase(Index owner_step, Index referenced_step);
  /// The partial BTRAN identified by Forrest & Tomlin as the way to compute the row-eta:
  /// e~ = e_step^T U_current^-1, i.e. a BTRAN through U ALONE (no L, no earlier retas),
  /// seeded at a single step. Shares the push logic with ft_forward_substitute() below.
  ///
  /// Writes into the members ft_btran_scratch_/ft_btran_touched_ rather than an out-parameter
  /// (issue #279's follow-up): the result is genuinely sparse - e~'s only possible nonzero
  /// positions are `step` and whatever POSITION >= step's own the elimination pushes reach,
  /// so the loop starts at step's own position (everything earlier is provably zero, not
  /// merely usually zero) and ft_btran_touched_ records exactly which positions the push
  /// actually set, so update_forrest_tomlin() never has to sweep all m to find them.
  void ft_btran_unit(Index step) const;
  void ft_apply_retas(double* residual_by_step) const;      ///< FTRAN: oldest first
  void ft_apply_retas_transposed(double* z_by_step) const;  ///< BTRAN: newest first
  void ft_back_substitute(double* residual_by_step, double* solution_by_step) const;
  void ft_forward_substitute(double* z_by_step) const;
  /// The hyper-sparse transposed-elimination pass (#243), shared by both update schemes
  /// since neither ever touches L: applies M_k^T in decreasing k to work_, indexed by step
  /// throughout, picking up wherever forward_u_transposed() or
  /// ft_forward_substitute()+ft_apply_retas_transposed() left it.
  void apply_transposed_l() const;
};

}  // namespace sankhya
