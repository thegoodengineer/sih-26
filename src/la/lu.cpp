// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse LU with Markowitz pivoting. See lu.hpp for the references and for why
// DenseLu is kept alive as the oracle for this file.
//
// ============================================================================================
// THE TWO SOLVES, DERIVED. Read this before changing either of them.
// ============================================================================================
//
// The elimination applies, at step k, the transformation
//
//     M_k = I - sum_i mult_{i,k} e_i e_{r_k}^T
//
// to the active submatrix, where mult_{i,k} = a[i][c_k] / a[r_k][c_k]. After all m steps
//
//     M_{m-1} ... M_1 M_0 A = U                                                        (1)
//
// where U is upper triangular once its rows are read in the order r_0, r_1, ... and its
// columns in the order c_0, c_1, ...: pivot row r_k has entries only in columns c_k .. c_m-1.
// Inverting (1),
//
//     A = M_0^-1 M_1^-1 ... M_{m-1}^-1 U,      M_k^-1 = I + sum_i mult_{i,k} e_i e_{r_k}^T
//
// FTRAN, solve A x = b.
//   Left-multiply by M_{m-1} ... M_0 and use (1):   U x = (M_{m-1} ... M_0) b.
//   Applying M_k to a vector v is  v[i] -= mult_{i,k} * v[r_k]  for each recorded i, and the
//   factors must be applied in INCREASING k, the order they were produced.
//   Then back-substitute for x in DECREASING k:
//       x[c_k] = ( b[r_k] - sum_{l>k} U[r_k][c_l] * x[c_l] ) / a[r_k][c_k]
//
// BTRAN, solve A^T y = b.
//   A^T = U^T M_{m-1}^-T ... M_0^-T, so with z = (M_{m-1}^-T ... M_0^-T) y we need first
//       U^T z = b
//   U^T is lower triangular in the same orderings, so this is a forward substitution in
//   INCREASING k. Written in push form, which needs U by row exactly as it is stored:
//       z[r_k] = b[c_k] / a[r_k][c_k]
//       then for each (l > k, u) in row r_k:   b[c_l] -= u * z[r_k]
//   Then recover y from z = (M_{m-1}^-T ... M_0^-T) y, i.e. y = M_0^T M_1^T ... M_{m-1}^T z.
//   M_k^T v subtracts sum_i mult_{i,k} v[i] from component r_k alone, and the outermost
//   factor is M_0^T, so these are applied in DECREASING k - the reverse of FTRAN.
//
// The asymmetry is the whole point: FTRAN is L-then-U ascending-then-descending, BTRAN is
// U-then-L ascending-then-descending. Swapping either order produces a plausible vector.
// ============================================================================================

#include "lu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

/// How many low-count columns and rows the pivot search inspects before settling. Suhl &
/// Suhl report that a small budget loses almost nothing against an exhaustive Markowitz
/// search while removing its quadratic cost; 4 is their recommendation and the value every
/// production code has converged on.
constexpr Index kCandidateBudget = 4;

/// Entries whose magnitude falls below this after an update are treated as exact
/// cancellation and removed. Keeping them would inflate the fill counts that drive the
/// refactorization trigger, and they carry no information.
constexpr double kDropTolerance = tol::kZeroDrop;

/// A basis update is rejected when its pivot element is small relative to the largest entry
/// of alpha. This is the update's counterpart to the Markowitz threshold in the
/// factorization: a tiny pivot divides through every subsequent solve and is how a product
/// form quietly loses accuracy over a few hundred iterations.
constexpr double kUpdatePivotThreshold = 1e-7;

/// Refactorize once the eta file reaches this many updates, whatever its size. Bounds the
/// worst-case drift by bounding how long any single factorization is trusted.
///
/// Measured (#68): 128 passes every correctness gate - 320 unit tests and the rational
/// oracle at 0 mismatches - while a sweep to 512 made d2q06c fail outright, so the ceiling
/// is real and this sits well inside it. The time-based break-even in the simplex normally
/// fires first; this is the backstop for a model whose solves are so cheap that it does not.
constexpr Index kMaxEtaCount = 128;

/// Rows or columns of the active submatrix grouped by count, as intrusive doubly linked
/// lists: one node per index, one list per count, every index in at most one list. Moving
/// an index between counts is O(1) and leaves nothing behind, so the pivot search walks
/// live candidates only.
///
/// WHY LISTS AND NOT VECTORS (#210). The buckets used to be vectors that were appended to
/// on every count change and never purged; a pivoted column's entry stayed in the bucket it
/// was found in. The search starts from the lowest counts at every step, and those are
/// exactly the buckets that fill with retired singletons - so the walk past stale entries
/// grew with the number of steps taken, and one factorization of an 18,000-row basis spent
/// 120 ms choosing pivots against 25 ms of arithmetic.
class CountLists {
 public:
  explicit CountLists(Index m)
      : head_(static_cast<std::size_t>(m) + 1, -1),
        tail_(static_cast<std::size_t>(m) + 1, -1),
        next_(static_cast<std::size_t>(m), -1),
        prev_(static_cast<std::size_t>(m), -1),
        count_(static_cast<std::size_t>(m), -1) {}

  [[nodiscard]] Index first(Index count) const {
    return head_[static_cast<std::size_t>(count)];
  }
  [[nodiscard]] Index after(Index index) const {
    return next_[static_cast<std::size_t>(index)];
  }

  /// Take `index` out of whichever list it is in, if any.
  void remove(Index index) {
    const auto u = static_cast<std::size_t>(index);
    const Index count = count_[u];
    if (count < 0) return;
    const auto c = static_cast<std::size_t>(count);
    const Index p = prev_[u];
    const Index n = next_[u];
    if (p >= 0) {
      next_[static_cast<std::size_t>(p)] = n;
    } else {
      head_[c] = n;
    }
    if (n >= 0) {
      prev_[static_cast<std::size_t>(n)] = p;
    } else {
      tail_[c] = p;
    }
    count_[u] = -1;
    prev_[u] = -1;
    next_[u] = -1;
  }

  /// File `index` under `count`, at the back of that list. An index already filed under
  /// this count keeps its place.
  void place(Index index, Index count) {
    const auto u = static_cast<std::size_t>(index);
    if (count_[u] == count) return;
    remove(index);
    const auto c = static_cast<std::size_t>(count);
    count_[u] = count;
    prev_[u] = tail_[c];
    next_[u] = -1;
    if (tail_[c] >= 0) {
      next_[static_cast<std::size_t>(tail_[c])] = index;
    } else {
      head_[c] = index;
    }
    tail_[c] = index;
  }

 private:
  std::vector<Index> head_, tail_, next_, prev_, count_;
};

}  // namespace

// =========================================================================================
// Workspace - the active submatrix during elimination
// =========================================================================================

struct SparseLu::Workspace {
  Index m = 0;

  /// Values live COLUMN-wise. The elimination updates whole columns at a time (see
  /// eliminate()), so this is the orientation that keeps the inner loop contiguous.
  std::vector<std::vector<Index>> col_rows;
  std::vector<std::vector<double>> col_values;

  /// Rows carry the PATTERN only, and it is allowed to go stale: an entry removed by
  /// cancellation is not hunted down here. Every consumer re-checks against the column
  /// storage, which is authoritative. Chasing exact row patterns costs more than it saves.
  std::vector<std::vector<Index>> row_cols;

  /// Exact active counts. These drive Markowitz, so unlike row_cols they are maintained
  /// precisely on every insertion and deletion.
  std::vector<Index> row_count;
  std::vector<Index> col_count;

  std::vector<char> row_active;
  std::vector<char> col_active;

  /// Dense scatter/gather buffers indexed by ROW, used one column at a time.
  std::vector<double> acc;
  std::vector<char> acc_present;

  /// Multipliers of the current step, and the pivot row's entries collected as the update
  /// sweeps its columns.
  std::vector<Index> mult_rows;
  std::vector<double> mult_values;
  std::vector<Index> u_cols;
  std::vector<double> u_vals;

  /// The pivot row's active columns, DEDUPLICATED. row_cols is allowed to go stale, and a
  /// column removed by cancellation and later re-created by fill appears in it twice. Both
  /// the count bookkeeping and the update below must visit each column exactly once, so the
  /// list is built through a marker array rather than iterated in place.
  std::vector<Index> pivot_row_columns;
  std::vector<char> column_seen;

  /// The active rows of the column being updated, recorded as it is scattered, so that the
  /// gather afterwards visits those rows and the multiplier rows and nothing else. It used
  /// to sweep all m rows for every column of every step - the other half of #210.
  std::vector<Index> old_rows;

  void init(Index dimension) {
    m = dimension;
    const auto u = static_cast<std::size_t>(dimension);
    col_rows.assign(u, {});
    col_values.assign(u, {});
    row_cols.assign(u, {});
    row_count.assign(u, 0);
    col_count.assign(u, 0);
    row_active.assign(u, 1);
    col_active.assign(u, 1);
    acc.assign(u, 0.0);
    acc_present.assign(u, 0);
    column_seen.assign(u, 0);
  }
};

// =========================================================================================
// Factorization
// =========================================================================================

bool SparseLu::factorize(const std::vector<LuColumn>& columns, Index m, double pivot_tolerance,
                         double markowitz_threshold, const ShouldStop& should_stop) {
  m_ = m;
  stopped_early_ = false;
  pivot_row_.clear();
  pivot_col_.clear();
  pivot_value_.clear();
  l_start_.assign(1, 0);
  l_rows_.clear();
  l_values_.clear();
  u_start_.assign(1, 0);
  u_steps_.clear();
  u_values_.clear();
  eta_start_.assign(1, 0);
  eta_rows_.clear();
  eta_values_.clear();
  eta_pivot_position_.clear();
  eta_pivot_value_.clear();
  base_nonzeros_ = 0;
  work_.assign(static_cast<std::size_t>(m), 0.0);
  smallest_pivot_ = 0.0;
  largest_pivot_ = 0.0;
  dependent_positions_.clear();
  uncovered_rows_.clear();

  if (m == 0) return true;
  if (static_cast<Index>(columns.size()) != m) return false;

  Workspace w;
  w.init(m);

  for (Index j = 0; j < m; ++j) {
    const LuColumn& column = columns[static_cast<std::size_t>(j)];
    const auto uj = static_cast<std::size_t>(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index row = column.rows[k];
      const double value = column.values[k];
      if (row < 0 || row >= m) return false;
      if (value == 0.0) continue;
      w.col_rows[uj].push_back(row);
      w.col_values[uj].push_back(value);
      w.row_cols[static_cast<std::size_t>(row)].push_back(j);
      ++w.col_count[uj];
      ++w.row_count[static_cast<std::size_t>(row)];
    }
  }

  if (!eliminate(w, pivot_tolerance, markowitz_threshold, should_stop)) return false;

  // U was recorded against column indices, because at the moment a pivot row is retired the
  // step at which each of its columns will itself be eliminated is not yet known. Translate
  // the whole array once, now that the permutation is complete.
  std::vector<Index> step_of_column(static_cast<std::size_t>(m), -1);
  for (Index k = 0; k < m; ++k) {
    step_of_column[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] = k;
  }
  for (Index& entry : u_steps_) {
    const Index step = step_of_column[static_cast<std::size_t>(entry)];
    if (step < 0) return false;  // a column that was never pivotal: structurally singular
    entry = step;
  }
  build_column_u();
  build_row_l();
  base_nonzeros_ = factor_nonzeros();
  return true;
}

void SparseLu::build_row_l() {
  // Transpose L from step-major (step k holds the rows its multipliers touch) into
  // row-major by STEP: for the row retired at step k, the earlier steps j < k that hold a
  // multiplier on it. A multiplier of step j lives on a row still active after step j, so
  // that row's own step is later than j and the map is well defined.
  const Index m = m_;
  std::vector<Index> step_of_row(static_cast<std::size_t>(m), -1);
  for (Index k = 0; k < m; ++k) {
    step_of_row[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] = k;
  }
  lr_start_.assign(static_cast<std::size_t>(m) + 1, 0);
  for (const Index row : l_rows_) {
    ++lr_start_[static_cast<std::size_t>(step_of_row[static_cast<std::size_t>(row)]) + 1];
  }
  for (Index k = 0; k < m; ++k) {
    lr_start_[static_cast<std::size_t>(k) + 1] += lr_start_[static_cast<std::size_t>(k)];
  }
  lr_steps_.assign(static_cast<std::size_t>(lr_start_[static_cast<std::size_t>(m)]), 0);
  lr_values_.assign(lr_steps_.size(), 0.0);
  std::vector<Index> fill(lr_start_.begin(), lr_start_.end() - 1);
  for (Index j = 0; j < m; ++j) {
    for (Index p = l_start_[static_cast<std::size_t>(j)];
         p < l_start_[static_cast<std::size_t>(j) + 1]; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index k = step_of_row[static_cast<std::size_t>(l_rows_[up])];
      const auto slot = static_cast<std::size_t>(fill[static_cast<std::size_t>(k)]++);
      lr_steps_[slot] = j;
      lr_values_[slot] = l_values_[up];
    }
  }
}

void SparseLu::build_column_u() {
  // Transpose the row-wise U (row-step k holds the steps j > k of its off-pivot entries)
  // into column form: column-step j holds the row-steps i < j that push into it.
  const Index m = m_;
  uc_start_.assign(static_cast<std::size_t>(m) + 1, 0);
  for (Index k = 0; k < m; ++k) {
    for (Index p = u_start_[static_cast<std::size_t>(k)];
         p < u_start_[static_cast<std::size_t>(k) + 1]; ++p) {
      const Index j = u_steps_[static_cast<std::size_t>(p)];
      if (j <= k) continue;
      ++uc_start_[static_cast<std::size_t>(j) + 1];
    }
  }
  for (Index j = 0; j < m; ++j) {
    uc_start_[static_cast<std::size_t>(j) + 1] += uc_start_[static_cast<std::size_t>(j)];
  }
  uc_steps_.assign(static_cast<std::size_t>(uc_start_[static_cast<std::size_t>(m)]), 0);
  uc_values_.assign(uc_steps_.size(), 0.0);
  std::vector<Index> fill(uc_start_.begin(), uc_start_.end() - 1);
  for (Index k = 0; k < m; ++k) {
    for (Index p = u_start_[static_cast<std::size_t>(k)];
         p < u_start_[static_cast<std::size_t>(k) + 1]; ++p) {
      const Index j = u_steps_[static_cast<std::size_t>(p)];
      if (j <= k) continue;
      const auto slot = static_cast<std::size_t>(fill[static_cast<std::size_t>(j)]++);
      uc_steps_[slot] = k;
      uc_values_[slot] = u_values_[static_cast<std::size_t>(p)];
    }
  }
}

// =========================================================================================
// Basis update
// =========================================================================================

bool SparseLu::update(Index leaving_position, const double* alpha) {
  if (m_ == 0) return false;
  if (leaving_position < 0 || leaving_position >= m_) return false;

  const auto pivot_index = static_cast<std::size_t>(leaving_position);
  const double pivot = alpha[pivot_index];

  // Reject rather than divide by something too small. The factorization is left untouched,
  // so the caller can refactorize and retry the same pivot on fresh factors.
  double largest = 0.0;
  for (Index i = 0; i < m_; ++i) {
    largest = std::max(largest, std::fabs(alpha[static_cast<std::size_t>(i)]));
  }
  if (!std::isfinite(pivot)) return false;
  if (std::fabs(pivot) < kUpdatePivotThreshold * std::max(1.0, largest)) return false;

  for (Index i = 0; i < m_; ++i) {
    const double value = alpha[static_cast<std::size_t>(i)];
    if (i == leaving_position || std::fabs(value) < kDropTolerance) continue;
    if (!std::isfinite(value)) return false;
    eta_rows_.push_back(i);
    eta_values_.push_back(value);
  }
  eta_start_.push_back(static_cast<Index>(eta_rows_.size()));
  eta_pivot_position_.push_back(leaving_position);
  eta_pivot_value_.push_back(pivot);
  return true;
}

bool SparseLu::should_refactorize() const noexcept {
  // Only the drift bound remains here. The fill-ratio rule that used to sit beside it was
  // replaced by the simplex's measured break-even (see primal_simplex.cpp): a fill ratio
  // assumes a fixed relationship between eta size and eta cost that no single constant
  // captured across instances.
  return eta_count() >= kMaxEtaCount;
}

bool SparseLu::eliminate(Workspace& w, double pivot_tolerance, double threshold,
                         const ShouldStop& should_stop) {
  const Index m = w.m;
  pivot_row_.reserve(static_cast<std::size_t>(m));
  pivot_col_.reserve(static_cast<std::size_t>(m));
  pivot_value_.reserve(static_cast<std::size_t>(m));

  smallest_pivot_ = std::numeric_limits<double>::max();
  largest_pivot_ = 0.0;

  // Active rows and columns filed by count, so the pivot search can start from the
  // sparsest without scanning everything. A retired row or column is taken out of its list
  // and a changed count moves it, so the lists hold live candidates only (see CountLists).
  CountLists col_lists(m);
  CountLists row_lists(m);
  for (Index j = 0; j < m; ++j) col_lists.place(j, w.col_count[static_cast<std::size_t>(j)]);
  for (Index i = 0; i < m; ++i) row_lists.place(i, w.row_count[static_cast<std::size_t>(i)]);

  // A count that drifts out of [0, m] is a bookkeeping bug, and the symptom is not a wrong
  // answer but an out-of-bounds write: a negative count casts to a huge size_t and indexes
  // past the end of the list heads. Assert on the way in rather than segfaulting later at
  // a place that says nothing about the cause.
  const auto rebucket_column = [&](Index j) {
    const Index count = w.col_count[static_cast<std::size_t>(j)];
    assert(count >= 0 && count <= m && "column count out of range");
    col_lists.place(j, count);
  };
  const auto rebucket_row = [&](Index i) {
    const Index count = w.row_count[static_cast<std::size_t>(i)];
    assert(count >= 0 && count <= m && "row count out of range");
    row_lists.place(i, count);
  };

  for (Index step = 0; step < m; ++step) {
    // ASKED BEFORE EVERY PIVOT, as the LDL^T asks before every elimination step (#197): on
    // Mittelmann's bdry2 one factorization of a 376,500-row basis ran for minutes, and the
    // simplex could only look at the clock once it returned - 648 s against a 300 s limit
    // (#208). A coarser cadence was measured wrong for the LDL^T (48 s past a 10 s limit
    // in one batch of 64 steps) and is not tried here. The check decides only whether an
    // unfinished factorization keeps running; it never touches the arithmetic.
    if (should_stop && should_stop()) {
      stopped_early_ = true;
      return false;
    }
    // ---- choose a pivot ------------------------------------------------------------------
    Index best_row = -1;
    Index best_col = -1;
    double best_value = 0.0;
    Index best_cost = std::numeric_limits<Index>::max();
    Index examined = 0;

    // A column of count 1 has Markowitz cost 0 whatever its row, and so does a row of
    // count 1. Sweeping the buckets in increasing count finds those first, which is the
    // singleton/triangular pre-pass falling out of the general rule rather than being
    // special-cased alongside it.
    for (Index count = 1; count <= m && examined < kCandidateBudget && best_cost > 0; ++count) {
      // -- candidate columns of this count
      for (Index j = col_lists.first(count); j >= 0 && examined < kCandidateBudget;
           j = col_lists.after(j)) {
        const auto uj = static_cast<std::size_t>(j);
        assert(w.col_active[uj] != 0 && w.col_count[uj] == count && "a stale list entry");
        ++examined;

        double column_max = 0.0;
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
        }
        if (column_max < pivot_tolerance) continue;

        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          const Index i = w.col_rows[uj][t];
          const auto ui = static_cast<std::size_t>(i);
          if (w.row_active[ui] == 0) continue;
          const double value = w.col_values[uj][t];
          if (std::fabs(value) < threshold * column_max) continue;
          if (std::fabs(value) < pivot_tolerance) continue;
          const Index cost = (w.row_count[ui] - 1) * (count - 1);
          if (cost < best_cost) {
            best_cost = cost;
            best_row = i;
            best_col = j;
            best_value = value;
            if (cost == 0) break;
          }
        }
      }

      // -- candidate rows of this count, which the column sweep can miss entirely when a
      //    row singleton sits in a dense column
      for (Index i = row_lists.first(count); i >= 0 && examined < kCandidateBudget;
           i = row_lists.after(i)) {
        const auto ui = static_cast<std::size_t>(i);
        assert(w.row_active[ui] != 0 && w.row_count[ui] == count && "a stale list entry");
        ++examined;

        for (const Index j : w.row_cols[ui]) {
          const auto uj = static_cast<std::size_t>(j);
          if (w.col_active[uj] == 0) continue;
          const Index cost = (count - 1) * (w.col_count[uj] - 1);
          if (cost >= best_cost) continue;

          double column_max = 0.0;
          double value = 0.0;
          bool found = false;
          for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
            const Index r = w.col_rows[uj][t];
            if (w.row_active[static_cast<std::size_t>(r)] == 0) continue;
            const double v = w.col_values[uj][t];
            column_max = std::max(column_max, std::fabs(v));
            if (r == i) {
              value = v;
              found = true;
            }
          }
          if (!found) continue;  // the row pattern was stale
          if (std::fabs(value) < pivot_tolerance) continue;
          if (std::fabs(value) < threshold * column_max) continue;

          best_cost = cost;
          best_row = i;
          best_col = j;
          best_value = value;
          if (cost == 0) break;
        }
      }
    }

    // BUDGETED SEARCH FAILURE IS NOT PROOF OF SINGULARITY (issue #143). kCandidateBudget
    // bounds the search to a handful of low-count buckets for speed, which is the right
    // trade on a healthy basis - but on THIS step it just means none of the columns and rows
    // LOOKED AT had an admissible entry, not that none exists anywhere in the active
    // submatrix. Bailing out here at the first such step is exactly the bug: the entire
    // REMAINING m - step columns get reported as "the singular set" when almost all of them
    // were never actually examined. Measured on grow15/pilot4/25fv47/perold/d6cube, that
    // inflated the reported defect from a handful of genuinely dependent columns to 33-94%
    // of the whole basis.
    //
    // So before concluding anything, fall back to an EXHAUSTIVE scan of every active column
    // still standing, unbounded by the budget. If that finds an admissible pivot, the
    // budgeted search merely got unlucky with bucket order and this step proceeds normally.
    // Only when the exhaustive scan ALSO finds nothing is the active submatrix genuinely
    // singular to working precision - every remaining entry is below pivot_tolerance or
    // below threshold * its column's max, which is the asserted condition below, not an
    // assumed one.
    if (best_row < 0) {
      for (Index j = 0; j < m; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0) continue;

        double column_max = 0.0;
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
        }
        if (column_max < pivot_tolerance) continue;

        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          const Index i = w.col_rows[uj][t];
          const auto ui = static_cast<std::size_t>(i);
          if (w.row_active[ui] == 0) continue;
          const double value = w.col_values[uj][t];
          if (std::fabs(value) < pivot_tolerance) continue;
          if (std::fabs(value) < threshold * column_max) continue;
          const Index cost = (w.row_count[ui] - 1) * (w.col_count[uj] - 1);
          if (cost < best_cost) {
            best_cost = cost;
            best_row = i;
            best_col = j;
            best_value = value;
            if (cost == 0) break;
          }
        }
        if (best_cost == 0) break;
      }
    }

    if (best_row < 0) {
      // GENUINELY SINGULAR, not a search artefact: assert the condition this conclusion
      // rests on rather than assume it, exactly because "continuing must not turn singular
      // into silently wrong" (issue #143). Every active entry must fail admissibility for
      // SOME reason - too small outright, or too small relative to its own column's max - and
      // this recomputes both per column to check it directly, rather than trusting that the
      // exhaustive scan above could not itself have a bug that skipped a live entry.
#ifndef NDEBUG
      for (Index j = 0; j < m; ++j) {
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0) continue;
        double column_max = 0.0;
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          column_max = std::max(column_max, std::fabs(w.col_values[uj][t]));
        }
        for (std::size_t t = 0; t < w.col_rows[uj].size(); ++t) {
          if (w.row_active[static_cast<std::size_t>(w.col_rows[uj][t])] == 0) continue;
          const double magnitude = std::fabs(w.col_values[uj][t]);
          assert((magnitude < pivot_tolerance || magnitude < threshold * column_max) &&
                 "eliminate() declared the active submatrix singular but an admissible pivot "
                 "was still present - the exhaustive fallback scan has a bug");
        }
      }
#endif

      // The residual: every row and column still active is exactly the rank defect the
      // repair (elsewhere) needs to know about, reported by POSITION in the original
      // `columns` array factorize() was given, and by original row index - not by
      // elimination step, because these columns never reached one.
      dependent_positions_.clear();
      uncovered_rows_.clear();
      for (Index j = 0; j < m; ++j) {
        if (w.col_active[static_cast<std::size_t>(j)] != 0) dependent_positions_.push_back(j);
      }
      for (Index i = 0; i < m; ++i) {
        if (w.row_active[static_cast<std::size_t>(i)] != 0) uncovered_rows_.push_back(i);
      }
      return false;
    }

    const auto pivot_r = static_cast<std::size_t>(best_row);
    const auto pivot_c = static_cast<std::size_t>(best_col);
    const double pivot = best_value;
    smallest_pivot_ = std::min(smallest_pivot_, std::fabs(pivot));
    largest_pivot_ = std::max(largest_pivot_, std::fabs(pivot));

    pivot_row_.push_back(best_row);
    pivot_col_.push_back(best_col);
    pivot_value_.push_back(pivot);

    // ---- multipliers ---------------------------------------------------------------------
    w.mult_rows.clear();
    w.mult_values.clear();
    for (std::size_t t = 0; t < w.col_rows[pivot_c].size(); ++t) {
      const Index i = w.col_rows[pivot_c][t];
      const auto ui = static_cast<std::size_t>(i);
      if (w.row_active[ui] == 0 || i == best_row) continue;
      const double value = w.col_values[pivot_c][t];
      if (value == 0.0) continue;
      w.mult_rows.push_back(i);
      w.mult_values.push_back(value / pivot);
    }

    // The pivot row and column leave the active submatrix now, so that neither the update
    // below nor any later step has to keep excluding them by hand.
    w.row_active[pivot_r] = 0;
    w.col_active[pivot_c] = 0;
    row_lists.remove(best_row);
    col_lists.remove(best_col);

    // Every row that had an entry in the pivot column loses it.
    for (const Index i : w.mult_rows) --w.row_count[static_cast<std::size_t>(i)];

    // Deduplicate the pivot row's active columns before touching any counts. See the note
    // on pivot_row_columns: a stale pattern can name the same column twice, and visiting it
    // twice would double-decrement col_count and emit two U entries for one coefficient.
    w.pivot_row_columns.clear();
    for (const Index j : w.row_cols[pivot_r]) {
      const auto uj = static_cast<std::size_t>(j);
      if (w.col_active[uj] == 0 || w.column_seen[uj] != 0) continue;
      w.column_seen[uj] = 1;
      w.pivot_row_columns.push_back(j);
    }
    for (const Index j : w.pivot_row_columns) w.column_seen[static_cast<std::size_t>(j)] = 0;
    // NOTE: col_count is deliberately NOT decremented here. The pivot row's pattern is
    // allowed to go stale, so it can name a column whose entry was cancelled away at an
    // earlier step. Decrementing on the strength of the pattern alone undercounts that
    // column, and the error accumulates until the count goes NEGATIVE - at which point
    // col_bucket[static_cast<std::size_t>(-1)] writes off the end of the bucket array. The
    // decrement therefore happens in the update loop below, once the column's own storage
    // has confirmed the entry is really there.

    // ---- update the remaining columns ----------------------------------------------------
    w.u_cols.clear();
    w.u_vals.clear();

    for (const Index j : w.pivot_row_columns) {
      const auto uj = static_cast<std::size_t>(j);

      std::vector<Index>& rows = w.col_rows[uj];
      std::vector<double>& values = w.col_values[uj];

      // Scatter the column, keeping only rows that are still active. The pivot row's own
      // entry is what becomes the U coefficient.
      double pivot_row_value = 0.0;
      bool pivot_row_present = false;
      w.old_rows.clear();
      for (std::size_t t = 0; t < rows.size(); ++t) {
        const Index i = rows[t];
        const auto ui = static_cast<std::size_t>(i);
        if (i == best_row) {
          pivot_row_value = values[t];
          pivot_row_present = true;
          continue;
        }
        if (w.row_active[ui] == 0) continue;
        w.acc[ui] = values[t];
        w.acc_present[ui] = 1;
        w.old_rows.push_back(i);
      }

      if (!pivot_row_present || pivot_row_value == 0.0) {
        // A stale row-pattern entry: nothing to eliminate against in this column. Undo the
        // scatter and move on.
        for (const Index i : w.old_rows) {
          const auto ui = static_cast<std::size_t>(i);
          w.acc[ui] = 0.0;
          w.acc_present[ui] = 0;
        }
        continue;
      }

      // The entry is confirmed present, so now the column really does lose it.
      --w.col_count[uj];

      w.u_cols.push_back(j);
      w.u_vals.push_back(pivot_row_value);

      for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
        const auto ui = static_cast<std::size_t>(w.mult_rows[t]);
        const double contribution = w.mult_values[t] * pivot_row_value;
        if (w.acc_present[ui] == 0) {
          // Fill-in. The row gains a column it did not have; the row pattern has to learn
          // about it or a later pivot search will never consider this entry.
          w.acc[ui] = -contribution;
          w.acc_present[ui] = 1;
          w.row_cols[ui].push_back(j);
          ++w.row_count[ui];
          ++w.col_count[uj];
        } else {
          w.acc[ui] -= contribution;
        }
      }

      // Gather back, dropping exact cancellations.
      rows.clear();
      values.clear();
      for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
        const auto ui = static_cast<std::size_t>(w.mult_rows[t]);
        if (w.acc_present[ui] == 0) continue;
        const double value = w.acc[ui];
        w.acc[ui] = 0.0;
        w.acc_present[ui] = 0;
        if (std::fabs(value) < kDropTolerance) {
          --w.row_count[ui];
          --w.col_count[uj];
          continue;
        }
        rows.push_back(w.mult_rows[t]);
        values.push_back(value);
      }
      // Whatever remains in the accumulator belongs to rows the pivot column did not touch:
      // rows of the column's old pattern, which the scatter recorded.
      for (const Index i : w.old_rows) {
        const auto ui = static_cast<std::size_t>(i);
        if (w.acc_present[ui] == 0) continue;
        rows.push_back(i);
        values.push_back(w.acc[ui]);
        w.acc[ui] = 0.0;
        w.acc_present[ui] = 0;
      }

      rebucket_column(j);
    }

    for (const Index i : w.mult_rows) rebucket_row(i);

    // ---- commit the step's factors -------------------------------------------------------
    for (std::size_t t = 0; t < w.mult_rows.size(); ++t) {
      l_rows_.push_back(w.mult_rows[t]);
      l_values_.push_back(w.mult_values[t]);
    }
    l_start_.push_back(static_cast<Index>(l_rows_.size()));

    for (std::size_t t = 0; t < w.u_cols.size(); ++t) {
      u_steps_.push_back(w.u_cols[t]);  // still a COLUMN here; translated in factorize()
      u_values_.push_back(w.u_vals[t]);
    }
    u_start_.push_back(static_cast<Index>(u_steps_.size()));
  }

  if (smallest_pivot_ == std::numeric_limits<double>::max()) smallest_pivot_ = 0.0;
  return true;
}

// =========================================================================================
// Solves
// =========================================================================================

void SparseLu::forward_l(double* b) const {
  // Apply the elimination factors in increasing k. See the derivation at the top.
  for (Index k = 0; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const double pivot_component = b[static_cast<std::size_t>(pivot_row_[uk])];
    if (pivot_component == 0.0) continue;  // hyper-sparsity, in its cheapest form
    const Index begin = l_start_[uk];
    const Index end = l_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      b[static_cast<std::size_t>(l_rows_[up])] -= l_values_[up] * pivot_component;
    }
  }
}

void SparseLu::apply_etas(double* b) const {
  // Then the recorded updates, OLDEST FIRST: x = E_k^-1 ... E_1^-1 (B_0^-1 b). Applying
  // E^-1 is z_p = y_p / alpha_p followed by z_i = y_i - alpha_i z_p.
  const Index etas = eta_count();
  for (Index k = 0; k < etas; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const auto pivot_index = static_cast<std::size_t>(eta_pivot_position_[uk]);
    const double scaled = b[pivot_index] / eta_pivot_value_[uk];
    b[pivot_index] = scaled;
    if (scaled == 0.0) continue;
    const Index begin = eta_start_[uk];
    const Index end = eta_start_[uk + 1];
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      b[static_cast<std::size_t>(eta_rows_[ut])] -= eta_values_[ut] * scaled;
    }
  }
}

void SparseLu::solve(double* b) const {
  if (m_ == 0) return;
  forward_l(b);

  // HYPER-SPARSE BACK-SUBSTITUTION (#68; Gilbert & Peierls 1988). U is walked by COLUMN in
  // decreasing step order, and a step whose result is exactly zero pushes nothing: on a
  // right-hand side with a handful of nonzeros - the entering column of a large sparse
  // basis - this touches a small fraction of U. The gather in solve_reference() reads all
  // of U regardless; the two must agree to rounding, and a test says so.
  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])];
  }
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = work_[uk] / pivot_value_[uk];
    work_[uk] = value;
    if (value == 0.0) continue;
    const Index begin = uc_start_[uk];
    const Index end = uc_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      work_[static_cast<std::size_t>(uc_steps_[up])] -= uc_values_[up] * value;
    }
  }
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] =
        work_[static_cast<std::size_t>(k)];
  }

  apply_etas(b);
}

void SparseLu::solve_reference(double* b) const {
  if (m_ == 0) return;
  forward_l(b);
  // Back-substitute in decreasing k. work_ holds x indexed by elimination step, so that the
  // inner loop can index U directly; it is scattered back to column order at the end.
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    double sum = b[static_cast<std::size_t>(pivot_row_[uk])];
    const Index begin = u_start_[uk];
    const Index end = u_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      if (step <= k) continue;  // the pivot entry itself
      sum -= u_values_[up] * work_[static_cast<std::size_t>(step)];
    }
    work_[uk] = sum / pivot_value_[uk];
  }

  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])] =
        work_[static_cast<std::size_t>(k)];
  }

  apply_etas(b);
}

void SparseLu::apply_etas_transposed(double* b) const {
  // The updates come FIRST here and in the REVERSE order to FTRAN:
  // x = B_0^-T (E_1^-T ... E_k^-T b). Applying E^-T touches one component,
  // v_p <- (v_p - sum_{i != p} alpha_i v_i) / alpha_p, leaving the rest alone.
  for (Index k = eta_count() - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index begin = eta_start_[uk];
    const Index end = eta_start_[uk + 1];
    double accumulated = 0.0;
    for (Index t = begin; t < end; ++t) {
      const auto ut = static_cast<std::size_t>(t);
      accumulated += eta_values_[ut] * b[static_cast<std::size_t>(eta_rows_[ut])];
    }
    const auto pivot_index = static_cast<std::size_t>(eta_pivot_position_[uk]);
    b[pivot_index] = (b[pivot_index] - accumulated) / eta_pivot_value_[uk];
  }
}

void SparseLu::forward_u_transposed() const {
  // Forward-substitute through U^T in increasing k, in push form, on work_ indexed by step.
  // A step whose result is exactly zero pushes nothing.
  std::vector<double>& z = work_;
  for (Index k = 0; k < m_; ++k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk] / pivot_value_[uk];
    z[uk] = value;
    if (value == 0.0) continue;
    const Index begin = u_start_[uk];
    const Index end = u_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      const Index step = u_steps_[up];
      if (step <= k) continue;
      z[static_cast<std::size_t>(step)] -= u_values_[up] * value;
    }
  }
}

void SparseLu::solve_transpose(double* b) const {
  if (m_ == 0) return;
  apply_etas_transposed(b);

  // work_ is indexed by step from here to the end: the right-hand side enters through the
  // pivot columns and the answer leaves through the pivot rows, and both triangular passes
  // run in step space in between, so nothing is scattered to row order and gathered back.
  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
  }
  forward_u_transposed();

  // HYPER-SPARSE TRANSPOSED ELIMINATION (#243; Gilbert & Peierls 1988). Apply M_k^T in
  // DECREASING k. M_k^T subtracts, from the component on pivot row r_k, the multipliers of
  // step k times the components on the rows they touch; every row a step-k multiplier
  // touches is retired at a LATER step, so in decreasing k the component on r_k is final
  // the moment step k is reached and can be pushed into every earlier step that holds a
  // multiplier on r_k - which is exactly what the row-wise L lists. A zero component
  // pushes nothing, so a sparse rho costs the rows it reaches rather than the whole of L,
  // which the gather in solve_transpose_reference() reads regardless.
  std::vector<double>& z = work_;
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const double value = z[uk];
    if (value == 0.0) continue;
    const Index begin = lr_start_[uk];
    const Index end = lr_start_[uk + 1];
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      z[static_cast<std::size_t>(lr_steps_[up])] -= lr_values_[up] * value;
    }
  }

  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] =
        z[static_cast<std::size_t>(k)];
  }
}

void SparseLu::solve_transpose_reference(double* b) const {
  if (m_ == 0) return;
  apply_etas_transposed(b);

  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
  }
  forward_u_transposed();

  // z is indexed by step and belongs on the pivot ROWS; place it there before applying the
  // transposed elimination factors, which are indexed by row.
  std::vector<double>& z = work_;
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] =
        z[static_cast<std::size_t>(k)];
  }

  // Apply M_k^T in DECREASING k - the reverse of FTRAN, and the half of this file most
  // likely to be "corrected" into agreement with solve() by someone who has not read the
  // derivation above. This is the gather form: every entry of L is read whatever the
  // density of the result.
  for (Index k = m_ - 1; k >= 0; --k) {
    const auto uk = static_cast<std::size_t>(k);
    const Index begin = l_start_[uk];
    const Index end = l_start_[uk + 1];
    double accumulated = 0.0;
    for (Index p = begin; p < end; ++p) {
      const auto up = static_cast<std::size_t>(p);
      accumulated += l_values_[up] * b[static_cast<std::size_t>(l_rows_[up])];
    }
    b[static_cast<std::size_t>(pivot_row_[uk])] -= accumulated;
  }
}

}  // namespace sankhya
