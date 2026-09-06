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
                         double markowitz_threshold) {
  m_ = m;
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

  if (!eliminate(w, pivot_tolerance, markowitz_threshold)) return false;

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
  base_nonzeros_ = factor_nonzeros();
  return true;
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

bool SparseLu::eliminate(Workspace& w, double pivot_tolerance, double threshold) {
  const Index m = w.m;
  pivot_row_.reserve(static_cast<std::size_t>(m));
  pivot_col_.reserve(static_cast<std::size_t>(m));
  pivot_value_.reserve(static_cast<std::size_t>(m));

  smallest_pivot_ = std::numeric_limits<double>::max();
  largest_pivot_ = 0.0;

  // Active rows and columns bucketed by count, so the pivot search can start from the
  // sparsest without scanning everything. Entries go stale as counts change and are
  // filtered on the way out rather than being relocated eagerly.
  std::vector<std::vector<Index>> col_bucket(static_cast<std::size_t>(m) + 1);
  std::vector<std::vector<Index>> row_bucket(static_cast<std::size_t>(m) + 1);
  for (Index j = 0; j < m; ++j) {
    col_bucket[static_cast<std::size_t>(w.col_count[static_cast<std::size_t>(j)])].push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    row_bucket[static_cast<std::size_t>(w.row_count[static_cast<std::size_t>(i)])].push_back(i);
  }

  // A count that drifts out of [0, m] is a bookkeeping bug, and the symptom is not a wrong
  // answer but an out-of-bounds write: a negative count casts to a huge size_t and indexes
  // past the end of the bucket array. Assert on the way in rather than segfaulting later at
  // a place that says nothing about the cause.
  const auto rebucket_column = [&](Index j) {
    const Index count = w.col_count[static_cast<std::size_t>(j)];
    assert(count >= 0 && count <= m && "column count out of range");
    col_bucket[static_cast<std::size_t>(count)].push_back(j);
  };
  const auto rebucket_row = [&](Index i) {
    const Index count = w.row_count[static_cast<std::size_t>(i)];
    assert(count >= 0 && count <= m && "row count out of range");
    row_bucket[static_cast<std::size_t>(count)].push_back(i);
  };

  for (Index step = 0; step < m; ++step) {
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
      const auto uc = static_cast<std::size_t>(count);

      // -- candidate columns of this count
      std::vector<Index>& cols = col_bucket[uc];
      for (std::size_t p = 0; p < cols.size() && examined < kCandidateBudget; ++p) {
        const Index j = cols[p];
        const auto uj = static_cast<std::size_t>(j);
        if (w.col_active[uj] == 0 || w.col_count[uj] != count) continue;  // stale
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
      std::vector<Index>& rows = row_bucket[uc];
      for (std::size_t p = 0; p < rows.size() && examined < kCandidateBudget; ++p) {
        const Index i = rows[p];
        const auto ui = static_cast<std::size_t>(i);
        if (w.row_active[ui] == 0 || w.row_count[ui] != count) continue;  // stale
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
      }

      if (!pivot_row_present || pivot_row_value == 0.0) {
        // A stale row-pattern entry: nothing to eliminate against in this column. Undo the
        // scatter and move on.
        for (std::size_t t = 0; t < rows.size(); ++t) {
          const auto ui = static_cast<std::size_t>(rows[t]);
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
      // Whatever remains in the accumulator belongs to rows the pivot column did not touch.
      for (Index i = 0; i < m; ++i) {
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

void SparseLu::solve(double* b) const {
  if (m_ == 0) return;

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

void SparseLu::solve_transpose(double* b) const {
  if (m_ == 0) return;

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

  // Forward-substitute through U^T in increasing k, in push form. work_ is indexed by step
  // and holds the right-hand side as it is consumed.
  for (Index k = 0; k < m_; ++k) {
    work_[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(pivot_col_[static_cast<std::size_t>(k)])];
  }

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

  // z is indexed by step and belongs on the pivot ROWS; place it there before applying the
  // transposed elimination factors, which are indexed by row.
  for (Index i = 0; i < m_; ++i) b[static_cast<std::size_t>(i)] = 0.0;
  for (Index k = 0; k < m_; ++k) {
    b[static_cast<std::size_t>(pivot_row_[static_cast<std::size_t>(k)])] =
        z[static_cast<std::size_t>(k)];
  }

  // Apply M_k^T in DECREASING k - the reverse of FTRAN, and the half of this file most
  // likely to be "corrected" into agreement with solve() by someone who has not read the
  // derivation above.
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
