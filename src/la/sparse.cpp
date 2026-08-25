// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse container implementation.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems" (SIAM, 2006), ch. 2.
// The triplet-to-CSC conversion and the transpose are both the standard two-pass counting
// sort: count per column, prefix-sum into starts, then place. That gives ordered row
// indices without ever calling a comparison sort, which matters because finalize() runs
// once per model read and once per presolve round on matrices with millions of entries.

#include "sankhya/sparse.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace sankhya {

// =========================================================================================
// SparseMatrix
// =========================================================================================

void SparseMatrix::reset(Index num_rows, Index num_cols) {
  assert(num_rows >= 0 && num_cols >= 0);
  num_rows_ = num_rows;
  num_cols_ = num_cols;
  frozen_ = false;
  column_starts_.clear();
  row_indices_.clear();
  values_.clear();
  build_rows_.clear();
  build_cols_.clear();
  build_values_.clear();
}

void SparseMatrix::reserve(std::size_t n) {
  if (frozen_) {
    row_indices_.reserve(n);
    values_.reserve(n);
  } else {
    build_rows_.reserve(n);
    build_cols_.reserve(n);
    build_values_.reserve(n);
  }
}

void SparseMatrix::add_entry(Index row, Index col, double value) {
  assert(!frozen_ && "add_entry on a frozen matrix; call unfreeze() first");
  assert(row >= 0 && row < num_rows_);
  assert(col >= 0 && col < num_cols_);
  build_rows_.push_back(row);
  build_cols_.push_back(col);
  build_values_.push_back(value);
}

void SparseMatrix::finalize(double drop_tol) {
  if (frozen_) return;

  const std::size_t nnz = build_values_.size();

  // Pass 1: count entries per column.
  column_starts_.assign(static_cast<std::size_t>(num_cols_) + 1, 0);
  for (std::size_t k = 0; k < nnz; ++k) {
    ++column_starts_[static_cast<std::size_t>(build_cols_[k]) + 1];
  }
  // Prefix sum turns counts into column starts.
  for (Index j = 0; j < num_cols_; ++j) {
    column_starts_[static_cast<std::size_t>(j) + 1] +=
        column_starts_[static_cast<std::size_t>(j)];
  }

  // Pass 2: place entries, tracking the next free slot per column.
  std::vector<Index> next(column_starts_.begin(), column_starts_.end() - 1);
  std::vector<Index> scratch_rows(nnz);
  std::vector<double> scratch_values(nnz);
  for (std::size_t k = 0; k < nnz; ++k) {
    const auto slot =
        static_cast<std::size_t>(next[static_cast<std::size_t>(build_cols_[k])]++);
    scratch_rows[slot] = build_rows_[k];
    scratch_values[slot] = build_values_[k];
  }

  // Pass 3: within each column sort by row, sum duplicates, drop structural zeros.
  row_indices_.clear();
  values_.clear();
  row_indices_.reserve(nnz);
  values_.reserve(nnz);

  std::vector<Index> order;
  std::vector<Index> final_starts(static_cast<std::size_t>(num_cols_) + 1, 0);
  for (Index j = 0; j < num_cols_; ++j) {
    const auto begin = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j)]);
    const auto end = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j) + 1]);
    final_starts[static_cast<std::size_t>(j)] = static_cast<Index>(row_indices_.size());
    if (begin == end) continue;

    order.resize(end - begin);
    std::iota(order.begin(), order.end(), static_cast<Index>(0));
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
      return scratch_rows[begin + static_cast<std::size_t>(a)] <
             scratch_rows[begin + static_cast<std::size_t>(b)];
    });

    std::size_t p = 0;
    while (p < order.size()) {
      const Index row = scratch_rows[begin + static_cast<std::size_t>(order[p])];
      double sum = 0.0;
      while (p < order.size() &&
             scratch_rows[begin + static_cast<std::size_t>(order[p])] == row) {
        sum += scratch_values[begin + static_cast<std::size_t>(order[p])];
        ++p;
      }
      // Written as the NEGATION of "negligible" rather than as `fabs(sum) >= drop_tol`,
      // and the difference is not stylistic. Every comparison involving NaN is false, so
      // the direct form quietly answers "no, do not keep this" for a NaN coefficient and
      // DELETES it - turning a corrupt model into a well formed but different one, which
      // then solves cleanly and reports a confident wrong optimum. This form keeps NaN and
      // infinity so that Model::validate() can reject them by name. Behaviour is unchanged
      // for every finite value.
      const bool negligible = std::fabs(sum) < drop_tol;
      if (!negligible) {
        row_indices_.push_back(row);
        values_.push_back(sum);
      }
    }
  }
  final_starts[static_cast<std::size_t>(num_cols_)] = static_cast<Index>(row_indices_.size());
  column_starts_.swap(final_starts);

  build_rows_.clear();
  build_cols_.clear();
  build_values_.clear();
  build_rows_.shrink_to_fit();
  build_cols_.shrink_to_fit();
  build_values_.shrink_to_fit();
  frozen_ = true;
}

void SparseMatrix::unfreeze() {
  if (!frozen_) return;
  const std::size_t nnz = values_.size();
  build_rows_.resize(nnz);
  build_cols_.resize(nnz);
  build_values_.resize(nnz);
  for (Index j = 0; j < num_cols_; ++j) {
    const auto begin = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j)]);
    const auto end = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j) + 1]);
    for (std::size_t k = begin; k < end; ++k) {
      build_rows_[k] = row_indices_[k];
      build_cols_[k] = j;
      build_values_[k] = values_[k];
    }
  }
  column_starts_.clear();
  row_indices_.clear();
  values_.clear();
  frozen_ = false;
}

Index SparseMatrix::num_nonzeros() const noexcept {
  return frozen_ ? static_cast<Index>(values_.size())
                 : static_cast<Index>(build_values_.size());
}

const std::vector<Index>& SparseMatrix::column_starts() const noexcept {
  ensure_frozen();
  return column_starts_;
}

const std::vector<Index>& SparseMatrix::row_indices() const noexcept {
  ensure_frozen();
  return row_indices_;
}

const std::vector<double>& SparseMatrix::values() const noexcept {
  ensure_frozen();
  return values_;
}

ColumnView SparseMatrix::column(Index j) const noexcept {
  ensure_frozen();
  assert(j >= 0 && j < num_cols_);
  const auto begin = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j)]);
  const auto end = static_cast<std::size_t>(column_starts_[static_cast<std::size_t>(j) + 1]);
  ColumnView view;
  view.size = static_cast<Index>(end - begin);
  if (view.size > 0) {
    view.rows = row_indices_.data() + begin;
    view.values = values_.data() + begin;
  }
  return view;
}

double SparseMatrix::at(Index row, Index col) const noexcept {
  const ColumnView c = column(col);
  const Index* found = std::lower_bound(c.rows, c.rows + c.size, row);
  if (found != c.rows + c.size && *found == row) {
    return c.values[found - c.rows];
  }
  return 0.0;
}

void SparseMatrix::multiply_add(const double* x, double* y, double alpha) const {
  ensure_frozen();
  for (Index j = 0; j < num_cols_; ++j) {
    const double xj = alpha * x[static_cast<std::size_t>(j)];
    if (xj == 0.0) continue;
    const ColumnView c = column(j);
    for (Index k = 0; k < c.size; ++k) {
      y[static_cast<std::size_t>(c.rows[k])] += c.values[k] * xj;
    }
  }
}

void SparseMatrix::transpose_multiply_add(const double* x, double* y, double alpha) const {
  ensure_frozen();
  for (Index j = 0; j < num_cols_; ++j) {
    const ColumnView c = column(j);
    double dot = 0.0;
    for (Index k = 0; k < c.size; ++k) {
      dot += c.values[k] * x[static_cast<std::size_t>(c.rows[k])];
    }
    y[static_cast<std::size_t>(j)] += alpha * dot;
  }
}

void SparseMatrix::multiply(const double* x, double* y, double alpha) const {
  std::fill(y, y + num_rows_, 0.0);
  multiply_add(x, y, alpha);
}

void SparseMatrix::transpose_multiply(const double* x, double* y, double alpha) const {
  std::fill(y, y + num_cols_, 0.0);
  transpose_multiply_add(x, y, alpha);
}

SparseMatrix SparseMatrix::transpose() const {
  ensure_frozen();
  SparseMatrix t;
  t.num_rows_ = num_cols_;
  t.num_cols_ = num_rows_;
  t.frozen_ = true;

  const std::size_t nnz = values_.size();
  t.column_starts_.assign(static_cast<std::size_t>(num_rows_) + 1, 0);
  for (std::size_t k = 0; k < nnz; ++k) {
    ++t.column_starts_[static_cast<std::size_t>(row_indices_[k]) + 1];
  }
  for (Index i = 0; i < num_rows_; ++i) {
    t.column_starts_[static_cast<std::size_t>(i) + 1] +=
        t.column_starts_[static_cast<std::size_t>(i)];
  }

  t.row_indices_.resize(nnz);
  t.values_.resize(nnz);
  std::vector<Index> next(t.column_starts_.begin(), t.column_starts_.end() - 1);
  // Walking source columns in ascending order makes the destination row indices ascending
  // too, so the result is ordered without a sort.
  for (Index j = 0; j < num_cols_; ++j) {
    const ColumnView c = column(j);
    for (Index k = 0; k < c.size; ++k) {
      const auto slot = static_cast<std::size_t>(next[static_cast<std::size_t>(c.rows[k])]++);
      t.row_indices_[slot] = j;
      t.values_[slot] = c.values[k];
    }
  }
  return t;
}

double SparseMatrix::max_abs_value() const noexcept {
  const std::vector<double>& v = frozen_ ? values_ : build_values_;
  double best = 0.0;
  for (const double x : v) best = std::max(best, std::fabs(x));
  return best;
}

// =========================================================================================
// CsrView
// =========================================================================================

void CsrView::build(const SparseMatrix& a) {
  num_rows_ = a.num_rows();
  num_cols_ = a.num_cols();
  const SparseMatrix t = a.transpose();
  row_starts_ = t.column_starts();
  column_indices_ = t.row_indices();
  values_ = t.values();
}

ColumnView CsrView::row(Index i) const noexcept {
  assert(i >= 0 && i < num_rows_);
  const auto begin = static_cast<std::size_t>(row_starts_[static_cast<std::size_t>(i)]);
  const auto end = static_cast<std::size_t>(row_starts_[static_cast<std::size_t>(i) + 1]);
  ColumnView view;
  view.size = static_cast<Index>(end - begin);
  if (view.size > 0) {
    view.rows = column_indices_.data() + begin;
    view.values = values_.data() + begin;
  }
  return view;
}

// =========================================================================================
// SparseVector
// =========================================================================================

void SparseVector::resize(Index dimension) {
  assert(dimension >= 0);
  dimension_ = dimension;
  dense_.assign(static_cast<std::size_t>(dimension), 0.0);
  marked_.assign(static_cast<std::size_t>(dimension), 0);
  pattern_.clear();
}

void SparseVector::clear() {
  for (const Index i : pattern_) {
    const auto u = static_cast<std::size_t>(i);
    dense_[u] = 0.0;
    marked_[u] = 0;
  }
  pattern_.clear();
}

void SparseVector::compress(double drop_tol) {
  std::size_t out = 0;
  for (std::size_t k = 0; k < pattern_.size(); ++k) {
    const Index i = pattern_[k];
    const auto u = static_cast<std::size_t>(i);
    if (std::fabs(dense_[u]) >= drop_tol) {
      pattern_[out++] = i;
    } else {
      dense_[u] = 0.0;
      marked_[u] = 0;
    }
  }
  pattern_.resize(out);
}

void SparseVector::scatter_to_dense(double* out) const {
  std::fill(out, out + dimension_, 0.0);
  for (const Index i : pattern_) {
    out[static_cast<std::size_t>(i)] = dense_[static_cast<std::size_t>(i)];
  }
}

void SparseVector::gather_from_dense(const double* in, double drop_tol) {
  clear();
  for (Index i = 0; i < dimension_; ++i) {
    const double v = in[static_cast<std::size_t>(i)];
    if (std::fabs(v) >= drop_tol) set(i, v);
  }
}

}  // namespace sankhya
