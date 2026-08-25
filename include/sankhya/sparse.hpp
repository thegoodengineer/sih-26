// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse linear algebra containers.
//
// Reference: Davis, "Direct Methods for Sparse Linear Systems" (SIAM, 2006), ch. 2, for the
// compressed-column layout and the two-pass counting-sort transpose used here.
//
// Design notes that matter downstream:
//  * SparseMatrix has two lifecycle states. In BUILD state it accepts unordered triplets
//    with duplicates. finalize() performs a counting sort into compressed columns, sums
//    duplicates and drops structural zeros; afterwards the matrix is FROZEN and the read
//    accessors return stable pointers. Every solver consumes frozen matrices only, so no
//    engine ever has to defend against a half-built matrix.
//  * SparseVector implements the dense-accumulator-plus-index-list pattern that FTRAN and
//    BTRAN need: O(1) scatter/accumulate into a dense array, with an explicit nonzero
//    pattern so the subsequent gather stays proportional to the result sparsity rather
//    than to the dimension. This is the container that makes hyper-sparsity (Phase 6)
//    possible; getting its interface right now avoids rewriting FTRAN later.
#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {

/// A read-only view of one column of a frozen SparseMatrix.
struct ColumnView {
  const Index* rows = nullptr;
  const double* values = nullptr;
  Index size = 0;

  [[nodiscard]] bool empty() const noexcept { return size == 0; }
};

/// Compressed-sparse-column matrix with a build path and a frozen read path.
class SparseMatrix {
 public:
  SparseMatrix() = default;
  SparseMatrix(Index num_rows, Index num_cols) { reset(num_rows, num_cols); }

  /// Discard all contents and re-enter BUILD state with the given dimensions.
  void reset(Index num_rows, Index num_cols);

  /// Reserve space for n triplets. Purely a performance hint.
  void reserve(std::size_t n);

  /// BUILD state only. Records A(row, col) += value. Duplicates are legal and are summed
  /// by finalize(); out-of-range indices are a programming error and assert.
  void add_entry(Index row, Index col, double value);

  /// Transition BUILD -> FROZEN. Sorts by (col, row), sums duplicates, and drops entries
  /// whose magnitude is below drop_tol. Idempotent on an already-frozen matrix.
  void finalize(double drop_tol = tol::kZeroDrop);

  /// Return to BUILD state, keeping the existing entries as triplets. Used by presolve,
  /// which mutates a model that a reader already froze.
  void unfreeze();

  [[nodiscard]] bool frozen() const noexcept { return frozen_; }
  [[nodiscard]] Index num_rows() const noexcept { return num_rows_; }
  [[nodiscard]] Index num_cols() const noexcept { return num_cols_; }
  [[nodiscard]] Index num_nonzeros() const noexcept;

  // ---- Frozen read path. All of these assert on a non-frozen matrix. -------------------

  [[nodiscard]] const std::vector<Index>& column_starts() const noexcept;
  [[nodiscard]] const std::vector<Index>& row_indices() const noexcept;
  [[nodiscard]] const std::vector<double>& values() const noexcept;

  /// View of column j. Entries are ordered by ascending row index.
  [[nodiscard]] ColumnView column(Index j) const noexcept;

  /// Look up a single entry. O(log nnz_in_column). Intended for tests and readers, not
  /// for inner loops.
  [[nodiscard]] double at(Index row, Index col) const noexcept;

  // ---- Kernels ------------------------------------------------------------------------

  /// y += alpha * A * x.  x has num_cols entries, y has num_rows entries.
  void multiply_add(const double* x, double* y, double alpha = 1.0) const;

  /// y += alpha * A^T * x.  x has num_rows entries, y has num_cols entries.
  void transpose_multiply_add(const double* x, double* y, double alpha = 1.0) const;

  /// y = alpha * A * x (y is overwritten).
  void multiply(const double* x, double* y, double alpha = 1.0) const;

  /// y = alpha * A^T * x (y is overwritten).
  void transpose_multiply(const double* x, double* y, double alpha = 1.0) const;

  /// Explicit transpose as a new frozen matrix. Counting sort, O(nnz + n).
  [[nodiscard]] SparseMatrix transpose() const;

  /// Largest magnitude entry, or 0.0 when empty.
  [[nodiscard]] double max_abs_value() const noexcept;

 private:
  void ensure_frozen() const noexcept {
    assert(frozen_ && "SparseMatrix must be finalized before reading");
  }

  Index num_rows_ = 0;
  Index num_cols_ = 0;
  bool frozen_ = false;

  // FROZEN representation. column_starts_ has num_cols_ + 1 entries.
  std::vector<Index> column_starts_;
  std::vector<Index> row_indices_;
  std::vector<double> values_;

  // BUILD representation.
  std::vector<Index> build_rows_;
  std::vector<Index> build_cols_;
  std::vector<double> build_values_;
};

/// Row-compressed view of a frozen SparseMatrix. Held separately rather than as a second
/// copy inside SparseMatrix because only some engines need it: the simplex works entirely
/// column-wise, while PDHG and presolve want both orientations.
class CsrView {
 public:
  CsrView() = default;
  explicit CsrView(const SparseMatrix& a) { build(a); }

  void build(const SparseMatrix& a);

  [[nodiscard]] Index num_rows() const noexcept { return num_rows_; }
  [[nodiscard]] Index num_cols() const noexcept { return num_cols_; }
  [[nodiscard]] const std::vector<Index>& row_starts() const noexcept { return row_starts_; }
  [[nodiscard]] const std::vector<Index>& column_indices() const noexcept {
    return column_indices_;
  }
  [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

  /// View of row i, ordered by ascending column index. Reuses the ColumnView layout; its
  /// rows member holds column indices here.
  [[nodiscard]] ColumnView row(Index i) const noexcept;

 private:
  Index num_rows_ = 0;
  Index num_cols_ = 0;
  std::vector<Index> row_starts_;
  std::vector<Index> column_indices_;
  std::vector<double> values_;
};

/// Dense accumulator plus explicit nonzero pattern.
///
/// Invariant: pattern_ lists exactly the indices i for which marked_[i] is true, in
/// insertion order, and dense_[i] is only meaningful for those i. Entries whose value
/// cancels to zero stay in the pattern until compress() is called - dropping them eagerly
/// would cost a linear scan on every accumulate.
class SparseVector {
 public:
  SparseVector() = default;
  explicit SparseVector(Index dimension) { resize(dimension); }

  /// Set the ambient dimension and clear the contents.
  void resize(Index dimension);

  /// Reset to the zero vector in O(nnz), not O(dimension).
  void clear();

  [[nodiscard]] Index dimension() const noexcept { return dimension_; }
  [[nodiscard]] Index num_nonzeros() const noexcept {
    return static_cast<Index>(pattern_.size());
  }
  [[nodiscard]] const std::vector<Index>& pattern() const noexcept { return pattern_; }

  /// Value at i. Defined for every index in range, zero outside the pattern.
  [[nodiscard]] double operator[](Index i) const noexcept {
    assert(i >= 0 && i < dimension_);
    const auto u = static_cast<std::size_t>(i);
    return marked_[u] != 0 ? dense_[u] : 0.0;
  }

  /// v[i] += value, registering i in the pattern on first touch. This is the hot call in
  /// FTRAN/BTRAN, so it is inline and branch-light.
  void add(Index i, double value) {
    assert(i >= 0 && i < dimension_);
    const auto u = static_cast<std::size_t>(i);
    if (marked_[u] == 0) {
      marked_[u] = 1;
      dense_[u] = value;
      pattern_.push_back(i);
    } else {
      dense_[u] += value;
    }
  }

  /// v[i] = value, registering i in the pattern on first touch.
  void set(Index i, double value) {
    assert(i >= 0 && i < dimension_);
    const auto u = static_cast<std::size_t>(i);
    if (marked_[u] == 0) {
      marked_[u] = 1;
      pattern_.push_back(i);
    }
    dense_[u] = value;
  }

  /// Drop pattern entries whose magnitude is below drop_tol. O(nnz).
  void compress(double drop_tol = tol::kZeroDrop);

  /// Overwrite out (length dimension()) with the dense contents, zeroing everything else.
  void scatter_to_dense(double* out) const;

  /// Replace the contents with the nonzeros of the dense array in (length dimension()).
  void gather_from_dense(const double* in, double drop_tol = tol::kZeroDrop);

 private:
  Index dimension_ = 0;
  std::vector<double> dense_;
  std::vector<char> marked_;  // char rather than bool: we want addressable bytes.
  std::vector<Index> pattern_;
};

}  // namespace sankhya
