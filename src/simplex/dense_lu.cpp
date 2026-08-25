// SPDX-License-Identifier: Apache-2.0
// SANKHYA - dense LU factorization, implementation.

#include "dense_lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sankhya {

bool DenseLu::factorize(std::vector<double> columns, Index m, double pivot_tolerance) {
  m_ = m;
  lu_ = std::move(columns);
  pivot_.assign(static_cast<std::size_t>(m), 0);
  smallest_pivot_ = std::numeric_limits<double>::infinity();
  largest_pivot_ = 0.0;
  if (m == 0) {
    smallest_pivot_ = 0.0;
    return true;
  }

  for (Index k = 0; k < m_; ++k) {
    // Partial pivoting: the largest magnitude in the remaining column. Partial rather than
    // complete pivoting because the growth factor bound it gives is enough here and the
    // column search is what keeps this O(m^3) rather than O(m^3) with a large constant.
    Index best_row = k;
    double best = std::fabs(at(k, k));
    for (Index i = k + 1; i < m_; ++i) {
      const double candidate = std::fabs(at(i, k));
      if (candidate > best) {
        best = candidate;
        best_row = i;
      }
    }

    if (best < pivot_tolerance) {
      smallest_pivot_ = best;
      return false;
    }
    smallest_pivot_ = std::min(smallest_pivot_, best);
    largest_pivot_ = std::max(largest_pivot_, best);

    pivot_[static_cast<std::size_t>(k)] = best_row;
    if (best_row != k) {
      for (Index j = 0; j < m_; ++j) std::swap(at(k, j), at(best_row, j));
    }

    const double diagonal = at(k, k);
    for (Index i = k + 1; i < m_; ++i) at(i, k) /= diagonal;

    for (Index j = k + 1; j < m_; ++j) {
      const double multiplier = at(k, j);
      if (multiplier == 0.0) continue;
      for (Index i = k + 1; i < m_; ++i) at(i, j) -= at(i, k) * multiplier;
    }
  }
  return true;
}

// B = P^T L U, so B z = b becomes L U z = P b: permute, forward-substitute through the
// unit lower factor, then back-substitute through the upper one.
void DenseLu::solve(double* b) const {
  for (Index k = 0; k < m_; ++k) {
    const Index p = pivot_[static_cast<std::size_t>(k)];
    if (p != k) std::swap(b[k], b[p]);
  }

  for (Index k = 0; k < m_; ++k) {
    const double value = b[k];
    if (value == 0.0) continue;
    for (Index i = k + 1; i < m_; ++i) b[i] -= at(i, k) * value;
  }

  for (Index k = m_ - 1; k >= 0; --k) {
    b[k] /= at(k, k);
    const double value = b[k];
    if (value == 0.0) continue;
    for (Index i = 0; i < k; ++i) b[i] -= at(i, k) * value;
  }
}

// B^T = U^T L^T P, so B^T z = b becomes: forward-substitute through U^T (which is lower
// triangular with U's diagonal), back-substitute through the unit upper L^T, then undo the
// permutation in reverse order.
void DenseLu::solve_transpose(double* b) const {
  for (Index k = 0; k < m_; ++k) {
    double sum = b[k];
    for (Index i = 0; i < k; ++i) sum -= at(i, k) * b[i];
    b[k] = sum / at(k, k);
  }

  for (Index k = m_ - 1; k >= 0; --k) {
    double sum = b[k];
    for (Index i = k + 1; i < m_; ++i) sum -= at(i, k) * b[i];
    b[k] = sum;
  }

  for (Index k = m_ - 1; k >= 0; --k) {
    const Index p = pivot_[static_cast<std::size_t>(k)];
    if (p != k) std::swap(b[k], b[p]);
  }
}

}  // namespace sankhya
