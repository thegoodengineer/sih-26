// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sparse symmetric LDL^T (#70). See ldl.hpp for the references; every step below
// is implemented from their description of the algorithm, none from another solver's code.

#include "la/ldl.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "sankhya/tolerances.hpp"

namespace sankhya {

// -----------------------------------------------------------------------------------------
// Ordering: minimum degree on the explicit elimination graph (Tinney & Walker 1967)
// -----------------------------------------------------------------------------------------
//
// The exact rule, not the approximate one production codes use: at each step eliminate the
// vertex of smallest degree, add the clique its neighbours form, and repeat. Kept simple
// deliberately - the elimination graph is stored explicitly as sorted neighbour lists, so
// the cost is proportional to the fill it creates, which is fine for the row counts this
// solver reaches today and is the part to replace (by AMD's quotient graph) when it is not.
// Ties go to the lowest index, so the ordering is deterministic.
bool SparseLdl::minimum_degree(const SparseMatrix& lower, const ShouldStop& should_stop) {
  const Index n = n_;
  std::vector<std::vector<Index>> adjacency(static_cast<std::size_t>(n));
  for (Index c = 0; c < n; ++c) {
    const ColumnView column = lower.column(c);
    for (Index p = 0; p < column.size; ++p) {
      const Index r = column.rows[p];
      if (r <= c) continue;  // the strict lower triangle defines the graph
      adjacency[static_cast<std::size_t>(r)].push_back(c);
      adjacency[static_cast<std::size_t>(c)].push_back(r);
    }
  }
  for (auto& list : adjacency) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  std::vector<bool> eliminated(static_cast<std::size_t>(n), false);
  perm_.clear();
  perm_.reserve(static_cast<std::size_t>(n));
  std::vector<Index> merged;
  for (Index step = 0; step < n; ++step) {
    // ASKED EVERY STEP. A coarser granularity was tried first and is not good enough: the
    // cost of one elimination step grows as fill accumulates, and on a 20,000-row model a
    // single batch of 64 steps ran for 48 seconds past a 10 second limit. One call through a
    // std::function per step is nothing beside the clique merging below (#193).
    if (should_stop && should_stop()) return false;

    // The vertex of minimum current degree; lowest index on a tie.
    Index best = -1;
    std::size_t best_degree = std::numeric_limits<std::size_t>::max();
    for (Index v = 0; v < n; ++v) {
      const auto u = static_cast<std::size_t>(v);
      if (eliminated[u]) continue;
      if (adjacency[u].size() < best_degree) {
        best_degree = adjacency[u].size();
        best = v;
      }
    }
    const auto p = static_cast<std::size_t>(best);
    perm_.push_back(best);
    eliminated[p] = true;
    // Eliminating p makes its neighbours a clique: every neighbour u gains p's other
    // neighbours and loses p.
    const std::vector<Index> neighbours = std::move(adjacency[p]);
    adjacency[p].clear();
    for (const Index v : neighbours) {
      const auto u = static_cast<std::size_t>(v);
      std::vector<Index>& own = adjacency[u];
      merged.clear();
      merged.reserve(own.size() + neighbours.size());
      std::set_union(own.begin(), own.end(), neighbours.begin(), neighbours.end(),
                     std::back_inserter(merged));
      merged.erase(std::remove_if(merged.begin(), merged.end(),
                                  [&](Index w) { return w == v || w == best; }),
                   merged.end());
      own.swap(merged);
    }
  }
  inverse_.assign(static_cast<std::size_t>(n), -1);
  for (Index k = 0; k < n; ++k)
    inverse_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])] = k;
  return true;
}

// -----------------------------------------------------------------------------------------
// The permuted matrix, upper triangle by column
// -----------------------------------------------------------------------------------------
//
// The up-looking factorization consumes row k of the lower triangle, which is column k of
// the upper one. Each original entry (r, c), r >= c, lands in permuted column max(pr, pc)
// at permuted row min(pr, pc). Duplicates from the caller are summed.
void SparseLdl::build_permuted_pattern(const SparseMatrix& lower) {
  const Index n = n_;
  std::vector<std::vector<std::pair<Index, double>>> columns(static_cast<std::size_t>(n));
  for (Index c = 0; c < n; ++c) {
    const ColumnView column = lower.column(c);
    for (Index p = 0; p < column.size; ++p) {
      const Index r = column.rows[p];
      if (r < c) continue;
      const Index pr = inverse_[static_cast<std::size_t>(r)];
      const Index pc = inverse_[static_cast<std::size_t>(c)];
      const Index hi = std::max(pr, pc);
      const Index lo = std::min(pr, pc);
      columns[static_cast<std::size_t>(hi)].push_back({lo, column.values[p]});
    }
  }
  a_starts_.assign(static_cast<std::size_t>(n) + 1, 0);
  a_rows_.clear();
  a_values_.clear();
  for (Index k = 0; k < n; ++k) {
    auto& column = columns[static_cast<std::size_t>(k)];
    std::sort(column.begin(), column.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    for (std::size_t p = 0; p < column.size(); ++p) {
      if (p > 0 && column[p].first == column[p - 1].first) {
        a_values_.back() += column[p].second;
        continue;
      }
      a_rows_.push_back(column[p].first);
      a_values_.push_back(column[p].second);
    }
    a_starts_[static_cast<std::size_t>(k) + 1] = static_cast<Index>(a_rows_.size());
  }
}

// -----------------------------------------------------------------------------------------
// Elimination tree (Liu 1990; Davis 2006, sec. 4.1)
// -----------------------------------------------------------------------------------------
//
// parent[j] is the smallest i > j with L(i, j) != 0. Computed by walking, for each column k
// and each entry A(i, k) with i < k, from i up through the tree built so far until a root,
// which becomes a child of k; `ancestor` compresses the paths so the walk stays near-linear.
void SparseLdl::elimination_tree() {
  const Index n = n_;
  parent_.assign(static_cast<std::size_t>(n), -1);
  std::vector<Index> ancestor(static_cast<std::size_t>(n), -1);
  for (Index k = 0; k < n; ++k) {
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      Index i = a_rows_[static_cast<std::size_t>(p)];
      while (i != -1 && i < k) {
        const Index next = ancestor[static_cast<std::size_t>(i)];
        ancestor[static_cast<std::size_t>(i)] = k;
        if (next == -1) {
          parent_[static_cast<std::size_t>(i)] = k;
          break;
        }
        i = next;
      }
    }
  }
}

// -----------------------------------------------------------------------------------------
// Symbolic pattern of L (Davis 2006, sec. 4.2-4.3)
// -----------------------------------------------------------------------------------------
//
// The nonzeros of row k of L are the nodes reached from the entries A(i, k), i < k, by
// walking up the elimination tree until a node already reached for this k. Every node j
// reached gets L(k, j) != 0, i.e. row k in column j. Rows are visited in increasing k, so the
// row lists of every column come out sorted with no extra work.
void SparseLdl::symbolic_pattern() {
  const Index n = n_;
  std::vector<Index> mark(static_cast<std::size_t>(n), -1);
  std::vector<Index> count(static_cast<std::size_t>(n), 0);
  std::vector<std::vector<Index>> rows(static_cast<std::size_t>(n));
  for (Index k = 0; k < n; ++k) {
    mark[static_cast<std::size_t>(k)] = k;
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      Index i = a_rows_[static_cast<std::size_t>(p)];
      while (i < k && mark[static_cast<std::size_t>(i)] != k) {
        mark[static_cast<std::size_t>(i)] = k;
        rows[static_cast<std::size_t>(i)].push_back(k);
        i = parent_[static_cast<std::size_t>(i)];
        if (i == -1) break;
      }
    }
  }
  l_starts_.assign(static_cast<std::size_t>(n) + 1, 0);
  l_rows_.clear();
  for (Index j = 0; j < n; ++j) {
    l_rows_.insert(l_rows_.end(), rows[static_cast<std::size_t>(j)].begin(),
                   rows[static_cast<std::size_t>(j)].end());
    l_starts_[static_cast<std::size_t>(j) + 1] = static_cast<Index>(l_rows_.size());
  }
  l_values_.assign(l_rows_.size(), 0.0);
  d_.assign(static_cast<std::size_t>(n), 0.0);
}

bool SparseLdl::analyze(const SparseMatrix& lower, const ShouldStop& should_stop) {
  analyzed_ = false;
  stopped_early_ = false;
  if (lower.num_rows() != lower.num_cols() || lower.num_rows() <= 0) return false;
  n_ = lower.num_rows();
  // The ordering is where the time goes: measured on generated instances, analyze() costs
  // three to four times a numeric factorization, and at 20,000 rows it is most of an
  // 813-second first iteration (#193). It is therefore the one phase that has to be
  // interruptible for a time limit to mean anything.
  if (!minimum_degree(lower, should_stop)) {
    stopped_early_ = true;
    return false;
  }
  build_permuted_pattern(lower);
  elimination_tree();
  symbolic_pattern();
  analyzed_ = true;
  return true;
}

// -----------------------------------------------------------------------------------------
// Numeric factorization, up-looking (Davis 2006, sec. 4.4, for LDL^T)
// -----------------------------------------------------------------------------------------
//
// Row k of L solves  L(0:k-1, 0:k-1) D L(k, 0:k-1)^T = A(0:k-1, k).  With y = D L(k, :)^T
// that is a sparse forward substitution over the reach of row k, columns ascending (the
// tree guarantees that order is topological); then L(k, j) = y_j / d_j and
// d_k = A(k, k) - sum_j L(k, j) y_j. Each column's rows are filled in increasing k, which is
// exactly the order the symbolic pattern listed them in.
bool SparseLdl::factorize(const SparseMatrix& lower, double regularization,
                          const ShouldStop& should_stop) {
  stopped_early_ = false;
  if (!analyzed_ || lower.num_rows() != n_ || lower.num_cols() != n_) return false;
  const Index n = n_;

  // Refresh the permuted values. The pattern may be a subset of the analyzed one; an entry
  // outside it means the caller changed the structure, which is a contract violation.
  std::fill(a_values_.begin(), a_values_.end(), 0.0);
  for (Index c = 0; c < n; ++c) {
    const ColumnView column = lower.column(c);
    for (Index p = 0; p < column.size; ++p) {
      const Index r = column.rows[p];
      if (r < c) continue;
      const Index pr = inverse_[static_cast<std::size_t>(r)];
      const Index pc = inverse_[static_cast<std::size_t>(c)];
      const Index hi = std::max(pr, pc);
      const Index lo = std::min(pr, pc);
      const auto begin = a_rows_.begin() + a_starts_[static_cast<std::size_t>(hi)];
      const auto end = a_rows_.begin() + a_starts_[static_cast<std::size_t>(hi) + 1];
      const auto slot = std::lower_bound(begin, end, lo);
      if (slot == end || *slot != lo) return false;
      a_values_[static_cast<std::size_t>(slot - a_rows_.begin())] += column.values[p];
    }
  }

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  std::vector<Index> mark(static_cast<std::size_t>(n), -1);
  std::vector<Index> reach;
  std::vector<Index> fill(static_cast<std::size_t>(n), 0);  // entries stored per column
  regularized_ = 0;
  smallest_pivot_ = std::numeric_limits<double>::infinity();
  largest_pivot_ = 0.0;

  for (Index k = 0; k < n; ++k) {
    // The same coarse deadline the ordering uses (#193). A numeric factorization is cheaper
    // than the ordering that preceded it, but on a large model it is still long enough to
    // outlast a time limit on its own.
    if (should_stop && should_stop()) {
      stopped_early_ = true;
      return false;
    }
    // Scatter A(0:k-1, k) and the diagonal; collect the reach.
    reach.clear();
    double diagonal = 0.0;
    mark[static_cast<std::size_t>(k)] = k;
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      const Index i = a_rows_[static_cast<std::size_t>(p)];
      const double value = a_values_[static_cast<std::size_t>(p)];
      if (i == k) {
        diagonal += value;
        continue;
      }
      x[static_cast<std::size_t>(i)] += value;
      Index j = i;
      while (j != -1 && j < k && mark[static_cast<std::size_t>(j)] != k) {
        mark[static_cast<std::size_t>(j)] = k;
        reach.push_back(j);
        j = parent_[static_cast<std::size_t>(j)];
      }
    }
    std::sort(reach.begin(), reach.end());

    // Forward substitution over the reach: y_j = x_j after every earlier column's update.
    for (const Index j : reach) {
      const auto uj = static_cast<std::size_t>(j);
      const double y = x[uj];
      const Index begin = l_starts_[uj];
      const Index stored = fill[uj];
      for (Index p = begin; p < begin + stored; ++p) {
        const Index i = l_rows_[static_cast<std::size_t>(p)];
        x[static_cast<std::size_t>(i)] -= l_values_[static_cast<std::size_t>(p)] * y;
      }
      const double l_kj = y / d_[uj];
      // Row k is the next row of column j by construction of the symbolic pattern.
      const auto slot = static_cast<std::size_t>(begin + stored);
      l_values_[slot] = l_kj;
      ++fill[uj];
      diagonal -= l_kj * y;
      x[uj] = 0.0;
    }

    // THE PIVOT, REGULARIZED (Altman & Gondzio). A pivot at or below the threshold is
    // replaced by it: the factors then belong to a matrix that differs from the given one
    // on that diagonal entry, by less than the threshold, which is the standard IPM remedy
    // for a normal-equations matrix that has become singular in the limit.
    if (!(diagonal > regularization)) {
      diagonal = regularization;
      ++regularized_;
    }
    d_[static_cast<std::size_t>(k)] = diagonal;
    smallest_pivot_ = std::min(smallest_pivot_, diagonal);
    largest_pivot_ = std::max(largest_pivot_, diagonal);
  }
  if (n == 0) smallest_pivot_ = 0.0;
  return true;
}

void SparseLdl::solve(double* b) const {
  const Index n = n_;
  std::vector<double> z(static_cast<std::size_t>(n));
  for (Index k = 0; k < n; ++k) {
    z[static_cast<std::size_t>(k)] =
        b[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])];
  }
  // L z = b
  for (Index j = 0; j < n; ++j) {
    const double zj = z[static_cast<std::size_t>(j)];
    if (zj == 0.0) continue;
    for (Index p = l_starts_[static_cast<std::size_t>(j)];
         p < l_starts_[static_cast<std::size_t>(j) + 1]; ++p) {
      z[static_cast<std::size_t>(l_rows_[static_cast<std::size_t>(p)])] -=
          l_values_[static_cast<std::size_t>(p)] * zj;
    }
  }
  // D z = z
  for (Index k = 0; k < n; ++k)
    z[static_cast<std::size_t>(k)] /= d_[static_cast<std::size_t>(k)];
  // L^T z = z
  for (Index j = n - 1; j >= 0; --j) {
    double sum = z[static_cast<std::size_t>(j)];
    for (Index p = l_starts_[static_cast<std::size_t>(j)];
         p < l_starts_[static_cast<std::size_t>(j) + 1]; ++p) {
      sum -= l_values_[static_cast<std::size_t>(p)] *
             z[static_cast<std::size_t>(l_rows_[static_cast<std::size_t>(p)])];
    }
    z[static_cast<std::size_t>(j)] = sum;
  }
  for (Index k = 0; k < n; ++k) {
    b[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])] =
        z[static_cast<std::size_t>(k)];
  }
}

// -----------------------------------------------------------------------------------------
// A Theta A^T + delta I, lower triangle
// -----------------------------------------------------------------------------------------
void normal_equations_lower(const SparseMatrix& a, const std::vector<double>& theta,
                            const std::vector<double>& row_shift, double delta,
                            SparseMatrix* out) {
  const Index m = a.num_rows();
  const Index n = a.num_cols();
  const bool have_shift = static_cast<Index>(row_shift.size()) == m;
  // Row-wise access to A, once.
  const CsrView by_row(a);
  std::vector<double> accumulator(static_cast<std::size_t>(m), 0.0);
  std::vector<Index> mark(static_cast<std::size_t>(m), -1);
  std::vector<Index> touched;
  out->reset(m, m);
  for (Index i = 0; i < m; ++i) {
    touched.clear();
    // M(r, i) for r >= i: sum over columns j in row i of theta_j a_ij a_rj.
    const ColumnView row = by_row.row(i);
    for (Index p = 0; p < row.size; ++p) {
      const Index j = row.rows[p];  // a CSR view stores COLUMN indices in `rows`
      if (j < 0 || j >= n) continue;
      const double scale = theta[static_cast<std::size_t>(j)] * row.values[p];
      if (scale == 0.0) continue;
      const ColumnView column = a.column(j);
      for (Index q = 0; q < column.size; ++q) {
        const Index r = column.rows[q];
        if (r < i) continue;
        const auto ur = static_cast<std::size_t>(r);
        if (mark[ur] != i) {
          mark[ur] = i;
          accumulator[ur] = 0.0;
          touched.push_back(r);
        }
        accumulator[ur] += scale * column.values[q];
      }
    }
    if (mark[static_cast<std::size_t>(i)] != i) {
      mark[static_cast<std::size_t>(i)] = i;
      accumulator[static_cast<std::size_t>(i)] = 0.0;
      touched.push_back(i);
    }
    accumulator[static_cast<std::size_t>(i)] +=
        delta + (have_shift ? row_shift[static_cast<std::size_t>(i)] : 0.0);
    std::sort(touched.begin(), touched.end());
    for (const Index r : touched) {
      out->add_entry(r, i, accumulator[static_cast<std::size_t>(r)]);
    }
  }
  out->finalize(0.0);
}

}  // namespace sankhya
