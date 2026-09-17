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
// Ordering: approximate minimum degree on the quotient graph (Amestoy, Davis & Duff 1996)
// -----------------------------------------------------------------------------------------
//
// The first version of this was exact minimum degree on an explicit elimination graph: at
// each step eliminate the vertex of smallest degree and merge its neighbours into a clique
// by set union. That is Tinney & Walker, and it is correct, and it is what #193 measured:
// on a 7,338-row normal-equations matrix from the random scale family the ordering took
// 66 s and was 99.7% of analyze(), growing as n^2.35, because every elimination step pays
// for the clique it creates, and the clique is the fill. The staircase family, with small
// separators, ordered the same size in 2.3 s. The graph representation was the cost, not
// the rule.
//
// AMD never forms a clique. An eliminated vertex becomes an ELEMENT, and the clique its
// neighbours would form is represented by the element itself: a variable's neighbourhood
// is its list of variables (A_i) plus its list of elements (E_i), and two variables that
// share an element are adjacent without either storing the other. Eliminating p forms
// L_p = A_p union (the members of every element in E_p) - those elements are absorbed into
// p and disappear - and the only per-step work is over L_p and the lists of its members.
// The degree is then APPROXIMATE: for i in L_p it is the smallest of three upper bounds,
//     n - k - 1,   d_i + |L_p \ i|,   |A_i \ L_p| + |L_p \ i| + sum over e in E_i of |L_e
//     \ L_p|,
// where |L_e \ L_p| is computed for every element touching L_p in one pass (w[e] starts at
// |L_e| and is decremented once for each member of L_p that lists e), which is the trick
// that makes the whole step linear in the lists it reads. An element with |L_e \ L_p| = 0
// is a subset of the new one and is absorbed at once (aggressive absorption). Variables
// wait in degree buckets, so choosing the next pivot is a scan from the current minimum
// upward rather than over every vertex.
//
// Not implemented, deliberately: supervariable detection (indistinguishable nodes merged
// and eliminated together) and dense-row postponement. Both matter on graphs with many
// identical rows and neither is what the two scale families exercise; they are the next
// step if a structured industrial model (#211) shows the need.
//
// Ties go to the most recently inserted vertex of the minimum degree, and every list is
// built in index order, so the ordering is deterministic: the same matrix gives the same
// permutation on every machine.
bool SparseLdl::minimum_degree(const SparseMatrix& lower, const ShouldStop& should_stop) {
  const Index n = n_;
  const auto un = static_cast<std::size_t>(n);
  const auto at = [](Index i) { return static_cast<std::size_t>(i); };

  // The quotient graph. adjacent[i] is A_i, elements[i] is E_i, members[e] is L_e; an
  // element reuses the index of the pivot that created it.
  std::vector<std::vector<Index>> adjacent(un);
  std::vector<std::vector<Index>> elements(un);
  std::vector<std::vector<Index>> members(un);
  for (Index c = 0; c < n; ++c) {
    const ColumnView column = lower.column(c);
    for (Index p = 0; p < column.size; ++p) {
      const Index r = column.rows[p];
      if (r <= c) continue;  // the strict lower triangle defines the graph
      adjacent[at(r)].push_back(c);
      adjacent[at(c)].push_back(r);
    }
  }
  // The storage the ordering holds live, in list entries, checked against the budget
  // (#246). Counted rather than measured because the allocator does not say, and counted
  // approximately: the three list families are what grows, and an element list that shrinks
  // by resize() gives its capacity back only when it is swapped away, so this undercounts
  // capacity and overcounts nothing.
  std::size_t live = 0;
  for (auto& list : adjacent) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
    live += list.size();
  }

  enum class Status : std::uint8_t { kVariable, kElement, kAbsorbed };
  std::vector<Status> status(un, Status::kVariable);
  std::vector<Index> degree(un, 0);
  for (Index i = 0; i < n; ++i) degree[at(i)] = static_cast<Index>(adjacent[at(i)].size());

  // Degree buckets: doubly linked lists, one per degree, head insertion.
  std::vector<Index> head(un + 1, -1);
  std::vector<Index> next(un, -1);
  std::vector<Index> prev(un, -1);
  const auto insert = [&](Index i) {
    const auto d = at(degree[at(i)]);
    next[at(i)] = head[d];
    prev[at(i)] = -1;
    if (head[d] >= 0) prev[at(head[d])] = i;
    head[d] = i;
  };
  const auto remove = [&](Index i) {
    const auto d = at(degree[at(i)]);
    if (prev[at(i)] >= 0) {
      next[at(prev[at(i)])] = next[at(i)];
    } else {
      head[d] = next[at(i)];
    }
    if (next[at(i)] >= 0) prev[at(next[at(i)])] = prev[at(i)];
  };
  for (Index i = 0; i < n; ++i) insert(i);
  Index min_degree = 0;

  std::vector<Index> in_lp(un, -1);  // stamp: in_lp[i] == k means i is in L_p at step k
  std::vector<Index> w(un, 0);       // |L_e \ L_p| for elements touching L_p
  std::vector<Index> w_stamp(un, -1);
  std::vector<Index> lp;
  perm_.clear();
  perm_.reserve(un);

  for (Index k = 0; k < n; ++k) {
    // ASKED EVERY STEP, as before: one call through a std::function is nothing beside the
    // list work, and the deadline is what makes a time limit mean anything here (#197).
    if (should_stop && should_stop()) return false;
    if (live > ordering_budget_) {
      ordering_too_large_ = true;
      return false;
    }

    while (head[at(min_degree)] < 0) ++min_degree;
    const Index p = head[at(min_degree)];
    remove(p);
    perm_.push_back(p);
    status[at(p)] = Status::kElement;

    // L_p: the live variables of A_p and of every element p touches; those elements are
    // absorbed into p.
    lp.clear();
    in_lp[at(p)] = k;
    for (const Index i : adjacent[at(p)]) {
      if (status[at(i)] != Status::kVariable || in_lp[at(i)] == k) continue;
      in_lp[at(i)] = k;
      lp.push_back(i);
    }
    for (const Index e : elements[at(p)]) {
      if (status[at(e)] != Status::kElement) continue;
      for (const Index i : members[at(e)]) {
        if (status[at(i)] != Status::kVariable || in_lp[at(i)] == k) continue;
        in_lp[at(i)] = k;
        lp.push_back(i);
      }
      status[at(e)] = Status::kAbsorbed;
      live -= members[at(e)].size();
      std::vector<Index>().swap(members[at(e)]);
    }
    live -= adjacent[at(p)].size() + elements[at(p)].size();
    std::vector<Index>().swap(adjacent[at(p)]);
    std::vector<Index>().swap(elements[at(p)]);
    members[at(p)] = lp;
    live += lp.size();

    // w[e] = |L_e \ L_p| for every live element adjacent to a member of L_p. A member i of
    // L_p lists e exactly when i is in L_e, so each such listing subtracts one.
    for (const Index i : lp) {
      for (const Index e : elements[at(i)]) {
        if (status[at(e)] != Status::kElement) continue;
        if (w_stamp[at(e)] != k) {
          w_stamp[at(e)] = k;
          w[at(e)] = static_cast<Index>(members[at(e)].size());
        }
        --w[at(e)];
      }
    }

    const auto lp_size = static_cast<Index>(lp.size());
    std::size_t visited = 0;
    for (const Index i : lp) {
      // A step under memory pressure is not short: on the model that motivated #246 the
      // intervals between the per-step checks above reached minutes while the machine
      // swapped. So the deadline is also asked inside the step, every few thousand members.
      if (should_stop && (++visited & 4095) == 0 && should_stop()) return false;
      remove(i);
      // A_i loses the members of L_p (they are reachable through p now) and anything
      // eliminated.
      std::size_t keep = 0;
      for (const Index v : adjacent[at(i)]) {
        if (status[at(v)] != Status::kVariable || in_lp[at(v)] == k) continue;
        adjacent[at(i)][keep++] = v;
      }
      live -= adjacent[at(i)].size() - keep;
      adjacent[at(i)].resize(keep);
      // E_i loses absorbed elements and gains p; an element wholly inside L_p is absorbed
      // here, before it can be counted.
      Index external = 0;
      keep = 0;
      for (const Index e : elements[at(i)]) {
        if (status[at(e)] != Status::kElement) continue;
        if (w[at(e)] <= 0) {
          status[at(e)] = Status::kAbsorbed;
          live -= members[at(e)].size();
          std::vector<Index>().swap(members[at(e)]);
          continue;
        }
        elements[at(i)][keep++] = e;
        external += w[at(e)];
      }
      live -= elements[at(i)].size() - keep;
      elements[at(i)].resize(keep);
      elements[at(i)].push_back(p);
      ++live;

      const Index remaining = n - k - 1;
      const Index by_old_degree = degree[at(i)] + (lp_size - 1);
      const Index by_lists =
          static_cast<Index>(adjacent[at(i)].size()) + (lp_size - 1) + external;
      degree[at(i)] = std::max<Index>(0, std::min({remaining, by_old_degree, by_lists}));
      insert(i);
      if (degree[at(i)] < min_degree) min_degree = degree[at(i)];
    }
  }
  inverse_.assign(un, -1);
  for (Index k = 0; k < n; ++k) inverse_[at(perm_[at(k)])] = k;
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
bool SparseLdl::symbolic_pattern(const ShouldStop& should_stop) {
  const Index n = n_;
  const auto un = static_cast<std::size_t>(n);
  std::vector<Index> mark(un, -1);
  // The walk that defines L's pattern: for row k, every column reached from an entry of row k
  // by climbing the elimination tree gets L(k, j) != 0. Run TWICE, first to count and then to
  // fill, rather than once into a vector of vectors: on a matrix whose fill is catastrophic
  // the counting pass is what lets the deadline and the factor budget stop it (#246) - the
  // 100,000-row random model built its pattern for 190 s past a 120 s limit and then ran the
  // machine out of memory in exactly this loop - and the filling pass then writes into one
  // exactly-sized array instead of n growing ones.
  const auto walk = [&](Index k, auto&& visit) {
    mark[static_cast<std::size_t>(k)] = k;
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      Index i = a_rows_[static_cast<std::size_t>(p)];
      while (i < k && mark[static_cast<std::size_t>(i)] != k) {
        mark[static_cast<std::size_t>(i)] = k;
        visit(i);
        i = parent_[static_cast<std::size_t>(i)];
        if (i == -1) break;
      }
    }
  };

  std::vector<std::size_t> count(un, 0);
  std::size_t total = 0;
  for (Index k = 0; k < n; ++k) {
    if (should_stop && (k & 255) == 0 && should_stop()) {
      stopped_early_ = true;
      return false;
    }
    walk(k, [&](Index i) {
      ++count[static_cast<std::size_t>(i)];
      ++total;
    });
    // The caller's budget for the factor, consulted as the count grows so that a factor ten
    // times too large is refused after a tenth of the work, not after all of it.
    if (total > factor_budget_) {
      factor_too_large_ = true;
      l_starts_.clear();
      l_rows_.clear();
      l_values_.clear();
      return false;
    }
  }
  // The factor can be far denser than the matrix - that is what fill-in means - so its size
  // is checked here rather than inherited from the input's (#305). l_starts_ holds Index
  // offsets, and a factor past kMaxNonzeros would wrap them; the ordering has already told us
  // the pattern, so refusing costs nothing and reaches the caller as a clean false.
  if (!nonzero_count_fits(total)) {
    pattern_too_large_ = true;
    l_starts_.clear();
    l_rows_.clear();
    l_values_.clear();
    return false;
  }

  l_starts_.assign(un + 1, 0);
  for (Index j = 0; j < n; ++j) {
    l_starts_[static_cast<std::size_t>(j) + 1] =
        l_starts_[static_cast<std::size_t>(j)] +
        static_cast<Index>(count[static_cast<std::size_t>(j)]);
  }
  l_rows_.assign(total, 0);
  std::vector<Index> next(l_starts_.begin(), l_starts_.end() - 1);
  std::fill(mark.begin(), mark.end(), -1);
  for (Index k = 0; k < n; ++k) {
    // Rows are visited in increasing k, so every column's row list comes out sorted.
    walk(k, [&](Index i) {
      l_rows_[static_cast<std::size_t>(next[static_cast<std::size_t>(i)]++)] = k;
    });
  }
  l_values_.assign(total, 0.0);
  d_.assign(un, 0.0);
  return true;
}

bool SparseLdl::analyze(const SparseMatrix& lower, const ShouldStop& should_stop) {
  analyzed_ = false;
  stopped_early_ = false;
  pattern_too_large_ = false;
  ordering_too_large_ = false;
  factor_too_large_ = false;
  if (lower.num_rows() != lower.num_cols() || lower.num_rows() <= 0) return false;
  n_ = lower.num_rows();
  // The ordering is where the time goes: measured on generated instances, analyze() costs
  // three to four times a numeric factorization, and at 20,000 rows it is most of an
  // 813-second first iteration (#193). It is therefore the one phase that has to be
  // interruptible for a time limit to mean anything.
  if (!minimum_degree(lower, should_stop)) {
    // Two reasons to give up, told apart for the caller: a deadline is a time limit, a
    // budget is a refusal with the number in it (#246).
    stopped_early_ = !ordering_too_large_;
    return false;
  }
  build_permuted_pattern(lower);
  elimination_tree();
  if (!symbolic_pattern(should_stop)) return false;
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

// -----------------------------------------------------------------------------------------
// Semidefiniteness, as a decision procedure (#303)
// -----------------------------------------------------------------------------------------
//
// Same up-looking pass as factorize(), three changes: no regularization, a zero pivot is kept
// at zero instead of lifted, and the substitution that would divide by a zero pivot first
// checks that its numerator is negligible. That last check is the whole difference between a
// test that decides semidefiniteness and one that merely completes: for a positive
// semidefinite matrix, d_j = 0 forces the whole of column j to be zero (Higham 1990,
// "Analysis of the Cholesky decomposition of a semi-definite matrix"), so a nonzero numerator
// over a zero pivot exhibits the indefinite direction rather than a rounding artefact.
SemidefiniteReport SparseLdl::check_semidefinite(const SparseMatrix& lower, double slack_factor,
                                                 const ShouldStop& should_stop) {
  SemidefiniteReport report;
  if (lower.num_rows() != lower.num_cols()) return report;
  if (lower.num_rows() == 0 || lower.num_nonzeros() == 0) {
    report.verdict = SemidefiniteReport::Verdict::kPositiveSemidefinite;
    return report;
  }
  if (!analyze(lower, should_stop)) return report;  // kUndecided: deadline, or too large

  const Index n = n_;
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
      if (slot == end || *slot != lo) return report;  // pattern mismatch: undecided
      a_values_[static_cast<std::size_t>(slot - a_rows_.begin())] += column.values[p];
    }
  }

  // The scale the slack is measured against is the largest diagonal magnitude, not an
  // absolute number: an indefinite direction in a badly scaled matrix would hide under a
  // fixed tolerance, and a well scaled one would see rounding reported as curvature.
  double largest_diagonal = 0.0;
  for (Index k = 0; k < n; ++k) {
    for (Index p = a_starts_[static_cast<std::size_t>(k)];
         p < a_starts_[static_cast<std::size_t>(k) + 1]; ++p) {
      if (a_rows_[static_cast<std::size_t>(p)] == k) {
        largest_diagonal =
            std::max(largest_diagonal, std::fabs(a_values_[static_cast<std::size_t>(p)]));
      }
    }
  }
  const double slack = slack_factor * std::max(1.0, largest_diagonal);

  std::vector<double> x(static_cast<std::size_t>(n), 0.0);
  std::vector<Index> mark(static_cast<std::size_t>(n), -1);
  std::vector<Index> reach;
  std::vector<Index> fill(static_cast<std::size_t>(n), 0);
  const auto original = [&](Index permuted) {
    return perm_[static_cast<std::size_t>(permuted)];
  };

  for (Index k = 0; k < n; ++k) {
    if (should_stop && should_stop()) {
      stopped_early_ = true;
      return report;  // kUndecided
    }
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

    for (const Index j : reach) {
      const auto uj = static_cast<std::size_t>(j);
      const double y = x[uj];
      const Index begin = l_starts_[uj];
      const Index stored = fill[uj];
      for (Index p = begin; p < begin + stored; ++p) {
        const Index i = l_rows_[static_cast<std::size_t>(p)];
        x[static_cast<std::size_t>(i)] -= l_values_[static_cast<std::size_t>(p)] * y;
      }

      double l_kj = 0.0;
      if (d_[uj] == 0.0) {
        // A zero pivot earlier in the factorization. For a semidefinite matrix everything
        // this column would have divided is zero as well; a numerator that is not is the
        // certificate, and dividing by zero here would have produced an infinity that the
        // pivot test below reads as a perfectly good positive number.
        if (std::fabs(y) > slack) {
          report.verdict = SemidefiniteReport::Verdict::kIndefinite;
          report.column = original(j);
          report.pivot = y;
          return report;
        }
      } else {
        l_kj = y / d_[uj];
      }
      const auto slot = static_cast<std::size_t>(begin + stored);
      l_values_[slot] = l_kj;
      ++fill[uj];
      diagonal -= l_kj * y;
      x[uj] = 0.0;
    }

    if (diagonal < -slack) {
      report.verdict = SemidefiniteReport::Verdict::kIndefinite;
      report.column = original(k);
      report.pivot = diagonal;
      return report;
    }
    d_[static_cast<std::size_t>(k)] = diagonal <= slack ? 0.0 : diagonal;
  }

  // The factors describe a matrix with zeros on D; nothing may solve with them afterwards.
  analyzed_ = false;
  report.verdict = SemidefiniteReport::Verdict::kPositiveSemidefinite;
  return report;
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
bool normal_equations_lower(const SparseMatrix& a, const std::vector<double>& theta,
                            const std::vector<double>& row_shift, double delta,
                            SparseMatrix* out, const SparseLdl::ShouldStop& should_stop) {
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
    // Every 256 rows: a row of the product costs the sum of its columns' lengths, so the
    // check is amortised over thousands of multiply-adds and never decides arithmetic.
    if (should_stop && (i & 255) == 0 && should_stop()) return false;
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
  return true;
}

}  // namespace sankhya
