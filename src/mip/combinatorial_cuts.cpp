// SPDX-License-Identifier: Apache-2.0
// SANKHYA - clique cuts and {0,1/2}-Chvatal-Gomory cuts (#358). References and the validity
// argument for each family are on the declarations.

#include "combinatorial_cuts.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <unordered_set>
#include <utility>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

/// Past this many conflict pairs the graph stops growing. Fewer conflicts only means fewer
/// cliques - a missing edge can never make a cut invalid - so the cap trades strength for
/// memory and never correctness.
constexpr int kMaxConflictEdges = 200000;
/// Clique separation starts from at most this many of the largest LP values.
constexpr int kCliqueStarts = 50;
/// {0,1/2}: partners tried per row, and row pairs tried in total.
constexpr int kPartnersPerRow = 10;
constexpr int kMaxRowPairs = 20000;
/// A cut must be violated at the LP point by more than this, relative to max(1, |rhs|).
constexpr double kMinViolation = 1e-6;

bool is_binary(const Model& model, const std::vector<double>& lower,
               const std::vector<double>& upper, std::size_t j) {
  return model.col_type[j] == VarType::kInteger && lower[j] == 0.0 && upper[j] == 1.0;
}

}  // namespace

// ---- Clique cuts
// -----------------------------------------------------------------------------

std::vector<Cut> generate_clique_cuts(const Model& model, const Solution& solution,
                                      const std::vector<double>& col_lower,
                                      const std::vector<double>& col_upper,
                                      CombinatorialCutStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  const auto un = static_cast<std::size_t>(n);
  if (solution.col_value.size() != un) return cuts;
  const std::vector<double>& x = solution.col_value;

  // ---- The conflict graph, from every row in <= orientation.
  std::vector<std::vector<Index>> adjacent(un);
  std::unordered_set<std::uint64_t> edge;
  const auto key = [n](Index p, Index q) {
    const Index a = std::min(p, q);
    const Index b = std::max(p, q);
    return static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(n) +
           static_cast<std::uint64_t>(b);
  };
  bool capped = false;
  const CsrView by_row(model.matrix);
  for (Index i = 0; i < model.num_rows() && !capped; ++i) {
    const ColumnView row = by_row.row(i);
    const auto ui = static_cast<std::size_t>(i);
    for (const double sign : {1.0, -1.0}) {
      // sign +1: a x <= u.  sign -1: -a x <= -l.
      const double bound = sign > 0.0 ? model.row_upper[ui] : -model.row_lower[ui];
      if (!std::isfinite(bound)) continue;
      // The least every column outside the positive binaries can contribute.
      std::vector<std::pair<double, Index>> positive;
      double least_others = 0.0;
      bool finite = true;
      for (Index k = 0; k < row.size && finite; ++k) {
        const Index j = row.rows[k];
        const auto u = static_cast<std::size_t>(j);
        const double a = sign * row.values[k];
        if (a > 0.0 && is_binary(model, col_lower, col_upper, u)) {
          positive.emplace_back(a, j);
        } else if (a != 0.0) {
          const double low = a > 0.0 ? col_lower[u] : col_upper[u];
          if (!std::isfinite(low)) finite = false;
          least_others += a * low;
        }
      }
      if (!finite || positive.size() < 2) continue;
      // Both at 1 is impossible when a_p + a_q + least_others > bound. The margin is a
      // feasibility tolerance, scaled, so a pair the solver would accept within tolerance is
      // never declared in conflict.
      const double room = bound - least_others + 1e-6 * std::max(1.0, std::fabs(bound));
      std::sort(positive.begin(), positive.end(), [](const auto& l, const auto& r) {
        return l.first > r.first || (l.first == r.first && l.second < r.second);
      });
      for (std::size_t p = 0; p < positive.size() && !capped; ++p) {
        for (std::size_t q = p + 1; q < positive.size(); ++q) {
          if (positive[p].first + positive[q].first <= room) break;  // sorted: none further
          const Index a = positive[p].second;
          const Index b = positive[q].second;
          if (edge.insert(key(a, b)).second) {
            adjacent[static_cast<std::size_t>(a)].push_back(b);
            adjacent[static_cast<std::size_t>(b)].push_back(a);
            if (static_cast<int>(edge.size()) >= kMaxConflictEdges) {
              capped = true;
              break;
            }
          }
        }
      }
    }
  }
  if (stats != nullptr) {
    stats->conflict_edges = static_cast<int>(edge.size());
    stats->conflict_graph_capped = capped;
  }
  if (edge.empty()) return cuts;
  const auto conflict = [&](Index p, Index q) { return edge.count(key(p, q)) > 0; };

  // ---- Greedy separation from the largest LP values.
  std::vector<Index> order;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (is_binary(model, col_lower, col_upper, u) && x[u] > 1e-6 && !adjacent[u].empty()) {
      order.push_back(j);
    }
  }
  std::sort(order.begin(), order.end(), [&](Index l, Index r) {
    const double xl = x[static_cast<std::size_t>(l)];
    const double xr = x[static_cast<std::size_t>(r)];
    return xl > xr || (xl == xr && l < r);
  });
  std::set<std::vector<Index>> seen;
  const std::size_t starts = std::min<std::size_t>(order.size(), kCliqueStarts);
  for (std::size_t s = 0; s < starts; ++s) {
    std::vector<Index> clique{order[s]};
    double weight = x[static_cast<std::size_t>(order[s])];
    for (const Index v : order) {
      if (v == order[s]) continue;
      if (std::all_of(clique.begin(), clique.end(), [&](Index c) { return conflict(v, c); })) {
        clique.push_back(v);
        weight += x[static_cast<std::size_t>(v)];
      }
    }
    if (weight <= 1.0 + kMinViolation) continue;
    // Extend to a maximal clique with the start's other neighbours: a larger clique is a
    // stronger cut, and every member still conflicts with every other.
    std::vector<Index> neighbours = adjacent[static_cast<std::size_t>(order[s])];
    std::sort(neighbours.begin(), neighbours.end());
    for (const Index v : neighbours) {
      if (std::find(clique.begin(), clique.end(), v) != clique.end()) continue;
      if (std::all_of(clique.begin(), clique.end(), [&](Index c) { return conflict(v, c); })) {
        clique.push_back(v);
      }
    }
    std::sort(clique.begin(), clique.end());
    if (!seen.insert(clique).second) continue;
    Cut cut;
    cut.coeff.assign(un, 0.0);
    for (const Index c : clique) cut.coeff[static_cast<std::size_t>(c)] = 1.0;
    cut.rhs = 1.0;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

// ---- {0,1/2}-Chvatal-Gomory cuts
// -------------------------------------------------------------

namespace {

/// A row in <= form over the substituted non-negative columns x' (x = lower + x', or
/// x = upper - x' for a column bounded above only).
struct IntegerRow {
  std::map<Index, double> coef;  // integer-valued, in x'
  double rhs = 0.0;
};

}  // namespace

std::vector<Cut> generate_zero_half_cuts(const Model& model, const Solution& solution,
                                         const std::vector<double>& col_lower,
                                         const std::vector<double>& col_upper,
                                         CombinatorialCutStats* stats) {
  std::vector<Cut> cuts;
  const Index n = model.num_cols();
  const auto un = static_cast<std::size_t>(n);
  if (solution.col_value.size() != un) return cuts;
  const std::vector<double>& x = solution.col_value;

  // The substitution that makes every column non-negative, chosen once per column so every
  // row is expressed in the same variables. 0: x = l + x'. 1: x = u - x'. 2: unusable.
  std::vector<int> shift(un, 2);
  std::vector<double> anchor(un, 0.0);
  for (std::size_t j = 0; j < un; ++j) {
    if (model.col_type[j] != VarType::kInteger) continue;
    if (std::isfinite(col_lower[j])) {
      shift[j] = 0;
      anchor[j] = std::ceil(col_lower[j] - 1e-9);
    } else if (std::isfinite(col_upper[j])) {
      shift[j] = 1;
      anchor[j] = std::floor(col_upper[j] + 1e-9);
    }
  }

  // Every eligible row side: pure-integer, integer coefficients, every column substitutable.
  std::vector<IntegerRow> rows;
  const CsrView by_row(model.matrix);
  for (Index i = 0; i < model.num_rows(); ++i) {
    const ColumnView row = by_row.row(i);
    const auto ui = static_cast<std::size_t>(i);
    bool eligible = row.size > 0;
    for (Index k = 0; k < row.size && eligible; ++k) {
      const auto u = static_cast<std::size_t>(row.rows[k]);
      eligible = shift[u] != 2 && row.values[k] == std::round(row.values[k]);
    }
    if (!eligible) continue;
    for (const double sign : {1.0, -1.0}) {
      const double bound = sign > 0.0 ? model.row_upper[ui] : -model.row_lower[ui];
      if (!std::isfinite(bound)) continue;
      IntegerRow r;
      r.rhs = bound;
      for (Index k = 0; k < row.size; ++k) {
        const Index j = row.rows[k];
        const auto u = static_cast<std::size_t>(j);
        const double a = sign * row.values[k];
        // a x = a (l + x') = a l + a x'   or   a (u - x') = a u - a x'
        r.rhs -= a * anchor[u];
        r.coef[j] = shift[u] == 0 ? a : -a;
      }
      rows.push_back(std::move(r));
    }
  }

  // Which rows touch each column, to pair rows that share one.
  std::vector<std::vector<int>> rows_of(un);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    for (const auto& [j, a] : rows[r].coef)
      rows_of[static_cast<std::size_t>(j)].push_back(static_cast<int>(r));
  }
  std::vector<std::pair<int, int>> sets;
  for (std::size_t r = 0; r < rows.size(); ++r) sets.emplace_back(static_cast<int>(r), -1);
  std::set<std::pair<int, int>> paired;
  for (std::size_t r = 0; r < rows.size() && static_cast<int>(paired.size()) < kMaxRowPairs;
       ++r) {
    int partners = 0;
    for (const auto& [j, a] : rows[r].coef) {
      for (const int s : rows_of[static_cast<std::size_t>(j)]) {
        if (s <= static_cast<int>(r) || partners >= kPartnersPerRow) continue;
        if (paired.insert({static_cast<int>(r), s}).second) {
          sets.emplace_back(static_cast<int>(r), s);
          ++partners;
        }
      }
      if (partners >= kPartnersPerRow) break;
    }
  }
  if (stats != nullptr) stats->candidate_row_sets = static_cast<int>(sets.size());

  std::set<std::pair<std::vector<std::pair<Index, double>>, double>> seen;
  for (const auto& [first, second] : sets) {
    std::map<Index, double> sum = rows[static_cast<std::size_t>(first)].coef;
    double rhs_sum = rows[static_cast<std::size_t>(first)].rhs;
    if (second >= 0) {
      for (const auto& [j, a] : rows[static_cast<std::size_t>(second)].coef) sum[j] += a;
      rhs_sum += rows[static_cast<std::size_t>(second)].rhs;
    }
    // floor(u'A) x' <= floor(u'b), u = 1/2 on each row of the set.
    std::vector<std::pair<Index, double>> in_x;  // back in the model's own columns
    double rhs = std::floor(rhs_sum / 2.0 + 1e-12);
    for (const auto& [j, a] : sum) {
      const double c = std::floor(a / 2.0);
      if (c == 0.0) continue;
      const auto u = static_cast<std::size_t>(j);
      // c x' with x' = x - l  ->  c x - c l ;   x' = u - x  ->  -c x + c u
      if (shift[u] == 0) {
        in_x.emplace_back(j, c);
        rhs += c * anchor[u];
      } else {
        in_x.emplace_back(j, -c);
        rhs -= c * anchor[u];
      }
    }
    if (in_x.empty()) continue;
    double activity = 0.0;
    for (const auto& [j, c] : in_x) activity += c * x[static_cast<std::size_t>(j)];
    if (activity <= rhs + kMinViolation * std::max(1.0, std::fabs(rhs))) continue;
    if (!seen.insert({in_x, rhs}).second) continue;
    Cut cut;
    cut.coeff.assign(un, 0.0);
    for (const auto& [j, c] : in_x) cut.coeff[static_cast<std::size_t>(j)] = c;
    cut.rhs = rhs;
    cuts.push_back(std::move(cut));
  }
  return cuts;
}

}  // namespace sankhya::mip
