// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover from the interior point to a vertex (#219). See crossover.hpp.

#include "crossover.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "la/lu.hpp"
#include "primal_simplex.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

/// How close to a bound, relative to the variable's own size, counts as "at" it. Loose on
/// purpose: the interior point holds every variable a little off its bound, by an amount
/// that shrinks with the barrier, and calling one basic that the vertex has at a bound only
/// costs a pivot.
constexpr double kAtBoundFraction = 1e-6;

/// One entry's case for being basic: its slack from the nearer bound, relative to its size.
struct Score {
  Index entry;
  double slack;
};

[[nodiscard]] double relative_slack(double value, double lower, double upper) {
  const double scale = std::max(1.0, std::fabs(value));
  double slack = std::numeric_limits<double>::infinity();
  if (is_finite_bound(lower)) slack = std::min(slack, (value - lower) / scale);
  if (is_finite_bound(upper)) slack = std::min(slack, (upper - value) / scale);
  return slack;
}

}  // namespace

CrossoverGuess crossover_guess(const Model& model, const Solution& interior) {
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  CrossoverGuess guess;
  guess.col_status.assign(static_cast<std::size_t>(n), BasisStatus::kUnknown);
  guess.row_status.assign(static_cast<std::size_t>(m), BasisStatus::kUnknown);
  const bool have_point = static_cast<Index>(interior.col_value.size()) == n &&
                          static_cast<Index>(interior.row_activity.size()) == m;
  if (!have_point) return guess;
  const bool have_duals = static_cast<Index>(interior.col_dual.size()) == n;
  const double sense = model.sense_multiplier();

  // Pass 1: entries the point holds at a bound, with the reduced cost agreeing, are
  // nonbasic there; everything else is a candidate for the basis, scored by slack.
  std::vector<Score> candidates;
  candidates.reserve(static_cast<std::size_t>(n + m));
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = model.col_lower[u];
    const double hi = model.col_upper[u];
    const double x = interior.col_value[u];
    if (lo == hi) {
      guess.col_status[u] = BasisStatus::kFixed;
      continue;
    }
    // Reduced cost in MINIMIZATION sense: positive means the lower bound is where the
    // objective wants this column, negative the upper.
    const double d = have_duals ? sense * interior.col_dual[u] : 0.0;
    const double scale = std::max(1.0, std::fabs(x));
    const bool at_lower = is_finite_bound(lo) && x - lo <= kAtBoundFraction * scale;
    const bool at_upper = is_finite_bound(hi) && hi - x <= kAtBoundFraction * scale;
    if (at_lower && d >= -tol::kDualFeasibility) {
      guess.col_status[u] = BasisStatus::kAtLower;
      continue;
    }
    if (at_upper && d <= tol::kDualFeasibility) {
      guess.col_status[u] = BasisStatus::kAtUpper;
      continue;
    }
    ++guess.interior;
    candidates.push_back({j, relative_slack(x, lo, hi)});
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double lo = model.row_lower[u];
    const double hi = model.row_upper[u];
    const double a = interior.row_activity[u];
    if (lo == hi) {
      guess.row_status[u] = BasisStatus::kFixed;
      continue;
    }
    const double scale = std::max(1.0, std::fabs(a));
    const bool at_lower = is_finite_bound(lo) && a - lo <= kAtBoundFraction * scale;
    const bool at_upper = is_finite_bound(hi) && hi - a <= kAtBoundFraction * scale;
    if (at_lower) {
      guess.row_status[u] = BasisStatus::kAtLower;
      continue;
    }
    if (at_upper) {
      guess.row_status[u] = BasisStatus::kAtUpper;
      continue;
    }
    ++guess.interior;
    candidates.push_back({n + i, relative_slack(a, lo, hi)});
  }

  // Pass 2: exactly m basic entries. The interior set is usually larger than m (the point
  // is not a vertex), so the guess has to choose - and it chooses for RANK first. An
  // interior row's logical is a unit column, and any set of them is independent; an
  // interior structural column is not, and a guess made of the m structurals with the most
  // slack came back singular on every model tried (7 of 27 unpivoted on afiro, 318 of 4,559
  // on the 5,000-row staircase), which the simplex answers by throwing the guess away and
  // starting cold. So interior rows go first, then structurals by slack, which is the crash
  // basis a simplex would build from the same information: a slack basis with the
  // structurals the point says are away from their bounds. What is still dependent is the
  // simplex's to repair on its first factorization.
  std::sort(candidates.begin(), candidates.end(), [n](const Score& x, const Score& y) {
    const bool x_row = x.entry >= n;
    const bool y_row = y.entry >= n;
    if (x_row != y_row) return x_row;
    if (x.slack != y.slack) return x.slack > y.slack;
    return x.entry < y.entry;
  });
  const auto status_of = [&](Index entry) -> BasisStatus& {
    return entry < n ? guess.col_status[static_cast<std::size_t>(entry)]
                     : guess.row_status[static_cast<std::size_t>(entry - n)];
  };
  const auto to_nearer_bound = [&](Index entry) {
    const bool column = entry < n;
    const auto u = static_cast<std::size_t>(column ? entry : entry - n);
    const double lo = column ? model.col_lower[u] : model.row_lower[u];
    const double hi = column ? model.col_upper[u] : model.row_upper[u];
    const double v = column ? interior.col_value[u] : interior.row_activity[u];
    BasisStatus status = BasisStatus::kNonbasicFree;
    if (is_finite_bound(lo) && is_finite_bound(hi)) {
      status = v - lo <= hi - v ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
    } else if (is_finite_bound(lo)) {
      status = BasisStatus::kAtLower;
    } else if (is_finite_bound(hi)) {
      status = BasisStatus::kAtUpper;
    }
    status_of(entry) = status;
  };
  for (const Score& candidate : candidates) {
    if (guess.basic < m) {
      status_of(candidate.entry) = BasisStatus::kBasic;
      ++guess.basic;
    } else {
      to_nearer_bound(candidate.entry);
    }
  }
  for (Index i = 0; i < m && guess.basic < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (guess.row_status[u] == BasisStatus::kBasic) continue;
    guess.row_status[u] = BasisStatus::kBasic;
    ++guess.basic;
  }

  // Pass 3: RANK. The m entries chosen above are chosen for slack, not independence, and
  // the interior structurals of a real model are dependent often enough that the simplex
  // refused the guess on every model tried and started cold (its narrow repair trusts a
  // defect of a few columns, not the hundreds an interior set carries). So the guess is
  // factorized here, with partial pivoting so that only a genuine defect counts, and each
  // column no pivot reached is evicted for the logical of a row no pivot covered - the
  // repair of Maros sec. 9.4 and Suhl & Suhl 1990, applied until the basis factorizes. A
  // logical is a unit column, so every round strictly raises the rank and the loop ends.
  std::vector<Index> basic_entries;
  basic_entries.reserve(static_cast<std::size_t>(m));
  for (Index j = 0; j < n; ++j) {
    if (guess.col_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic)
      basic_entries.push_back(j);
  }
  for (Index i = 0; i < m; ++i) {
    if (guess.row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic)
      basic_entries.push_back(n + i);
  }
  if (static_cast<Index>(basic_entries.size()) != m) return guess;
  std::vector<Index> logical_rows(static_cast<std::size_t>(m));
  std::vector<double> logical_values(static_cast<std::size_t>(m), -1.0);
  for (Index i = 0; i < m; ++i) logical_rows[static_cast<std::size_t>(i)] = i;
  std::vector<LuColumn> columns(static_cast<std::size_t>(m));
  const auto fill_columns = [&]() {
    for (Index slot = 0; slot < m; ++slot) {
      const Index k = basic_entries[static_cast<std::size_t>(slot)];
      LuColumn& target = columns[static_cast<std::size_t>(slot)];
      if (k < n) {
        const ColumnView column = model.matrix.column(k);
        target.rows = column.rows;
        target.values = column.values;
        target.size = column.size;
      } else {
        const auto row = static_cast<std::size_t>(k - n);
        target.rows = logical_rows.data() + row;
        target.values = logical_values.data() + row;
        target.size = 1;
      }
    }
  };
  SparseLu lu;
  for (int round = 0; round < 64; ++round) {
    fill_columns();
    if (lu.factorize(columns, m, tol::kPivotTolerance, 1.0)) break;
    const std::vector<Index> dependent = lu.dependent_positions();
    const std::vector<Index> uncovered = lu.uncovered_rows();
    if (dependent.empty() || dependent.size() != uncovered.size()) break;
    for (std::size_t t = 0; t < dependent.size(); ++t) {
      const Index slot = dependent[t];
      const Index logical = n + uncovered[t];
      if (slot < 0 || slot >= m) continue;
      if (guess.row_status[static_cast<std::size_t>(uncovered[t])] == BasisStatus::kBasic)
        continue;
      const Index evicted = basic_entries[static_cast<std::size_t>(slot)];
      to_nearer_bound(evicted);
      basic_entries[static_cast<std::size_t>(slot)] = logical;
      guess.row_status[static_cast<std::size_t>(uncovered[t])] = BasisStatus::kBasic;
      ++guess.repaired;
    }
  }
  return guess;
}

Solution crossover_to_vertex(const Model& model, Solution interior, const Options& options,
                             Logger& logger, SolveControl* control, const Timer& timer) {
  if (interior.status != SolveStatus::kOptimal) return interior;
  const CrossoverGuess guess = crossover_guess(model, interior);
  if (guess.basic != model.num_rows()) {
    interior.message += "; crossover skipped: no basis guess could be built";
    return interior;
  }

  Options pivots = options;
  pivots.set_string("algorithm", "dual-simplex");
  const double time_limit = options.get_double("time_limit");
  if (time_limit > 0.0 && std::isfinite(time_limit) && time_limit < 1e300) {
    const double remaining = time_limit - timer.elapsed_seconds();
    if (remaining <= 0.0) {
      interior.message += "; no time left for crossover, the interior point's answer stands";
      return interior;
    }
    pivots.set_double("time_limit", remaining);
  }
  WarmStart warm;
  warm.col_status = guess.col_status;
  warm.row_status = guess.row_status;
  logger.info(
      "Crossover: {} of {} entries interior, basis guess of {} from the interior point, {} "
      "evicted for rank",
      guess.interior, model.num_cols() + model.num_rows(), guess.basic, guess.repaired);
  Timer pivot_clock;
  Solution vertex = solve_dual_simplex(model, pivots, logger, control, &warm);
  if (vertex.status != SolveStatus::kOptimal) {
    interior.message += fmt::format(
        "; crossover did not reach a vertex ({} after {} pivots, {:.2f}s), the interior "
        "point's answer stands",
        to_string(vertex.status), vertex.iterations, pivot_clock.elapsed_seconds());
    logger.warning("Crossover: {} after {} pivots; keeping the interior point's answer",
                   to_string(vertex.status), vertex.iterations);
    return interior;
  }
  const Count pivot_count = vertex.iterations;
  vertex.iterations += interior.iterations;
  vertex.algorithm = interior.algorithm + "+crossover";
  const std::string note = fmt::format(
      "crossover: {} pivots from the interior point's basis guess in {:.2f}s, after {} "
      "interior point iterations",
      pivot_count, pivot_clock.elapsed_seconds(), interior.iterations);
  vertex.message = vertex.message.empty() ? note : vertex.message + "; " + note;
  logger.info("Crossover: optimal vertex after {} pivots in {:.2f}s", pivot_count,
              pivot_clock.elapsed_seconds());
  return vertex;
}

}  // namespace sankhya
