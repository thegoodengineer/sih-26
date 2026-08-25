// SPDX-License-Identifier: Apache-2.0
// SANKHYA - random LP generators. TESTS ONLY.

#include "oracles/lp_generator.hpp"

#include <algorithm>

namespace sankhya::oracle {
namespace {

Index pick_dimension(std::mt19937_64& rng, Index low, Index high) {
  std::uniform_int_distribution<int> pick(static_cast<int>(low), static_cast<int>(high));
  return static_cast<Index>(pick(rng));
}

std::int64_t pick_coefficient(std::mt19937_64& rng, std::int64_t magnitude) {
  std::uniform_int_distribution<std::int64_t> pick(-magnitude, magnitude);
  return pick(rng);
}

bool happens(std::mt19937_64& rng, double probability) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  return unit(rng) < probability;
}

/// Allocate the shared skeleton: dimensions, a zeroed matrix, and unbounded columns.
GeneratedLp make_skeleton(std::mt19937_64& rng, const GeneratorConfig& config) {
  GeneratedLp lp;
  lp.num_rows = pick_dimension(rng, config.min_rows, config.max_rows);
  lp.num_cols = pick_dimension(rng, config.min_cols, config.max_cols);
  // Constructed rather than assigned. `assign` on a vector of vectors runs libstdc++'s
  // _M_erase_at_end -> _Destroy path, and GCC 16 cannot prove the (empty) storage pointer is
  // non-null there, so -Wnull-dereference fires as a false positive attributed to this line
  // rather than to the header it actually occurs in. `lp` is freshly default-constructed, so
  // there is nothing to erase and the two forms are equivalent. CI's older GCC does not warn;
  // this keeps the local GCC 16 build clean without weakening the warning set for real code.
  lp.a = std::vector<std::vector<std::int64_t>>(
      static_cast<std::size_t>(lp.num_rows),
      std::vector<std::int64_t>(static_cast<std::size_t>(lp.num_cols), 0));
  lp.b.assign(static_cast<std::size_t>(lp.num_rows), 0);
  lp.c.assign(static_cast<std::size_t>(lp.num_cols), 0);
  lp.upper.assign(static_cast<std::size_t>(lp.num_cols), kNoUpperBound);
  return lp;
}

void fill_matrix(std::mt19937_64& rng, const GeneratorConfig& config, GeneratedLp* lp) {
  for (Index i = 0; i < lp->num_rows; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    bool any = false;
    for (Index j = 0; j < lp->num_cols; ++j) {
      if (!happens(rng, config.density)) continue;
      const std::int64_t value = pick_coefficient(rng, config.magnitude);
      lp->a[ui][static_cast<std::size_t>(j)] = value;
      if (value != 0) any = true;
    }
    // An all-zero row is a degenerate special case that says nothing about the pivot loop,
    // so force at least one entry.
    if (!any) {
      const Index j = pick_dimension(rng, 0, lp->num_cols - 1);
      std::int64_t value = pick_coefficient(rng, config.magnitude);
      if (value == 0) value = 1;
      lp->a[ui][static_cast<std::size_t>(j)] = value;
    }
  }
}

void assign_upper_bounds(std::mt19937_64& rng, const GeneratorConfig& config, GeneratedLp* lp,
                         std::int64_t minimum) {
  std::uniform_int_distribution<std::int64_t> extra(0, config.magnitude);
  for (Index j = 0; j < lp->num_cols; ++j) {
    if (!happens(rng, config.bounded_column_probability)) continue;
    lp->upper[static_cast<std::size_t>(j)] = minimum + extra(rng);
  }
}

}  // namespace

GeneratedLp random_lp(std::mt19937_64& rng, const GeneratorConfig& config) {
  GeneratedLp lp = make_skeleton(rng, config);
  fill_matrix(rng, config, &lp);
  for (Index i = 0; i < lp.num_rows; ++i) {
    lp.b[static_cast<std::size_t>(i)] = pick_coefficient(rng, config.magnitude);
  }
  for (Index j = 0; j < lp.num_cols; ++j) {
    lp.c[static_cast<std::size_t>(j)] = pick_coefficient(rng, config.magnitude);
  }
  assign_upper_bounds(rng, config, &lp, 0);
  return lp;
}

GeneratedLp degenerate_lp(std::mt19937_64& rng, const GeneratorConfig& config) {
  GeneratedLp lp = make_skeleton(rng, config);
  fill_matrix(rng, config, &lp);

  // Choose a point and make MOST rows exactly tight there. Every tight row is an active
  // constraint at the same vertex, which is precisely the configuration that stalls a
  // simplex and, without an anti-cycling rule, can cycle it forever.
  std::vector<std::int64_t> point(static_cast<std::size_t>(lp.num_cols));
  std::uniform_int_distribution<std::int64_t> coordinate(0, 4);
  for (std::int64_t& value : point) value = coordinate(rng);

  for (Index i = 0; i < lp.num_rows; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    std::int64_t activity = 0;
    for (Index j = 0; j < lp.num_cols; ++j) {
      activity += lp.a[ui][static_cast<std::size_t>(j)] * point[static_cast<std::size_t>(j)];
    }
    // 80% of rows tight at the point, the rest strictly slack, so the instance is feasible
    // and heavily degenerate rather than merely tight.
    lp.b[ui] = happens(rng, 0.8) ? activity : activity - 1 - (coordinate(rng));
  }
  for (Index j = 0; j < lp.num_cols; ++j) {
    lp.c[static_cast<std::size_t>(j)] = pick_coefficient(rng, config.magnitude);
  }

  // Upper bounds must not cut the point off, or the instance stops being feasible.
  const std::int64_t largest = *std::max_element(point.begin(), point.end());
  assign_upper_bounds(rng, config, &lp, largest);
  return lp;
}

KktInstance kkt_lp(std::mt19937_64& rng, const GeneratorConfig& config) {
  KktInstance instance;
  GeneratedLp lp = make_skeleton(rng, config);
  fill_matrix(rng, config, &lp);

  const auto cols = static_cast<std::size_t>(lp.num_cols);
  const auto rows = static_cast<std::size_t>(lp.num_rows);

  // x* >= 0, with some coordinates deliberately at zero so that complementary slackness has
  // something to say about their reduced costs.
  std::vector<std::int64_t> x(cols, 0);
  std::uniform_int_distribution<std::int64_t> value(0, 5);
  for (std::size_t j = 0; j < cols; ++j) {
    x[j] = happens(rng, 0.6) ? value(rng) : 0;
  }

  // y* >= 0, nonzero only on the rows we will make active.
  std::vector<std::int64_t> y(rows, 0);
  std::vector<char> active(rows, 0);
  for (std::size_t i = 0; i < rows; ++i) {
    if (happens(rng, 0.5)) {
      active[i] = 1;
      y[i] = value(rng);  // may be zero, which is a legal degenerate multiplier
    }
  }

  // d >= 0 with d_j = 0 wherever x*_j > 0, so that d_j * x*_j = 0 termwise.
  std::vector<std::int64_t> d(cols, 0);
  for (std::size_t j = 0; j < cols; ++j) {
    d[j] = (x[j] > 0) ? 0 : value(rng);
  }

  // c = A^T y* + d makes the dual feasible by construction.
  for (std::size_t j = 0; j < cols; ++j) {
    std::int64_t cost = d[j];
    for (std::size_t i = 0; i < rows; ++i) {
      cost += lp.a[i][j] * y[i];
    }
    lp.c[j] = cost;
  }

  // b_i = A_i x* on active rows; strictly below it elsewhere, so an inactive row has slack
  // and its zero multiplier is consistent.
  for (std::size_t i = 0; i < rows; ++i) {
    std::int64_t activity = 0;
    for (std::size_t j = 0; j < cols; ++j) activity += lp.a[i][j] * x[j];
    lp.b[i] = active[i] != 0 ? activity : activity - 1 - value(rng);
  }

  // Upper bounds must not cut off x*, or x* stops being feasible and the promised optimum
  // becomes a lie.
  std::uniform_int_distribution<std::int64_t> slack(0, 3);
  for (std::size_t j = 0; j < cols; ++j) {
    if (happens(rng, config.bounded_column_probability)) {
      lp.upper[j] = x[j] + slack(rng);
    }
  }

  std::int64_t objective = 0;
  for (std::size_t j = 0; j < cols; ++j) objective += lp.c[j] * x[j];

  instance.lp = std::move(lp);
  instance.optimal_objective = objective;
  instance.optimal_x = std::move(x);
  return instance;
}

}  // namespace sankhya::oracle
