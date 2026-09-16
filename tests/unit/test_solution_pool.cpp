// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solution pool (#225).
//
// The pool's promise under pool_complete is exact: the k best integer assignments, best
// first, no two alike. That is checkable by brute force on models small enough to enumerate,
// so every expectation below is computed by enumerating every assignment in the test, not
// read off a previous run. The fuzz test is the gate; the named tests pin the cases a reader
// would ask about.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

#include "sankhya/io.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

/// A pure-binary model: `rows` dense, `lower <= A x <= upper`, costs `cost`.
Model binary_model(const std::vector<std::vector<double>>& rows,
                   const std::vector<double>& lower, const std::vector<double>& upper,
                   const std::vector<double>& cost, ObjSense sense) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.sense = sense;
  model.col_cost = cost;
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.row_lower = lower;
  model.row_upper = upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double a = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (a != 0.0) model.matrix.add_entry(i, j, a);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

struct Enumerated {
  double objective;
  std::vector<int> x;
};

/// Every feasible 0/1 assignment, best first in the model's sense.
std::vector<Enumerated> enumerate(const std::vector<std::vector<double>>& rows,
                                  const std::vector<double>& lower,
                                  const std::vector<double>& upper,
                                  const std::vector<double>& cost, ObjSense sense) {
  const std::size_t n = cost.size();
  std::vector<Enumerated> out;
  for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
    std::vector<int> x(n);
    for (std::size_t j = 0; j < n; ++j) x[j] = static_cast<int>((mask >> j) & 1u);
    bool feasible = true;
    for (std::size_t i = 0; i < rows.size() && feasible; ++i) {
      double activity = 0.0;
      for (std::size_t j = 0; j < n; ++j) activity += rows[i][j] * x[j];
      feasible = activity >= lower[i] - 1e-9 && activity <= upper[i] + 1e-9;
    }
    if (!feasible) continue;
    double objective = 0.0;
    for (std::size_t j = 0; j < n; ++j) objective += cost[j] * x[j];
    out.push_back({objective, x});
  }
  std::stable_sort(out.begin(), out.end(), [&](const Enumerated& a, const Enumerated& b) {
    return sense == ObjSense::kMaximize ? a.objective > b.objective : a.objective < b.objective;
  });
  return out;
}

Options pool_options(std::int64_t size, bool complete, bool diversity = false) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("pool_size", size);
  options.set_bool("pool_complete", complete);
  options.set_bool("pool_diversity", diversity);
  options.set_int("node_limit", 200000);
  return options;
}

std::vector<long> assignment(const std::vector<double>& x) {
  std::vector<long> out;
  for (const double v : x) out.push_back(std::lround(v));
  return out;
}

std::size_t hamming(const std::vector<double>& a, const std::vector<double>& b) {
  std::size_t d = 0;
  for (std::size_t j = 0; j < a.size(); ++j) d += std::lround(a[j]) != std::lround(b[j]);
  return d;
}

double mean_pairwise_hamming(const std::vector<Solution::PoolEntry>& pool) {
  double total = 0.0;
  int pairs = 0;
  for (std::size_t a = 0; a < pool.size(); ++a) {
    for (std::size_t b = a + 1; b < pool.size(); ++b) {
      total += static_cast<double>(hamming(pool[a].col_value, pool[b].col_value));
      ++pairs;
    }
  }
  return pairs == 0 ? 0.0 : total / pairs;
}

/// What every pool must satisfy whatever the options: the solution first, ordered, distinct,
/// every member feasible and integral with its objective right.
void expect_pool_invariants(const Model& model, const Solution& s) {
  ASSERT_FALSE(s.pool.empty());
  EXPECT_EQ(s.pool.front().col_value, s.col_value) << "pool[0] must be the solution";
  EXPECT_EQ(s.pool.front().objective, s.objective);
  const double sigma = model.sense == ObjSense::kMaximize ? -1.0 : 1.0;
  std::set<std::vector<long>> seen;
  for (std::size_t k = 0; k < s.pool.size(); ++k) {
    const Solution::PoolEntry& member = s.pool[k];
    EXPECT_TRUE(seen.insert(assignment(member.col_value)).second) << "duplicate at " << k;
    if (k > 0) {
      EXPECT_GE(sigma * member.objective, sigma * s.pool[k - 1].objective - 1e-9)
          << "out of order at " << k;
    }
    EXPECT_NEAR(member.objective, model.evaluate_objective(member.col_value.data()), 1e-9);
    Solution measured;
    measured.col_value = member.col_value;
    measured.recompute_quality(model);
    EXPECT_LE(measured.primal_infeasibility, tol::kPrimalFeasibility) << "member " << k;
    EXPECT_LE(measured.integrality_violation, tol::kIntegrality) << "member " << k;
  }
}

// =========================================================================================

TEST(SolutionPool, CompleteSearchHoldsTheKBestInOrderWithNoDuplicates) {
  // values 10 7 4 3 2, weights 5 4 3 2 1, capacity 9: a knapsack with 20-odd feasible fillings
  // and several ties in value, so "the k best" is a claim about objectives, checked as one.
  const std::vector<std::vector<double>> rows = {{5.0, 4.0, 3.0, 2.0, 1.0}};
  const std::vector<double> lower = {-kInfinity};
  const std::vector<double> upper = {9.0};
  const std::vector<double> cost = {10.0, 7.0, 4.0, 3.0, 2.0};
  const Model model = binary_model(rows, lower, upper, cost, ObjSense::kMaximize);
  const std::vector<Enumerated> truth =
      enumerate(rows, lower, upper, cost, ObjSense::kMaximize);
  ASSERT_GE(truth.size(), 6u);

  const Solution s = solve(model, pool_options(6, /*complete=*/true));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  expect_pool_invariants(model, s);
  ASSERT_EQ(s.pool.size(), 6u);
  for (std::size_t k = 0; k < 6; ++k) {
    EXPECT_NEAR(s.pool[k].objective, truth[k].objective, 1e-9) << "rank " << k + 1;
  }
}

TEST(SolutionPool, CompleteSearchAgreesWithEnumerationOnRandomModels) {
  // The gate. Random pure-binary models, two or three rows of mixed sign, costs of both signs,
  // both senses; the pool's objectives must equal the enumeration's k best exactly.
  std::mt19937 rng(20260916);
  std::uniform_int_distribution<int> coefficient(-6, 9);
  std::uniform_int_distribution<int> columns(4, 8);
  std::uniform_int_distribution<int> row_count(1, 3);
  int compared = 0;
  for (int instance = 0; instance < 60; ++instance) {
    const auto n = static_cast<std::size_t>(columns(rng));
    const auto m = static_cast<std::size_t>(row_count(rng));
    std::vector<std::vector<double>> rows(m, std::vector<double>(n));
    std::vector<double> lower(m);
    std::vector<double> upper(m);
    for (std::size_t i = 0; i < m; ++i) {
      double total = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        rows[i][j] = coefficient(rng);
        total += std::fabs(rows[i][j]);
      }
      upper[i] = std::floor(total * 0.4);
      lower[i] = (instance % 3 == 0) ? -std::floor(total * 0.3) : -kInfinity;
    }
    std::vector<double> cost(n);
    for (std::size_t j = 0; j < n; ++j) cost[j] = coefficient(rng);
    const ObjSense sense = instance % 2 == 0 ? ObjSense::kMinimize : ObjSense::kMaximize;
    const std::size_t k = 1 + static_cast<std::size_t>(instance % 7);

    const Model model = binary_model(rows, lower, upper, cost, sense);
    const std::vector<Enumerated> truth = enumerate(rows, lower, upper, cost, sense);
    const Solution s = solve(model, pool_options(static_cast<std::int64_t>(k), true));
    if (truth.empty()) {
      EXPECT_EQ(s.status, SolveStatus::kInfeasible) << "instance " << instance;
      continue;
    }
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << "instance " << instance << ": " << s.message;
    expect_pool_invariants(model, s);
    ASSERT_EQ(s.pool.size(), std::min(k, truth.size())) << "instance " << instance;
    for (std::size_t r = 0; r < s.pool.size(); ++r) {
      EXPECT_NEAR(s.pool[r].objective, truth[r].objective, 1e-9)
          << "instance " << instance << " rank " << r + 1;
    }
    ++compared;
  }
  EXPECT_GE(compared, 40) << "too few feasible instances to be evidence";
}

TEST(SolutionPool, TheDefaultSearchReportsWhatItFoundWithoutChangingTheAnswer) {
  const std::vector<std::vector<double>> rows = {{5.0, 4.0, 3.0, 2.0, 1.0}};
  const Model model =
      binary_model(rows, {-kInfinity}, {9.0}, {10.0, 7.0, 4.0, 3.0, 2.0}, ObjSense::kMaximize);
  Options off = pool_options(0, false);
  const Solution without = solve(model, off);
  const Solution with = solve(model, pool_options(10, false));
  EXPECT_TRUE(without.pool.empty());
  ASSERT_EQ(with.status, without.status);
  EXPECT_EQ(with.objective, without.objective);
  EXPECT_EQ(with.col_value, without.col_value);
  EXPECT_EQ(with.nodes, without.nodes) << "the default pool must not change the search";
  expect_pool_invariants(model, with);
}

TEST(SolutionPool, DiversitySpreadsThePoolOut) {
  // One heavy column and seven light ones. The k best all set the heavy column and differ by
  // dropping one light column - near-copies at Hamming distance 1 or 2 of each other. With
  // diversity on, the pool keeps plans that differ in many columns instead.
  const std::vector<std::vector<double>> rows = {{1, 1, 1, 1, 1, 1, 1, 1}};
  const std::vector<double> cost = {100, 1, 1, 1, 1, 1, 1, 1};
  const Model model = binary_model(rows, {-kInfinity}, {8.0}, cost, ObjSense::kMaximize);

  const Solution plain = solve(model, pool_options(4, true, /*diversity=*/false));
  const Solution diverse = solve(model, pool_options(4, true, /*diversity=*/true));
  ASSERT_EQ(plain.status, SolveStatus::kOptimal);
  ASSERT_EQ(diverse.status, SolveStatus::kOptimal);
  expect_pool_invariants(model, plain);
  expect_pool_invariants(model, diverse);
  ASSERT_EQ(plain.pool.size(), 4u);
  ASSERT_EQ(diverse.pool.size(), 4u);
  EXPECT_EQ(diverse.objective, plain.objective) << "diversity must not change the solution";
  EXPECT_GT(mean_pairwise_hamming(diverse.pool), mean_pairwise_hamming(plain.pool));
}

TEST(SolutionPool, PoolGapDropsMembersTooFarFromTheSolution) {
  const std::vector<std::vector<double>> rows = {{5.0, 4.0, 3.0, 2.0, 1.0}};
  const Model model =
      binary_model(rows, {-kInfinity}, {9.0}, {10.0, 7.0, 4.0, 3.0, 2.0}, ObjSense::kMaximize);
  Options options = pool_options(20, true);
  options.set_double("pool_gap", 0.1);  // within 10% of the optimum 17: objective >= 15.3
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal);
  expect_pool_invariants(model, s);
  for (const Solution::PoolEntry& member : s.pool) {
    EXPECT_GE(member.objective, 17.0 - 0.1 * 17.0 - 1e-9);
  }
  // 17 (items 0,1), 16 (0,2,4 / 0,3,4... ) - enumerate rather than list by hand.
  const std::vector<Enumerated> truth =
      enumerate(rows, {-kInfinity}, {9.0}, {10.0, 7.0, 4.0, 3.0, 2.0}, ObjSense::kMaximize);
  const auto within = static_cast<std::size_t>(
      std::count_if(truth.begin(), truth.end(),
                    [](const Enumerated& e) { return e.objective >= 17.0 - 1.7 - 1e-9; }));
  EXPECT_EQ(s.pool.size(), within);
}

TEST(SolutionPool, TheSolFileCarriesThePoolIntegersOnly) {
  const std::vector<std::vector<double>> rows = {{5.0, 4.0, 3.0, 2.0, 1.0}};
  Model model =
      binary_model(rows, {-kInfinity}, {9.0}, {10.0, 7.0, 4.0, 3.0, 2.0}, ObjSense::kMaximize);
  const Options options = pool_options(3, true);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.pool.size(), 3u);
  const TempFile file("", ".sol");
  std::string error;
  ASSERT_TRUE(io::write_solution(file.path(), model, s, options, &error)) << error;
  std::string text;
  if (std::FILE* in = std::fopen(file.path().c_str(), "rb")) {
    char buffer[4096];
    std::size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof buffer, in)) > 0) text.append(buffer, got);
    std::fclose(in);
  }
  EXPECT_NE(text.find("begin pool 3 5\n"), std::string::npos) << text;
  EXPECT_NE(text.find("solution 1 "), std::string::npos);
  EXPECT_NE(text.find("solution 3 "), std::string::npos);
  EXPECT_NE(text.find("end pool\n"), std::string::npos);
}

TEST(SolutionPool, PoolWriteAllColumnsWritesTheContinuousColumnsToo) {
  // maximise z - 4y, z - 10y <= 0, y binary, 0 <= z <= 10: two plans, (1, 10) and (0, 0).
  Model model;
  model.sense = ObjSense::kMaximize;
  model.col_cost = {-4.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {1.0, 10.0};
  model.col_type = {VarType::kInteger, VarType::kContinuous};
  model.row_lower = {-kInfinity};
  model.row_upper = {0.0};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, -10.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  model.hessian.reset(2, 2);
  model.hessian.finalize();

  Options options = pool_options(5, true);
  const Solution s = solve(model, options);
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  expect_pool_invariants(model, s);
  ASSERT_EQ(s.pool.size(), 2u);

  const auto written = [&](const Options& with) {
    const TempFile file("", ".sol");
    std::string error;
    EXPECT_TRUE(io::write_solution(file.path(), model, s, with, &error)) << error;
    std::string text;
    if (std::FILE* in = std::fopen(file.path().c_str(), "rb")) {
      char buffer[4096];
      std::size_t got = 0;
      while ((got = std::fread(buffer, 1, sizeof buffer, in)) > 0) text.append(buffer, got);
      std::fclose(in);
    }
    return text;
  };
  EXPECT_NE(written(options).find("begin pool 2 1\n"), std::string::npos);
  options.set_bool("pool_write_all_columns", true);
  EXPECT_NE(written(options).find("begin pool 2 2\n"), std::string::npos);
}

}  // namespace
}  // namespace sankhya
