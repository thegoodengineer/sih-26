// SPDX-License-Identifier: Apache-2.0
// SANKHYA - node selection policies (#293).
//
// The correctness gate for these lives in test_branch_and_bound.cpp, where every policy is
// run against the exact rational oracle: order may change, answers may not. What is left for
// this file is the part the oracle cannot see - that the policies actually DIFFER, that ties
// are broken the same way twice, and that the search reports which one it used.
//
// A policy framework whose policies all behave identically would pass every correctness test
// ever written and be worth nothing, so the tests below fail if the tree does not change
// shape between them.

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

Options with_policy(const char* policy) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_int("node_limit", 200000);
  options.set_string("mip_node_selection", policy);
  return options;
}

/// A strongly correlated knapsack, the family that makes branch and bound work: each item's
/// value is its weight plus a constant, so every subset of the same weight is worth the same
/// and the relaxation cannot tell the good packings from the bad ones. Deterministic weights
/// rather than a random generator, so the tree is the same on every machine.
///
/// At 18 columns this costs about a tenth of a second and a few hundred nodes - enough tree
/// for the policies to disagree about, little enough for a unit test.
Model correlated_knapsack(int columns) {
  Model model;
  const auto n = static_cast<Index>(columns);
  model.sense = ObjSense::kMaximize;
  model.col_cost.resize(static_cast<std::size_t>(n));
  model.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  model.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double weight = 20.0 + static_cast<double>((j * 37) % 51);
    model.col_cost[u] = weight + 10.0;
    model.matrix.add_entry(0, j, weight);
    total += weight;
  }
  model.matrix.finalize();
  model.row_lower = {-kInfinity};
  model.row_upper = {std::floor(total / 2.0)};
  model.hessian.reset(n, n);
  model.hessian.finalize();
  EXPECT_EQ(model.validate(), "");
  return model;
}

// =========================================================================================

TEST(NodeSelection, EveryPolicyReachesTheSameOptimum) {
  const Model model = correlated_knapsack(18);
  const Solution hybrid = solve(model, with_policy("hybrid"));
  ASSERT_EQ(hybrid.status, SolveStatus::kOptimal) << hybrid.message;

  for (const char* policy : {"best-bound", "depth-first", "best-estimate"}) {
    const Solution other = solve(model, with_policy(policy));
    EXPECT_EQ(other.status, SolveStatus::kOptimal) << policy << ": " << other.message;
    EXPECT_NEAR(other.objective, hybrid.objective, 1e-9) << policy;
    EXPECT_LE(other.integrality_violation, 1e-6) << policy;
  }
}

TEST(NodeSelection, ThePoliciesExploreDifferentTrees) {
  // If they did not, there would be no framework here - only four names for one policy.
  // Measured on this model: best-bound 579 nodes, depth-first 641, best-estimate 737. The
  // test asserts they differ rather than pinning those numbers, which any change to
  // branching or pruning would move without anything being wrong.
  const Model model = correlated_knapsack(18);
  const Solution best_bound = solve(model, with_policy("best-bound"));
  const Solution depth_first = solve(model, with_policy("depth-first"));
  const Solution best_estimate = solve(model, with_policy("best-estimate"));
  ASSERT_EQ(best_bound.status, SolveStatus::kOptimal) << best_bound.message;
  ASSERT_EQ(depth_first.status, SolveStatus::kOptimal) << depth_first.message;
  ASSERT_EQ(best_estimate.status, SolveStatus::kOptimal) << best_estimate.message;

  EXPECT_NE(best_bound.nodes, depth_first.nodes)
      << "best-bound and depth-first explored the same number of nodes, which means the "
         "policy is not reaching the search";
  EXPECT_NE(best_bound.nodes, best_estimate.nodes)
      << "best-estimate ordered the tree exactly as best-bound did, which means the "
         "pseudocost estimate is not being used";
}

TEST(NodeSelection, TheSamePolicyTwiceExploresTheSameTree) {
  // Ties are broken by node index rather than by whichever node the scan happened to reach
  // first, so two runs of one model take the same nodes in the same order. Without that a
  // degenerate MILP - where equal bounds are most of the tree - would explore differently on
  // every run and no node count could be compared against anything.
  const Model model = correlated_knapsack(18);
  for (const char* policy : {"hybrid", "best-bound", "depth-first", "best-estimate"}) {
    const Solution first = solve(model, with_policy(policy));
    const Solution second = solve(model, with_policy(policy));
    EXPECT_EQ(first.nodes, second.nodes) << policy;
    EXPECT_EQ(first.objective, second.objective) << policy;
    EXPECT_EQ(first.col_value, second.col_value) << policy;
  }
}

TEST(NodeSelection, TheHybridFollowsBestBoundOnceTheDiveIsOver) {
  // The hybrid is a dive followed by best-bound, so on a model whose dive ends at the root
  // the two agree. This is what makes the default a considered choice rather than a fourth
  // policy: it differs from best-bound only in how it reaches the first incumbent.
  const Model model = correlated_knapsack(18);
  const Solution hybrid = solve(model, with_policy("hybrid"));
  const Solution best_bound = solve(model, with_policy("best-bound"));
  ASSERT_EQ(hybrid.status, SolveStatus::kOptimal);
  EXPECT_NEAR(hybrid.objective, best_bound.objective, 1e-9);
  EXPECT_EQ(hybrid.nodes, best_bound.nodes);
}

TEST(NodeSelection, AnUnknownPolicyIsRefusedByTheOptionRegistry) {
  Options options;
  std::string error;
  EXPECT_FALSE(options.set_from_string("mip_node_selection", "cheapest", &error));
  EXPECT_FALSE(error.empty());
  // And the four that exist are accepted.
  for (const char* policy : {"hybrid", "best-bound", "depth-first", "best-estimate"}) {
    EXPECT_TRUE(options.set_from_string("mip_node_selection", policy, &error)) << policy;
  }
}

TEST(NodeSelection, AnLpIsUnaffectedByTheOption) {
  // The option belongs to branch and bound; a model with no integer columns must not notice
  // it at all.
  Model model = correlated_knapsack(8);
  model.col_type.assign(model.col_type.size(), VarType::kContinuous);
  const Solution a = solve(model, with_policy("hybrid"));
  const Solution b = solve(model, with_policy("depth-first"));
  EXPECT_EQ(a.status, b.status);
  EXPECT_NEAR(a.objective, b.objective, 1e-9);
}

}  // namespace
}  // namespace sankhya
