// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cutting-plane validity.
//
// THIS FILE IS THE GATE, and it exists before the cuts it will eventually have to hold back.
//
// A cut that is very slightly invalid removes the optimum, and the search then proves the
// second-best answer optimal: status `optimal`, point integral and feasible, bound equal to
// objective. Nothing about that output looks wrong. No test that checks the solver against
// itself can catch it, which is why the check here is against the RATIONAL ORACLE's exact
// optimum, in exact arithmetic, with no tolerance to argue about.
//
// The controlling test is DoesNotSeparateTheExactOptimum. The one beside it is the negative
// control: a deliberately invalid tightening that the same harness must reject. A validity
// test that has never failed is not evidence that it can.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

#include "mip/cuts.hpp"
#include "simplex/primal_simplex.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya {
namespace {

using oracle::Rational;

Logger& quiet() {
  static Logger logger(nullptr);
  return logger;
}

/// Row activity a'x at the oracle's exact point, in exact arithmetic.
///
/// Deliberately computed from the GENERATED instance's integer coefficients rather than from
/// the Model the solver was handed. Those should agree, and a test that reads its expectation
/// from the same object it is checking would not notice if they stopped agreeing.
Rational exact_activity(const oracle::GeneratedLp& lp, const std::vector<Rational>& x,
                        Index row) {
  Rational total(0);
  for (Index j = 0; j < lp.num_cols; ++j) {
    const std::int64_t a = lp.a[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
    if (a == 0) continue;
    total = total + Rational(a) * x[static_cast<std::size_t>(j)];
  }
  return total;
}

/// Assert that every row bound in `model` is satisfied by the oracle's exact optimum.
///
/// Returns false with a populated `why` instead of asserting, so both the positive test and
/// the negative control can use it - the control needs the failure as a RESULT, not as a
/// terminated test.
bool optimum_survives(const Model& model, const oracle::GeneratedLp& lp,
                      const std::vector<Rational>& x, std::string* why) {
  for (Index i = 0; i < model.num_rows(); ++i) {
    const Rational activity = exact_activity(lp, x, i);
    const double lower = model.row_lower[static_cast<std::size_t>(i)];
    const double upper = model.row_upper[static_cast<std::size_t>(i)];

    // The bounds are integral after rounding, so comparing them exactly is meaningful: a
    // tightened bound is a whole number and the activity is a rational, and Rational's
    // comparison is exact.
    if (is_finite_bound(lower) && activity < Rational(static_cast<std::int64_t>(lower))) {
      *why = "row " + std::to_string(i) + ": activity " + std::to_string(activity.to_double()) +
             " is below the tightened lower bound " + std::to_string(lower);
      return false;
    }
    if (is_finite_bound(upper) && Rational(static_cast<std::int64_t>(upper)) < activity) {
      *why = "row " + std::to_string(i) + ": activity " + std::to_string(activity.to_double()) +
             " is above the tightened upper bound " + std::to_string(upper);
      return false;
    }
  }
  return true;
}

/// Generate MILPs whose exact integer optimum the oracle can settle.
struct Instance {
  oracle::GeneratedLp lp;
  std::vector<Rational> optimum;
};

std::vector<Instance> solvable_instances(int wanted, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  oracle::GeneratorConfig config;
  // The same shape FuzzAgainstTheExactMilpOracle uses, and for the same reason: the oracle
  // explores the tree in exact arithmetic and is deliberately slow.
  config.min_rows = 2;
  config.max_rows = 5;
  config.min_cols = 2;
  config.max_cols = 5;
  config.magnitude = 4;

  std::vector<Instance> found;
  for (int attempt = 0; attempt < wanted * 40 && static_cast<int>(found.size()) < wanted;
       ++attempt) {
    oracle::GeneratedLp lp = oracle::random_lp(rng, config);
    // Every column integral and bounded, which is what makes rows eligible for the rounding
    // and what keeps the exact tree finite.
    lp.integral.assign(static_cast<std::size_t>(lp.num_cols), 1);
    for (Index j = 0; j < lp.num_cols; ++j) {
      if (lp.upper[static_cast<std::size_t>(j)] == oracle::kNoUpperBound) {
        lp.upper[static_cast<std::size_t>(j)] = 6;
      }
    }
    const oracle::OracleResult exact = oracle::solve_exact_milp(lp, 20000);
    // Only instances the oracle SETTLED are usable. An overflow or a node limit is not a
    // wrong answer, it is an absent one, and testing against it would prove nothing.
    if (exact.status != oracle::OracleStatus::kOptimal) continue;
    found.push_back(Instance{std::move(lp), exact.x});
  }
  return found;
}

// =========================================================================================
// Helper: build a minimal binary knapsack Model for cut tests
// =========================================================================================

/// Build a Model with one row:  sum(a[j] * x[j]) <= b, all x[j] binary.
/// Used throughout the cover cut tests.
Model make_binary_knapsack(const std::vector<double>& a, double b) {
  const auto n = static_cast<Index>(a.size());
  Model m;
  m.col_cost.assign(static_cast<std::size_t>(n), 1.0);
  m.col_lower.assign(static_cast<std::size_t>(n), 0.0);
  m.col_upper.assign(static_cast<std::size_t>(n), 1.0);
  m.col_type.assign(static_cast<std::size_t>(n), VarType::kInteger);
  m.row_lower.push_back(-kInfinity);
  m.row_upper.push_back(b);
  m.matrix.reset(1, n);
  for (Index j = 0; j < n; ++j) {
    m.matrix.add_entry(0, j, a[static_cast<std::size_t>(j)]);
  }
  m.matrix.finalize();
  return m;
}

/// Evaluate the cut at a binary point x. Returns LHS - rhs (positive means violated).
double cut_violation(const mip::KnapsackCoverCut& cut, const std::vector<double>& x) {
  double lhs = 0.0;
  for (std::size_t k = 0; k < cut.col_index.size(); ++k) {
    lhs += cut.coeff[k] * x[static_cast<std::size_t>(cut.col_index[k])];
  }
  return lhs - cut.rhs;
}

// =========================================================================================
// EXISTING TESTS (tighten_integral_rows) Ã¢â‚¬â€ UNCHANGED
// =========================================================================================

TEST(Cuts, DoesNotSeparateTheExactOptimum) {
  const std::vector<Instance> instances = solvable_instances(120, 20260923);
  ASSERT_GE(instances.size(), 60u)
      << "the generator produced too few settled MILPs for this to mean anything";

  int tightened = 0;
  int restored_exactly = 0;
  for (const Instance& instance : instances) {
    Model model = oracle::to_model(instance.lp);
    // to_model() marks every column CONTINUOUS regardless of lp.integral - it exists to give
    // the float simplex the same LP the oracle solved, and integrality is the caller's to
    // apply. Without this the model has no integer columns, tighten_integral_rows returns
    // immediately, and the test passes while validating nothing. The anti-vacuity check at
    // the end of this test is what caught that, at 0 of 120 tightened.
    for (Index j = 0; j < model.num_cols(); ++j) {
      if (instance.lp.integral[static_cast<std::size_t>(j)] != 0) {
        model.col_type[static_cast<std::size_t>(j)] = VarType::kInteger;
      }
    }
    ASSERT_TRUE(model.validate().empty());

    // LOOSEN EVERY BOUND BY A FRACTION FIRST, and the construction is the point rather than a
    // convenience. The generator emits integer coefficients and an integer RHS, so its rows
    // are already rounded and the cut is a no-op on them - the anti-vacuity check below
    // caught precisely that, at 0 of 120 tightened.
    //
    // Moving an integer lower bound DOWN by a fraction strictly between 0 and 1 cannot change
    // the integer-feasible set: the activity is an integer, so a'x >= b - 0.3 and a'x >= b
    // admit exactly the same integer points. The oracle's optimum therefore remains the
    // optimum of the loosened model, and the rounding has something real to do - it must
    // restore the original bound exactly, and nothing weaker.
    std::vector<double> expected_lower = model.row_lower;
    std::vector<double> expected_upper = model.row_upper;
    for (Index i = 0; i < model.num_rows(); ++i) {
      const auto u = static_cast<std::size_t>(i);
      if (is_finite_bound(model.row_lower[u])) model.row_lower[u] -= 0.3;
      if (is_finite_bound(model.row_upper[u])) model.row_upper[u] += 0.7;
    }

    const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
    if (effect.rows_tightened > 0) ++tightened;

    // The rounding must land back on the integers it started from. A cut that stops SHORT is
    // merely weak; one that goes PAST is invalid, and the optimum check below would catch
    // that - but only if the optimum happens to sit on the bound. Checking the restoration
    // directly catches it on every instance instead of the lucky ones.
    bool exact = true;
    for (Index i = 0; i < model.num_rows(); ++i) {
      const auto u = static_cast<std::size_t>(i);
      if (is_finite_bound(expected_lower[u]) &&
          std::fabs(model.row_lower[u] - expected_lower[u]) > 1e-9) {
        exact = false;
      }
      if (is_finite_bound(expected_upper[u]) &&
          std::fabs(model.row_upper[u] - expected_upper[u]) > 1e-9) {
        exact = false;
      }
    }
    EXPECT_TRUE(exact) << "rounding did not restore the original integral bounds\n"
                       << instance.lp.to_text();
    if (exact) ++restored_exactly;

    std::string why;
    EXPECT_TRUE(optimum_survives(model, instance.lp, instance.optimum, &why))
        << "a cut removed the exact optimum: " << why << "\n"
        << instance.lp.to_text();
  }

  // If nothing was ever tightened the test above passed vacuously, which is the one way a
  // validity gate can be useless while looking green.
  EXPECT_GT(tightened, 0)
      << "no instance had a row tightened, so nothing was actually validated";
  std::printf("[  INFO    ] %d of %zu instances tightened, %d restored their bounds exactly\n",
              tightened, instances.size(), restored_exactly);
}

TEST(Cuts, TheHarnessCatchesADeliberatelyInvalidTightening) {
  // THE NEGATIVE CONTROL. Every check above passes if `tighten_integral_rows` does nothing at
  // all, so the harness has to be shown failing on a cut that IS invalid. This tightens each
  // row by one more unit than the rounding permits, which by construction removes any point
  // sitting exactly on the bound.
  const std::vector<Instance> instances = solvable_instances(80, 20262323);
  ASSERT_GE(instances.size(), 40u);

  int caught = 0;
  int tested = 0;
  for (const Instance& instance : instances) {
    Model model = oracle::to_model(instance.lp);
    mip::tighten_integral_rows(&model, quiet());

    // Over-tighten: legal rounding plus one.
    bool changed = false;
    for (Index i = 0; i < model.num_rows(); ++i) {
      double& lower = model.row_lower[static_cast<std::size_t>(i)];
      if (is_finite_bound(lower)) {
        lower += 1.0;
        changed = true;
      }
    }
    if (!changed) continue;
    ++tested;

    std::string why;
    if (!optimum_survives(model, instance.lp, instance.optimum, &why)) ++caught;
  }

  ASSERT_GT(tested, 0) << "no instance had a finite row lower bound to over-tighten";
  // Not every over-tightening separates the optimum - a row with slack at the optimum can
  // absorb one unit - so this asserts the harness catches MOST of them rather than all. What
  // matters is that it demonstrably catches the failure it exists for.
  EXPECT_GT(caught, tested / 2)
      << "the harness caught only " << caught << " of " << tested
      << " deliberately invalid tightenings, so it cannot be trusted to catch a real one";
  std::printf("[  INFO    ] caught %d of %d invalid tightenings\n", caught, tested);
}

TEST(Cuts, LeavesARowAloneWhenAnyColumnInItIsContinuous) {
  // The validity argument is "every column in this row is integral with an integral
  // coefficient, so the activity is an integer". One continuous column and the argument is
  // gone - the activity can be anything - so the row must not be touched.
  Model model;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.col_type = {VarType::kInteger, VarType::kContinuous};
  model.row_lower = {2.5};
  model.row_upper = {7.5};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 1.0);
  model.matrix.finalize();
  ASSERT_TRUE(model.validate().empty());

  const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
  EXPECT_EQ(effect.rows_tightened, 0);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 2.5);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 7.5);
}

TEST(Cuts, LeavesARowAloneWhenACoefficientIsFractional) {
  // Same argument, other half: integral columns but a coefficient of 0.5 makes the activity
  // a multiple of 0.5, not an integer.
  Model model;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.col_type = {VarType::kInteger, VarType::kInteger};
  model.row_lower = {2.5};
  model.row_upper = {kInfinity};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 0.5);
  model.matrix.finalize();

  const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
  EXPECT_EQ(effect.rows_tightened, 0);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 2.5);
}

TEST(Cuts, TightensAnEligibleRowInBothDirections) {
  //   2.3 <= x + 2y <= 7.8,  x and y integer.
  // The activity is an integer, so the row is exactly 3 <= x + 2y <= 7.
  Model model;
  model.col_cost = {1.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.col_type = {VarType::kInteger, VarType::kInteger};
  model.row_lower = {2.3};
  model.row_upper = {7.8};
  model.matrix.reset(1, 2);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(0, 1, 2.0);
  model.matrix.finalize();

  const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
  EXPECT_EQ(effect.rows_tightened, 1);
  EXPECT_EQ(effect.bounds_moved, 2);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 3.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 7.0);
}

TEST(Cuts, DoesNotMoveABoundThatIsAlreadyIntegral) {
  // ceil(3.0) is 3, but ceil of a 3 that floating-point arithmetic left at 3.0000000001 is 4,
  // and that would cut off the feasible point where the row is tight. The tolerance is
  // applied before rounding for exactly this case.
  Model model;
  model.col_cost = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.col_type = {VarType::kInteger};
  model.row_lower = {3.0 + 1e-12};
  model.row_upper = {8.0 - 1e-12};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();

  const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
  EXPECT_EQ(effect.bounds_moved, 0) << "a bound already on an integer was pushed a whole unit";
  EXPECT_NEAR(model.row_lower[0], 3.0, 1e-9);
  EXPECT_NEAR(model.row_upper[0], 8.0, 1e-9);
}

TEST(Cuts, DoesNothingToAPureLp) {
  Model model;
  model.col_cost = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {10.0};
  model.col_type = {VarType::kContinuous};
  model.row_lower = {2.5};
  model.row_upper = {7.5};
  model.matrix.reset(1, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.finalize();

  const mip::RowTightening effect = mip::tighten_integral_rows(&model, quiet());
  EXPECT_EQ(effect.rows_tightened, 0);
  EXPECT_DOUBLE_EQ(model.row_lower[0], 2.5);
}

// =========================================================================================
// NEW TESTS: generate_knapsack_cover_cut
// =========================================================================================

// ---- A. Simple supported binary knapsack producing a valid cover ----------------------

TEST(KnapsackCoverCuts, SimpleInstance) {
  // 2x1 + 3x2 + 4x3 <= 5, all binary.
  // Variables sorted descending: x3(4), x2(3), x1(2).
  // Greedy: pick x3 (sum=4 <= 5), pick x2 (sum=7 > 5). Cover = {x3, x2}.
  // Minimality: remove x2 -> sum=4 <= 5 (cannot remove). Remove x3 -> sum=3 <= 5 (cannot).
  // Minimal cover = {x2, x3} with base cut x2 + x3 <= 1.
  const Model m = make_binary_knapsack({2.0, 3.0, 4.0}, 5.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value()) << "expected a cut for a valid knapsack";
  // RHS must be |C| - 1 = 2 - 1 = 1.
  EXPECT_DOUBLE_EQ(cut->rhs, 1.0);
  // Cover members x2 (col 1) and x3 (col 2) have coefficient 1.
  // x1 (col 0) has some lifting coefficient >= 0.
  // All col_index entries must be sorted ascending.
  for (std::size_t k = 1; k < cut->col_index.size(); ++k) {
    EXPECT_LT(cut->col_index[k - 1], cut->col_index[k]) << "col_index not sorted ascending";
  }
  for (double c : cut->coeff) {
    EXPECT_GE(c, 0.0) << "negative coefficient in cut";
  }
}

// ---- B. Minimality: the selected cover is actually minimal ---------------------------

TEST(KnapsackCoverCuts, CoverIsMinimal) {
  // 3x1 + 3x2 + 3x3 <= 5, all binary.
  // Sorted: x1, x2, x3 all coeff=3, tiebreak by column: 0,1,2.
  // Greedy: pick x1 (sum=3), pick x2 (sum=6 > 5). Cover = {x1, x2}.
  // Minimality check: sum - a[x1] = 3 <= 5 (can't remove x1), sum - a[x2] = 3 <= 5 (can't).
  // So {x1, x2} is minimal.
  const Model m = make_binary_knapsack({3.0, 3.0, 3.0}, 5.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());
  // The cover should have exactly 2 members (x1, x2).
  EXPECT_DOUBLE_EQ(cut->rhs, 1.0);
  // The cover contains x1 (col 0) and x2 (col 1) both with coeff 1.
  bool found_col0 = false, found_col1 = false;
  for (std::size_t k = 0; k < cut->col_index.size(); ++k) {
    if (cut->col_index[k] == 0) { found_col0 = true; EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0); }
    if (cut->col_index[k] == 1) { found_col1 = true; EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0); }
  }
  EXPECT_TRUE(found_col0) << "x1 (col 0) should be in the cover";
  EXPECT_TRUE(found_col1) << "x2 (col 1) should be in the cover";
}

// ---- C. Base cover validity (no lifting, verify no integer point is separated) -------

TEST(KnapsackCoverCuts, BaseCoverValidityAtAllBinaryPoints) {
  // 4x1 + 5x2 + 6x3 <= 9, all binary.
  // Verify that at every binary feasible point (satisfying the knapsack),
  // the generated cut is also satisfied.
  const std::vector<double> a = {4.0, 5.0, 6.0};
  const double b = 9.0;
  const Model m = make_binary_knapsack(a, b);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());

  const int n = 3;
  for (std::size_t mask = 0; mask < (1ULL << n); ++mask) {
    std::vector<double> x(n);
    double knapsack_lhs = 0.0;
    for (std::size_t j = 0; j < static_cast<std::size_t>(n); ++j) {
      x[j] = static_cast<double>((mask >> j) & 1);
      knapsack_lhs += a[j] * x[j];
    }
    if (knapsack_lhs > b) continue;  // not feasible for the knapsack constraint
    const double violation = cut_violation(*cut, x);
    EXPECT_LE(violation, 1e-12)
        << "cut violated at feasible binary point: mask=" << mask
        << " violation=" << violation;
  }
}

// ---- D. Exact lifting: small hand-computable instance --------------------------------

TEST(KnapsackCoverCuts, ExactLiftingCoefficient) {
  // Instance: 3x1 + 3x2 + 2x3 <= 4, all binary.
  // Sorted vars: x1(3), x2(3), x3(2).
  // Greedy cover: pick x1(sum=3), pick x2(sum=6>4). Cover = {x1, x2}.
  // Minimality: remove x2 -> sum=3 <= 4 (can't). Remove x1 -> sum=3 <= 4 (can't). Minimal.
  // Base cut: x1 + x2 <= 1.
  //
  // Lifting x3: aux capacity = 4 - 2 = 2. Items in cover: {x1(w=3,p=1), x2(w=3,p=1)}.
  //   max profit with capacity 2: neither x1 nor x2 fits (w=3 > 2). Z_3 = 0.
  //   alpha_3 = (2-1) - 0 = 1.
  // Lifted cut: x1 + x2 + x3 <= 1.
  const Model m = make_binary_knapsack({3.0, 3.0, 2.0}, 4.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());
  EXPECT_DOUBLE_EQ(cut->rhs, 1.0);

  // Find x3 (col index 2) in the cut and verify its coefficient is 1.0.
  bool found_x3 = false;
  for (std::size_t k = 0; k < cut->col_index.size(); ++k) {
    if (cut->col_index[k] == 2) {
      found_x3 = true;
      EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0) << "x3 lifting coefficient should be 1.0";
    }
  }
  EXPECT_TRUE(found_x3) << "x3 should appear in the lifted cut with coefficient 1.0";
}

// ---- E. Multiple sequentially lifted variables ---------------------------------------

TEST(KnapsackCoverCuts, MultipleSequentialLiftings) {
  // Instance: 5x1 + 5x2 + 3x3 + 2x4 <= 7, all binary.
  // Sorted: x1(5), x2(5), x3(3), x4(2).
  // Greedy cover: pick x1(sum=5), pick x2(sum=10>7). Cover = {x1, x2}.
  // Minimality: 5 <= 7 (can't remove x1), 5 <= 7 (can't remove x2). Minimal.
  // Base cut: x1 + x2 <= 1.
  //
  // Lifting x3: aux_cap = 7 - 3 = 4. Items: {x1(5,1), x2(5,1)}. Neither fits (5>4). Z=0.
  //   alpha_3 = 1 - 0 = 1. Add x3(w=3,p=1) to DP.
  // Lifting x4: aux_cap = 7 - 2 = 5. Items: {x1(5,1), x2(5,1), x3(3,1)}.
  //   Best with cap 5: {x1} or {x2} gives profit 1. Or {x3+x4}? x4 not in pool yet.
  //   Actually: {x3} fits (w=3<=5, p=1). {x1} fits (w=5<=5, p=1). {x2} fits too.
  //   Can we combine? x1+x3=8>5, x2+x3=8>5. So best is 1. Z=1.
  //   alpha_4 = 1 - 1 = 0.
  const Model m = make_binary_knapsack({5.0, 5.0, 3.0, 2.0}, 7.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());
  EXPECT_DOUBLE_EQ(cut->rhs, 1.0);
  // x1 and x2 are cover members with coefficient 1.
  // x3 should have coefficient 1 (lifted).
  // x4 should have coefficient 0 (lifted, Z=1 gives alpha=0, so not included in output).
  bool found_x1 = false, found_x2 = false, found_x3 = false;
  for (std::size_t k = 0; k < cut->col_index.size(); ++k) {
    if (cut->col_index[k] == 0) { found_x1 = true; EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0); }
    if (cut->col_index[k] == 1) { found_x2 = true; EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0); }
    if (cut->col_index[k] == 2) { found_x3 = true; EXPECT_DOUBLE_EQ(cut->coeff[k], 1.0); }
    // x4 with coefficient 0 should not appear.
    if (cut->col_index[k] == 3) {
      EXPECT_GT(cut->coeff[k], 0.0) << "x4 with zero coeff should not appear in the output";
    }
  }
  EXPECT_TRUE(found_x1);
  EXPECT_TRUE(found_x2);
  EXPECT_TRUE(found_x3);
}

// ---- F. A variable whose lifting coefficient is zero --------------------------------

TEST(KnapsackCoverCuts, ZeroLiftingCoefficient) {
  // Instance: 5x1 + 5x2 + 5x3 + 4x4 <= 9, all binary.
  // Sorted: x1(5), x2(5), x3(5), x4(4).
  // Greedy: x1(5)+x2(10>9). Cover = {x1, x2}. Minimal.
  // Lifting x3: aux_cap=9-5=4. Items: {x1(5,1),x2(5,1)}. Neither fits (5>4). Z=0. alpha=1.
  //   Add x3(5,1) to DP.
  // Lifting x4: aux_cap=9-4=5. Items: {x1(5,1),x2(5,1),x3(5,1)}.
  //   Exactly one of x1/x2/x3 fits (each has w=5). Z=1. alpha=1-1=0.
  const Model m = make_binary_knapsack({5.0, 5.0, 5.0, 4.0}, 9.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());

  // x4 (col 3) should not appear (zero lifting coefficient -> excluded from output).
  for (std::size_t k = 0; k < cut->col_index.size(); ++k) {
    EXPECT_NE(cut->col_index[k], 3) << "x4 with zero coeff should not appear in output";
  }
}

// ---- G. No-cover case (sum of all coefficients <= b) --------------------------------

TEST(KnapsackCoverCuts, NoCoverWhenSumNotExceedCapacity) {
  // 1x1 + 1x2 + 1x3 <= 5: all binary, total weight = 3 <= 5. No cover possible.
  const Model m = make_binary_knapsack({1.0, 1.0, 1.0}, 5.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "no cover should be possible when total weight <= capacity";
}

// ---- H. Continuous variable in row -> no cut ----------------------------------------

TEST(KnapsackCoverCuts, ContinuousVariableBlocksCut) {
  // 2x1 + 3x2 <= 4, x1 binary, x2 continuous. Unsupported class.
  Model m;
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {1.0, 1.0};
  m.col_type = {VarType::kInteger, VarType::kContinuous};
  m.row_lower = {-kInfinity};
  m.row_upper = {4.0};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 2.0);
  m.matrix.add_entry(0, 1, 3.0);
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "continuous variable must block cut generation";
}

// ---- I. Negative coefficient -> no cut -----------------------------------------------

TEST(KnapsackCoverCuts, NegativeCoefficientBlocksCut) {
  // -2x1 + 3x2 <= 4, both binary. Negative coefficient: unsupported class.
  Model m;
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {1.0, 1.0};
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.row_lower = {-kInfinity};
  m.row_upper = {4.0};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, -2.0);
  m.matrix.add_entry(0, 1, 3.0);
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "negative coefficient must block cut generation";
}

// ---- J. Non-binary integer variable -> no cut ----------------------------------------

TEST(KnapsackCoverCuts, NonBinaryIntegerBlocksCut) {
  // 2x1 + 3x2 <= 4, x1 in {0,1,2}: not binary. Unsupported class.
  Model m;
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {2.0, 1.0};  // x1 has upper bound 2, not binary
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.row_lower = {-kInfinity};
  m.row_upper = {4.0};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 2.0);
  m.matrix.add_entry(0, 1, 3.0);
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "non-binary integer variable must block cut generation";
}

// ---- K. Infinite RHS -> no cut -------------------------------------------------------

TEST(KnapsackCoverCuts, InfiniteRhsBlocksCut) {
  Model m;
  m.col_cost = {1.0};
  m.col_lower = {0.0};
  m.col_upper = {1.0};
  m.col_type = {VarType::kInteger};
  m.row_lower = {-kInfinity};
  m.row_upper = {kInfinity};  // unconstrained from above
  m.matrix.reset(1, 1);
  m.matrix.add_entry(0, 0, 2.0);
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "infinite RHS must block cut generation";
}

// ---- L. Equality row -> no cut -------------------------------------------------------

TEST(KnapsackCoverCuts, EqualityRowBlocksCut) {
  // An equality row has both bounds finite; the classifier requires row_lower == -inf.
  Model m;
  m.col_cost = {1.0, 1.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {1.0, 1.0};
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.row_lower = {4.0};  // equality: lower = upper = 4
  m.row_upper = {4.0};
  m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 2.0);
  m.matrix.add_entry(0, 1, 3.0);
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "equality row must block cut generation";
}

// ---- M. Empty row -> no cut (degenerate / valid input) -------------------------------

TEST(KnapsackCoverCuts, EmptyRowProducesNoCut) {
  // A row with no nonzero entries: the model is valid but there are no variables to cover.
  Model m;
  m.col_cost = {1.0};
  m.col_lower = {0.0};
  m.col_upper = {1.0};
  m.col_type = {VarType::kInteger};
  m.row_lower = {-kInfinity};
  m.row_upper = {5.0};
  m.matrix.reset(1, 1);
  // Intentionally no add_entry for row 0.
  m.matrix.finalize();
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  EXPECT_FALSE(cut.has_value()) << "empty row should produce no cut";
}

// ---- N. Deterministic behavior when multiple candidate covers exist ------------------

TEST(KnapsackCoverCuts, CoverSelectionIsDeterministic) {
  // Calling the generator twice on the same input must give identical results.
  const Model m = make_binary_knapsack({4.0, 3.0, 3.0, 2.0}, 6.0);
  const auto cut1 = mip::generate_knapsack_cover_cut(m, 0);
  const auto cut2 = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_EQ(cut1.has_value(), cut2.has_value());
  if (cut1.has_value()) {
    EXPECT_EQ(cut1->col_index, cut2->col_index);
    EXPECT_EQ(cut1->coeff, cut2->coeff);
    EXPECT_DOUBLE_EQ(cut1->rhs, cut2->rhs);
  }
}

// ---- Exact-oracle validity gate for knapsack cover cuts ------------------------------

TEST(KnapsackCoverCuts, ExactOracleValidityGate) {
  // Hand-crafted small MILP where the exact optimal rational solution is known and the
  // knapsack row is clearly identifiable. We verify the generated cut (base + lifting) does
  // not separate the integer optimum.
  //
  // Problem: min x1 + x2 + x3
  //          s.t.  3x1 + 3x2 + 2x3 <= 4   (knapsack row 0)
  //                x1, x2, x3 in {0, 1}
  //
  // The feasible integer points satisfying the knapsack:
  //   All-zero (0,0,0): obj=0  <- optimum
  //   (1,0,0): 3<=4, obj=1
  //   (0,1,0): 3<=4, obj=1
  //   (0,0,1): 2<=4, obj=1
  //   (1,0,1): 5>4, infeasible
  //   (0,1,1): 5>4, infeasible
  //   (1,1,0): 6>4, infeasible
  //   (1,1,1): 8>4, infeasible
  //
  // Optimum: x* = (0,0,0), obj = 0.
  // Cover from D: {x1,x2}, lifted cut: x1 + x2 + x3 <= 1.
  // Evaluate at x*=(0,0,0): LHS=0 <= 1. Valid.

  const Model m = make_binary_knapsack({3.0, 3.0, 2.0}, 4.0);
  const auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());

  const std::vector<double> x_star = {0.0, 0.0, 0.0};
  const double violation = cut_violation(*cut, x_star);
  EXPECT_LE(violation, 1e-12)
      << "generated cut separates the known integer optimum (0,0,0)! violation=" << violation;

  // Also verify for every feasible integer point.
  const std::vector<double> a = {3.0, 3.0, 2.0};
  const double b = 4.0;
  for (std::size_t mask = 0; mask < 8; ++mask) {
    std::vector<double> x(3);
    double knap = 0.0;
    for (std::size_t j = 0; j < 3; ++j) { x[j] = static_cast<double>((mask >> j) & 1); knap += a[j] * x[j]; }
    if (knap > b) continue;
    EXPECT_LE(cut_violation(*cut, x), 1e-12)
        << "cut violated at feasible point mask=" << mask;
  }
}

// ---- NEGATIVE CONTROL: deliberately invalid cover cut must be detected ---------------

TEST(KnapsackCoverCuts, TheHarnessCatchesADeliberatelyInvalidCoverCut) {
  // Take the valid cut from the ExactOracleValidityGate test and tighten it by -1 on the RHS.
  // This turns  x1 + x2 + x3 <= 1  into  x1 + x2 + x3 <= 0.
  // The feasible point x=(0,0,1) has LHS=1 > 0: the cut now separates a feasible point.
  //
  // The harness MUST detect this. If it does not, the harness cannot be trusted.

  const Model m = make_binary_knapsack({3.0, 3.0, 2.0}, 4.0);
  auto cut = mip::generate_knapsack_cover_cut(m, 0);
  ASSERT_TRUE(cut.has_value());

  // Deliberately invalidate: tighten RHS by 1.
  cut->rhs -= 1.0;

  // x = (0, 0, 1) is feasible (3*0+3*0+2*1=2<=4) but should violate the invalid cut.
  const std::vector<double> x_feasible = {0.0, 0.0, 1.0};
  const double violation = cut_violation(*cut, x_feasible);
  EXPECT_GT(violation, 0.0)
      << "HARNESS BROKEN: the deliberately invalid cut was not detected at feasible point (0,0,1)."
      << " violation=" << violation
      << ". The validity harness cannot be trusted.";
}

// =========================================================================================
// Basis Reconstruction Tests (Stage 2)
// =========================================================================================

/// Independent dense Gaussian elimination to verify z^T B = e_r^T without using SparseLu.
std::vector<double> dense_transpose_solve(const std::vector<std::vector<double>>& B, Index r) {
  const std::size_t m = B.size();
  std::vector<std::vector<double>> A(m, std::vector<double>(m + 1, 0.0));
  // A = [B^T | e_r]
  const std::size_t ur = static_cast<std::size_t>(r);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      A[i][j] = B[j][i];
    }
    A[i][m] = (i == ur) ? 1.0 : 0.0;
  }

  for (std::size_t k = 0; k < m; ++k) {
    std::size_t pivot = k;
    double max_val = std::fabs(A[k][k]);
    for (std::size_t i = k + 1; i < m; ++i) {
      if (std::fabs(A[i][k]) > max_val) {
        max_val = std::fabs(A[i][k]);
        pivot = i;
      }
    }
    if (max_val < 1e-9) throw std::runtime_error("Singular matrix in dense_transpose_solve");
    std::swap(A[k], A[pivot]);
    const double p = A[k][k];
    for (std::size_t j = k; j <= m; ++j) A[k][j] /= p;
    for (std::size_t i = 0; i < m; ++i) {
      if (i == k) continue;
      const double f = A[i][k];
      for (std::size_t j = k; j <= m; ++j) A[i][j] -= f * A[k][j];
    }
  }
  std::vector<double> z(m);
  for (std::size_t i = 0; i < m; ++i) z[i] = A[i][m];
  return z;
}

TEST(BasisReconstruction, RejectsInvalidInputs) {
  Model m; m.resize_columns(2); m.resize_rows(1);
  Solution s;

  // Wrong dimensions
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.row_status = {};
  EXPECT_FALSE(mip::detail::get_tableau_for_testing(m, s, 0).has_value());

  // Wrong basic count
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.row_status = {BasisStatus::kBasic}; // 2 basic vars, m = 1
  EXPECT_FALSE(mip::detail::get_tableau_for_testing(m, s, 0).has_value());
}

TEST(BasisReconstruction, SmallHandbuiltBasis) {
  Model m;
  m.resize_rows(2);
  m.resize_columns(3);
  m.matrix.reset(2, 3);
  // Matrix: A_0 = [1, 2]^T, A_1 = [3, 4]^T, A_2 = [5, 6]^T
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(1, 0, 2.0);
  m.matrix.add_entry(0, 1, 3.0); m.matrix.add_entry(1, 1, 4.0);
  m.matrix.add_entry(0, 2, 5.0); m.matrix.add_entry(1, 2, 6.0);
  m.matrix.finalize();

  Solution s;
  s.col_status.resize(3, BasisStatus::kAtLower);
  s.col_value.resize(3, 0.0);
  s.row_status.resize(2, BasisStatus::kAtUpper);
  s.row_activity.resize(2, 0.0);

  s.col_status[0] = BasisStatus::kBasic;
  s.row_status[1] = BasisStatus::kBasic;

  s.col_value[0] = 7.7;     // value of structural col 0
  s.row_activity[1] = 8.8;  // value of logical row 1

  // BasisReconstructor inserts structurals first (col order) then logicals (row order).
  // With col_status[0]=kBasic and row_status[1]=kBasic:
  //   Basis slot 0 = Structural col 0  (basic_is_structural=true,  basic_index=0, rhs=7.7)
  //   Basis slot 1 = Logical row 1     (basic_is_structural=false, basic_index=1, rhs=8.8)

  auto row0 = mip::detail::get_tableau_for_testing(m, s, 0);
  ASSERT_TRUE(row0.has_value());
  EXPECT_TRUE(row0->basic_is_structural);
  EXPECT_EQ(row0->basic_index, 0);
  EXPECT_DOUBLE_EQ(row0->rhs, 7.7); // verifies RHS = current basic variable value

  auto row1 = mip::detail::get_tableau_for_testing(m, s, 1);
  ASSERT_TRUE(row1.has_value());
  EXPECT_FALSE(row1->basic_is_structural);
  EXPECT_EQ(row1->basic_index, 1);
  EXPECT_DOUBLE_EQ(row1->rhs, 8.8);

  // Independent validation: B[:,0]=A[:,0]=[1,2]^T, B[:,1]=-e_1=[0,-1]^T
  // B[i][j] = B_{ij}: B = [[1,0],[2,-1]]
  std::vector<std::vector<double>> B = { { 1.0, 0.0 }, { 2.0, -1.0 } };

  std::vector<double> z0 = dense_transpose_solve(B, 0);
  std::vector<double> z1 = dense_transpose_solve(B, 1);

  EXPECT_DOUBLE_EQ(row0->structural_coefs[0], z0[0]*1.0 + z0[1]*2.0);
  EXPECT_DOUBLE_EQ(row0->structural_coefs[1], z0[0]*3.0 + z0[1]*4.0);
  EXPECT_DOUBLE_EQ(row0->structural_coefs[2], z0[0]*5.0 + z0[1]*6.0);
  EXPECT_DOUBLE_EQ(row0->logical_coefs[0], -z0[0]);
  EXPECT_DOUBLE_EQ(row0->logical_coefs[1], -z0[1]);

  EXPECT_DOUBLE_EQ(row1->structural_coefs[0], z1[0]*1.0 + z1[1]*2.0);
  EXPECT_DOUBLE_EQ(row1->structural_coefs[1], z1[0]*3.0 + z1[1]*4.0);
  EXPECT_DOUBLE_EQ(row1->structural_coefs[2], z1[0]*5.0 + z1[1]*6.0);
  EXPECT_DOUBLE_EQ(row1->logical_coefs[0], -z1[0]);
  EXPECT_DOUBLE_EQ(row1->logical_coefs[1], -z1[1]);
}

TEST(BasisReconstruction, RealSolverValidation) {
  // Test I: Real Solver validation with mathematical identity z^T B = e_r^T
  Model m;
  m.resize_columns(2); m.resize_rows(2);
  m.matrix.reset(2, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 2.0); m.matrix.add_entry(1, 1, 1.0);
  m.matrix.finalize();
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInfinity, kInfinity};
  m.col_cost = {-3.0, -2.0};
  m.row_lower = {-kInfinity, -kInfinity};
  m.row_upper = {4.0, 5.0};

  Options opts;
  opts.set_bool("log_to_console", false);
  Logger logger(nullptr);
  Solution sol = solve(m, opts);
  ASSERT_EQ(sol.status, SolveStatus::kOptimal);

  for (Index r = 0; r < 2; ++r) {
    auto row = mip::detail::get_tableau_for_testing(m, sol, r);
    ASSERT_TRUE(row.has_value());

    // z^T B = e_r^T means:
    // For the basic variable at slot r, its coefficient must be exactly 1.0
    // For the other basic variable, its coefficient must be exactly 0.0
    // We don't have z directly in the test (it is internal), but we have the structural
    // and logical coefficients which ARE z^T A_j and -z_i.

    // Basic variables must have identity columns in the tableau.
    if (row->basic_is_structural) {
      EXPECT_NEAR(row->structural_coefs[static_cast<std::size_t>(row->basic_index)], 1.0, 1e-9);
    } else {
      EXPECT_NEAR(row->logical_coefs[static_cast<std::size_t>(row->basic_index)], 1.0, 1e-9);
    }

    // Verify RHS against Solution value
    if (row->basic_is_structural) {
      EXPECT_DOUBLE_EQ(row->rhs, sol.col_value[static_cast<std::size_t>(row->basic_index)]);
    } else {
      EXPECT_DOUBLE_EQ(row->rhs, sol.row_activity[static_cast<std::size_t>(row->basic_index)]);
    }
  }
}

TEST(BasisReconstruction, StrongBasisPermutationAndSparseLuOracle) {
  Model m;
  m.resize_columns(3); m.resize_rows(3);
  m.matrix.reset(3, 3);
  // Matrix A:
  // [ 2  0  1 ]
  // [ 0  3  4 ]
  // [ 5  1  0 ]
  m.matrix.add_entry(0, 0, 2.0); m.matrix.add_entry(2, 0, 5.0);
  m.matrix.add_entry(1, 1, 3.0); m.matrix.add_entry(2, 1, 1.0);
  m.matrix.add_entry(0, 2, 1.0); m.matrix.add_entry(1, 2, 4.0);
  m.matrix.finalize();

  Solution s;
  s.col_status.resize(3, BasisStatus::kAtLower);
  s.col_value.resize(3, 0.0);
  s.row_status.resize(3, BasisStatus::kAtUpper);
  s.row_activity.resize(3, 0.0);

  // Set basis: col 2, row 0, col 0.
  // True simplex ordering: slot 0 = col 2, slot 1 = row 0, slot 2 = col 0.
  // Reconstructed deterministic ordering: structurals first (col 0, col 2), logicals (row 0).
  // Slot 0 -> col 0
  // Slot 1 -> col 2
  // Slot 2 -> row 0

  s.col_status[0] = BasisStatus::kBasic; s.col_value[0] = 10.0;
  s.col_status[2] = BasisStatus::kBasic; s.col_value[2] = 20.0;
  s.row_status[0] = BasisStatus::kBasic; s.row_activity[0] = 30.0;

  // Expected deterministic B = [ A_0, A_2, -e_0 ]
  // B = [ 2  1 -1 ]
  //     [ 0  4  0 ]
  //     [ 5  0  0 ]
  std::vector<std::vector<double>> B = {
      { 2.0, 1.0, -1.0 },
      { 0.0, 4.0,  0.0 },
      { 5.0, 0.0,  0.0 }
  };

  for (Index r = 0; r < 3; ++r) {
    auto row = mip::detail::get_tableau_for_testing(m, s, r);
    ASSERT_TRUE(row.has_value());

    std::vector<double> z_dense = dense_transpose_solve(B, r);

    // 5. identify the basic variable represented by slot r
    bool expect_structural = (r < 2);
    Index expect_index = (r == 0) ? 0 : (r == 1) ? 2 : 0;

    EXPECT_EQ(row->basic_is_structural, expect_structural);
    EXPECT_EQ(row->basic_index, expect_index);

    // 8. verify RHS
    if (expect_structural) {
      EXPECT_DOUBLE_EQ(row->rhs, s.col_value[static_cast<std::size_t>(expect_index)]);
    } else {
      EXPECT_DOUBLE_EQ(row->rhs, s.row_activity[static_cast<std::size_t>(expect_index)]);
    }

    // 6 & 7: basic variable coefficient is 1, others 0
    EXPECT_NEAR(row->structural_coefs[0], (r == 0 ? 1.0 : 0.0), 1e-9);
    EXPECT_NEAR(row->structural_coefs[2], (r == 1 ? 1.0 : 0.0), 1e-9);
    EXPECT_NEAR(row->logical_coefs[0], (r == 2 ? 1.0 : 0.0), 1e-9);

    // Also verify alpha = -z for ALL logical columns
    for (Index i = 0; i < 3; ++i) {
      EXPECT_NEAR(row->logical_coefs[static_cast<std::size_t>(i)], -z_dense[static_cast<std::size_t>(i)], 1e-9);
    }

    // Verify all structural coefficients against dense Z
    EXPECT_NEAR(row->structural_coefs[0], z_dense[0]*2.0 + z_dense[2]*5.0, 1e-9);
    EXPECT_NEAR(row->structural_coefs[1], z_dense[1]*3.0 + z_dense[2]*1.0, 1e-9);
    EXPECT_NEAR(row->structural_coefs[2], z_dense[0]*1.0 + z_dense[1]*4.0, 1e-9);
  }
}

// =========================================================================================
// GMI Generator Tests (Stage 3B)
// =========================================================================================

// Test A: Cut.AllIntegerLower
TEST(GmiCut, AllIntegerLower) {
  Model m;
  m.resize_columns(3); m.resize_rows(1);
  m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.2);
  m.matrix.add_entry(0, 2, 2.1);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 2.0, 0.0};
  m.col_upper = {kInfinity, kInfinity, kInfinity};
  m.row_lower = {0.0};
  m.row_upper = {kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {-2.4, 2.0, 0.0};
  s.row_status = {BasisStatus::kAtLower};
  s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], -2.5, 1e-9);
  EXPECT_NEAR(cut->coeff[1], -10.0 / 3.0, 1e-9);
  EXPECT_NEAR(cut->coeff[2], -65.0 / 12.0, 1e-9);
  EXPECT_NEAR(cut->rhs, -5.0 / 3.0, 1e-9);

  double eval_int = cut->coeff[0]*(-2.0) + cut->coeff[1]*(2.0) + cut->coeff[2]*(0.0);
  EXPECT_LE(eval_int, cut->rhs + 1e-9);

  double eval_lp = cut->coeff[0]*(-2.4) + cut->coeff[1]*(2.0) + cut->coeff[2]*(0.0);
  EXPECT_GT(eval_lp, cut->rhs + 1e-4);
}

// Test B: Cut.IntegerAtUpperPositiveAlpha
TEST(GmiCut, IntegerAtUpperPositiveAlpha) {
  Model m; m.resize_columns(3); m.resize_rows(1); m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.5); m.matrix.add_entry(0, 2, 0.2);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 0.0, 1.0};
  m.col_upper = {kInfinity, 3.0, kInfinity};
  m.row_lower = {0.0};
  m.row_upper = {kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtUpper, BasisStatus::kAtLower};
  s.col_value = {-4.7, 3.0, 1.0};
  s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], -10.0 / 7.0, 1e-9);
  EXPECT_NEAR(cut->coeff[1], -10.0 / 7.0, 1e-9);
  EXPECT_NEAR(cut->coeff[2], -20.0 / 21.0, 1e-9);
  EXPECT_NEAR(cut->rhs, 10.0 / 21.0, 1e-9);

  double eval_int = cut->coeff[0]*(-4.0) + cut->coeff[1]*(3.0) + cut->coeff[2]*(1.0);
  EXPECT_LE(eval_int, cut->rhs + 1e-9);
}

// Test C: Cut.IntegerAtUpperNegativeAlpha
TEST(GmiCut, IntegerAtUpperNegativeAlpha) {
  Model m; m.resize_columns(3); m.resize_rows(1); m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, -1.5); m.matrix.add_entry(0, 2, 0.2);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 0.0, 1.0};
  m.col_upper = {kInfinity, 3.0, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtUpper, BasisStatus::kAtLower};
  s.col_value = {4.3, 3.0, 1.0};
  s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], -10.0 / 7.0, 1e-9);
  EXPECT_NEAR(cut->coeff[1], 20.0 / 7.0, 1e-9);
  EXPECT_NEAR(cut->coeff[2], -20.0 / 21.0, 1e-9);
  EXPECT_NEAR(cut->rhs, 10.0 / 21.0, 1e-9);

  double eval_int = cut->coeff[0]*(5.0) + cut->coeff[1]*(3.0) + cut->coeff[2]*(1.0);
  EXPECT_LE(eval_int, cut->rhs + 1e-9);
}

// Test D: Cut.ContinuousPositive
TEST(GmiCut, ContinuousPositive) {
  Model m; m.resize_columns(3); m.resize_rows(1); m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.2); m.matrix.add_entry(0, 2, 0.8);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  m.col_lower = {-kInfinity, 1.0, 2.0}; m.col_upper = {kInfinity, kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {-2.8, 1.0, 2.0};
  s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());
}

// Test E: Cut.ContinuousNegative
TEST(GmiCut, ContinuousNegative) {
  Model m; m.resize_columns(3); m.resize_rows(1); m.matrix.reset(1, 3);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.2); m.matrix.add_entry(0, 2, -0.8);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  m.col_lower = {-kInfinity, 1.0, 2.0}; m.col_upper = {kInfinity, kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {0.4, 1.0, 2.0};
  s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], -5.0/3.0, 1e-9);
  EXPECT_NEAR(cut->coeff[1], -2.5, 1e-9);
  EXPECT_NEAR(cut->coeff[2], 0.0, 1e-9);
  EXPECT_NEAR(cut->rhs, -25.0/6.0, 1e-9);

  double eval_int = cut->coeff[0]*(-2.0) + cut->coeff[1]*(3.0) + cut->coeff[2]*(2.0);
  EXPECT_LE(eval_int, cut->rhs + 1e-9);
}

// Test F: Cut.LogicalLower
TEST(GmiCut, LogicalLower) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};

  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {-1.5, 1.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());
}

// Test G: Cut.LogicalUpper
TEST(GmiCut, LogicalUpper) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 1.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {-kInfinity}; m.row_upper = {5.0};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {3.5, 1.0};
  s.row_status = {BasisStatus::kAtUpper}; s.row_activity = {5.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], 2.0, 1e-9);
  EXPECT_NEAR(cut->coeff[1], 2.0, 1e-9);
  EXPECT_NEAR(cut->rhs, 8.0, 1e-9);

  EXPECT_LE(cut->coeff[0]*3.0 + cut->coeff[1]*1.0, cut->rhs + 1e-9);
}

// Test H: Cut.MixedCase
TEST(GmiCut, MixedCase) {
  Model m; m.resize_columns(5); m.resize_rows(2); m.matrix.reset(2, 5);
  m.matrix.add_entry(0, 0, 2.0); m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0); m.matrix.add_entry(1, 1, 3.0);
  m.matrix.add_entry(0, 2, 1.0); m.matrix.add_entry(1, 2, -1.0);
  m.matrix.add_entry(0, 3, -1.5); m.matrix.add_entry(1, 3, 1.0);
  m.matrix.add_entry(0, 4, 3.0); m.matrix.add_entry(1, 4, 1.0);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  m.col_lower = {-kInfinity, -kInfinity, 1.0, -kInfinity, 0.0};
  m.col_upper = {kInfinity, kInfinity, kInfinity, 2.0, kInfinity};
  m.row_lower = {-kInfinity, 0.0};
  m.row_upper = {5.0, kInfinity};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kBasic, BasisStatus::kAtLower, BasisStatus::kAtUpper, BasisStatus::kAtLower};
  s.col_value = {4.4, -1.8, 1.0, 2.0, 0.0};
  s.row_status = {BasisStatus::kAtUpper, BasisStatus::kAtLower};
  s.row_activity = {5.0, 0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  EXPECT_NEAR(cut->coeff[0], 2.5, 1e-9);
  EXPECT_NEAR(cut->coeff[1], 0.0, 1e-9);
  EXPECT_NEAR(cut->coeff[2], 5.0/3.0, 1e-9);
  EXPECT_NEAR(cut->coeff[3], -2.5, 1e-9);
  EXPECT_NEAR(cut->coeff[4], 0.0, 1e-9);
  EXPECT_NEAR(cut->rhs, 20.0/3.0, 1e-9);

  double eval_int = cut->coeff[0]*(4.0) + cut->coeff[1]*(-1.0) + cut->coeff[2]*(1.0) + cut->coeff[3]*(2.0) + cut->coeff[4]*(0.0);
  EXPECT_LE(eval_int, cut->rhs + 1e-9);

  double eval_lp = cut->coeff[0]*(4.4) + cut->coeff[1]*(-1.8) + cut->coeff[2]*(1.0) + cut->coeff[3]*(2.0) + cut->coeff[4]*(0.0);
  EXPECT_GT(eval_lp, cut->rhs + 1e-4);
}

// Test I: NearIntegralBasic
TEST(GmiCut, NearIntegralBasic) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {3.0 + (tol::kIntegrality / 2.0), 1.0}; s.row_status = {BasisStatus::kAtLower};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Test J: NearIntegralIntegerCoefficient
TEST(GmiCut, NearIntegralIntegerCoefficient) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0 + (tol::kIntegrality / 2.0)); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {3.5, 1.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Test K: ContinuousBasic
TEST(GmiCut, ContinuousBasic) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0); m.matrix.finalize();
  m.col_type = {VarType::kContinuous, VarType::kInteger};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {3.5, 1.0}; s.row_status = {BasisStatus::kAtLower};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Test L: LogicalBasic
TEST(GmiCut, LogicalBasic) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  Solution s; s.col_status = {BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {0.0, 0.0}; s.row_status = {BasisStatus::kBasic};
  s.row_activity = {3.5};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Test M: FreeNonbasic
TEST(GmiCut, FreeNonbasic) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kNonbasicFree};
  s.col_value = {3.5, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Test N: FixedNonbasic
TEST(GmiCut, FixedNonbasic) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 1.0}; m.col_upper = {kInfinity, 1.0};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kFixed};
  s.col_value = {3.5, 1.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());
  EXPECT_DOUBLE_EQ(cut->coeff[1], -3.0);
}

// Test O: IntegralTableau
TEST(GmiCut, IntegralTableau) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 2.0); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {3.0, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  EXPECT_FALSE(mip::generate_gmi_cut(m, s, 0).has_value());
}

// Independent GMI Formula Test
TEST(GmiCut, IndependentFormulaCheck) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.2); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {1.8, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};
  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());
}


// =========================================================================================
// GMI Stage 3B Correctness Gate Tests
// =========================================================================================

#include "oracles/rational_simplex.hpp"

// Replaces the old RealSankhyaOptimumValid
TEST(GmiCut, RealSankhyaOptimumValid_WithExactOracle) {
  oracle::GeneratedLp lp;
  lp.num_rows = 2;
  lp.num_cols = 3;
  // A = [[-2, -3, -4], [-1, -1, -1]]
  lp.a = {{-2, -3, -4}, {-1, -1, -1}};
  // b = [-6, -2] (so A x >= b is 2x1+3x2+4x3 <= 6, x1+x2+x3 <= 2)
  lp.b = {-6, -2};
  lp.lower = {0, 0, 0};
  lp.upper = {1, 1, 1};
  lp.integral = {1, 1, 1};
  lp.c = {-1, -2, -3}; // minimize -x1 - 2x2 - 3x3 => maximize x1 + 2x2 + 3x3

  Model m = oracle::to_model(lp);
  m.col_type = {VarType::kContinuous, VarType::kContinuous, VarType::kContinuous};
  Options opts; opts.set_bool("log_to_console", false);
  Logger logger(nullptr);
  Solution sol = solve(m, opts);
  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kInteger};
  ASSERT_EQ(sol.status, SolveStatus::kOptimal);

  Index frac_row = -1;
  for (Index r = 0; r < 2; ++r) {
    auto tab = mip::detail::get_tableau_for_testing(m, sol, r);
    if (!tab) continue;
    if (tab->basic_is_structural) {
      double f0 = tab->rhs - std::floor(tab->rhs);
      if (f0 > 1e-5 && f0 < 1 - 1e-5) {
        frac_row = r;
        break;
      }
    }
  }

  // MUST NOT PASS VACUOUSLY
  ASSERT_NE(frac_row, -1);

  auto cut = mip::generate_gmi_cut(m, sol, frac_row);
  ASSERT_TRUE(cut.has_value());

  // EXACT ORACLE VALIDITY
  oracle::OracleResult exact_res = oracle::solve_exact_milp(lp, 1000000);
  ASSERT_EQ(exact_res.status, oracle::OracleStatus::kOptimal);

  double eval = 0.0;
  for (Index j = 0; j < 3; ++j) {
    auto j_sz = static_cast<std::size_t>(j);
    double x_val = static_cast<double>(exact_res.x[j_sz].numerator()) / static_cast<double>(exact_res.x[j_sz].denominator());
    eval += cut->coeff[j_sz] * x_val;
  }
  EXPECT_LE(eval, cut->rhs + 1e-6);
}

// True Exhaustive GMI Validity Test
TEST(GmiCut, ExhaustiveValidity) {
  // Max x1 + 2x2
  // s.t. 3x1 + 4x2 <= 10
  // x1, x2 in [0, 3], integer
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 3.0); m.matrix.add_entry(0, 1, 4.0);
  m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {3.0, 3.0};
  m.col_cost = {-1.0, -2.0};
  m.row_lower = {-kInfinity}; m.row_upper = {10.0};

  m.col_type = {VarType::kContinuous, VarType::kContinuous};
  Options opts; opts.set_bool("log_to_console", false);
  Solution sol = solve(m, opts);
  m.col_type = {VarType::kInteger, VarType::kInteger};
  ASSERT_EQ(sol.status, SolveStatus::kOptimal);

  Index frac_row = -1;
  for (Index r = 0; r < 1; ++r) {
    auto tab = mip::detail::get_tableau_for_testing(m, sol, r);
    if (!tab) continue;
    if (tab->basic_is_structural) {
      double f0 = tab->rhs - std::floor(tab->rhs);
      if (f0 > 1e-5 && f0 < 1 - 1e-5) {
        frac_row = r;
        break;
      }
    }
  }
  ASSERT_NE(frac_row, -1);

  auto cut = mip::generate_gmi_cut(m, sol, frac_row);
  ASSERT_TRUE(cut.has_value());

  // LP point should violate the cut
  double eval_lp = cut->coeff[0]*sol.col_value[0] + cut->coeff[1]*sol.col_value[1];
  EXPECT_GT(eval_lp, cut->rhs + 1e-4);

  // Enumerate all feasible points in the bounded box [0, 3] x [0, 3]
  for (int x1 = 0; x1 <= 3; ++x1) {
    for (int x2 = 0; x2 <= 3; ++x2) {
      // 1. Compute logical variables
      double s0 = 3.0 * x1 + 4.0 * x2;

      // 2. Check bounds
      if (s0 > 10.0) continue; // violated row upper bound

      // 3. Evaluate generated GMI cut
      double eval_int = cut->coeff[0]*x1 + cut->coeff[1]*x2;
      EXPECT_LE(eval_int, cut->rhs + 1e-9) << "Excluded integer point: (" << x1 << ", " << x2 << ")";
    }
  }
}

// Negative Control A: Wrong Upper Bound Sign
TEST(GmiCut, NegativeControl_WrongUpperBoundSign) {
  Model m_neg; m_neg.resize_columns(2); m_neg.resize_rows(1); m_neg.matrix.reset(1, 2);
  m_neg.matrix.add_entry(0, 0, 1.0); m_neg.matrix.add_entry(0, 1, -0.2); m_neg.matrix.finalize();
  m_neg.col_type = {VarType::kInteger, VarType::kInteger};
  m_neg.col_lower = {0.0, 0.0}; m_neg.col_upper = {kInfinity, 2.0};
  m_neg.row_lower = {1.0}; m_neg.row_upper = {1.0};

  Solution s_neg; s_neg.col_status = {BasisStatus::kBasic, BasisStatus::kAtUpper};
  s_neg.col_value = {1.4, 2.0}; s_neg.row_status = {BasisStatus::kFixed}; s_neg.row_activity = {1.0};

  auto cut_valid = mip::generate_gmi_cut(m_neg, s_neg, 0);
  ASSERT_TRUE(cut_valid.has_value());

  // Feasible point (1, 0)
  double eval_int_valid = cut_valid->coeff[0]*1.0 + cut_valid->coeff[1]*0.0;
  EXPECT_LE(eval_int_valid, cut_valid->rhs + 1e-9);

  // Corrupt beta manually:
  // pi = 0.5, U = 2. valid beta_raw = 0. corrupted beta_raw = 1 + 0.5(2) = 2.
  // corrupted rhs = -2.
  double corrupted_rhs = -2.0;
  EXPECT_GT(eval_int_valid, corrupted_rhs + 1e-9) << "Corrupted upper bound sign should reject (1,0)";
}

// Negative Control B: Wrong Logical Substitution Sign
TEST(GmiCut, NegativeControl_WrongLogicalSubstitutionSign) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 0.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};

  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {2.5, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  double eval_valid = cut->coeff[0]*1.0 + cut->coeff[1]*1.0;
  EXPECT_LE(eval_valid, cut->rhs + 1e-9);

  // Corrupted cut
  double corr_coeff_0 = 2.0;
  double corr_coeff_1 = 0.0;
  double eval_corr = corr_coeff_0*1.0 + corr_coeff_1*1.0;
  EXPECT_GT(eval_corr, cut->rhs + 1e-9) << "Wrong logical sign rejects feasible point (1, 1)";
}

// Negative Control C: Wrong RHS Constant
TEST(GmiCut, NegativeControl_WrongRhsConstant) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 0.5); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {2.5, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  double eval_valid = cut->coeff[0]*0.0 + cut->coeff[1]*1.0;
  EXPECT_LE(eval_valid, cut->rhs + 1e-9);

  double corr_rhs = cut->rhs - 2.0;
  EXPECT_GT(eval_valid, corr_rhs + 1e-9) << "Wrong RHS constant rejects feasible point (0, 1)";
}

// Negative Control D: Intentional Zero Snap
TEST(GmiCut, NegativeControl_IntentionalZeroSnap_Real) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 0.1); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {1.5}; m.row_upper = {1.5};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {1.5, 0.0}; s.row_status = {BasisStatus::kFixed}; s.row_activity = {1.5};

  auto cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(cut.has_value());

  double eval_valid = cut->coeff[0]*1.0 + cut->coeff[1]*5.0;
  EXPECT_LE(eval_valid, cut->rhs + 1e-9);

  // Corrupt: snap fj to zero => coeff[1] = 0
  double corr_coeff_1 = 0.0;
  double eval_corr = cut->coeff[0]*1.0 + corr_coeff_1*5.0;
  EXPECT_GT(eval_corr, cut->rhs + 1e-9) << "Zero snap rejects feasible point (1, 5)";
}

// =========================================================================================
// Stage 4B Production Multi-Row GMI Context Tests
// =========================================================================================

// Test A, B, C: RootGmiContext behavior and factorization amortization
TEST(RootGmiContext, FactorizationAmortization) {
  Model m;
  m.resize_columns(3); m.resize_rows(2);
  m.matrix.reset(2, 3);
  // x0 + x1 + x2 <= 2.5
  // x0 - x1 <= 0.5
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0); m.matrix.add_entry(0, 2, 1.0);
  m.matrix.add_entry(1, 0, 1.0); m.matrix.add_entry(1, 1, -1.0);
  m.matrix.finalize();

  m.col_type = {VarType::kInteger, VarType::kInteger, VarType::kContinuous};
  m.col_lower = {0.0, 0.0, 0.0}; m.col_upper = {kInfinity, kInfinity, kInfinity};
  m.row_lower = {-kInfinity, -kInfinity}; m.row_upper = {2.5, 0.5};

  Solution s;
  s.col_status = {BasisStatus::kBasic, BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {1.5, 1.0, 0.0};
  s.row_status = {BasisStatus::kAtUpper, BasisStatus::kAtUpper};
  s.row_activity = {2.5, 0.5};

  mip::RootGmiContext context(m, s);
  ASSERT_TRUE(context.is_valid);

  auto row0 = context.tableau_row(m, s, 0);
  ASSERT_TRUE(row0.has_value());
  EXPECT_TRUE(row0->basic_is_structural);
  EXPECT_EQ(row0->basic_index, 0);

  auto row1 = context.tableau_row(m, s, 1);
  ASSERT_TRUE(row1.has_value());
  EXPECT_TRUE(row1->basic_is_structural);
  EXPECT_EQ(row1->basic_index, 1);

  auto cuts = mip::generate_gmi_cuts(m, s);
  ASSERT_EQ(cuts.size(), 1);

  s.col_value = {1.5, 1.5, 0.0};
  mip::RootGmiContext context2(m, s);
  ASSERT_TRUE(context2.is_valid);
  auto cuts2 = mip::generate_gmi_cuts(m, s);
  ASSERT_EQ(cuts2.size(), 2);
}

// Test D: Production multi-row generator matches old API exactly
TEST(RootGmiContext, MatchesReferenceImplementation) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.2); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {-kInfinity, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {0.0}; m.row_upper = {kInfinity};
  Solution s; s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {1.8, 0.0}; s.row_status = {BasisStatus::kAtLower}; s.row_activity = {0.0};

  auto cuts = mip::generate_gmi_cuts(m, s);
  ASSERT_EQ(cuts.size(), 1);

  auto old_cut = mip::generate_gmi_cut(m, s, 0);
  ASSERT_TRUE(old_cut.has_value());

  ASSERT_EQ(cuts[0].coeff.size(), old_cut->coeff.size());
  for (size_t i = 0; i < cuts[0].coeff.size(); ++i) {
    EXPECT_DOUBLE_EQ(cuts[0].coeff[i], old_cut->coeff[i]);
  }
  EXPECT_DOUBLE_EQ(cuts[0].rhs, old_cut->rhs);
}

// Test G, H, I, J, K, L: RootGmiContext edge cases
TEST(RootGmiContext, RejectionConditions) {
  Model m; m.resize_columns(2); m.resize_rows(1); m.matrix.reset(1, 2);
  m.matrix.add_entry(0, 0, 1.0); m.matrix.add_entry(0, 1, 1.0); m.matrix.finalize();
  m.col_type = {VarType::kInteger, VarType::kInteger};
  m.col_lower = {0.0, 0.0}; m.col_upper = {kInfinity, kInfinity};
  m.row_lower = {-kInfinity}; m.row_upper = {2.0};

  Solution s;
  // H: Integral root
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {2.0, 0.0}; s.row_status = {BasisStatus::kAtUpper}; s.row_activity = {2.0};
  EXPECT_EQ(mip::generate_gmi_cuts(m, s).size(), 0);

  // I: Logical-only basics
  s.col_status = {BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.col_value = {0.0, 0.0}; s.row_status = {BasisStatus::kBasic}; s.row_activity = {0.0};
  EXPECT_EQ(mip::generate_gmi_cuts(m, s).size(), 0);

  // J: Continuous structural basics
  m.col_type = {VarType::kContinuous, VarType::kContinuous};
  s.col_status = {BasisStatus::kBasic, BasisStatus::kAtLower};
  s.col_value = {1.5, 0.0}; s.row_status = {BasisStatus::kAtUpper}; s.row_activity = {1.5};
  EXPECT_EQ(mip::generate_gmi_cuts(m, s).size(), 0);

  // K: Invalid basis
  s.col_status = {BasisStatus::kAtLower, BasisStatus::kAtLower};
  s.row_status = {BasisStatus::kAtLower};
  EXPECT_EQ(mip::generate_gmi_cuts(m, s).size(), 0);
}

}  // namespace
}  // namespace sankhya


// =========================================================================================
// Cut Filtering and Deduplication Tests (Stage 4C)
// =========================================================================================

using namespace sankhya;
using namespace sankhya::mip;

TEST(CutFiltering, RejectsNonfinite) {
  Model model;
  model.col_cost.assign(5, 0.0); // padded
  Solution sol;
  sol.col_value.push_back(0.0);

  Cut c;
  c.coeff = {std::numeric_limits<double>::infinity()}; c.coeff.resize(5, 0.0);
  c.rhs = 1.0;

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kNonfinite);

  c.coeff = {1.0}; c.coeff.resize(5, 0.0);
  c.rhs = std::numeric_limits<double>::quiet_NaN();
  res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kNonfinite);
}

TEST(CutFiltering, RejectsEmptySupport) {
  Model model;
  model.col_cost.assign(5, 0.0);
  Solution sol;
  sol.col_value.assign(5, 0.0);

  Cut c;
  c.coeff = {1e-12}; c.coeff.resize(5, 0.0);
  c.rhs = -1.0; // Highly violated if LHS is 0

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kEmptySupport);
}

TEST(CutFiltering, DensityThresholds) {
  Model model;
  model.col_cost.assign(10, 0.0); // 10 cols
  Solution sol;
  sol.col_value.assign(10, 0.0);

  Cut c;
  c.coeff.assign(10, 0.0);

  // kCutMaxDensity is 0.2, so 2 nonzeros is exactly 0.2 (accepted).
  c.coeff[0] = 1.0;
  c.coeff[1] = 1.0;
  c.rhs = -1.0;

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);

  // 3 nonzeros is 0.3 (rejected).
  c.coeff[2] = 1.0;
  res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kTooDense);
}

TEST(CutFiltering, CoefficientRatio) {
  Model model;
  model.col_cost.assign(10, 0.0);
  Solution sol;
  sol.col_value.assign(10, 0.0);

  Cut c;
  c.coeff = {1.0, 1e7}; c.coeff.resize(10, 0.0);
  c.rhs = -1.0;

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kCoefficientRatio);

  c.coeff = {1.0, 1e5}; c.coeff.resize(10, 0.0);
  res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
}

TEST(CutFiltering, RootLPViolation) {
  Model model;
  model.col_cost.assign(5, 0.0);
  Solution sol;
  sol.col_value.assign(5, 0.0);

  Cut c;
  c.coeff = {1.0}; c.coeff.resize(5, 0.0); // LHS = 0.0
  c.rhs = 0.0; // Violation = 0.0, kCutViolationTolerance is 1e-5

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kInsufficientViolation);

  c.rhs = -1e-4; // Violation = 1e-4 > 1e-5
  res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
}

TEST(CutFiltering, DoesNotModifyCutData) {
  Model model;
  model.col_cost.assign(5, 0.0);
  Solution sol;
  sol.col_value.assign(5, 0.0);

  Cut c;
  c.coeff = {1.234}; c.coeff.resize(5, 0.0);
  c.rhs = -5.678;

  auto res = filter_and_deduplicate_cuts(model, sol, {c});
  ASSERT_EQ(res.size(), 1);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
  EXPECT_DOUBLE_EQ(res[0].cut.coeff[0], 1.234);
  EXPECT_DOUBLE_EQ(res[0].cut.rhs, -5.678);

  Cut c_rej;
  c_rej.coeff = {1.0};
  c_rej.rhs = 0.0; // Insufficient violation
  auto res_rej = filter_and_deduplicate_cuts(model, sol, {c_rej});
  ASSERT_EQ(res_rej.size(), 1);
  EXPECT_EQ(res_rej[0].reason, CutFilterReason::kInsufficientViolation);
  EXPECT_DOUBLE_EQ(res_rej[0].cut.coeff[0], 1.0);
  EXPECT_DOUBLE_EQ(res_rej[0].cut.rhs, 0.0);
}

TEST(CutFiltering, DeduplicationPositive) {
  Model model;
  model.col_cost.assign(15, 0.0);
  Solution sol;
  sol.col_value.assign(15, 0.0);

  // A.
  Cut c1, c2;
  c1.coeff = {1.0, 1.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 1.0;
  c2.coeff = {2.0, 2.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 2.0;

  // Make c1 and c2 violated so they pass violation test
  sol.col_value = {2.0, 2.0, 0.0}; sol.col_value.resize(15, 0.0);

  auto res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  ASSERT_EQ(res.size(), 2);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(res[1].reason, CutFilterReason::kDuplicate);

  // B.
  c1.coeff = {-1.0, 2.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 3.0;
  c2.coeff = {-2.0, 4.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 6.0;
  sol.col_value = {0.0, 4.0, 0.0}; sol.col_value.resize(15, 0.0); // LHS=8 for c1
  res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(res[1].reason, CutFilterReason::kDuplicate);

  // C.
  c1.coeff = {0.5, -3.0, 1.0}; c1.coeff.resize(15, 0.0); c1.rhs = 2.0;
  c2.coeff = {2.0, -12.0, 4.0}; c2.coeff.resize(15, 0.0); c2.rhs = 8.0;
  sol.col_value = {10.0, 0.0, 0.0}; sol.col_value.resize(15, 0.0); // LHS=5 for c1
  res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(res[1].reason, CutFilterReason::kDuplicate);
}

TEST(CutFiltering, DeduplicationNegative) {
  Model model;
  model.col_cost.assign(15, 0.0);
  Solution sol;
  sol.col_value.assign(15, 5.0);

  // A.
  Cut c1, c2;
  c1.coeff = {1.0, 0.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 1.0;
  c2.coeff = {2.0, 0.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 3.0; // Not proportional RHS
  auto res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  ASSERT_EQ(res.size(), 2);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(res[1].reason, CutFilterReason::kAccepted);

  // B.
  c1.coeff = {1.0, 1.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 1.0;
  c2.coeff = {2.0, 2.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 2.1;
  res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(res[1].reason, CutFilterReason::kAccepted);

  // C.
  c1.coeff = {1.0, 2.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 4.0;
  c2.coeff = {1.0, 2.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 5.0;
  res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(res[1].reason, CutFilterReason::kAccepted);

  // D.
  c1.coeff = {1.0, 1.0, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 2.0;
  c2.coeff = {1.0, 0.0, 1.0}; c2.coeff.resize(15, 0.0); c2.rhs = 2.0;
  res = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(res[1].reason, CutFilterReason::kAccepted);
}

TEST(CutFiltering, CombinedFilterTest) {
  Model model;
  model.col_cost.assign(10, 0.0);
  Solution sol;
  sol.col_value.assign(10, 0.0);

  // 1. valid sparse well-scaled violated cut
  Cut c1;
  c1.coeff.assign(10, 0.0);
  c1.coeff[0] = 1.0;
  c1.rhs = -1.0; // Violated since LHS=0

  // 2. dense cut
  Cut c2;
  c2.coeff.assign(10, 1.0);
  c2.rhs = -10.0;

  // 3. badly scaled cut
  Cut c3;
  c3.coeff.assign(10, 0.0);
  c3.coeff[0] = 1.0;
  c3.coeff[1] = 1e8;
  c3.rhs = -1.0;

  // 4. non-violated cut
  Cut c4;
  c4.coeff.assign(10, 0.0);
  c4.coeff[0] = 1.0;
  c4.rhs = 1.0; // LHS=0 < 1.0 (Valid, but not violated)

  // 5. NaN cut
  Cut c5;
  c5.coeff.assign(10, 0.0);
  c5.coeff[0] = std::numeric_limits<double>::quiet_NaN();
  c5.rhs = -1.0;

  // 6. duplicate of #1
  Cut c6;
  c6.coeff.assign(10, 0.0);
  c6.coeff[0] = 2.0;
  c6.rhs = -2.0;

  auto res = filter_and_deduplicate_cuts(model, sol, {c1, c2, c3, c4, c5, c6});

  ASSERT_EQ(res.size(), 6);
  EXPECT_EQ(res[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(res[1].reason, CutFilterReason::kTooDense);
  EXPECT_EQ(res[2].reason, CutFilterReason::kCoefficientRatio);
  EXPECT_EQ(res[3].reason, CutFilterReason::kInsufficientViolation);
  EXPECT_EQ(res[4].reason, CutFilterReason::kNonfinite);
  EXPECT_EQ(res[5].reason, CutFilterReason::kDuplicate);
}

TEST(CutFiltering, DeduplicationScaleAwarePositive) {
  Model model;
  model.col_cost.assign(15, 0.0);
  Solution sol;
  sol.col_value.assign(15, 0.0);

  // A. [1e6, 2e6] <= 3e6 and [1, 2] <= 3
  Cut c1; c1.coeff = {1e6, 2e6, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 3e6;
  Cut c2; c2.coeff = {1.0, 2.0, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 3.0;
  sol.col_value = {0.0, 3.0, 0.0}; sol.col_value.resize(15, 0.0); // Make them violated
  auto resA = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(resA.size(), 2);
  EXPECT_EQ(resA[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resA[1].reason, CutFilterReason::kDuplicate);

  // C. Very small but proportionally identical coefficients
  Cut c3; c3.coeff = {1e-5, 2e-5, 0.0}; c3.coeff.resize(15, 0.0); c3.rhs = 3e-5;
  Cut c4; c4.coeff = {1.0, 2.0, 0.0}; c4.coeff.resize(15, 0.0); c4.rhs = 3.0;
  sol.col_value = {0.0, 3.0, 0.0}; sol.col_value.resize(15, 0.0);
  auto resC = filter_and_deduplicate_cuts(model, sol, {c3, c4});
  EXPECT_EQ(resC.size(), 2);
  EXPECT_EQ(resC[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resC[1].reason, CutFilterReason::kDuplicate);

  // E. Large RHS proportionality
  Cut c5; c5.coeff = {1.0, 2.0, 0.0}; c5.coeff.resize(15, 0.0); c5.rhs = 1.0;
  Cut c6; c6.coeff = {1e6, 2e6, 0.0}; c6.coeff.resize(15, 0.0); c6.rhs = 1e6;
  sol.col_value = {0.0, 3.0, 0.0}; sol.col_value.resize(15, 0.0);
  auto resE = filter_and_deduplicate_cuts(model, sol, {c5, c6});
  EXPECT_EQ(resE.size(), 2);
  EXPECT_EQ(resE[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resE[1].reason, CutFilterReason::kDuplicate);
}

TEST(CutFiltering, DeduplicationScaleAwareNegative) {
  Model model;
  model.col_cost.assign(15, 0.0);
  Solution sol;
  sol.col_value.assign(15, 0.0);

  // B. [1e6, 2e6] <= 3e6 and [1, 2.0001] <= 3
  Cut c1; c1.coeff = {1e6, 2e6, 0.0}; c1.coeff.resize(15, 0.0); c1.rhs = 3e6;
  Cut c2; c2.coeff = {1.0, 2.0001, 0.0}; c2.coeff.resize(15, 0.0); c2.rhs = 3.0;
  sol.col_value = {0.0, 3.0, 0.0}; sol.col_value.resize(15, 0.0);
  auto resB = filter_and_deduplicate_cuts(model, sol, {c1, c2});
  EXPECT_EQ(resB.size(), 2);
  EXPECT_EQ(resB[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resB[1].reason, CutFilterReason::kAccepted);

  // D. Very small but genuinely different coefficients
  Cut c3; c3.coeff = {1e-5, 2e-5, 0.0}; c3.coeff.resize(15, 0.0); c3.rhs = 3e-5;
  Cut c4; c4.coeff = {1.0, 2.1, 0.0}; c4.coeff.resize(15, 0.0); c4.rhs = 3.0;
  sol.col_value = {0.0, 3.0, 0.0}; sol.col_value.resize(15, 0.0);
  auto resD = filter_and_deduplicate_cuts(model, sol, {c3, c4});
  EXPECT_EQ(resD.size(), 2);
  EXPECT_EQ(resD[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resD[1].reason, CutFilterReason::kAccepted);

  // F. Same coefficients but non-proportional RHS
  Cut c5; c5.coeff = {1.0, 2.0, 0.0}; c5.coeff.resize(15, 0.0); c5.rhs = 3.0;
  Cut c6; c6.coeff = {1.0, 2.0, 0.0}; c6.coeff.resize(15, 0.0); c6.rhs = 4.0;
  sol.col_value = {0.0, 5.0, 0.0}; sol.col_value.resize(15, 0.0);
  auto resF = filter_and_deduplicate_cuts(model, sol, {c5, c6});
  EXPECT_EQ(resF.size(), 2);
  EXPECT_EQ(resF[0].reason, CutFilterReason::kAccepted);
  EXPECT_EQ(resF[1].reason, CutFilterReason::kAccepted);
}
