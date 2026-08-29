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
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"

#include "mip/cuts.hpp"

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

}  // namespace
}  // namespace sankhya
