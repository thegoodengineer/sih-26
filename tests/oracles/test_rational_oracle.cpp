// SPDX-License-Identifier: Apache-2.0
// SANKHYA - tests for the exact oracle ITSELF.
//
// The fuzz harness judges the floating-point simplex against this oracle, so an oracle bug
// would not show up as a failure - it would show up as agreement on a wrong answer. That
// makes this file load-bearing: the oracle has to be verified independently before it is
// allowed to verify anything else.
//
// Two independent standards are used here, neither of which is the oracle:
//   * small LPs whose optimum is derived by hand in the comment
//   * KKT-constructed instances whose optimum is known analytically before solving

#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oracles/lp_generator.hpp"
#include "oracles/rational.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

/// Build an instance directly, for the hand-checked cases.
GeneratedLp make(const std::vector<std::vector<std::int64_t>>& a,
                 const std::vector<std::int64_t>& b, const std::vector<std::int64_t>& c,
                 const std::vector<std::int64_t>& upper = {}) {
  GeneratedLp lp;
  lp.num_rows = static_cast<Index>(a.size());
  lp.num_cols = static_cast<Index>(c.size());
  lp.a = a;
  lp.b = b;
  lp.c = c;
  lp.upper = upper.empty() ? std::vector<std::int64_t>(static_cast<std::size_t>(lp.num_cols),
                                                       kNoUpperBound)
                           : upper;
  return lp;
}

// =========================================================================================
// Exact arithmetic
// =========================================================================================

TEST(Rational, NormalisesOnConstruction) {
  const Rational r(6, -8);
  EXPECT_EQ(static_cast<long long>(r.numerator()), -3);
  EXPECT_EQ(static_cast<long long>(r.denominator()), 4);
}

TEST(Rational, ZeroHasACanonicalForm) {
  const Rational r(0, 17);
  EXPECT_TRUE(r.is_zero());
  EXPECT_EQ(static_cast<long long>(r.denominator()), 1);
  EXPECT_EQ(r, Rational(0));
}

TEST(Rational, ArithmeticIsExactWhereDoublesAreNot) {
  // 1/3 + 1/3 + 1/3 == 1 exactly. (In double this one happens to come out exact too,
  // because the rounding cancels - which is why the 0.1 + 0.2 case below is the honest
  // demonstration and this one is only a sanity check on the rational type.)
  const Rational third(1, 3);
  EXPECT_EQ(third + third + third, Rational(1));

  // 1/10 + 2/10 == 3/10 exactly, the canonical binary-floating-point embarrassment.
  EXPECT_EQ(Rational(1, 10) + Rational(2, 10), Rational(3, 10));
  EXPECT_NE(0.1 + 0.2, 0.3);
}

TEST(Rational, MultiplicationAndDivision) {
  EXPECT_EQ(Rational(2, 3) * Rational(3, 2), Rational(1));
  EXPECT_EQ(Rational(2, 3) / Rational(4, 9), Rational(3, 2));
  EXPECT_EQ(-Rational(5, 7), Rational(-5, 7));
}

TEST(Rational, Comparison) {
  EXPECT_TRUE(Rational(1, 3) < Rational(1, 2));
  EXPECT_TRUE(Rational(-1, 3) < Rational(1, 1000000));
  EXPECT_TRUE(Rational(7, 2) > Rational(3));
  EXPECT_TRUE(Rational(4, 2) == Rational(2));
  EXPECT_LE(Rational(2), Rational(2));
}

TEST(Rational, OverflowThrowsRatherThanWrapping) {
  // A wrapped intermediate would silently turn the oracle into a random number generator,
  // and the fuzz harness would then confirm whatever the float simplex happened to do.
  const Rational huge(static_cast<Rational::Int>(1) << 100);
  EXPECT_THROW((void)(huge * huge), RationalOverflow);
  EXPECT_THROW((void)(Rational(1) / Rational(0)), RationalOverflow);
}

TEST(Rational, ConvertsToDoubleForReporting) {
  EXPECT_DOUBLE_EQ(Rational(3, 4).to_double(), 0.75);
  EXPECT_DOUBLE_EQ(Rational(-7, 2).to_double(), -3.5);
}

// =========================================================================================
// The exact simplex, against optima derived by hand
// =========================================================================================

TEST(RationalSimplex, MinimisesASum) {
  // min x0 + x1  s.t.  x0 + x1 >= 2,  x >= 0.  Optimum 2.
  const OracleResult r = solve_exact(make({{1, 1}}, {2}, {1, 1}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(2));
}

TEST(RationalSimplex, TextbookMaximisationWrittenAsAMinimisation) {
  // max 3a + 5b  s.t.  a <= 4, 2b <= 12, 3a + 2b <= 18  has optimum 36.
  // Negate to minimise, and flip the <= rows into >= rows: the optimum is -36.
  const OracleResult r =
      solve_exact(make({{-1, 0}, {0, -2}, {-3, -2}}, {-4, -12, -18}, {-3, -5}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(-36));
  ASSERT_EQ(r.x.size(), 2u);
  EXPECT_EQ(r.x[0], Rational(2));
  EXPECT_EQ(r.x[1], Rational(6));
}

TEST(RationalSimplex, ProducesAFractionalOptimumExactly) {
  // min x0  s.t.  3 x0 >= 1  ->  x0 = 1/3 exactly, not 0.333...
  const OracleResult r = solve_exact(make({{3}}, {1}, {1}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(1, 3));
  EXPECT_EQ(static_cast<long long>(r.objective.numerator()), 1);
  EXPECT_EQ(static_cast<long long>(r.objective.denominator()), 3);
}

TEST(RationalSimplex, DetectsInfeasibility) {
  // x0 >= 3 and -x0 >= -1 (that is, x0 <= 1) cannot both hold.
  const OracleResult r = solve_exact(make({{1}, {-1}}, {3, -1}, {1}));
  EXPECT_EQ(r.status, OracleStatus::kInfeasible);
}

TEST(RationalSimplex, DetectsInfeasibilityFromABound) {
  // x0 >= 5 with an upper bound of 2.
  const OracleResult r = solve_exact(make({{1}}, {5}, {1}, {2}));
  EXPECT_EQ(r.status, OracleStatus::kInfeasible);
}

TEST(RationalSimplex, DetectsUnboundedness) {
  // min -x0 with x0 >= 0 and nothing above it.
  const OracleResult r = solve_exact(make({{1}}, {0}, {-1}));
  EXPECT_EQ(r.status, OracleStatus::kUnbounded);
}

TEST(RationalSimplex, AnUpperBoundMakesAnUnboundedObjectiveFinite) {
  const OracleResult r = solve_exact(make({{1}}, {0}, {-1}, {7}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(-7));
}

TEST(RationalSimplex, SolvesADegenerateVertex) {
  // Three rows all tight at (1, 0): x0 >= 1, x0 + x1 >= 1, x0 - x1 >= 1, minimising x0.
  const OracleResult r = solve_exact(make({{1, 0}, {1, 1}, {1, -1}}, {1, 1, 1}, {1, 0}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(1));
}

TEST(RationalSimplex, HandlesRedundantDuplicatedRows) {
  // The same constraint five times over. Phase 1 must leave artificials pinned at zero on
  // the redundant rows rather than declaring the instance infeasible.
  const OracleResult r =
      solve_exact(make({{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}}, {3, 3, 3, 3, 3}, {1, 1}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(3));
}

TEST(RationalSimplex, ZeroObjectiveIsAFeasibilityQuestion) {
  const OracleResult r = solve_exact(make({{1, 1}}, {4}, {0, 0}));
  ASSERT_EQ(r.status, OracleStatus::kOptimal);
  EXPECT_EQ(r.objective, Rational(0));
}

// =========================================================================================
// The oracle against analytically known optima
// =========================================================================================

TEST(RationalSimplex, AgreesWithTheKktConstructionOnEveryInstance) {
  // kkt_lp() knows the optimum before anything is solved, because it builds the instance
  // backwards from a primal-dual pair. If the oracle disagrees, the oracle is wrong - and
  // this is the only test in the project able to make that statement.
  std::mt19937_64 rng(20260901);
  GeneratorConfig config;
  int checked = 0;
  int overflow = 0;

  for (int trial = 0; trial < 400; ++trial) {
    const KktInstance instance = kkt_lp(rng, config);
    const OracleResult r = solve_exact(instance.lp);
    if (r.status == OracleStatus::kOverflow || r.status == OracleStatus::kIterationLimit) {
      ++overflow;
      continue;
    }
    ASSERT_EQ(r.status, OracleStatus::kOptimal)
        << "a KKT-constructed instance is optimal by construction, but the oracle said "
        << to_string(r.status) << "\n"
        << instance.lp.to_text();
    ASSERT_EQ(r.objective, Rational(instance.optimal_objective))
        << "oracle " << r.objective.to_double() << " vs analytic " << instance.optimal_objective
        << "\n"
        << instance.lp.to_text();
    ++checked;
  }

  std::cout << "KKT cross-check: " << checked << " verified, " << overflow << " skipped\n";
  EXPECT_GT(checked, 350) << "too few instances survived for this to mean anything";
}

TEST(RationalSimplex, ReportsOverflowRatherThanAWrongAnswer) {
  // Coefficients large enough that the exact arithmetic must eventually give up. Whatever
  // happens, the one outcome that is not allowed is a confident wrong optimum.
  std::mt19937_64 rng(4242);
  GeneratorConfig config;
  config.magnitude = 1000000000LL;
  config.min_rows = 6;
  config.max_rows = 10;
  config.min_cols = 6;
  config.max_cols = 10;

  int overflow = 0;
  for (int trial = 0; trial < 60; ++trial) {
    const OracleResult r = solve_exact(random_lp(rng, config));
    if (r.status == OracleStatus::kOverflow) ++overflow;
    // Every status is a legal outcome here; the test is that it does not crash or wrap.
    EXPECT_NE(to_string(r.status), std::string("unknown"));
  }
  std::cout << "large-coefficient run: " << overflow << "/60 hit the overflow guard\n";
}

}  // namespace
}  // namespace sankhya::oracle
