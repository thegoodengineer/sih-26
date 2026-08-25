// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the floating-point simplex against the exact oracle.
//
// THIS IS THE GATE. Every other test in the suite checks that the solver does what its
// author expected. This one checks that it does what the mathematics requires, on instances
// nobody chose, against an answer computed without a single rounding error.
//
// A mismatch here is the most important output the project can produce, so when one occurs
// the failure prints the whole instance, both objectives and the difference. A pass rate
// with no reproducible failing case is a claim, not evidence.
//
// The two engines see ONE instance, not two: GeneratedLp is handed to solve_exact() and to
// to_model() -> solve(), so a disagreement cannot be blamed on the file format.

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "oracles/lp_generator.hpp"
#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {
namespace {

/// Tally of one fuzz campaign. Every instance lands in exactly one bucket, and the buckets
/// are printed so that a generator drifting towards trivial instances is visible rather than
/// quietly making the whole run vacuous.
struct Tally {
  int optimal_agreed = 0;
  int infeasible_agreed = 0;
  int unbounded_agreed = 0;
  int oracle_skipped = 0;  // exact arithmetic overflowed, or the oracle hit its cap
  int mismatched = 0;
  std::vector<std::string> failures;

  [[nodiscard]] int compared() const {
    return optimal_agreed + infeasible_agreed + unbounded_agreed + mismatched;
  }
};

Solution solve_float(const Model& model) {
  Options options;
  options.set_bool("log_to_console", false);
  return solve(model, options);
}

/// Compare one instance. Returns true when the two engines agree.
bool compare(const GeneratedLp& lp, Tally* tally) {
  const OracleResult exact = solve_exact(lp);
  if (exact.status == OracleStatus::kOverflow ||
      exact.status == OracleStatus::kIterationLimit) {
    ++tally->oracle_skipped;
    return true;  // the oracle abstained; this instance says nothing either way
  }

  const Model model = to_model(lp);
  const Solution approximate = solve_float(model);

  const auto disagree = [&](const std::string& why) {
    ++tally->mismatched;
    if (tally->failures.size() < 5) {
      tally->failures.push_back(
          why + "\n  oracle:  " + to_string(exact.status) +
          (exact.status == OracleStatus::kOptimal
               ? "  objective " + std::to_string(exact.objective.to_double())
               : "") +
          "\n  solver:  " + std::string(to_string(approximate.status)) +
          (approximate.status == SolveStatus::kOptimal
               ? "  objective " + std::to_string(approximate.objective)
               : "") +
          "\n" + lp.to_text());
    }
    return false;
  };

  switch (exact.status) {
    case OracleStatus::kOptimal: {
      if (approximate.status != SolveStatus::kOptimal) {
        return disagree("the exact optimum exists but the solver did not find it");
      }
      const double expected = exact.objective.to_double();
      const double difference = std::fabs(approximate.objective - expected);
      const double scale = std::max(1.0, std::fabs(expected));
      if (difference > 1e-6 * scale) {
        return disagree("objectives differ by " + std::to_string(difference));
      }
      ++tally->optimal_agreed;
      return true;
    }
    case OracleStatus::kInfeasible:
      if (approximate.status != SolveStatus::kInfeasible) {
        return disagree("the instance is exactly infeasible");
      }
      ++tally->infeasible_agreed;
      return true;
    case OracleStatus::kUnbounded:
      if (approximate.status != SolveStatus::kUnbounded) {
        return disagree("the instance is exactly unbounded");
      }
      ++tally->unbounded_agreed;
      return true;
    case OracleStatus::kOverflow:
    case OracleStatus::kIterationLimit: return true;
  }
  return true;
}

void report(const std::string& title, const Tally& tally) {
  std::cout << "\n=== " << title << " ===\n"
            << "  agreed optimal      " << tally.optimal_agreed << "\n"
            << "  agreed infeasible   " << tally.infeasible_agreed << "\n"
            << "  agreed unbounded    " << tally.unbounded_agreed << "\n"
            << "  oracle abstained    " << tally.oracle_skipped << "\n"
            << "  MISMATCHED          " << tally.mismatched << "\n"
            << "  compared            " << tally.compared() << "\n";
  for (const std::string& failure : tally.failures) {
    std::cout << "\n--- failing instance ---\n" << failure << "\n";
  }
}

// =========================================================================================

TEST(FuzzAgainstOracle, RandomInstances) {
  std::mt19937_64 rng(20260903);
  GeneratorConfig config;
  Tally tally;

  for (int trial = 0; trial < 1000; ++trial) {
    compare(random_lp(rng, config), &tally);
  }

  report("1000 random instances", tally);
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.compared(), 800) << "the oracle abstained too often to prove anything";
  // Unstructured instances should produce a real spread of statuses; if they collapse to one
  // bucket the generator has drifted and the run means much less than its count suggests.
  EXPECT_GT(tally.optimal_agreed, 50);
  EXPECT_GT(tally.infeasible_agreed + tally.unbounded_agreed, 50);
}

TEST(FuzzAgainstOracle, DegenerateInstances) {
  // Degeneracy is where a simplex stalls or cycles. These instances are built so that most
  // constraints are active at one point, which is the configuration Bland's rule exists for.
  std::mt19937_64 rng(777000777);
  GeneratorConfig config;
  Tally tally;

  for (int trial = 0; trial < 1000; ++trial) {
    compare(degenerate_lp(rng, config), &tally);
  }

  report("1000 deliberately degenerate instances", tally);
  EXPECT_EQ(tally.mismatched, 0);
  EXPECT_GT(tally.compared(), 800);
  EXPECT_GT(tally.optimal_agreed, 400) << "degenerate instances should mostly be feasible";
}

TEST(FuzzAgainstOracle, KktInstancesAgainstTheAnalyticOptimum) {
  // The float solver against an optimum known in closed form - no oracle in the loop at all,
  // so this cannot be fooled by the oracle and the solver sharing a mistake.
  std::mt19937_64 rng(31337007);
  GeneratorConfig config;
  int agreed = 0;
  int wrong = 0;
  double worst_relative_error = 0.0;
  std::vector<std::string> failures;

  for (int trial = 0; trial < 500; ++trial) {
    const KktInstance instance = kkt_lp(rng, config);
    const Model model = to_model(instance.lp);
    const Solution solution = solve_float(model);

    const auto expected = static_cast<double>(instance.optimal_objective);
    if (solution.status != SolveStatus::kOptimal) {
      ++wrong;
      if (failures.size() < 5) {
        failures.push_back(std::string("status ") + to_string(solution.status) +
                           " on an instance optimal by construction (expected objective " +
                           std::to_string(expected) + ")\n" + instance.lp.to_text());
      }
      continue;
    }
    const double scale = std::max(1.0, std::fabs(expected));
    const double relative = std::fabs(solution.objective - expected) / scale;
    worst_relative_error = std::max(worst_relative_error, relative);
    if (relative > 1e-6) {
      ++wrong;
      if (failures.size() < 5) {
        failures.push_back("objective " + std::to_string(solution.objective) +
                           " but the construction guarantees " + std::to_string(expected) +
                           "\n" + instance.lp.to_text());
      }
      continue;
    }
    ++agreed;
  }

  std::cout << "\n=== 500 KKT-constructed instances (analytic optimum) ===\n"
            << "  agreed              " << agreed << "\n"
            << "  WRONG               " << wrong << "\n"
            << "  worst relative error " << worst_relative_error << "\n";
  for (const std::string& failure : failures) {
    std::cout << "\n--- failing instance ---\n" << failure << "\n";
  }
  EXPECT_EQ(wrong, 0);
  EXPECT_GT(agreed, 450);
}

}  // namespace
}  // namespace sankhya::oracle
