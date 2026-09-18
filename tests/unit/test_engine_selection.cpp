// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP engine selection behind `algorithm=auto` (#284). Every rule in
// src/core/engine_selection.hpp is pinned here with a model shaped to fire it, so a
// changed threshold shows up as a failed test and not as a quietly different default.

#include <gtest/gtest.h>

#include "core/engine_selection.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

/// A feasible, bounded LP with `rows` identity-like rows and `cols` columns, plus enough
/// extra entries to reach `nonzeros`. Only the shape matters to the selector.
Model shaped_lp(Index rows, Index cols, Count nonzeros) {
  Model m;
  m.resize_columns(cols);
  m.resize_rows(rows);
  m.matrix.reset(rows, cols);
  Count placed = 0;
  for (Index i = 0; i < rows && placed < nonzeros; ++i) {
    m.matrix.add_entry(i, i % cols, 1.0);
    ++placed;
  }
  // Fill the remainder along diagonals so no entry repeats.
  for (Index shift = 1; placed < nonzeros && shift < cols; ++shift) {
    for (Index i = 0; i < rows && placed < nonzeros; ++i) {
      m.matrix.add_entry(i, (i + shift) % cols, 1.0);
      ++placed;
    }
  }
  m.matrix.finalize();
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    m.col_cost[u] = 1.0;
    m.col_lower[u] = 0.0;
    m.col_upper[u] = 1.0;
  }
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    m.row_lower[u] = -kInfinity;
    m.row_upper[u] = 10.0;
  }
  return m;
}

Options auto_options() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

TEST(EngineSelection, ARequestedEngineIsHonouredAsGiven) {
  Options o = auto_options();
  o.set_string("algorithm", "pdhg");
  const EngineSelection s = select_engine(shaped_lp(3, 3, 3), o, false);
  EXPECT_EQ(s.algorithm, "pdhg");
  EXPECT_EQ(s.rule, "requested");
}

TEST(EngineSelection, AStartingBasisForcesTheDualSimplexWhateverTheSize) {
  const EngineSelection s =
      select_engine(shaped_lp(kPdhgRowFloor + 1, 10, 10), auto_options(), true);
  EXPECT_EQ(s.algorithm, "dual-simplex");
  EXPECT_EQ(s.rule, "warm-start");
}

TEST(EngineSelection, ANetlibSizedModelGoesToTheDualSimplex) {
  const EngineSelection s = select_engine(shaped_lp(2000, 5000, 30000), auto_options(), false);
  EXPECT_EQ(s.algorithm, "dual-simplex");
  EXPECT_EQ(s.rule, "default:dual-simplex");
  EXPECT_NE(s.reason.find("netlib-full"), std::string::npos) << s.reason;
}

TEST(EngineSelection, ADenseModelBelowTheRowLimitGoesToTheInteriorPoint) {
  // maros-r7's shape: 3,136 rows, 9,408 columns, 144,848 nonzeros.
  const EngineSelection s =
      select_engine(shaped_lp(3136, 9408, kIpmNonzeroFloor + 1), auto_options(), false);
  EXPECT_EQ(s.algorithm, "ipm");
  EXPECT_EQ(s.rule, "density:ipm");
}

TEST(EngineSelection, FromTheRowLimitTheInteriorPointIsChosen) {
  const EngineSelection s =
      select_engine(shaped_lp(kDualSimplexRowLimit, 100, 200), auto_options(), false);
  EXPECT_EQ(s.algorithm, "ipm");
  EXPECT_EQ(s.rule, "size:ipm");
  const EngineSelection below =
      select_engine(shaped_lp(kDualSimplexRowLimit - 1, 100, 200), auto_options(), false);
  EXPECT_EQ(below.algorithm, "dual-simplex");
}

TEST(EngineSelection, FromThePdhgFloorTheFirstOrderMethodIsChosen) {
  const EngineSelection s =
      select_engine(shaped_lp(kPdhgRowFloor, 100, 200), auto_options(), false);
  EXPECT_EQ(s.algorithm, "pdhg");
  EXPECT_EQ(s.rule, "size:pdhg");
  EXPECT_EQ(s.rows, kPdhgRowFloor);
}

TEST(EngineSelection, TheAnswerCarriesTheRuleAndTheReason) {
  // End to end: a small LP under `auto` is solved by the dual simplex, and the Solution
  // says which rule chose it, through presolve and all.
  Model m = shaped_lp(4, 6, 12);
  const Solution s = solve(m, auto_options());
  ASSERT_EQ(s.status, SolveStatus::kOptimal);
  EXPECT_EQ(s.engine_rule, "default:dual-simplex");
  EXPECT_NE(s.engine_reason.find("4 rows, 6 columns, 12 nonzeros"), std::string::npos)
      << s.engine_reason;
  Options explicit_primal = auto_options();
  explicit_primal.set_string("algorithm", "simplex");
  const Solution p = solve(m, explicit_primal);
  EXPECT_EQ(p.engine_rule, "requested");
}

TEST(EngineSelection, AnInteriorPointThatDeclinesFallsBackToTheDualSimplex) {
  // The selector can only choose; it cannot promise the interior point finishes. When the
  // chosen engine returns a failure that is not a limit, the dual simplex runs from scratch
  // and the message says so. The factor budget forced to zero makes the interior point
  // decline immediately on any model that reaches it; presolve is off so the model does,
  // and two entries per row give the normal equations a factor far beyond a budget of one.
  Model m = shaped_lp(kDualSimplexRowLimit, 200, 2 * kDualSimplexRowLimit);
  Options o = auto_options();
  o.set_bool("presolve", false);
  o.set_int("ipm_max_factor_nonzeros", 1);
  const Solution s = solve(m, o);
  EXPECT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.engine_rule, "size:ipm");
  EXPECT_NE(s.message.find("fell back to the dual simplex"), std::string::npos) << s.message;
}

}  // namespace
}  // namespace sankhya
