// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover from the interior point to a vertex (#219).
//
// THE CLAIM UNDER TEST IS THE VERTEX, not the guess. crossover_guess() is a heuristic and is
// checked only for its contract - m basic entries, every nonbasic one on a bound it has; the
// answer is checked the way any simplex answer is: optimal status, the published objective,
// exactly m basic entries in the reported basis, and every nonbasic entry sitting on its
// bound within the feasibility tolerance. A degenerate model - one where the interior point
// stops at the analytic centre of a whole optimal face - is where a crossover earns its
// keep, so that is the first model.

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex/crossover.hpp"

namespace sankhya {
namespace {

Options ipm_options(bool crossover) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("algorithm", "ipm");
  options.set_bool("crossover", crossover);
  // Presolve off: the vertex checks below count basic entries against m, and postsolve's
  // status vectors currently carry more than m basic entries (filed as its own issue), which
  // is a property of postsolve and not of the vertex crossover reaches.
  options.set_bool("presolve", false);
  return options;
}

Model make_lp(const std::vector<std::vector<double>>& rows,
              const std::vector<double>& row_lower, const std::vector<double>& row_upper,
              const std::vector<double>& cost, const std::vector<double>& col_lower,
              const std::vector<double>& col_upper) {
  Model model;
  const auto n = static_cast<Index>(cost.size());
  const auto m = static_cast<Index>(rows.size());
  model.col_cost = cost;
  model.col_lower = col_lower;
  model.col_upper = col_upper;
  model.col_type.assign(static_cast<std::size_t>(n), VarType::kContinuous);
  model.row_lower = row_lower;
  model.row_upper = row_upper;
  model.matrix.reset(m, n);
  for (Index i = 0; i < m; ++i) {
    for (Index j = 0; j < n; ++j) {
      const double v = rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
      if (v != 0.0) model.matrix.add_entry(i, j, v);
    }
  }
  model.matrix.finalize();
  model.hessian.reset(n, n);
  model.hessian.finalize();
  return model;
}

/// A vertex: exactly m basic entries, and every nonbasic entry on the bound its status names.
void expect_a_vertex(const Model& model, const Solution& s) {
  ASSERT_EQ(static_cast<Index>(s.col_status.size()), model.num_cols()) << s.message;
  ASSERT_EQ(static_cast<Index>(s.row_status.size()), model.num_rows()) << s.message;
  Index basic = 0;
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double x = s.col_value[u];
    switch (s.col_status[u]) {
      case BasisStatus::kBasic: ++basic; break;
      case BasisStatus::kAtLower:
        EXPECT_NEAR(x, model.col_lower[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kAtUpper:
        EXPECT_NEAR(x, model.col_upper[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kFixed:
        EXPECT_NEAR(x, model.col_lower[u], tol::kPrimalFeasibility) << "column " << j;
        break;
      case BasisStatus::kNonbasicFree: EXPECT_NEAR(x, 0.0, tol::kPrimalFeasibility); break;
      case BasisStatus::kUnknown: ADD_FAILURE() << "column " << j << " has no status"; break;
    }
  }
  for (Index i = 0; i < model.num_rows(); ++i) {
    if (s.row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic) ++basic;
  }
  EXPECT_EQ(basic, model.num_rows()) << "a basis has exactly m basic entries";
}

TEST(Crossover, ADegenerateModelEndsAtAVertexNotAtTheAnalyticCentre) {
  // min x1 + x2 + x3 subject to x1 + x2 + x3 >= 3, each x in [0, 3]: every point of the
  // face x1 + x2 + x3 = 3 is optimal. The interior point stops at the centre (1, 1, 1);
  // a vertex has two coordinates at a bound.
  const Model model = make_lp({{1.0, 1.0, 1.0}}, {3.0}, {kInfinity}, {1.0, 1.0, 1.0},
                              {0.0, 0.0, 0.0}, {3.0, 3.0, 3.0});
  const Solution interior = solve(model, ipm_options(false));
  ASSERT_EQ(interior.status, SolveStatus::kOptimal) << interior.message;
  for (const BasisStatus status : interior.col_status) {
    EXPECT_EQ(status, BasisStatus::kUnknown) << "the interior point reports no basis";
  }

  const CrossoverGuess guess = crossover_guess(model, interior);
  EXPECT_EQ(guess.basic, model.num_rows());

  const Solution vertex = solve(model, ipm_options(true));
  ASSERT_EQ(vertex.status, SolveStatus::kOptimal) << vertex.message;
  EXPECT_NEAR(vertex.objective, 3.0, 1e-7);
  EXPECT_NE(vertex.algorithm.find("crossover"), std::string::npos) << vertex.algorithm;
  EXPECT_NE(vertex.message.find("crossover:"), std::string::npos) << vertex.message;
  expect_a_vertex(model, vertex);
  Index at_bound = 0;
  for (const double x : vertex.col_value) {
    if (std::fabs(x) <= tol::kPrimalFeasibility ||
        std::fabs(x - 3.0) <= tol::kPrimalFeasibility) {
      ++at_bound;
    }
  }
  EXPECT_GE(at_bound, 2) << "a vertex of this face has at least two coordinates at a bound";
}

TEST(Crossover, OffLeavesTheInteriorPointsAnswerAlone) {
  const Model model = make_lp({{1.0, 1.0}, {1.0, -1.0}}, {1.0, -kInfinity}, {kInfinity, 1.0},
                              {1.0, 2.0}, {0.0, 0.0}, {kInfinity, kInfinity});
  const Solution s = solve(model, ipm_options(false));
  ASSERT_EQ(s.status, SolveStatus::kOptimal) << s.message;
  EXPECT_EQ(s.algorithm.find("crossover"), std::string::npos) << s.algorithm;
  for (const BasisStatus status : s.col_status) EXPECT_EQ(status, BasisStatus::kUnknown);
}

TEST(Crossover, NetlibInstancesReachAVertexAtThePublishedObjective) {
  // The committed small set, through ipm + crossover: the objective the published table
  // gives, with a basis, from a start that is already optimal to 1e-8. The pivot count is
  // in the message; the claim that it is small is what the Netlib CSV records (#219).
  struct Instance {
    const char* name;
    double published;
  };
  const Instance instances[] = {{"afiro", -464.75314286},   {"sc50b", -70.0},
                                {"share2b", -415.73224074}, {"stocfor1", -41131.976219},
                                {"adlittle", 225494.96316}, {"israel", -896644.82186}};
  for (const Instance& instance : instances) {
    Model model;
    const std::string path =
        (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() /
         "data/netlib" / (std::string(instance.name) + ".mps"))
            .string();
    const io::ReadResult read = io::read_model(path, &model);
    ASSERT_TRUE(read.ok) << path << ": " << read.error;
    const Solution s = solve(model, ipm_options(true));
    ASSERT_EQ(s.status, SolveStatus::kOptimal) << instance.name << ": " << s.message;
    EXPECT_NEAR(s.objective, instance.published,
                1e-6 * std::max(1.0, std::fabs(instance.published)))
        << instance.name;
    EXPECT_NE(s.algorithm.find("crossover"), std::string::npos)
        << instance.name << ": " << s.message;
    expect_a_vertex(model, s);
  }
}

}  // namespace
}  // namespace sankhya
