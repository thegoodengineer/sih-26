// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch-and-bound checkpoint and resume (#287).
//
// The acceptance test is the issue's own: an uninterrupted solve against solve -> checkpoint
// at a node limit -> resume -> completion, which must reach the same optimum. Then every way
// a checkpoint can be wrong for the solve reading it - corrupt, edited, another format
// version, another model, another integrality tolerance - must be refused before anything
// runs, and a write that fails must leave the previous checkpoint exactly as it was.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "mip/checkpoint.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"

namespace sankhya {
namespace {

/// A strongly correlated knapsack: a real tree, a few hundred nodes, well under a second.
Model knapsack(int columns, int seed) {
  Model m;
  const auto n = static_cast<Index>(columns);
  const auto un = static_cast<std::size_t>(n);
  m.sense = ObjSense::kMaximize;
  m.col_cost.resize(un);
  m.col_lower.assign(un, 0.0);
  m.col_upper.assign(un, 1.0);
  m.col_type.assign(un, VarType::kInteger);
  m.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double w = 20.0 + static_cast<double>((j * 37 + seed * 11) % 51);
    m.col_cost[static_cast<std::size_t>(j)] = w + 10.0;
    m.matrix.add_entry(0, j, w);
    total += w;
  }
  m.matrix.finalize();
  m.row_lower = {-kInfinity};
  m.row_upper = {std::floor(total / 2.0)};
  m.hessian.reset(n, n);
  m.hessian.finalize();
  return m;
}

Options base_options() {
  Options o;
  o.set_bool("log_to_console", false);
  return o;
}

std::string read_all(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

void write_all(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
}

TEST(Checkpoint, AResumedSearchReachesTheUninterruptedOptimum) {
  // The first leg stops EARLY, at a thirtieth of the uninterrupted node count, so that on some
  // seeds its incumbent is still short of the optimum and the resumed search has to FIND the
  // rest. Mutation testing showed why that matters: with the first leg stopped at a third,
  // every incumbent was already optimal, and a resume that silently dropped half its open
  // nodes still reported the right objective. The count of seeds that needed improving is
  // asserted, so the test cannot drift back into only checking a proof.
  int improved_after_resume = 0;
  for (int seed = 0; seed < 12; ++seed) {
    const Model model = knapsack(18, seed);
    Options plain = base_options();
    plain.set_bool("presolve", false);
    const Solution whole = solve(model, plain);
    ASSERT_EQ(whole.status, SolveStatus::kOptimal) << whole.message;
    ASSERT_GT(whole.nodes, 20) << "the model should need a real tree";

    testing::TempFile file("", ".chk");
    Options first = plain;
    first.set_string("checkpoint", file.path());
    first.set_int("node_limit", std::max<Count>(2, whole.nodes / 30));
    const Solution stopped = solve(model, first);
    ASSERT_NE(stopped.status, SolveStatus::kOptimal)
        << "seed " << seed << ": the first leg must stop short: " << stopped.message;
    ASSERT_FALSE(read_all(file.path()).empty())
        << "a limit-stopped search writes its checkpoint";
    if (!(std::fabs(stopped.objective - whole.objective) <= 1e-9)) ++improved_after_resume;

    Options second = plain;
    second.set_string("resume", file.path());
    const Solution resumed = solve(model, second);
    ASSERT_EQ(resumed.status, SolveStatus::kOptimal)
        << "seed " << seed << ": " << resumed.message;
    EXPECT_NEAR(resumed.objective, whole.objective, 1e-9) << "seed " << seed;
    EXPECT_GE(resumed.nodes, stopped.nodes) << "the node count continues from the checkpoint";
  }
  EXPECT_GT(improved_after_resume, 2)
      << "too few seeds left the resumed search anything to find; the test would only check a "
         "proof";
}

TEST(Checkpoint, PeriodicCheckpointsAreWrittenOnTheNodeSchedule) {
  const Model model = knapsack(16, 1);
  testing::TempFile file("", ".chk");
  Options options = base_options();
  options.set_string("checkpoint", file.path());
  options.set_int("checkpoint_nodes", 10);
  const Solution solved = solve(model, options);
  ASSERT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
  mip::TreeCheckpoint c;
  std::string error;
  ASSERT_TRUE(mip::read_checkpoint(file.path(), &c, &error)) << error;
  EXPECT_GT(c.nodes_explored, 0);
  EXPECT_EQ(c.nodes_explored % 10, 0) << "written on the schedule";
}

class RefusedCheckpoint : public ::testing::Test {
 protected:
  void SetUp() override {
    Options first = base_options();
    first.set_string("checkpoint", file.path());
    first.set_int("node_limit", 10);
    (void)solve(model, first);
    original = read_all(file.path());
    ASSERT_FALSE(original.empty());
  }
  std::string resume_message(const Model& m, Options options = base_options()) {
    options.set_string("resume", file.path());
    const Solution s = solve(m, options);
    EXPECT_EQ(s.status, SolveStatus::kNotSolved) << s.message;
    return s.message;
  }
  Model model = knapsack(16, 2);
  testing::TempFile file{"", ".chk"};
  std::string original;
};

TEST_F(RefusedCheckpoint, ACorruptedFileIsRefused) {
  std::string damaged = original;
  const std::size_t key = damaged.find("\"nodes_explored\"");
  ASSERT_NE(key, std::string::npos);
  const std::size_t digit = damaged.find_first_of("0123456789", key);
  ASSERT_NE(digit, std::string::npos);
  damaged[digit] = damaged[digit] == '1' ? '2' : '1';  // one digit of the state, still JSON
  write_all(file.path(), damaged);
  EXPECT_NE(resume_message(model).find("checksum"), std::string::npos);

  write_all(file.path(), original.substr(0, original.size() / 2));  // truncated
  EXPECT_NE(resume_message(model).find("does not parse"), std::string::npos);
}

TEST_F(RefusedCheckpoint, AnotherFormatVersionIsRefused) {
  std::string edited = original;
  const std::size_t at = edited.find("\"version\": 1");
  ASSERT_NE(at, std::string::npos) << edited.substr(0, 200);
  edited.replace(at, 12, "\"version\": 2");
  write_all(file.path(), edited);
  EXPECT_NE(resume_message(model).find("version"), std::string::npos);
}

TEST_F(RefusedCheckpoint, AnotherModelIsRefused) {
  EXPECT_NE(resume_message(knapsack(16, 3)).find("different model"), std::string::npos);
}

TEST(Checkpoint, ADifferentPresolveSettingIsRefusedWhenItChangesTheTreesModel) {
  // The tree sees the model AFTER presolve. On a model presolve leaves alone, resuming with
  // presolve off is the same search and is rightly accepted; on one it reduces, the two
  // searches run on different models and the fingerprint refuses the mix. A column fixed at
  // 1 is removed by presolve, which makes the difference.
  Model model = knapsack(16, 2);
  model.col_cost.push_back(5.0);
  model.col_lower.push_back(1.0);
  model.col_upper.push_back(1.0);
  model.col_type.push_back(VarType::kInteger);
  Model grown;
  grown = model;
  grown.matrix.reset(1, model.num_cols());
  for (Index j = 0; j + 1 < model.num_cols(); ++j) {
    grown.matrix.add_entry(0, j, knapsack(16, 2).matrix.at(0, j));
  }
  grown.matrix.finalize();
  grown.hessian.reset(model.num_cols(), model.num_cols());
  grown.hessian.finalize();

  testing::TempFile file("", ".chk");
  Options first = base_options();
  first.set_string("checkpoint", file.path());
  first.set_int("node_limit", 8);
  (void)solve(grown, first);
  ASSERT_FALSE(read_all(file.path()).empty());

  Options second = base_options();
  second.set_string("resume", file.path());
  second.set_bool("presolve", false);
  const Solution s = solve(grown, second);
  EXPECT_EQ(s.status, SolveStatus::kNotSolved) << s.message;
}

TEST_F(RefusedCheckpoint, AnotherIntegralityToleranceIsRefused) {
  Options other = base_options();
  other.set_double("integrality_tolerance", 1e-4);
  EXPECT_NE(resume_message(model, other).find("integrality_tolerance"), std::string::npos);
}

TEST(Checkpoint, AFailedWriteLeavesThePreviousCheckpointIntact) {
  const Model model = knapsack(16, 0);
  testing::TempFile file("", ".chk");
  Options first = base_options();
  first.set_string("checkpoint", file.path());
  first.set_int("node_limit", 10);
  (void)solve(model, first);
  const std::string before = read_all(file.path());
  ASSERT_FALSE(before.empty());

  // Make the temporary name a DIRECTORY, so the next write cannot open it.
  const std::string temporary = file.path() + ".tmp";
  std::filesystem::create_directory(temporary);
  Options second = base_options();
  second.set_string("checkpoint", file.path());
  second.set_int("node_limit", 20);
  const Solution s = solve(model, second);
  EXPECT_NE(s.status, SolveStatus::kModelError)
      << "a failed checkpoint does not fail the solve";
  EXPECT_EQ(read_all(file.path()), before) << "the previous checkpoint must be untouched";
  std::filesystem::remove(temporary);
}

TEST(Checkpoint, TheFormatRoundTripsExactly) {
  mip::TreeCheckpoint c;
  c.model_fingerprint = "0123456789abcdef";
  c.problem_class = "milp";
  c.num_rows = 3;
  c.num_cols = 2;
  c.integrality_tolerance = 1e-6;
  c.have_incumbent = true;
  c.incumbent_x = {0.1, 1.0 / 3.0};
  c.nodes_explored = 42;
  c.nodes_pruned = 7;
  c.pseudo_down_sum = {0.5, 1e-300};
  c.pseudo_up_sum = {2.0 / 7.0, 0.0};
  c.pseudo_down_count = {1, 2};
  c.pseudo_up_count = {3, 4};
  c.open.push_back({{{1, true, 0.0}, {0, false, 1.0}}, -kInfinity, 2, 3.25});
  testing::TempFile file("", ".chk");
  std::string error;
  ASSERT_TRUE(mip::write_checkpoint(file.path(), c, &error)) << error;
  mip::TreeCheckpoint back;
  ASSERT_TRUE(mip::read_checkpoint(file.path(), &back, &error)) << error;
  EXPECT_EQ(back.incumbent_x, c.incumbent_x) << "doubles round-trip to the bit";
  EXPECT_EQ(back.pseudo_down_sum, c.pseudo_down_sum);
  EXPECT_EQ(back.pseudo_up_sum, c.pseudo_up_sum);
  ASSERT_EQ(back.open.size(), 1u);
  EXPECT_TRUE(std::isinf(back.open[0].bound) && back.open[0].bound < 0)
      << "an infinity survives";
  EXPECT_EQ(back.open[0].domain.size(), 2u);
  EXPECT_EQ(back.nodes_explored, 42);
}

}  // namespace
}  // namespace sankhya
