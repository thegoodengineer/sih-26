// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solve profiler (#285).
//
// The unit half builds region trees directly, with durations passed in as numbers, so nothing
// here depends on how fast the machine is. The solve half checks the two things a profiler
// can get wrong in ways that matter: that turning it on changes no answer, and that what it
// reports is the tree the solve actually walked.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "support/temp_file.hpp"
#include "util/profiler.hpp"

namespace sankhya {
namespace {

int find(const Profiler& p, const std::string& name) {
  for (std::size_t k = 0; k < p.regions().size(); ++k) {
    if (p.regions()[k].name == name) return static_cast<int>(k);
  }
  return -1;
}

// ---- The profiler on its own ---------------------------------------------------------------

TEST(Profiler, OffRecordsNothingAndANullProfilerIsSafe) {
  Profiler off(ProfileMode::kOff);
  {
    ProfileScope a(&off, "solve");
    ProfileScope b(nullptr, "anything");
  }
  off.count("iterations", 5);
  off.record("pricing", 1.0);
  EXPECT_TRUE(off.regions().empty());
  EXPECT_TRUE(off.counters().empty());
}

TEST(Profiler, ScopesNestAndSameNamedSiblingsAccumulate) {
  Profiler p(ProfileMode::kBasic);
  const int solve = p.enter("solve");
  for (int k = 0; k < 3; ++k) {
    const int engine = p.enter("engine");
    const int factor = p.enter("factor");
    p.leave(factor, 0.5);
    p.leave(engine, 2.0);
  }
  p.leave(solve, 7.0);

  ASSERT_EQ(p.regions().size(), 3u) << "one engine region entered three times, not three";
  const int engine = find(p, "engine");
  const int factor = find(p, "factor");
  EXPECT_EQ(p.regions()[static_cast<std::size_t>(engine)].calls, 3);
  EXPECT_DOUBLE_EQ(p.regions()[static_cast<std::size_t>(engine)].inclusive_seconds, 6.0);
  EXPECT_DOUBLE_EQ(p.exclusive_seconds(engine), 6.0 - 1.5);
  EXPECT_EQ(p.regions()[static_cast<std::size_t>(factor)].parent, engine);
  EXPECT_DOUBLE_EQ(p.exclusive_seconds(find(p, "solve")), 1.0);
}

TEST(Profiler, ExclusiveTimeNeverGoesNegative) {
  // A child measured by its own clock can come out a tick longer than its parent.
  Profiler p(ProfileMode::kBasic);
  const int parent = p.enter("parent");
  p.record("child", 1.0000001);
  p.leave(parent, 1.0);
  EXPECT_EQ(p.exclusive_seconds(parent), 0.0);
}

TEST(Profiler, ABasicProfilerSkipsDetailedScopes) {
  Profiler p(ProfileMode::kBasic);
  {
    ProfileScope phase(&p, "engine");
    ProfileScope inner(&p, "pricing", ProfileMode::kDetailed);
  }
  EXPECT_NE(find(p, "engine"), -1);
  EXPECT_EQ(find(p, "pricing"), -1);

  Profiler d(ProfileMode::kDetailed);
  {
    ProfileScope phase(&d, "engine");
    ProfileScope inner(&d, "pricing", ProfileMode::kDetailed);
  }
  EXPECT_NE(find(d, "pricing"), -1);
}

TEST(Profiler, RecordAttachesMeasuredTimeUnderTheOpenRegion) {
  Profiler p(ProfileMode::kDetailed);
  const int engine = p.enter("engine");
  p.record("pricing", 0.25, 100);
  p.record("pricing", 0.25, 50);
  p.leave(engine, 1.0);
  const int pricing = find(p, "pricing");
  ASSERT_NE(pricing, -1);
  EXPECT_EQ(p.regions()[static_cast<std::size_t>(pricing)].parent, engine);
  EXPECT_DOUBLE_EQ(p.regions()[static_cast<std::size_t>(pricing)].inclusive_seconds, 0.5);
  EXPECT_EQ(p.regions()[static_cast<std::size_t>(pricing)].calls, 150);
}

TEST(Profiler, MergeFoldsByPathAndSumsCounters) {
  // Two workers' profilers, as a parallel search would have: same paths, different amounts.
  Profiler a(ProfileMode::kBasic);
  Profiler b(ProfileMode::kBasic);
  for (Profiler* p : {&a, &b}) {
    const int solve = p->enter("solve");
    const int node = p->enter("node LP");
    p->leave(node, 1.0);
    p->leave(solve, 2.0);
    p->count("nodes", 10);
  }
  // A region only b has.
  const int extra = b.enter("heuristics");
  b.leave(extra, 0.5);

  a.merge(b);
  EXPECT_DOUBLE_EQ(a.regions()[static_cast<std::size_t>(find(a, "node LP"))].inclusive_seconds,
                   2.0);
  EXPECT_EQ(a.regions()[static_cast<std::size_t>(find(a, "solve"))].calls, 2);
  EXPECT_NE(find(a, "heuristics"), -1);
  ASSERT_EQ(a.counters().size(), 1u);
  EXPECT_EQ(a.counters()[0].value, 20);

  const std::size_t before = a.regions().size();
  a.merge(a);  // folding a tree into itself must not walk a vector it is growing
  EXPECT_EQ(a.regions().size(), before);
}

TEST(Profiler, BothReportsCarryTheSameTree) {
  Profiler p(ProfileMode::kBasic);
  const int solve = p.enter("solve");
  const int presolve = p.enter("presolve");
  p.leave(presolve, 0.1);
  p.leave(solve, 1.0);
  p.count("iterations", 42);

  const std::string text = p.format_text();
  EXPECT_NE(text.find("solve"), std::string::npos);
  EXPECT_NE(text.find("  presolve"), std::string::npos) << "children are indented";
  EXPECT_NE(text.find("42"), std::string::npos);

  const nlohmann::json blob = nlohmann::json::parse(p.format_json());
  EXPECT_EQ(blob["mode"], "basic");
  ASSERT_EQ(blob["regions"].size(), 2u);
  EXPECT_EQ(blob["regions"][1]["path"], "solve/presolve");
  EXPECT_EQ(blob["regions"][1]["parent_path"], "solve");
  EXPECT_DOUBLE_EQ(blob["regions"][0]["exclusive_seconds"].get<double>(), 0.9);
  EXPECT_EQ(blob["counters"]["iterations"], 42);
}

TEST(Profiler, ModesParseAndAnythingElseIsRefused) {
  ProfileMode mode = ProfileMode::kOff;
  EXPECT_TRUE(parse_profile_mode("detailed", &mode));
  EXPECT_EQ(mode, ProfileMode::kDetailed);
  EXPECT_FALSE(parse_profile_mode("verbose", &mode));
  Options options;
  std::string error;
  EXPECT_FALSE(options.set_from_string("profile", "everything", &error)) << "a closed set";
}

// ---- Through solve() -----------------------------------------------------------------------

Model small_lp() {
  // max 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18, as a minimisation.
  Model m;
  m.col_cost = {-3.0, -5.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {kInfinity, kInfinity};
  m.col_type = {VarType::kContinuous, VarType::kContinuous};
  m.matrix.reset(3, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(1, 1, 2.0);
  m.matrix.add_entry(2, 0, 3.0);
  m.matrix.add_entry(2, 1, 2.0);
  m.matrix.finalize();
  m.row_lower = {-kInfinity, -kInfinity, -kInfinity};
  m.row_upper = {4.0, 12.0, 18.0};
  m.hessian.reset(2, 2);
  m.hessian.finalize();
  return m;
}

Model small_milp() {
  Model m;
  const Index n = 14;
  const auto un = static_cast<std::size_t>(n);
  m.col_cost.assign(un, 0.0);
  m.col_lower.assign(un, 0.0);
  m.col_upper.assign(un, 1.0);
  m.col_type.assign(un, VarType::kInteger);
  m.matrix.reset(1, n);
  double total = 0.0;
  for (Index j = 0; j < n; ++j) {
    const double w = 17.0 + static_cast<double>((j * 29) % 41);
    m.col_cost[static_cast<std::size_t>(j)] = -(w + 3.0);
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

nlohmann::json profile_of(const Model& model, Options options, const char* mode,
                          Solution* out = nullptr) {
  testing::TempFile file("", ".json");
  options.set_bool("log_to_console", false);
  options.set_string("profile", mode);
  options.set_string("profile_out", file.path());
  const Solution solved = solve(model, options);
  if (out != nullptr) *out = solved;
  std::ifstream in(file.path());
  std::stringstream text;
  text << in.rdbuf();
  return text.str().empty() ? nlohmann::json() : nlohmann::json::parse(text.str());
}

bool has_path(const nlohmann::json& blob, const std::string& path) {
  for (const auto& r : blob["regions"]) {
    if (r["path"] == path) return true;
  }
  return false;
}

TEST(Profiler, ProfilingChangesNoAnswer) {
  for (const Model& model : {small_lp(), small_milp()}) {
    Options plain;
    plain.set_bool("log_to_console", false);
    const Solution off = solve(model, plain);
    for (const char* mode : {"basic", "detailed"}) {
      Solution on;
      (void)profile_of(model, Options{}, mode, &on);
      EXPECT_EQ(on.status, off.status) << mode;
      EXPECT_EQ(on.objective, off.objective) << mode << ": to the bit";
      EXPECT_EQ(on.iterations, off.iterations) << mode;
      EXPECT_EQ(on.nodes, off.nodes) << mode;
      EXPECT_EQ(on.col_value, off.col_value) << mode;
    }
  }
}

TEST(Profiler, ABasicProfileOfAnLpShowsTheSolvePhases) {
  Options options;
  options.set_string("algorithm", "dual-simplex");
  const nlohmann::json blob = profile_of(small_lp(), options, "basic");
  ASSERT_FALSE(blob.is_null()) << "profile_out was not written";
  EXPECT_TRUE(has_path(blob, "solve"));
  EXPECT_TRUE(has_path(blob, "solve/presolve"));
  EXPECT_TRUE(has_path(blob, "solve/verification"));
  EXPECT_TRUE(blob["counters"].contains("iterations"));
  for (const auto& r : blob["regions"]) {
    EXPECT_NE(r["path"].get<std::string>().find("solve"), std::string::npos)
        << "every region hangs under the solve";
    EXPECT_GE(r["inclusive_seconds"].get<double>(), 0.0);
  }
}

TEST(Profiler, ADetailedProfileOfAMilpShowsTheSearch) {
  Options options;
  options.set_bool("presolve", false);  // keep the search, not presolve, doing the work
  const nlohmann::json blob = profile_of(small_milp(), options, "detailed");
  ASSERT_FALSE(blob.is_null());
  EXPECT_TRUE(has_path(blob, "solve/engine/node LP")) << blob.dump(2);
  EXPECT_TRUE(has_path(blob, "solve/engine/branching")) << blob.dump(2);
  EXPECT_TRUE(blob["counters"].contains("nodes"));
  EXPECT_TRUE(blob["counters"].contains("nodes pruned"));
}

TEST(Profiler, ADetailedProfileOfTheInteriorPointShowsTheFactorization) {
  Options options;
  options.set_string("algorithm", "ipm");
  options.set_bool("presolve", false);
  const nlohmann::json blob = profile_of(small_lp(), options, "detailed");
  ASSERT_FALSE(blob.is_null());
  EXPECT_TRUE(has_path(blob, "solve/engine/factorization")) << blob.dump(2);
  EXPECT_TRUE(has_path(blob, "solve/engine/ordering")) << blob.dump(2);
}

TEST(Profiler, AnUnwritableProfileIsAWarningNotAFailedSolve) {
  Options options;
  options.set_bool("log_to_console", false);
  options.set_string("profile", "basic");
  options.set_string("profile_out", "no_such_directory/deeper/profile.json");
  const Solution solved = solve(small_lp(), options);
  EXPECT_EQ(solved.status, SolveStatus::kOptimal) << solved.message;
}

}  // namespace
}  // namespace sankhya
