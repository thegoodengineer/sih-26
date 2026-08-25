// SPDX-License-Identifier: Apache-2.0
// SANKHYA - option table tests.
//
// The registry is a single source of truth for three surfaces, so the tests that matter
// most are the structural ones: no duplicate names, every default inside its own declared
// range, and every string default among its own choices. Those catch a bad registry row at
// build time rather than as a mystified user report.

#include <algorithm>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya {
namespace {

TEST(Options, RegistryHasNoDuplicateNames) {
  std::set<std::string> seen;
  for (const OptionSpec& spec : Options::registry()) {
    EXPECT_TRUE(seen.insert(spec.name).second) << "duplicate option name: " << spec.name;
  }
  EXPECT_FALSE(Options::registry().empty());
}

TEST(Options, EveryDefaultIsWithinItsOwnDeclaredRange) {
  for (const OptionSpec& spec : Options::registry()) {
    if (spec.type == OptionType::Double) {
      const double v = std::get<double>(spec.default_value);
      EXPECT_GE(v, spec.min_value) << spec.name;
      EXPECT_LE(v, spec.max_value) << spec.name;
    } else if (spec.type == OptionType::Int) {
      const auto v = static_cast<double>(std::get<std::int64_t>(spec.default_value));
      EXPECT_GE(v, spec.min_value) << spec.name;
      EXPECT_LE(v, spec.max_value) << spec.name;
    } else if (spec.type == OptionType::String && !spec.choices.empty()) {
      const std::string& v = std::get<std::string>(spec.default_value);
      EXPECT_NE(std::find(spec.choices.begin(), spec.choices.end(), v), spec.choices.end())
          << spec.name << " default '" << v << "' is not among its own choices";
    }
  }
}

TEST(Options, EverySpecHasADescription) {
  for (const OptionSpec& spec : Options::registry()) {
    EXPECT_FALSE(spec.description.empty()) << spec.name << " has no description";
  }
}

TEST(Options, DeclaredTypeMatchesTheStoredDefault) {
  for (const OptionSpec& spec : Options::registry()) {
    switch (spec.type) {
      case OptionType::Bool:
        EXPECT_TRUE(std::holds_alternative<bool>(spec.default_value)) << spec.name;
        break;
      case OptionType::Int:
        EXPECT_TRUE(std::holds_alternative<std::int64_t>(spec.default_value)) << spec.name;
        break;
      case OptionType::Double:
        EXPECT_TRUE(std::holds_alternative<double>(spec.default_value)) << spec.name;
        break;
      case OptionType::String:
        EXPECT_TRUE(std::holds_alternative<std::string>(spec.default_value)) << spec.name;
        break;
    }
  }
}

TEST(Options, DefaultsComeFromTolerancesHeader) {
  // CLAUDE.md: one numerical source of truth. If someone re-types a literal into the
  // registry, this test is what catches the drift.
  const Options options;
  EXPECT_DOUBLE_EQ(options.get_double("primal_feasibility_tolerance"), tol::kPrimalFeasibility);
  EXPECT_DOUBLE_EQ(options.get_double("dual_feasibility_tolerance"), tol::kDualFeasibility);
  EXPECT_DOUBLE_EQ(options.get_double("integrality_tolerance"), tol::kIntegrality);
  EXPECT_DOUBLE_EQ(options.get_double("mip_relative_gap"), tol::kMipRelativeGap);
  EXPECT_DOUBLE_EQ(options.get_double("mip_absolute_gap"), tol::kMipAbsoluteGap);
  EXPECT_DOUBLE_EQ(options.get_double("pdhg_tolerance"), tol::kPdhgLoose);
}

TEST(Options, UnknownNameIsRejectedWithAMessage) {
  Options options;
  std::string error;
  EXPECT_FALSE(options.set_from_string("no_such_option", "1", &error));
  EXPECT_NE(error.find("no_such_option"), std::string::npos);
  EXPECT_FALSE(Options::exists("no_such_option"));
}

TEST(Options, BooleanSpellings) {
  Options options;
  std::string error;
  for (const char* text : {"1", "true", "TRUE", "on", "yes"}) {
    ASSERT_TRUE(options.set_from_string("presolve", text, &error)) << text << ": " << error;
    EXPECT_TRUE(options.get_bool("presolve")) << text;
  }
  for (const char* text : {"0", "false", "OFF", "no"}) {
    ASSERT_TRUE(options.set_from_string("presolve", text, &error)) << text << ": " << error;
    EXPECT_FALSE(options.get_bool("presolve")) << text;
  }
  EXPECT_FALSE(options.set_from_string("presolve", "maybe", &error));
}

TEST(Options, IntegerParsingRejectsTrailingGarbage) {
  Options options;
  std::string error;
  EXPECT_TRUE(options.set_from_string("threads", "8", &error));
  EXPECT_EQ(options.get_int("threads"), 8);

  EXPECT_FALSE(options.set_from_string("threads", "8x", &error));
  EXPECT_FALSE(options.set_from_string("threads", "", &error));
  EXPECT_FALSE(options.set_from_string("threads", "3.5", &error));
  // A rejected assignment must not have changed the stored value.
  EXPECT_EQ(options.get_int("threads"), 8);
}

TEST(Options, NumericRangeIsEnforced) {
  Options options;
  std::string error;
  EXPECT_FALSE(options.set_from_string("threads", "-1", &error));
  EXPECT_NE(error.find("threads"), std::string::npos);
  EXPECT_FALSE(options.set_from_string("mip_relative_gap", "2.0", &error));
  EXPECT_FALSE(options.set_from_string("primal_feasibility_tolerance", "1.0", &error));
  EXPECT_TRUE(options.set_from_string("primal_feasibility_tolerance", "1e-9", &error));
  EXPECT_DOUBLE_EQ(options.get_double("primal_feasibility_tolerance"), 1e-9);
}

TEST(Options, DoubleParsingAcceptsScientificNotation) {
  Options options;
  std::string error;
  ASSERT_TRUE(options.set_from_string("time_limit", "6.5e1", &error)) << error;
  EXPECT_DOUBLE_EQ(options.get_double("time_limit"), 65.0);
  EXPECT_FALSE(options.set_from_string("time_limit", "sixty", &error));
}

TEST(Options, StringChoicesAreEnforcedAndCaseInsensitive) {
  Options options;
  std::string error;
  ASSERT_TRUE(options.set_from_string("algorithm", "PDHG", &error)) << error;
  EXPECT_EQ(options.get_string("algorithm"), "pdhg");
  EXPECT_FALSE(options.set_from_string("algorithm", "quantum", &error));
  EXPECT_NE(error.find("algorithm"), std::string::npos);
  EXPECT_EQ(options.get_string("algorithm"), "pdhg");
}

TEST(Options, NameLookupIgnoresCaseAndSurroundingSpace) {
  Options options;
  std::string error;
  ASSERT_TRUE(options.set_from_string("  Presolve ", " false ", &error)) << error;
  EXPECT_FALSE(options.get_bool("presolve"));
}

TEST(Options, ModifiedTracking) {
  Options options;
  EXPECT_TRUE(options.modified_names().empty());
  EXPECT_FALSE(options.is_modified("threads"));

  options.set_int("threads", 4);
  EXPECT_TRUE(options.is_modified("threads"));
  const std::vector<std::string> modified = options.modified_names();
  ASSERT_EQ(modified.size(), 1u);
  EXPECT_EQ(modified[0], "threads");

  // Setting a value back to its default clears the modified flag.
  options.set_int("threads", 1);
  EXPECT_FALSE(options.is_modified("threads"));
  EXPECT_TRUE(options.modified_names().empty());
}

TEST(Options, ValueAsStringRoundTrips) {
  Options options;
  std::string error;
  ASSERT_TRUE(options.set_from_string("threads", "12", &error));
  ASSERT_TRUE(options.set_from_string("algorithm", "ipm", &error));
  ASSERT_TRUE(options.set_from_string("presolve", "false", &error));

  EXPECT_EQ(options.value_as_string("threads"), "12");
  EXPECT_EQ(options.value_as_string("algorithm"), "ipm");
  EXPECT_EQ(options.value_as_string("presolve"), "false");

  Options reloaded;
  for (const std::string& name : options.modified_names()) {
    ASSERT_TRUE(reloaded.set_from_string(name, options.value_as_string(name), &error))
        << name << ": " << error;
  }
  EXPECT_EQ(reloaded.get_int("threads"), 12);
  EXPECT_EQ(reloaded.get_string("algorithm"), "ipm");
  EXPECT_FALSE(reloaded.get_bool("presolve"));
}

TEST(Options, FindSpecReturnsNullForUnknown) {
  EXPECT_EQ(Options::find_spec("not_an_option"), nullptr);
  ASSERT_NE(Options::find_spec("time_limit"), nullptr);
  EXPECT_EQ(Options::find_spec("time_limit")->type, OptionType::Double);
}

TEST(Options, InstancesAreIndependent) {
  Options a;
  Options b;
  a.set_int("threads", 16);
  EXPECT_EQ(b.get_int("threads"), 1);
}

}  // namespace
}  // namespace sankhya
