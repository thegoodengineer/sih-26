// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MPS reader tests.
//
// These are the tests CLAUDE.md says must never be cut. RANGES and BOUNDS are the two
// places in the MPS format where a plausible misreading yields a well-formed model of a
// DIFFERENT problem, which then solves cleanly and reports a confident wrong optimum. Every
// row type crossed with every range sign, and every bound type including the two whose
// conventions are counter-intuitive, is pinned down here by construction rather than by
// reference to any existing reader.

#include <cstdio>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#ifdef SANKHYA_WITH_ZLIB
#include <zlib.h>
#endif

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"

#include "support/temp_file.hpp"

namespace sankhya {
namespace {

using testing::TempFile;

[[nodiscard]] Model parse_or_fail(const std::string& text) {
  const TempFile file(text);
  Model model;
  const io::ReadResult result = io::read_mps(file.path(), &model);
  EXPECT_TRUE(result.ok) << result.error;
  return model;
}

[[nodiscard]] std::string parse_expecting_failure(const std::string& text) {
  const TempFile file(text);
  Model model;
  const io::ReadResult result = io::read_mps(file.path(), &model);
  EXPECT_FALSE(result.ok) << "expected this file to be rejected";
  return result.error;
}

/// Index of a row by name, or -1.
[[nodiscard]] Index row_of(const Model& model, const std::string& name) {
  for (std::size_t i = 0; i < model.row_names.size(); ++i) {
    if (model.row_names[i] == name) return static_cast<Index>(i);
  }
  return -1;
}

[[nodiscard]] Index col_of(const Model& model, const std::string& name) {
  for (std::size_t j = 0; j < model.col_names.size(); ++j) {
    if (model.col_names[j] == name) return static_cast<Index>(j);
  }
  return -1;
}

// =========================================================================================
// Baseline
// =========================================================================================

TEST(MpsReader, ReadsASmallCompleteModel) {
  const Model model = parse_or_fail(
      "* a comment line, ignored\n"
      "NAME          TESTLP\n"
      "ROWS\n"
      " N  COST\n"
      " L  LIM1\n"
      " G  LIM2\n"
      " E  MYEQN\n"
      "COLUMNS\n"
      "    XONE      COST         1.0   LIM1         1.0\n"
      "    XONE      LIM2         1.0\n"
      "    YTWO      COST         2.0   LIM1         1.0\n"
      "    YTWO      MYEQN       -1.0\n"
      "    ZTHREE    COST         3.0   LIM2         1.0\n"
      "    ZTHREE    MYEQN        1.0\n"
      "RHS\n"
      "    RHS       LIM1         4.0   LIM2         1.0\n"
      "    RHS       MYEQN        7.0\n"
      "ENDATA\n");

  EXPECT_EQ(model.name, "TESTLP");
  EXPECT_EQ(model.num_rows(), 3);
  EXPECT_EQ(model.num_cols(), 3);
  EXPECT_EQ(model.num_nonzeros(), 6);
  EXPECT_EQ(model.sense, ObjSense::kMinimize);

  EXPECT_DOUBLE_EQ(model.col_cost[0], 1.0);
  EXPECT_DOUBLE_EQ(model.col_cost[1], 2.0);
  EXPECT_DOUBLE_EQ(model.col_cost[2], 3.0);

  const Index lim1 = row_of(model, "LIM1");
  const Index lim2 = row_of(model, "LIM2");
  const Index eqn = row_of(model, "MYEQN");
  ASSERT_GE(lim1, 0);
  ASSERT_GE(lim2, 0);
  ASSERT_GE(eqn, 0);

  EXPECT_TRUE(is_infinite(model.row_lower[static_cast<std::size_t>(lim1)]));
  EXPECT_DOUBLE_EQ(model.row_upper[static_cast<std::size_t>(lim1)], 4.0);
  EXPECT_DOUBLE_EQ(model.row_lower[static_cast<std::size_t>(lim2)], 1.0);
  EXPECT_TRUE(is_infinite(model.row_upper[static_cast<std::size_t>(lim2)]));
  EXPECT_DOUBLE_EQ(model.row_lower[static_cast<std::size_t>(eqn)], 7.0);
  EXPECT_DOUBLE_EQ(model.row_upper[static_cast<std::size_t>(eqn)], 7.0);

  // Default column bounds are [0, +inf) with no BOUNDS section.
  for (Index j = 0; j < model.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    EXPECT_DOUBLE_EQ(model.col_lower[u], 0.0);
    EXPECT_TRUE(is_infinite(model.col_upper[u]));
    EXPECT_EQ(model.col_type[u], VarType::kContinuous);
  }

  EXPECT_DOUBLE_EQ(model.matrix.at(lim1, 0), 1.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(eqn, 1), -1.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(lim2, 2), 1.0);
}

// =========================================================================================
// RANGES - one test per (row type, sign of R). This is the table from the IBM spec.
// =========================================================================================

/// Build a one-row model of the given type with RHS 10 and the given RANGES value.
[[nodiscard]] Model ranged_row(const char* type, const char* range_value) {
  return parse_or_fail(std::string("NAME          RANGETEST\n"
                                   "ROWS\n"
                                   " N  COST\n"
                                   " ") +
                       type +
                       "  R1\n"
                       "COLUMNS\n"
                       "    X         COST         1.0   R1           1.0\n"
                       "RHS\n"
                       "    RHS       R1          10.0\n"
                       "RANGES\n"
                       "    RNG       R1          " +
                       range_value +
                       "\n"
                       "ENDATA\n");
}

TEST(MpsRanges, GreaterRowWithPositiveRange) {
  // G row: rhs is the lower bound, and the row extends UPWARD by |R|.
  const Model model = ranged_row("G", "4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 10.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 14.0);
}

TEST(MpsRanges, GreaterRowWithNegativeRangeUsesTheMagnitude) {
  // The sign is IGNORED on a G row. A reader that propagates it produces [10, 6], an empty
  // interval, and reports the model infeasible.
  const Model model = ranged_row("G", "-4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 10.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 14.0);
}

TEST(MpsRanges, LessRowWithPositiveRange) {
  // L row: rhs is the upper bound, and the row extends DOWNWARD by |R|.
  const Model model = ranged_row("L", "4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 6.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}

TEST(MpsRanges, LessRowWithNegativeRangeUsesTheMagnitude) {
  const Model model = ranged_row("L", "-4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 6.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}

TEST(MpsRanges, EqualityRowWithPositiveRangeExtendsUpward) {
  // The E row is the only case where the SIGN decides which side moves. Positive R widens
  // upward from the rhs.
  const Model model = ranged_row("E", "4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 10.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 14.0);
}

TEST(MpsRanges, EqualityRowWithNegativeRangeExtendsDownward) {
  // ... and negative R widens downward. Getting this backwards is the single most common
  // MPS reader bug, and it is invisible: the model still solves.
  const Model model = ranged_row("E", "-4.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 6.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}

TEST(MpsRanges, EqualityRowWithZeroRangeStaysAnEquality) {
  const Model model = ranged_row("E", "0.0");
  EXPECT_DOUBLE_EQ(model.row_lower[0], 10.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}

TEST(MpsRanges, RangeOnTheObjectiveRowIsRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RANGES\n"
      "    RNG       COST         4.0\n"
      "ENDATA\n");
  EXPECT_NE(error.find("objective row"), std::string::npos) << error;
}

// =========================================================================================
// BOUNDS - one test per type, plus the two conventions that surprise people.
// =========================================================================================

[[nodiscard]] Model bounded_column(const std::string& bounds_lines) {
  return parse_or_fail(
      "NAME          BOUNDTEST\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "    Y         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "BOUNDS\n" +
      bounds_lines + "ENDATA\n");
}

TEST(MpsBounds, UpWithAPositiveValue) {
  const Model model = bounded_column(" UP BND       X            5.0\n");
  const Index x = col_of(model, "X");
  ASSERT_GE(x, 0);
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 5.0);
}

TEST(MpsBounds, UpWithANegativeValueImpliesAFreeLowerBound) {
  // THE TRAP. [0, -5] is empty. The established convention is that a negative UP bound on a
  // continuous column whose lower bound is still the implicit 0 means the modeller wants a
  // negative variable, so the lower bound becomes -inf. A reader that skips this reports a
  // perfectly good model as infeasible.
  const Model model = bounded_column(" UP BND       X           -5.0\n");
  const Index x = col_of(model, "X");
  ASSERT_GE(x, 0);
  EXPECT_TRUE(is_infinite(model.col_lower[static_cast<std::size_t>(x)]));
  EXPECT_LT(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], -5.0);
}

TEST(MpsBounds, UpWithANegativeValueAfterAnExplicitLowerBoundLeavesItAlone) {
  // The convention only fires when the lower bound is still the DEFAULT zero. Once the file
  // has stated a lower bound, the modeller has said what they meant.
  const Model model = bounded_column(
      " LO BND       X          -20.0\n"
      " UP BND       X           -5.0\n");
  const Index x = col_of(model, "X");
  ASSERT_GE(x, 0);
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], -20.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], -5.0);
}

TEST(MpsBounds, LoSetsTheLowerBound) {
  const Model model = bounded_column(" LO BND       X            2.5\n");
  const Index x = col_of(model, "X");
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 2.5);
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(x)]));
}

TEST(MpsBounds, FxFixesTheColumn) {
  const Model model = bounded_column(" FX BND       X            3.0\n");
  const Index x = col_of(model, "X");
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 3.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 3.0);
  EXPECT_TRUE(model.is_fixed_column(x));
}

TEST(MpsBounds, FrMakesTheColumnFree) {
  const Model model = bounded_column(" FR BND       X\n");
  const Index x = col_of(model, "X");
  EXPECT_TRUE(is_infinite(model.col_lower[static_cast<std::size_t>(x)]));
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(x)]));
  EXPECT_LT(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_GT(model.col_upper[static_cast<std::size_t>(x)], 0.0);
}

TEST(MpsBounds, MiSetsOnlyTheLowerBound) {
  // Obsolete readers also forced the upper bound to zero. That would silently remove the
  // whole positive half of the feasible region.
  const Model model = bounded_column(" MI BND       X\n");
  const Index x = col_of(model, "X");
  EXPECT_TRUE(is_infinite(model.col_lower[static_cast<std::size_t>(x)]));
  EXPECT_LT(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(x)]));
  EXPECT_GT(model.col_upper[static_cast<std::size_t>(x)], 0.0);
}

TEST(MpsBounds, PlRestoresAnInfiniteUpperBound) {
  const Model model = bounded_column(
      " UP BND       X            5.0\n"
      " PL BND       X\n");
  const Index x = col_of(model, "X");
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(x)]));
}

TEST(MpsBounds, BvMakesABinaryColumn) {
  const Model model = bounded_column(" BV BND       X\n");
  const Index x = col_of(model, "X");
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 0.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 1.0);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(x)], VarType::kInteger);
}

TEST(MpsBounds, LiAndUiMakeTheColumnInteger) {
  const Model model = bounded_column(
      " LI BND       X            2.0\n"
      " UI BND       X            9.0\n");
  const Index x = col_of(model, "X");
  EXPECT_DOUBLE_EQ(model.col_lower[static_cast<std::size_t>(x)], 2.0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 9.0);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(x)], VarType::kInteger);
}

TEST(MpsBounds, SemiContinuousIsRejectedRatherThanMisread) {
  const std::string error = parse_expecting_failure(
      "NAME          SC\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "BOUNDS\n"
      " SC BND       X            5.0\n"
      "ENDATA\n");
  EXPECT_NE(error.find("semi-continuous"), std::string::npos) << error;
}

TEST(MpsBounds, TheVectorNameMayBeOmitted) {
  const Model model = bounded_column(" UP X            5.0\n");
  const Index x = col_of(model, "X");
  ASSERT_GE(x, 0);
  EXPECT_DOUBLE_EQ(model.col_upper[static_cast<std::size_t>(x)], 5.0);
}

TEST(MpsBounds, BoundOnAnUnknownColumnIsRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      "COLUMNS\n"
      "    X         COST         1.0\n"
      "BOUNDS\n"
      " UP BND       NOSUCH       5.0\n"
      "ENDATA\n");
  EXPECT_NE(error.find("NOSUCH"), std::string::npos) << error;
}

// =========================================================================================
// Integrality markers, objective sense and the objective constant
// =========================================================================================

TEST(MpsReader, MarkerRecordsToggleIntegrality) {
  const Model model = parse_or_fail(
      "NAME          MARK\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    C1        COST         1.0   R1           1.0\n"
      "    M1        'MARKER'                 'INTORG'\n"
      "    I1        COST         2.0   R1           1.0\n"
      "    I2        COST         3.0   R1           1.0\n"
      "    M2        'MARKER'                 'INTEND'\n"
      "    C2        COST         4.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");

  EXPECT_EQ(model.num_cols(), 4);
  EXPECT_EQ(model.num_integer_columns(), 2);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "C1"))],
            VarType::kContinuous);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "I1"))], VarType::kInteger);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "I2"))], VarType::kInteger);
  EXPECT_EQ(model.col_type[static_cast<std::size_t>(col_of(model, "C2"))],
            VarType::kContinuous);

  // An INTORG column with no BOUNDS entry keeps the [0, +inf) default. Forcing it to [0, 1]
  // would turn a general integer model into a binary one.
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(col_of(model, "I1"))]));
}

TEST(MpsReader, ObjsenseOnTheSameLine) {
  const Model model = parse_or_fail(
      "NAME          MAXTEST\n"
      "OBJSENSE      MAX\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");
  EXPECT_EQ(model.sense, ObjSense::kMaximize);
  EXPECT_DOUBLE_EQ(model.sense_multiplier(), -1.0);
}

TEST(MpsReader, ObjsenseAsItsOwnSection) {
  const Model model = parse_or_fail(
      "NAME          MAXTEST\n"
      "OBJSENSE\n"
      "    MAXIMIZE\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");
  EXPECT_EQ(model.sense, ObjSense::kMaximize);
}

TEST(MpsReader, RhsOnTheObjectiveRowIsNegatedIntoTheOffset) {
  // The MPS convention: the RHS entry on the objective row is the NEGATIVE of the constant
  // term. A reader that copies it through shifts every reported objective by 2c.
  const Model model = parse_or_fail(
      "NAME          OFFSET\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1           2.0   COST        -7.0\n"
      "ENDATA\n");
  EXPECT_DOUBLE_EQ(model.objective_offset, 7.0);

  // x = 2 at the optimum, so the objective is 2 + 7 = 9.
  const std::vector<double> x{2.0};
  EXPECT_DOUBLE_EQ(model.evaluate_objective(x.data()), 9.0);
}

TEST(MpsReader, ExtraFreeRowsAreDropped) {
  const Model model = parse_or_fail(
      "NAME          FREEROWS\n"
      "ROWS\n"
      " N  COST\n"
      " N  IGNORED\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   IGNORED      9.0\n"
      "    X         R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");
  EXPECT_EQ(model.num_rows(), 1);
  EXPECT_EQ(model.num_nonzeros(), 1);
  EXPECT_DOUBLE_EQ(model.col_cost[0], 1.0);
}

// =========================================================================================
// Structural rejection
// =========================================================================================

TEST(MpsReader, DuplicateMatrixEntriesAreRejected) {
  // A generic triplet builder sums duplicates. MPS has no accumulate semantics, so summing
  // silently doubles a coefficient.
  const std::string error = parse_expecting_failure(
      "NAME          DUP\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "    X         R1           2.0\n"
      "ENDATA\n");
  EXPECT_NE(error.find("duplicate"), std::string::npos) << error;
}

TEST(MpsReader, DuplicateRowNamesAreRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          DUP\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      " G  R1\n"
      "ENDATA\n");
  EXPECT_NE(error.find("duplicate row name"), std::string::npos) << error;
}

TEST(MpsReader, MissingEndataIsRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          TRUNC\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n");
  EXPECT_NE(error.find("ENDATA"), std::string::npos) << error;
}

TEST(MpsReader, AnUnknownRowNameInColumnsIsRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      "COLUMNS\n"
      "    X         NOSUCH       1.0\n"
      "ENDATA\n");
  EXPECT_NE(error.find("NOSUCH"), std::string::npos) << error;
}

TEST(MpsReader, AnUnparsableNumberIsRejected) {
  const std::string error = parse_expecting_failure(
      "NAME          BAD\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           notanumber\n"
      "ENDATA\n");
  EXPECT_NE(error.find("not a number"), std::string::npos) << error;
}

TEST(MpsReader, MpsScaleInfinityIsNormalised) {
  // 1e30 is the format's de-facto infinity. Leaving it as a finite 1e30 turns every ratio
  // test and every scaling pass into nonsense.
  const Model model = bounded_column(" UP BND       X          1.0e30\n");
  const Index x = col_of(model, "X");
  EXPECT_TRUE(is_infinite(model.col_upper[static_cast<std::size_t>(x)]));
}

TEST(MpsReader, FortranStyleExponentsAreAccepted) {
  const Model model = parse_or_fail(
      "NAME          FORTRAN\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST      1.5D+02   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");
  EXPECT_DOUBLE_EQ(model.col_cost[0], 150.0);
}

// =========================================================================================
// Free versus fixed dialect
// =========================================================================================

/// Lay out one fixed-format record at the column positions the IBM spec gives:
/// fields at columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61 (1-based).
[[nodiscard]] std::string fixed_line(const std::string& f1, const std::string& f2,
                                     const std::string& f3, const std::string& f4,
                                     const std::string& f5 = {}, const std::string& f6 = {}) {
  static constexpr std::size_t kOffsets[6] = {1, 4, 14, 24, 39, 49};
  const std::string* fields[6] = {&f1, &f2, &f3, &f4, &f5, &f6};
  std::string line;
  for (std::size_t k = 0; k < 6; ++k) {
    if (fields[k]->empty()) continue;
    if (line.size() < kOffsets[k]) line.resize(kOffsets[k], ' ');
    line += *fields[k];
  }
  return line + "\n";
}

TEST(MpsReader, FixedFormatNamesMayContainBlanks) {
  // This is the ONLY case where the two dialects disagree, and it is why the fixed reader
  // exists at all. Free tokenization would see "MY COL" as two tokens and reject the file.
  const std::string text =
      "NAME          FIXED\n"
      "ROWS\n" +
      fixed_line("N", "MY COST", "", "") + fixed_line("L", "MY ROW", "", "") + "COLUMNS\n" +
      fixed_line("", "MY COL", "MY COST", "1.0") + fixed_line("", "MY COL", "MY ROW", "2.0") +
      "RHS\n" + fixed_line("", "RHS", "MY ROW", "10.0") + "ENDATA\n";

  const TempFile file(text);
  Model model;
  io::MpsFormat used = io::MpsFormat::kAuto;
  const io::ReadResult result = io::read_mps(file.path(), &model, io::MpsFormat::kAuto, &used);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(used, io::MpsFormat::kFixed);
  EXPECT_EQ(model.num_rows(), 1);
  EXPECT_EQ(model.num_cols(), 1);
  EXPECT_EQ(model.col_names[0], "MY COL");
  EXPECT_EQ(model.row_names[0], "MY ROW");
  EXPECT_DOUBLE_EQ(model.col_cost[0], 1.0);
  EXPECT_DOUBLE_EQ(model.matrix.at(0, 0), 2.0);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}

TEST(MpsReader, AutoDetectionPrefersTheFreeDialect) {
  const TempFile file(
      "NAME          FREE\n"
      "ROWS\n"
      " N COST\n"
      " L R1\n"
      "COLUMNS\n"
      " X COST 1.0 R1 1.0\n"
      "RHS\n"
      " RHS R1 10.0\n"
      "ENDATA\n");
  Model model;
  io::MpsFormat used = io::MpsFormat::kFixed;
  const io::ReadResult result = io::read_mps(file.path(), &model, io::MpsFormat::kAuto, &used);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_EQ(used, io::MpsFormat::kFree);
  EXPECT_EQ(model.num_rows(), 1);
  EXPECT_EQ(model.num_cols(), 1);
}

TEST(MpsReader, CarriageReturnsAreStripped) {
  // A Windows-authored file read as binary leaves \r glued to the last token on the line,
  // which turns a bound type "UP" into "UP\r" and an unknown-type rejection.
  const TempFile file(
      "NAME          CRLF\r\n"
      "ROWS\r\n"
      " N  COST\r\n"
      " L  R1\r\n"
      "COLUMNS\r\n"
      "    X         COST         1.0   R1           1.0\r\n"
      "RHS\r\n"
      "    RHS       R1          10.0\r\n"
      "BOUNDS\r\n"
      " UP BND       X            5.0\r\n"
      "ENDATA\r\n");
  Model model;
  const io::ReadResult result = io::read_mps(file.path(), &model);
  ASSERT_TRUE(result.ok) << result.error;
  EXPECT_DOUBLE_EQ(model.col_upper[0], 5.0);
}

// =========================================================================================
// Compressed input
// =========================================================================================

#ifdef SANKHYA_WITH_ZLIB
/// Write `contents` through zlib, producing a real gzip file, and delete it on destruction.
class TempGzFile {
 public:
  explicit TempGzFile(const std::string& contents) {
    static int counter = 0;
    path_ = "sankhya_test_gz_" + std::to_string(counter++) + ".mps.gz";
    gzFile out = gzopen(path_.c_str(), "wb");
    if (out == nullptr) {
      ADD_FAILURE() << "cannot create " << path_;
      return;
    }
    gzwrite(out, contents.data(), static_cast<unsigned>(contents.size()));
    gzclose(out);
  }
  ~TempGzFile() {
    if (!path_.empty()) std::remove(path_.c_str());
  }
  TempGzFile(const TempGzFile&) = delete;
  TempGzFile& operator=(const TempGzFile&) = delete;
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  std::string path_;
};

TEST(MpsReader, ReadsGzipCompressedInput) {
  // MIPLIB and the larger Netlib instances ship compressed, so this path carries real
  // benchmark input rather than being a convenience. It had only ever been exercised by
  // hand from the command line before this test.
  const std::string text =
      "NAME          GZTEST\n"
      "ROWS\n"
      " N  COST\n"
      " G  R1\n"
      " E  R2\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "    X         R2           2.0\n"
      "    Y         COST         3.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1           5.0   R2           4.0\n"
      "RANGES\n"
      "    RNG       R1           3.0\n"
      "BOUNDS\n"
      " UP BND       Y            9.0\n"
      "ENDATA\n";

  const TempGzFile compressed(text);
  Model from_gz;
  const io::ReadResult gz_result = io::read_mps(compressed.path(), &from_gz);
  ASSERT_TRUE(gz_result.ok) << gz_result.error;

  // Identical to the same bytes read uncompressed - decompression must be transparent, not
  // merely successful.
  const Model plain = parse_or_fail(text);
  ASSERT_EQ(from_gz.num_rows(), plain.num_rows());
  ASSERT_EQ(from_gz.num_cols(), plain.num_cols());
  ASSERT_EQ(from_gz.num_nonzeros(), plain.num_nonzeros());
  EXPECT_EQ(from_gz.name, plain.name);
  for (Index i = 0; i < plain.num_rows(); ++i) {
    const auto u = static_cast<std::size_t>(i);
    EXPECT_DOUBLE_EQ(from_gz.row_lower[u], plain.row_lower[u]);
    EXPECT_DOUBLE_EQ(from_gz.row_upper[u], plain.row_upper[u]);
  }
  for (Index j = 0; j < plain.num_cols(); ++j) {
    const auto u = static_cast<std::size_t>(j);
    EXPECT_DOUBLE_EQ(from_gz.col_cost[u], plain.col_cost[u]);
    EXPECT_DOUBLE_EQ(from_gz.col_upper[u], plain.col_upper[u]);
  }
  // The RANGES row survived compression: G row with rhs 5 and range 3 is [5, 8].
  EXPECT_DOUBLE_EQ(from_gz.row_lower[0], 5.0);
  EXPECT_DOUBLE_EQ(from_gz.row_upper[0], 8.0);
}

TEST(MpsReader, ReadsAnUncompressedFileThroughTheSameCodePath) {
  // zlib reads a plain file transparently, which is why there is only one input path. If
  // that ever stopped being true, every uncompressed instance would fail at once.
  const Model model = parse_or_fail(
      "NAME          PLAIN\n"
      "ROWS\n"
      " N  COST\n"
      " L  R1\n"
      "COLUMNS\n"
      "    X         COST         1.0   R1           1.0\n"
      "RHS\n"
      "    RHS       R1          10.0\n"
      "ENDATA\n");
  EXPECT_EQ(model.num_rows(), 1);
  EXPECT_DOUBLE_EQ(model.row_upper[0], 10.0);
}
#endif  // SANKHYA_WITH_ZLIB

}  // namespace
}  // namespace sankhya
