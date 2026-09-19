// SPDX-License-Identifier: Apache-2.0
// SANKHYA - sensitivity ranging, the IIS, and the solution pool, as a C caller receives them
// (#261, the follow-up to #207/#254 that test_c_api_certificates.cpp covers). Same shape as
// that file: the model is built and solved through sankhya.h only, and what comes back is
// checked against either an independent C++ solve of the SAME model (ranging - there is no
// separate checker function the way farkas_proves_infeasible exists for certificates) or an
// arithmetic re-derivation from the returned numbers themselves (IIS witnesses, pool
// objectives) - so a C accessor wired to the wrong field cannot pass by accident.
//
// test_c_api.cpp keeps its promise of never including model.hpp; that is why this is a
// separate file, exactly as test_c_api_certificates.cpp already is.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/sankhya.h"

namespace {

struct ModelHandle {
  sankhya_model* handle = sankhya_model_create();
  ~ModelHandle() { sankhya_model_free(handle); }
  operator sankhya_model*() const { return handle; }
};

struct SolutionHandle {
  sankhya_solution* handle = nullptr;
  ~SolutionHandle() { sankhya_solution_free(handle); }
};

struct OptionsHandle {
  sankhya_options* handle = sankhya_options_create();
  ~OptionsHandle() { sankhya_options_free(handle); }
  operator sankhya_options*() const { return handle; }
};

// =========================================================================================
// Sensitivity ranging (#220 via the C API)
// =========================================================================================

// min -x1 - 2*x2;  x1+x2<=4 (row0), x1<=3 (row1), x2<=3 (row2); x1,x2>=0.
// Optimal x1=1, x2=3, objective=-7 - the exact model test_ranging.cpp uses, so its own
// hand-worked comments there are the derivation backing the numbers checked here too.
void build_ranging_lp(sankhya_model* model) {
  const double inf = sankhya_infinity();
  int x1 = -1, x2 = -1, r0 = -1, r1 = -1, r2 = -1;
  ASSERT_EQ(sankhya_model_add_column(model, -1.0, 0.0, inf, 0, "x1", &x1), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_column(model, -2.0, 0.0, inf, 0, "x2", &x2), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -inf, 4.0, "r0", &r0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -inf, 3.0, "r1", &r1), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -inf, 3.0, "r2", &r2), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, x1, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, x2, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r1, x1, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r2, x2, 1.0), SANKHYA_OK);
}

sankhya::Model ranging_lp_in_cpp() {
  sankhya::Model m;
  m.resize_columns(2);
  m.resize_rows(3);
  m.col_cost = {-1.0, -2.0};
  m.col_lower = {0.0, 0.0};
  m.col_upper = {sankhya::kInfinity, sankhya::kInfinity};
  m.row_lower = {-sankhya::kInfinity, -sankhya::kInfinity, -sankhya::kInfinity};
  m.row_upper = {4.0, 3.0, 3.0};
  m.matrix.reset(3, 2);
  m.matrix.add_entry(0, 0, 1.0);
  m.matrix.add_entry(0, 1, 1.0);
  m.matrix.add_entry(1, 0, 1.0);
  m.matrix.add_entry(2, 1, 1.0);
  m.matrix.finalize();
  return m;
}

TEST(CApiRanging, CrossesTheBoundaryAndMatchesAnIndependentCppSolve) {
  ModelHandle model;
  build_ranging_lp(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  ASSERT_EQ(sankhya_options_set_bool(options, "ranging", 1), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_OPTIMAL);
  ASSERT_EQ(sankhya_solution_has_ranging(solution.handle), 1);

  // The independent C++ solve: a fresh Model built separately above, solved through
  // sankhya::solve() directly rather than through the handle. Equal ranges here mean the C
  // accessors read the fields their names say, not e.g. each other's or col_dual's.
  sankhya::Options cpp_options;
  cpp_options.set_bool("ranging", true);
  const sankhya::Solution cpp_solution = sankhya::solve(ranging_lp_in_cpp(), cpp_options);
  ASSERT_EQ(cpp_solution.status, sankhya::SolveStatus::kOptimal);
  ASSERT_FALSE(cpp_solution.col_ranging_lower.empty());

  std::vector<double> col_lo(2), col_hi(2);
  ASSERT_EQ(sankhya_solution_col_ranging_lower(solution.handle, col_lo.data(), 2), SANKHYA_OK);
  ASSERT_EQ(sankhya_solution_col_ranging_upper(solution.handle, col_hi.data(), 2), SANKHYA_OK);
  for (int j = 0; j < 2; ++j) {
    EXPECT_DOUBLE_EQ(col_lo[static_cast<std::size_t>(j)],
                     cpp_solution.col_ranging_lower[static_cast<std::size_t>(j)]);
    EXPECT_DOUBLE_EQ(col_hi[static_cast<std::size_t>(j)],
                     cpp_solution.col_ranging_upper[static_cast<std::size_t>(j)]);
  }

  std::vector<double> row_lo(3), row_hi(3);
  ASSERT_EQ(sankhya_solution_row_ranging_lower(solution.handle, row_lo.data(), 3), SANKHYA_OK);
  ASSERT_EQ(sankhya_solution_row_ranging_upper(solution.handle, row_hi.data(), 3), SANKHYA_OK);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(row_lo[static_cast<std::size_t>(i)],
                     cpp_solution.row_ranging_lower[static_cast<std::size_t>(i)]);
    EXPECT_DOUBLE_EQ(row_hi[static_cast<std::size_t>(i)],
                     cpp_solution.row_ranging_upper[static_cast<std::size_t>(i)]);
  }
  EXPECT_EQ(sankhya_solution_ranging_basis_degenerate(solution.handle),
            cpp_solution.ranging_basis_degenerate ? 1 : 0);

  // NEGATIVE CONTROL: col_ranging_lower must not be col_dual under another name - x1 and x2
  // are both basic here (neither sits on a bound), so col_dual is all zero while the ranges
  // computed above are not.
  std::vector<double> col_duals(2, 0.0);
  ASSERT_EQ(sankhya_solution_col_duals(solution.handle, col_duals.data(), 2), SANKHYA_OK);
  EXPECT_NE(col_lo, col_duals);

  // The copy contract matches col_values: a wrong count is refused, not truncated.
  EXPECT_EQ(sankhya_solution_col_ranging_lower(solution.handle, col_lo.data(), 1),
            SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_row_ranging_upper(solution.handle, nullptr, 3),
            SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiRanging, IsAbsentAndLengthZeroWhenTheOptionIsOff) {
  ModelHandle model;
  build_ranging_lp(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  EXPECT_EQ(sankhya_solution_has_ranging(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_col_ranging_lower(solution.handle, nullptr, 0), SANKHYA_OK);
  EXPECT_EQ(sankhya_solution_ranging_basis_degenerate(solution.handle), 0);
}

// =========================================================================================
// The IIS (#217 via the C API)
// =========================================================================================

// x free, x >= 5 (row0) and x <= 2 (row1): the same contradictory pair
// test_c_api_certificates.cpp uses for the Farkas certificate. Both rows are necessary, so
// the IIS is exactly {row0, row1} - the model IS its own minimal infeasible subsystem.
void build_contradictory_pair(sankhya_model* model) {
  const double inf = sankhya_infinity();
  int x = -1, r0 = -1, r1 = -1;
  ASSERT_EQ(sankhya_model_add_column(model, 1.0, -inf, inf, 0, "x", &x), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, 5.0, inf, "lo", &r0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -inf, 2.0, "hi", &r1), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, x, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r1, x, 1.0), SANKHYA_OK);
}

TEST(CApiIis, CrossesTheBoundaryAndTheWitnessesProveEachRowNecessary) {
  ModelHandle model;
  build_contradictory_pair(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  ASSERT_EQ(sankhya_options_set_bool(options, "compute_iis", 1), SANKHYA_OK);
  // Presolve off, so the dual simplex runs and produces the Farkas dual compute_iis needs
  // (test_iis.cpp: when presolve detects the infeasibility instead, compute_iis returns
  // early - a known limitation, not what this file is testing).
  ASSERT_EQ(sankhya_options_set_bool(options, "presolve", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_INFEASIBLE);
  EXPECT_EQ(sankhya_solution_iis_inconclusive(solution.handle), 0);

  ASSERT_EQ(sankhya_solution_iis_row_count(solution.handle), 2) << "both rows are necessary";
  EXPECT_EQ(sankhya_solution_iis_col_lower_count(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_iis_col_upper_count(solution.handle), 0);

  std::vector<int> rows(2, -1);
  ASSERT_EQ(sankhya_solution_iis_rows(solution.handle, rows.data(), 2), SANKHYA_OK);
  EXPECT_EQ(std::min(rows[0], rows[1]), 0);
  EXPECT_EQ(std::max(rows[0], rows[1]), 1);

  // Two witnesses, one per row, one column each (the model has one column, x).
  ASSERT_EQ(sankhya_solution_iis_witness_count(solution.handle), 2);
  for (int k = 0; k < 2; ++k) {
    double x = 0.0;
    ASSERT_EQ(sankhya_solution_iis_witness(solution.handle, k, &x, 1), SANKHYA_OK)
        << sankhya_last_error();
    // ARITHMETIC RE-DERIVATION, not a peek at the internal Solution: a witness for row r must
    // satisfy the OTHER row's bound and violate row r's own - x >= 5 is row 0, x <= 2 is
    // row 1, so a witness for row 0 has x < 5 (violating it) while x <= 2 (satisfying row 1),
    // and vice versa. Either witness identity is a valid deletion-filter trial, so accept
    // whichever the row-index copy above paired it with.
    const int row = rows[static_cast<std::size_t>(k)];
    if (row == 0) {
      EXPECT_LT(x, 5.0) << "witness for row 0 (x >= 5) must violate it";
      EXPECT_LE(x, 2.0) << "witness for row 0 must still satisfy row 1 (x <= 2)";
    } else {
      EXPECT_GT(x, 2.0) << "witness for row 1 (x <= 2) must violate it";
      EXPECT_GE(x, 5.0) << "witness for row 1 must still satisfy row 0 (x >= 5)";
    }
  }

  // The copy contract: a wrong count and an out-of-range index are both refused.
  double one = 0.0;
  EXPECT_EQ(sankhya_solution_iis_witness(solution.handle, 0, &one, 0), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_iis_witness(solution.handle, 2, &one, 1), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_iis_rows(solution.handle, rows.data(), 1), SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiIis, IsAbsentAndLengthZeroWhenTheOptionIsOff) {
  ModelHandle model;
  build_contradictory_pair(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_INFEASIBLE);
  EXPECT_EQ(sankhya_solution_iis_row_count(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_iis_witness_count(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_iis_inconclusive(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_iis_rows(solution.handle, nullptr, 0), SANKHYA_OK);
}

// =========================================================================================
// The solution pool (#225 via the C API)
// =========================================================================================

// maximize x + y subject to x + y <= 1, x,y in {0,1} - i.e. minimise -x-y. Two equally good
// assignments, (1,0) and (0,1), both objective -1; (0,0) is feasible but worse. A clean tie
// to check the pool keeps both rather than only the one branch-and-bound happened to find
// first.
void build_tied_binary_pair(sankhya_model* model) {
  int x = -1, y = -1, r0 = -1;
  ASSERT_EQ(sankhya_model_add_column(model, -1.0, 0.0, 1.0, 1, "x", &x), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_column(model, -1.0, 0.0, 1.0, 1, "y", &y), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -sankhya_infinity(), 1.0, "r0", &r0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, x, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, y, 1.0), SANKHYA_OK);
}

TEST(CApiPool, CrossesTheBoundaryWithBothTiedAssignmentsAndMemberZeroIsTheReportedSolution) {
  ModelHandle model;
  build_tied_binary_pair(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  ASSERT_EQ(sankhya_options_set_bool(options, "pool_complete", 1), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_OPTIMAL);

  const int size = sankhya_solution_pool_size(solution.handle);
  ASSERT_GE(size, 2) << "both tied assignments should be kept";

  std::vector<double> reported(2, 0.0);
  ASSERT_EQ(sankhya_solution_col_values(solution.handle, reported.data(), 2), SANKHYA_OK);
  double reported_objective = sankhya_solution_objective(solution.handle);

  double member0_objective = 0.0;
  ASSERT_EQ(sankhya_solution_pool_objective(solution.handle, 0, &member0_objective),
            SANKHYA_OK);
  std::vector<double> member0(2, 0.0);
  ASSERT_EQ(sankhya_solution_pool_col_values(solution.handle, 0, member0.data(), 2),
            SANKHYA_OK);
  EXPECT_DOUBLE_EQ(member0_objective, reported_objective)
      << "pool member 0 is always the reported solution";
  EXPECT_EQ(member0, reported);

  // ARITHMETIC RE-DERIVATION: every member's objective must equal -(x+y) for ITS OWN
  // col_values, and every member's assignment must be integer-feasible and satisfy x+y<=1 -
  // checked from the copied numbers, not the internal PoolEntry.
  bool saw_x = false, saw_y = false;
  for (int k = 0; k < size; ++k) {
    double objective = 0.0;
    ASSERT_EQ(sankhya_solution_pool_objective(solution.handle, k, &objective), SANKHYA_OK);
    std::vector<double> values(2, 0.0);
    ASSERT_EQ(sankhya_solution_pool_col_values(solution.handle, k, values.data(), 2),
              SANKHYA_OK);
    const double x = values[0];
    const double y = values[1];
    EXPECT_NEAR(std::round(x), x, 1e-9);
    EXPECT_NEAR(std::round(y), y, 1e-9);
    EXPECT_LE(x + y, 1.0 + 1e-9);
    EXPECT_NEAR(objective, -(x + y), 1e-9)
        << "member " << k << "'s objective does not match its own assignment";
    if (x > 0.5 && y < 0.5) saw_x = true;
    if (y > 0.5 && x < 0.5) saw_y = true;
  }
  EXPECT_TRUE(saw_x) << "the (1,0) assignment should be in the pool";
  EXPECT_TRUE(saw_y) << "the (0,1) assignment should be in the pool";

  // The copy contract: an out-of-range index and a wrong count are both refused.
  double dummy_objective = 0.0;
  EXPECT_EQ(sankhya_solution_pool_objective(solution.handle, size, &dummy_objective),
            SANKHYA_ERROR_ARGUMENT);
  std::vector<double> dummy_values(2, 0.0);
  EXPECT_EQ(sankhya_solution_pool_col_values(solution.handle, 0, dummy_values.data(), 1),
            SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_pool_col_values(solution.handle, size, dummy_values.data(), 2),
            SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiPool, IsEmptyForAPureLp) {
  ModelHandle model;
  const double inf = sankhya_infinity();
  int x = -1;
  ASSERT_EQ(sankhya_model_add_column(model, 1.0, 0.0, inf, 0, "x", &x), SANKHYA_OK);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  EXPECT_EQ(sankhya_solution_pool_size(solution.handle), 0);
  double objective = 0.0;
  EXPECT_EQ(sankhya_solution_pool_objective(solution.handle, 0, &objective),
            SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiRangingIisPool, NullSolutionIsRefusedOrReadsAsEmpty) {
  EXPECT_EQ(sankhya_solution_has_ranging(nullptr), 0);
  EXPECT_EQ(sankhya_solution_ranging_basis_degenerate(nullptr), 0);
  EXPECT_EQ(sankhya_solution_col_ranging_lower(nullptr, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_iis_row_count(nullptr), 0);
  EXPECT_EQ(sankhya_solution_iis_inconclusive(nullptr), 0);
  EXPECT_EQ(sankhya_solution_iis_witness_count(nullptr), 0);
  EXPECT_EQ(sankhya_solution_iis_rows(nullptr, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_iis_witness(nullptr, 0, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_pool_size(nullptr), 0);
  double objective = 0.0;
  EXPECT_EQ(sankhya_solution_pool_objective(nullptr, 0, &objective), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_pool_col_values(nullptr, 0, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
}

}  // namespace
