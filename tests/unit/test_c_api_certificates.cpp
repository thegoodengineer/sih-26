// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the certificates, as a C caller receives them (#207).
//
// test_certificate.cpp proves the certificates are proofs inside the library. This file
// proves they survive the boundary: the model is built and solved through sankhya.h only,
// and the vectors that come back out are then checked by sankhya::farkas_proves_infeasible
// and sankhya::ray_proves_unbounded against an EQUIVALENT model built separately in C++.
// That is the one place this file reaches past the header, and it is on purpose: a C
// accessor wired to the wrong field (col_duals instead of farkas_dual, say) returns a vector
// of the right length that proves nothing, and only the checker can tell.
//
// test_c_api.cpp keeps its promise of never including model.hpp; that is why this is a
// separate file.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/certificate.hpp"
#include "sankhya/model.hpp"
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

/// x free, x >= 5 and x <= 2, as two rows. The contradiction needs both rows.
void build_contradictory_pair(sankhya_model* model) {
  const double inf = sankhya_infinity();
  int x = -1;
  int r0 = -1;
  int r1 = -1;
  ASSERT_EQ(sankhya_model_add_column(model, 1.0, -inf, inf, 0, "x", &x), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, 5.0, inf, "lo", &r0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_add_row(model, -inf, 2.0, "hi", &r1), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r0, x, 1.0), SANKHYA_OK);
  ASSERT_EQ(sankhya_model_set_coefficient(model, r1, x, 1.0), SANKHYA_OK);
}

sankhya::Model contradictory_pair_in_cpp() {
  sankhya::Model model;
  model.col_cost = {1.0};
  model.col_lower = {-sankhya::kInfinity};
  model.col_upper = {sankhya::kInfinity};
  model.col_type = {sankhya::VarType::kContinuous};
  model.row_lower = {5.0, -sankhya::kInfinity};
  model.row_upper = {sankhya::kInfinity, 2.0};
  model.matrix.reset(2, 1);
  model.matrix.add_entry(0, 0, 1.0);
  model.matrix.add_entry(1, 0, 1.0);
  model.matrix.finalize();
  model.hessian.reset(1, 1);
  model.hessian.finalize();
  return model;
}

/// minimise -x with x >= 1 and no rows: the objective runs to minus infinity along +x.
void build_unbounded(sankhya_model* model) {
  int x = -1;
  ASSERT_EQ(sankhya_model_add_column(model, -1.0, 1.0, sankhya_infinity(), 0, "x", &x),
            SANKHYA_OK);
}

sankhya::Model unbounded_in_cpp() {
  sankhya::Model model;
  model.col_cost = {-1.0};
  model.col_lower = {1.0};
  model.col_upper = {sankhya::kInfinity};
  model.col_type = {sankhya::VarType::kContinuous};
  model.matrix.reset(0, 1);
  model.matrix.finalize();
  model.hessian.reset(1, 1);
  model.hessian.finalize();
  return model;
}

TEST(CApiCertificates, AFarkasCertificateCrossesTheBoundaryAndStillProvesInfeasibility) {
  ModelHandle model;
  build_contradictory_pair(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);
  // Presolve off, so the vector comes from the simplex; the next test takes the presolve
  // route to the same verdict and expects a certificate there too (#253).
  ASSERT_EQ(sankhya_options_set_bool(options, "presolve", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_INFEASIBLE);
  EXPECT_EQ(sankhya_solution_claims_a_point(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_primal_ray_length(solution.handle), 0);

  const int length = sankhya_solution_farkas_dual_length(solution.handle);
  ASSERT_EQ(length, 2) << "one multiplier per row";
  std::vector<double> y(static_cast<std::size_t>(length), 0.0);
  ASSERT_EQ(sankhya_solution_farkas_dual(solution.handle, y.data(), length), SANKHYA_OK)
      << sankhya_last_error();

  std::string why;
  EXPECT_TRUE(sankhya::farkas_proves_infeasible(contradictory_pair_in_cpp(), y, &why)) << why;

  // NEGATIVE CONTROL, committed rather than done once by hand: the row duals have the same
  // length as the Farkas vector, so an accessor wired to row_dual would pass every length
  // check above. They are not a proof here, and the two vectors must differ.
  std::vector<double> row_duals(static_cast<std::size_t>(length), 0.0);
  ASSERT_EQ(sankhya_solution_row_duals(solution.handle, row_duals.data(), length), SANKHYA_OK)
      << sankhya_last_error();
  EXPECT_NE(y, row_duals) << "the Farkas accessor returns the row duals";
  EXPECT_FALSE(sankhya::farkas_proves_infeasible(contradictory_pair_in_cpp(), row_duals))
      << "the row duals would prove infeasibility too, so this control cannot catch a "
         "mis-wiring";

  // The copy contract is the same as col_values: a wrong count is refused, not truncated.
  EXPECT_EQ(sankhya_solution_farkas_dual(solution.handle, y.data(), 1), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_farkas_dual(solution.handle, nullptr, length),
            SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiCertificates, ThePresolveVerdictCarriesACertificateAndAnAbsentRayIsLengthZero) {
  // Default options, so presolve proves the contradiction itself (#253): the verdict now
  // carries the Farkas vector built from the two rows, checked against the original model
  // before it reaches the boundary. A ray, which no infeasible model has, reads as "nothing
  // attached" - length zero and a copy of nothing succeeds - not as an error.
  ModelHandle model;
  build_contradictory_pair(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_INFEASIBLE);

  const int length = sankhya_solution_farkas_dual_length(solution.handle);
  ASSERT_EQ(length, 2) << "presolve's proof, one multiplier per row";
  std::vector<double> y(static_cast<std::size_t>(length), 0.0);
  ASSERT_EQ(sankhya_solution_farkas_dual(solution.handle, y.data(), length), SANKHYA_OK)
      << sankhya_last_error();
  std::string why;
  EXPECT_TRUE(sankhya::farkas_proves_infeasible(contradictory_pair_in_cpp(), y, &why)) << why;

  EXPECT_EQ(sankhya_solution_primal_ray_length(solution.handle), 0);
  EXPECT_EQ(sankhya_solution_primal_ray(solution.handle, nullptr, 0), SANKHYA_OK)
      << sankhya_last_error();
  double one = 0.0;
  EXPECT_EQ(sankhya_solution_primal_ray(solution.handle, &one, 1), SANKHYA_ERROR_ARGUMENT);
}

TEST(CApiCertificates, AnUnboundedRayCrossesTheBoundaryAndStillProvesUnboundedness) {
  ModelHandle model;
  build_unbounded(model);
  OptionsHandle options;
  ASSERT_EQ(sankhya_options_set_bool(options, "log_to_console", 0), SANKHYA_OK);

  SolutionHandle solution;
  ASSERT_EQ(sankhya_solve(model, options, &solution.handle), SANKHYA_OK)
      << sankhya_last_error();
  ASSERT_EQ(sankhya_solution_status(solution.handle), SANKHYA_UNBOUNDED);
  // Unbounded carries a feasible starting point, which is half of the proof.
  EXPECT_EQ(sankhya_solution_claims_a_point(solution.handle), 1);
  EXPECT_EQ(sankhya_solution_farkas_dual_length(solution.handle), 0);

  const int length = sankhya_solution_primal_ray_length(solution.handle);
  ASSERT_EQ(length, 1) << "one entry per column";
  std::vector<double> d(1, 0.0);
  ASSERT_EQ(sankhya_solution_primal_ray(solution.handle, d.data(), length), SANKHYA_OK)
      << sankhya_last_error();

  std::string why;
  EXPECT_TRUE(sankhya::ray_proves_unbounded(unbounded_in_cpp(), d, &why)) << why;
}

TEST(CApiCertificates, NullSolutionIsRefusedOrReadsAsEmpty) {
  EXPECT_EQ(sankhya_solution_claims_a_point(nullptr), 0);
  EXPECT_EQ(sankhya_solution_farkas_dual_length(nullptr), 0);
  EXPECT_EQ(sankhya_solution_primal_ray_length(nullptr), 0);
  EXPECT_EQ(sankhya_solution_farkas_dual(nullptr, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
  EXPECT_EQ(sankhya_solution_primal_ray(nullptr, nullptr, 0), SANKHYA_ERROR_ARGUMENT);
}

}  // namespace
