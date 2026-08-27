// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity test for a quadratic objective.
//
// A convex QP has a global minimum that first-order and interior-point methods actually
// converge to. A NON-convex one has local minima, saddle points and possibly no finite
// infimum at all, and every method here would still produce a point, print it, and call it
// optimal. That is the failure mode CLAUDE.md opens with, so a Hessian that is not provably
// positive semidefinite is REFUSED rather than solved to whatever the iteration lands on.
//
// The test is an LDL^T factorization with symmetric pivoting on the DIAGONAL only. Q is
// positive semidefinite exactly when such a factorization exists with every d_i >= 0, and
// the factorization fails (or produces a negative d_i) otherwise. That is a decision
// procedure rather than a heuristic, which matters: guessing "convex" on a non-convex model
// is precisely the wrong error to make.
//
// Reference: Golub & Van Loan, "Matrix Computations" (4th ed.), section 4.1, for LDL^T and
// its relationship to definiteness; Higham, "Analysis of the Cholesky decomposition of a
// semi-definite matrix" (1990), for the semidefinite case and why a small negative pivot
// must be treated as rounding rather than as evidence of indefiniteness.
#pragma once

#include <string>

#include "sankhya/model.hpp"

namespace sankhya::qp {

/// What the convexity test concluded.
enum class Convexity {
  kConvex,      ///< Q is positive semidefinite; the QP has a global minimum
  kIndefinite,  ///< Q has a provably negative eigenvalue; the model is non-convex
  kUnverified,  ///< too large to decide with the dense test; treated as a refusal
};

struct ConvexityResult {
  Convexity verdict = Convexity::kUnverified;
  /// Human-readable detail naming the column and the pivot that decided it.
  std::string detail;
};

/// Decide whether `model.hessian` is positive semidefinite.
///
/// The Hessian is stored as the lower triangle of a symmetric matrix and the objective term
/// is 0.5 x^T Q x, so this tests Q itself rather than the stored triangle.
[[nodiscard]] ConvexityResult check_convexity(const Model& model);

}  // namespace sankhya::qp
