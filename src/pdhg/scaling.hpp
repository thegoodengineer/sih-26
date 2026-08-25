// SPDX-License-Identifier: Apache-2.0
// SANKHYA - diagonal preconditioning for the first-order LP engine.
//
// References:
//   Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in matrices",
//     RAL-TR-2001-034 (2001)
//   Pock & Chambolle, "Diagonal preconditioning for first order primal-dual algorithms in
//     convex optimization", ICCV 2011, section 4 (the alpha-family of diagonal scalings)
//   Applegate et al., "Practical Large-Scale Linear Programming using Primal-Dual Hybrid
//     Gradient" (PDLP), section 4.1 - the combination used here, Ruiz followed by
//     Pock-Chambolle with alpha = 1
//
// Written from the papers. No solver's source was consulted (CLAUDE.md red line).
//
// WHY THIS EXISTS AT ALL. PDHG converges at a rate governed by the operator norm of the
// constraint matrix, so on a badly scaled model - a refinery LP mixing flows in tonnes with
// qualities in parts per million is the canonical case - it does not converge in any useful
// number of iterations. The simplex is largely indifferent to scaling because it pivots on
// exact ratios; a first-order method is not. Preconditioning is not an optimisation here,
// it is what makes the method work.
#pragma once

#include <vector>

#include "sankhya/model.hpp"

namespace sankhya::pdhg {

/// A diagonal rescaling of a model:  Ahat = Dr A Dc.
///
/// The scaled problem solved internally is
///     min chat' xhat   s.t.   rlhat <= Ahat xhat <= ruhat,   lhat <= xhat <= uhat
/// and the mapping back is
///     x = Dc xhat        y = Dr yhat        reduced cost d = Dc^-1 dhat
struct Scaling {
  /// Strictly positive row and column multipliers.
  std::vector<double> row;
  std::vector<double> column;

  SparseMatrix matrix;  ///< Ahat
  std::vector<double> cost;
  std::vector<double> col_lower;
  std::vector<double> col_upper;
  std::vector<double> row_lower;
  std::vector<double> row_upper;

  /// Largest and smallest absolute entry of Ahat, reported in the log so that a model the
  /// scaling could not equilibrate is visible rather than merely slow.
  double max_abs = 0.0;
  double min_abs = 0.0;
};

/// Build the scaled problem: `ruiz_iterations` rounds of Ruiz equilibration in the infinity
/// norm, followed by one Pock-Chambolle diagonal pass with alpha = 1.
///
/// Costs are taken in MINIMISE space: the caller has already folded the objective sense in,
/// so this function never looks at model.sense.
[[nodiscard]] Scaling build_scaling(const Model& model,
                                    const std::vector<double>& min_space_cost,
                                    int ruiz_iterations);

/// Largest singular value of `matrix`, by power iteration on A^T A. PDHG's step size is
/// bounded by 1 / ||A||_2, so this sets the starting point that the adaptive rule then
/// refines. An underestimate diverges, so the estimate is deliberately rounded up.
[[nodiscard]] double estimate_spectral_norm(const SparseMatrix& matrix, int iterations,
                                            unsigned seed);

}  // namespace sankhya::pdhg
