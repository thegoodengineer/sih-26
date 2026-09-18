// SPDX-License-Identifier: Apache-2.0
// SANKHYA - mixed-integer rounding (MIR) cuts (#221).
//
// A Gomory cut comes from a row of the simplex tableau, which is a dense, numerically
// fragile object; that is why the root Gomory cuts cost a proof on the MIPLIB A/B and are
// off by default. An MIR cut comes from a row of the MODEL: bound-substitute the variables so
// every one is non-negative, move the continuous part to the right-hand side, divide by a
// divisor, and apply the mixed-integer rounding inequality. The coefficients are the model's
// own, which is why MIR is the workhorse cut of every modern MILP solver (Marchand & Wolsey,
// "Aggregation and mixed integer rounding to solve MIPs", Operations Research 49 (2001);
// Nemhauser & Wolsey, "A recursive procedure to generate all cuts for 0-1 mixed integer
// programs", Math. Programming 46 (1990) for the inequality itself).
//
// THE INEQUALITY. For  sum_j a_j y_j <= b + s,  y integer and non-negative, s >= 0, with
// f0 = frac(b) in (0, 1) and f_j = frac(a_j):
//     sum_j ( floor(a_j) + max(0, f_j - f0) / (1 - f0) ) y_j  <=  floor(b) + s / (1 - f0).
// Dividing the base inequality by a divisor delta before rounding gives a family; the
// divisor is tried at 1 and at the coefficients of the fractional integer variables, and
// the most violated member is returned, in the model's original variables and in the
// `Cut` convention (sum coeff_j x_j <= rhs). Single model rows only in this version; row
// aggregation to eliminate continuous variables is the next step the issue names.
#pragma once

#include <vector>

#include "cuts.hpp"
#include "sankhya/model.hpp"

namespace sankhya::mip {

/// MIR cuts from the model's rows at the LP point `solution`, each violated by that point.
/// Every cut is valid for the integer feasible set by construction; the caller's filter
/// decides which are worth adding.
[[nodiscard]] std::vector<Cut> generate_mir_cuts(const Model& model, const Solution& solution);

/// The same, with the column bounds to substitute given explicitly (#221): a node of the
/// tree passes the GLOBAL bounds so the cut is valid everywhere, while the point it is
/// separated at is the node's own. `model.col_lower/upper` are ignored.
[[nodiscard]] std::vector<Cut> generate_mir_cuts(const Model& model, const Solution& solution,
                                                 const std::vector<double>& col_lower,
                                                 const std::vector<double>& col_upper);

/// The MIR inequality on one already bound-substituted base inequality, exposed for the
/// textbook test: `coefficient[j]` and `is_integer[j]` describe sum a_j y_j <= rhs over
/// non-negative y; the continuous part is folded into s. Returns the rounded coefficients
/// and right-hand side in the same y variables, or false when f0 is too close to 0 or 1.
[[nodiscard]] bool mir_inequality(const std::vector<double>& coefficient,
                                  const std::vector<bool>& is_integer, double rhs,
                                  double divisor, std::vector<double>* cut_coefficient,
                                  double* cut_rhs);

}  // namespace sankhya::mip
