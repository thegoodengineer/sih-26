// SPDX-License-Identifier: Apache-2.0
// SANKHYA — every numerical tolerance in the solver, in one place.
//
// CLAUDE.md rule: no magic numbers in the numerical core. If a comparison against a small
// constant appears anywhere in src/, the constant is declared here with a justification.
// Anything tunable at runtime is ALSO an entry in the option table (src/util/options.cpp)
// whose default is one of these constants; these are the defaults, not the law.
#pragma once

#include "sankhya/types.hpp"

namespace sankhya::tol {

// ---------------------------------------------------------------------------------------
// Feasibility and optimality
// ---------------------------------------------------------------------------------------

/// Max allowed violation of a row or column bound before a point is called infeasible.
inline constexpr double kPrimalFeasibility = 1e-7;

/// Max allowed violation of dual feasibility (sign conditions on reduced costs).
inline constexpr double kDualFeasibility = 1e-7;

/// Max allowed distance from an integer before a value is called fractional.
inline constexpr double kIntegrality = 1e-6;

/// MIP termination: stop when (incumbent - dual bound) / |incumbent| falls below this.
inline constexpr double kMipRelativeGap = 1e-4;

/// MIP termination: stop when (incumbent - dual bound) falls below this in absolute terms.
inline constexpr double kMipAbsoluteGap = 1e-6;

/// LP optimality check used by the independent verifier: primal objective must equal dual
/// objective to this relative accuracy. Tighter than feasibility on purpose — a converged
/// simplex basis should reproduce strong duality far better than it satisfies bounds.
inline constexpr double kDualityGap = 1e-9;

// ---------------------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------------------

/// Structural zero threshold. Values below this in magnitude are dropped when a sparse
/// matrix is finalised, and are never accepted as pivots.
inline constexpr double kZeroDrop = 1e-11;

/// Markowitz threshold for sparse LU pivoting (Phase 6). A candidate pivot must be at
/// least this fraction of the largest magnitude in its column to be numerically eligible.
/// 0.01 is the standard simplex compromise between sparsity and stability (Suhl & Suhl).
inline constexpr double kMarkowitzThreshold = 0.01;

/// Below this, a computed pivot element is treated as a singular basis rather than a pivot.
inline constexpr double kPivotTolerance = 1e-9;

// ---------------------------------------------------------------------------------------
// Simplex
// ---------------------------------------------------------------------------------------

/// Reduced cost must beat this in magnitude to be an eligible entering candidate.
inline constexpr double kDualPricing = 1e-7;

/// Number of consecutive degenerate iterations after which the primal simplex switches to
/// Bland's rule. Bland's rule is provably non-cycling but prices badly, so it is a fallback
/// and not the default (Chvatal, "Linear Programming", ch. 3).
inline constexpr int kBlandSwitchIterations = 50;

/// Feasibility tolerance used inside the ratio test, deliberately looser than
/// kPrimalFeasibility so that a marginally infeasible basic variable does not block a pivot.
inline constexpr double kRatioTestFeasibility = 1e-9;

// ---------------------------------------------------------------------------------------
// First-order method (PDHG)
// ---------------------------------------------------------------------------------------

/// PDHG results are reported at BOTH of these, separately, never blended (CLAUDE.md).
inline constexpr double kPdhgLoose = 1e-4;
inline constexpr double kPdhgTight = 1e-8;

}  // namespace sankhya::tol
