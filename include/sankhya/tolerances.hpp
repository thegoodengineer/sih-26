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

/// Diving heuristic (root node only): maximum integer columns fixed in one dive before it
/// gives up. Bounds the heuristic's own cost independently of instance size - Achterberg,
/// "Constraint Integer Programming" (thesis, 2007), ch. 6, notes a dive's payoff is
/// concentrated in its first handful of fixes, so capping it well short of the full integer
/// column count keeps a dive on a large MILP from itself becoming the expensive part of
/// solving the root node.
inline constexpr int kDivingMaxDepth = 50;

/// Diving heuristic (root node only): maximum LP re-solves in one dive. In this
/// implementation every fixed column costs exactly one re-solve, so this moves together
/// with kDivingMaxDepth today - kept as its own constant because the two bound different
/// things (how much of the box the dive may fix vs. how much simplex work it may spend
/// doing so), and a future dive that backtracks or retries would resolve LPs without fixing
/// a new column.
inline constexpr int kDivingMaxLpResolves = 50;

/// Reliability branching (#69; Achterberg, Koch & Martin, "Branching rules revisited",
/// Operations Research Letters 33 (2005), 42-54). A column's pseudocost in a direction is
/// trusted once it has been observed this many times; until then the column is a
/// candidate for strong branching, which measures the two children directly.
/// eta_rel = 8 is the paper's recommendation, and the value SCIP ships.
inline constexpr int kPseudocostReliability = 8;

/// Strong branching evaluates at most this many unreliable candidates per node, the most
/// fractional first. Each costs two warm-started dual simplex solves; the paper's
/// "lookahead" bound of 8 stops after that many candidates fail to improve the best score,
/// which this simpler cap approximates.
inline constexpr int kStrongBranchingCandidates = 10;

/// Iteration cap for one strong-branching child LP. A dual simplex stopped at this cap
/// still reports a valid bound (its objective is dual feasible throughout), so a capped
/// probe measures a lower estimate of the gain rather than nothing.
inline constexpr int kStrongBranchingIterations = 50;

/// LP optimality check used by the independent verifier: primal objective must equal dual
/// objective to this relative accuracy. Tighter than feasibility on purpose - a converged
/// simplex basis should reproduce strong duality far better than it satisfies bounds.
///
/// THIS CONSTANT IS NOT INDEPENDENT OF kDualFeasibility, and the two were originally chosen
/// as though it were. The duality gap at a primal-feasible point is bounded by the
/// complementarity residual, which is bounded in turn by the dual infeasibility times the
/// size of the primal solution:
///
///     relative gap  <=  kDualFeasibility * ||x||_1 / |objective|
///
/// So a fixed 1e-9 is only attainable when ||x||_1 / |objective| is small. Netlib `etamacro`
/// is where that surfaced (issue #52). Measured there:
///
///     dual infeasibility     1.345e-07      (just over kDualFeasibility)
///     sum |x_j|, 688 columns 2721
///     |objective|            755.7
///     => implied bound on the relative gap    4.84e-07
///
/// which is roughly FIVE HUNDRED TIMES looser than the 1e-9 promised here. The observed gap
/// was 3.25e-09 - far better than the bound, but still over this constant, and no amount of
/// tightening the gap test can fix a point whose duals have not converged.
///
/// DECISION (#52): neither number moves.
///
///   * The verifier is NOT loosened. It is the independent check the whole evidence story
///     rests on, and tuning it so we pass inverts its purpose. `CLAUDE.md` says so directly.
///   * kDualityGap is NOT relaxed to cover the worst case either. As an expectation for a
///     genuinely converged basis, 1e-9 is right; the models that miss it are models whose
///     duals are not converged, and hiding that behind a looser constant is the same
///     mistake in the other direction.
///
/// The honest outcome for such a point is the status kFeasible - a usable primal point,
/// optimality not proven - which solve() now assigns automatically when the measured dual
/// infeasibility exceeds tolerance (issue #27). `etamacro` takes that path today and the
/// verifier accepts it, skipping the strong-duality test because no optimality is claimed.
///
/// What is actually needed for `etamacro` to reach kOptimal is better dual convergence, not
/// a different threshold: Devex pricing (#66) and the Harris ratio test (#67). This constant
/// should be revisited only if those land and instances still miss it.
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

/// Bound relaxation for the Harris two-pass ratio test (issue #67). Pass one finds the
/// tightest step allowed if every candidate row's bound were loosened by this much; pass two
/// then picks, among the rows whose EXACT (unrelaxed) step still fits under that limit, the
/// one with the largest pivot magnitude - buying numerical stability at the cost of a
/// controlled amount of new infeasibility.
///
/// PROVABLY TIGHTER THAN kPrimalFeasibility, which is what makes the trade safe: the
/// realised step is capped at the relaxed limit (see ratio_test()), so a Harris-relaxed pivot
/// alone can move a basic variable at most kHarrisRelaxation + kRatioTestFeasibility past its
/// bound - an order of magnitude inside kPrimalFeasibility, so it can never by itself turn a
/// point recompute_quality() would call feasible into one it calls infeasible.
inline constexpr double kHarrisRelaxation = 0.1 * kPrimalFeasibility;

// ---------------------------------------------------------------------------------------
// First-order method (PDHG)
// ---------------------------------------------------------------------------------------

/// PDHG results are reported at BOTH of these, separately, never blended (CLAUDE.md).
inline constexpr double kPdhgLoose = 1e-4;
inline constexpr double kPdhgTight = 1e-8;

}  // namespace sankhya::tol
