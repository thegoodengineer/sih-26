// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bounded-variable revised dual simplex (issue #65).
//
// References
//   Lemke, C.E. (1954), "The dual method of solving the linear programming problem", Naval
//     Research Logistics Quarterly 1, 36-47 - the method.
//   Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), ch. 10 - the
//     bounded-variable dual, and the bound-flipping ratio test.
//   Koberstein, A. (2005), "The dual simplex method, techniques for a fast and stable
//     implementation", PhD thesis, Universitaet Paderborn - the artificial-bound treatment
//     of dual infeasibility (sec. 4.5) and the practical shape of the whole algorithm.
//   Forrest, J.J. and Goldfarb, D. (1992), "Steepest-edge simplex algorithms for linear
//     programming", Mathematical Programming 57, 341-374 - the dual devex weights.
//
// WHY A SECOND METHOD. The primal simplex keeps a primal feasible point and improves the
// objective; the dual keeps every reduced cost sign-admissible and reduces the primal
// infeasibility. Branch and bound is where that difference is worth a second implementation:
// a child node is the parent with ONE bound moved, so the parent's optimal basis is still
// dual feasible at the child but no longer primal feasible - exactly the state the dual
// starts from, and typically a few pivots from the child's optimum. The primal has to
// re-solve from the slack basis, or run a phase 1 from a basis that is now infeasible.
// Measured on the MIPLIB subset before this landed: every node was a cold primal solve.
//
// THE SHAPE. Everything below runs over the state simplex_core.hpp declares: the same
// working problem in [A | -I] form, the same LU factors and product-form update, the same
// refactorization ladder and basis repair, the same reporting. What is specific to the
// dual is the choice of leaving ROW first (dual pricing, by primal infeasibility weighted
// with dual devex), the pivot row r of B^-1 [A | -I] from one BTRAN, the ratio test over
// the REDUCED COSTS with bound flipping, and what to do about a starting basis that is not
// dual feasible.
//
// DUAL INFEASIBILITY AT THE START. A nonbasic column at its lower bound with a negative
// reduced cost is dual infeasible; if it has an upper bound it is simply moved there, and if
// it has none an ARTIFICIAL upper bound is invented so that it can be (Koberstein sec. 4.5).
// The dual then solves that boxed problem. If at its optimum no column sits on an artificial
// bound, the boxes changed nothing and the answer stands; if one does, the boxes are removed
// and the PRIMAL loop finishes from that basis, which is primal feasible for the true bounds
// on every row but the ones the moved columns touch. No answer is ever reported about the
// boxed problem: the boxes are gone before finish() runs, on every exit.
//
// WHAT IS NOT HERE. Cost perturbation against dual degeneracy, and the Harris variant of the
// dual ratio test. A dual stall hands the basis to the primal loop instead, which has its
// own anti-degeneracy machinery, so a stall costs iterations and never a wrong answer.

#include "primal_simplex.hpp"
#include "simplex_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::detail {

namespace {

/// Width of an artificial bound, in the units of the (scaled) working problem.
///
/// Large enough that a column the true optimum leaves interior is not pinned by it, small
/// enough that the boxed problem is not itself ill-conditioned: 1e6 is Koberstein's value.
/// Its exact size does not affect the answer - a box that binds at the dual's optimum is
/// removed and the primal loop finishes - only how often that hand-over happens.
constexpr double kArtificialBound = 1e6;

/// Restart the dual devex reference framework once any row weight passes this. Same role
/// and same order as kDevexResetThreshold for the primal weights.
constexpr double kDualWeightResetThreshold = 1e6;

/// Relative disagreement between the pivot element as computed along the pivot ROW (from
/// the BTRAN of e_r) and along the entering COLUMN (from the FTRAN of a_q) beyond which the
/// factors are not trusted for this pivot. The two are the same number computed through
/// different solves; on faithful factors they agree to rounding.
///
/// 1e-8, MEASURED. pilot87 hands over on "pivot -5.304e+01 along the column against
/// -5.304e+01 along the row on fresh factors": seven digits of agreement, on a basis whose
/// smallest factorization pivot is 1.6e-7. Loosening this to 1e-6 to let that pivot through
/// was tried: the dual then took it, and the bases it produced from there went singular past
/// the repair limit in 6 s, on both the scaled and the unscaled attempt - a numerical error
/// where the hand-over reaches a verified feasible point. The disagreement is a symptom of
/// the basis, not of the tolerance, and the primal loop is the right place to be on it.
constexpr double kPivotAgreement = 1e-8;

/// COST PERTURBATION, the dual's analogue of the primal loop's bound perturbation. A dual
/// degenerate vertex has many reduced costs at exactly zero, so the dual ratio test ties
/// and the dual step is zero; dfl001 spent 1001 consecutive iterations that way and was
/// handed to the primal, which then needed 15,000 phase-1 iterations. Each NONBASIC cost is
/// shifted by this fraction of max(1, |c_j|), times the same deterministic per-variable
/// factor the bound perturbation uses, in the direction that keeps its reduced cost dual
/// feasible. Nonbasic only: y = B^-T c_B is untouched, so every other reduced cost is too.
///
/// 1e-6 and not kPerturbationSize's 1e-9: the shift has to exceed the dual tolerance the
/// ratio test judges ties by (1e-7), or it changes nothing the test can see.
constexpr double kDualCostPerturbation = 1e-6;

}  // namespace

// -----------------------------------------------------------------------------------------
// Dual feasibility and the artificial bounds
// -----------------------------------------------------------------------------------------

void Simplex::make_dual_feasible() {
  artificial_lower_.assign(static_cast<std::size_t>(total_), 0);
  artificial_upper_.assign(static_cast<std::size_t>(total_), 0);
  Count boxed = 0;
  Count flipped = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    const double d = reduced_cost_[u];
    switch (status_[u]) {
      case BasisStatus::kAtLower:
        if (d < -dual_tolerance_) {
          if (!is_finite_bound(upper_[u])) {
            upper_[u] = lower_[u] + kArtificialBound;
            artificial_upper_[u] = 1;
            ++boxed;
          }
          status_[u] = BasisStatus::kAtUpper;
          nonbasic_value_[u] = upper_[u];
          ++flipped;
        }
        break;
      case BasisStatus::kAtUpper:
        if (d > dual_tolerance_) {
          if (!is_finite_bound(lower_[u])) {
            lower_[u] = upper_[u] - kArtificialBound;
            artificial_lower_[u] = 1;
            ++boxed;
          }
          status_[u] = BasisStatus::kAtLower;
          nonbasic_value_[u] = lower_[u];
          ++flipped;
        }
        break;
      case BasisStatus::kNonbasicFree:
        // A free column's reduced cost must be zero. One that is not is boxed on both
        // sides and parked on the side its sign admits; it enters the basis at the first
        // pivot that can use it, and stays there as a rule.
        if (std::fabs(d) > dual_tolerance_) {
          lower_[u] = -kArtificialBound;
          upper_[u] = kArtificialBound;
          artificial_lower_[u] = 1;
          artificial_upper_[u] = 1;
          ++boxed;
          if (d > 0.0) {
            status_[u] = BasisStatus::kAtLower;
            nonbasic_value_[u] = lower_[u];
          } else {
            status_[u] = BasisStatus::kAtUpper;
            nonbasic_value_[u] = upper_[u];
          }
          ++flipped;
        }
        break;
      case BasisStatus::kFixed:
      case BasisStatus::kBasic:
      case BasisStatus::kUnknown: break;
    }
  }
  if (flipped > 0) {
    logger_.verbose(
        "dual simplex: {} column(s) moved to their other bound and {} given an "
        "artificial bound to make the starting basis dual feasible",
        flipped, boxed);
    compute_basic_values();
  }
}

bool Simplex::any_artificial_bound() const {
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (artificial_lower_[u] != 0 || artificial_upper_[u] != 0) return true;
  }
  return false;
}

bool Simplex::any_artificial_bound_active() const {
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kAtLower && artificial_lower_[u] != 0) return true;
    if (status_[u] == BasisStatus::kAtUpper && artificial_upper_[u] != 0) return true;
  }
  return false;
}

void Simplex::remove_artificial_bounds() {
  bool moved = false;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (artificial_lower_[u] == 0 && artificial_upper_[u] == 0) continue;
    if (k < n_) {
      lower_[u] = model_.col_lower[u];
      upper_[u] = model_.col_upper[u];
    } else {
      const auto r = static_cast<std::size_t>(k - n_);
      lower_[u] = model_.row_lower[r];
      upper_[u] = model_.row_upper[r];
    }
    artificial_lower_[u] = 0;
    artificial_upper_[u] = 0;
    if (basis_position_[u] >= 0) continue;
    // Parked on a bound that no longer exists: the nearest real one, or free at zero.
    // make_nonbasic() is the same choice a basis repair makes.
    make_nonbasic(k, nonbasic_value_[u]);
    moved = true;
  }
  if (moved) compute_basic_values();
}

// -----------------------------------------------------------------------------------------
// Dual pricing
// -----------------------------------------------------------------------------------------

void Simplex::reset_dual_weights() {
  dual_weight_.assign(static_cast<std::size_t>(m_), 1.0);
  ++dual_weight_resets_;
}

Index Simplex::choose_leaving_row() const {
  // The basic variable furthest outside its bounds, in the dual devex metric: infeasibility
  // squared over the row's reference weight. The weight approximates the norm of row r of
  // B^-1, which is what turns a primal infeasibility into a dual step length - the dual
  // analogue of why the primal prices on d^2 / w rather than on |d|.
  Index best = -1;
  double best_score = 0.0;
  for (Index slot = 0; slot < m_; ++slot) {
    const auto s = static_cast<std::size_t>(slot);
    const Index k = basis_[s];
    const double x = x_basic_[s];
    const double lo = lower_[static_cast<std::size_t>(k)];
    const double hi = upper_[static_cast<std::size_t>(k)];
    double infeasibility = 0.0;
    if (is_finite_bound(lo) && x < lo - primal_tolerance_) {
      infeasibility = lo - x;
    } else if (is_finite_bound(hi) && x > hi + primal_tolerance_) {
      infeasibility = x - hi;
    } else {
      continue;
    }
    const double score = infeasibility * infeasibility / dual_weight_[s];
    if (score > best_score) {
      best_score = score;
      best = slot;
    }
  }
  return best;
}

void Simplex::compute_pivot_row(Index leaving_slot) {
  std::fill(rho_.begin(), rho_.end(), 0.0);
  rho_[static_cast<std::size_t>(leaving_slot)] = 1.0;
  lu_.solve_transpose(rho_.data());
  // pivot_row_[k] = e_r^T B^-1 a_k = rho . a_k; for a logical, a_k = -e_i, so it is -rho_i.
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) {
      pivot_row_[u] = 0.0;
      continue;
    }
    double dot = 0.0;
    for_each_entry(k, [&](Index row, double coefficient) {
      dot += rho_[static_cast<std::size_t>(row)] * coefficient;
    });
    pivot_row_[u] = dot;
  }
}

void Simplex::update_dual_weights(Index leaving_slot, double pivot) {
  if (std::fabs(pivot) < tol::kZeroDrop) return;
  const auto r = static_cast<std::size_t>(leaving_slot);
  const double weight_r = dual_weight_[r];
  const double inverse_pivot = 1.0 / pivot;
  double largest = 1.0;
  // w_i <- max(w_i, (alpha_iq / alpha_rq)^2 w_r) for every other row; the row the entering
  // column takes over gets w_r / alpha_rq^2, floored at 1. Exact steepest edge subtracts a
  // cross term that needs a second FTRAN; devex drops it and keeps the max, so the weights
  // only grow inside a reference framework and are reset once they drift too far.
  for (Index slot = 0; slot < m_; ++slot) {
    const auto s = static_cast<std::size_t>(slot);
    if (s == r) continue;
    const double a = alpha_[s];
    if (a == 0.0) continue;
    const double ratio = a * inverse_pivot;
    const double candidate = ratio * ratio * weight_r;
    if (candidate > dual_weight_[s]) dual_weight_[s] = candidate;
    if (dual_weight_[s] > largest) largest = dual_weight_[s];
  }
  dual_weight_[r] = std::max(weight_r * inverse_pivot * inverse_pivot, 1.0);
  if (dual_weight_[r] > largest) largest = dual_weight_[r];
  if (largest > kDualWeightResetThreshold) reset_dual_weights();
}

// -----------------------------------------------------------------------------------------
// The dual ratio test
// -----------------------------------------------------------------------------------------

DualRatioResult Simplex::dual_ratio_test(Index leaving_slot, bool leaving_to_upper) const {
  // The leaving variable x_r is outside its bounds. Moving the duals by t along rho changes
  // every nonbasic reduced cost by d_j <- d_j - s t alpha_rj, where s = +1 when x_r is above
  // its upper bound and -1 when below its lower one (so that the leaving variable's own new
  // reduced cost, -s t, has the sign its destination bound demands). A nonbasic column
  // blocks when its reduced cost would cross zero: at its lower bound when s alpha_rj > 0,
  // at its upper bound when s alpha_rj < 0, and a free column for either sign. The ratio is
  // d_j / (s alpha_rj), never negative - a reduced cost a hair on the wrong side of zero is
  // treated as zero rather than as a step backwards.
  DualRatioResult result;
  const double s = leaving_to_upper ? 1.0 : -1.0;

  struct Candidate {
    Index column;
    double ratio;
    double alpha_abs;
    double range;  ///< upper - lower when both are finite, else infinite
  };
  std::vector<Candidate> candidates;
  candidates.reserve(64);

  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kFixed) continue;
    const double a = s * pivot_row_[u];
    if (std::fabs(a) <= tol::kPivotTolerance) continue;
    bool eligible = false;
    switch (status_[u]) {
      case BasisStatus::kAtLower: eligible = a > 0.0; break;
      case BasisStatus::kAtUpper: eligible = a < 0.0; break;
      case BasisStatus::kNonbasicFree: eligible = true; break;
      case BasisStatus::kFixed:
      case BasisStatus::kBasic:
      case BasisStatus::kUnknown: break;
    }
    if (!eligible) continue;
    double ratio = reduced_cost_[u] / a;
    if (ratio < 0.0) ratio = 0.0;
    const double lo = lower_[u];
    const double hi = upper_[u];
    const double range = is_finite_bound(lo) && is_finite_bound(hi)
                             ? hi - lo
                             : std::numeric_limits<double>::infinity();
    candidates.push_back({k, ratio, std::fabs(a), range});
  }

  if (candidates.empty()) {
    result.dual_unbounded = true;
    return result;
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& x, const Candidate& y) { return x.ratio < y.ratio; });

  // BOUND FLIPPING (Maros ch. 10, Koberstein sec. 3.3). The dual objective along the step
  // is piecewise linear and concave: its slope starts at the leaving row's primal
  // infeasibility and drops by |alpha_rj| * (upper_j - lower_j) each time a BOXED column's
  // reduced cost crosses zero - because that column can simply be moved to its other bound
  // and stay dual feasible. So the step runs past every boxed breakpoint while the slope
  // stays positive, flipping those columns, and stops at the first breakpoint it cannot
  // pass: a column with only one bound, or one whose flip would turn the slope negative.
  // That column enters. On a MILP relaxation nearly every column is boxed, and one dual
  // pivot then does the work of many.
  const Index leaving = basis_[static_cast<std::size_t>(leaving_slot)];
  const double x_r = x_basic_[static_cast<std::size_t>(leaving_slot)];
  double slope = leaving_to_upper ? x_r - upper_[static_cast<std::size_t>(leaving)]
                                  : lower_[static_cast<std::size_t>(leaving)] - x_r;

  std::size_t stop = 0;
  while (stop < candidates.size()) {
    const Candidate& candidate = candidates[stop];
    if (!std::isfinite(candidate.range)) break;
    const double drop = candidate.alpha_abs * candidate.range;
    // Flip only while the row stays infeasible beyond tolerance afterwards; a flip that
    // lands the row inside its bounds is the whole step, and the pivot is not needed.
    if (slope - drop <= primal_tolerance_) break;
    slope -= drop;
    result.flips.push_back(candidate.column);
    ++stop;
  }
  if (stop == candidates.size()) {
    // Every candidate was boxed and passed, and the dual objective still climbs: the dual
    // ray is unbounded, so the primal is infeasible. The flips are moot.
    result.flips.clear();
    result.dual_unbounded = true;
    return result;
  }

  // Among the candidates tied at the stopping ratio, the largest pivot - the same tie-break
  // the primal ratio test uses, for the same reason: a tiny pivot element is how a basis
  // decays.
  const double stop_ratio = candidates[stop].ratio;
  std::size_t best = stop;
  for (std::size_t i = stop; i < candidates.size(); ++i) {
    if (candidates[i].ratio > stop_ratio + tol::kRatioTestFeasibility) break;
    if (candidates[i].alpha_abs > candidates[best].alpha_abs) best = i;
  }
  result.entering = candidates[best].column;
  return result;
}

// -----------------------------------------------------------------------------------------
// The iteration loop
// -----------------------------------------------------------------------------------------

std::optional<Solution> Simplex::dual_loop(Timer& timer, Count* iterations_io) {
  Count& iterations = *iterations_io;
  pivot_row_.assign(static_cast<std::size_t>(total_), 0.0);
  compute_reduced_costs(false);
  make_dual_feasible();
  reset_dual_weights();

  int degenerate_run = 0;
  const auto hand_over = [&](const std::string& why) -> std::optional<Solution> {
    logger_.info(
        "Dual simplex: {} at iteration {}; the primal simplex continues from this "
        "basis",
        why, iterations);
    remove_cost_perturbation();
    remove_artificial_bounds();
    return std::nullopt;
  };
  const auto singular = [&]() {
    remove_cost_perturbation();
    remove_artificial_bounds();
    return finish(SolveStatus::kNumericalError,
                  fmt::format("basis became singular at iteration {}", iterations), iterations,
                  timer.elapsed_seconds());
  };
  // Refactorize and recompute everything the loop reads. Used wherever a claim reached
  // through the eta file must be re-examined on fresh factors before it is acted on.
  const auto refresh = [&]() {
    if (!refactorize()) return false;
    ++refactorizations_;
    compute_basic_values();
    compute_reduced_costs(false);
    return true;
  };

  for (;;) {
    iterations_seen_ = iterations;
    if (iterations % 20 == 0) {
      logger_.iteration(iterations, minimization_objective(), max_infeasibility(), -1.0,
                        timer.elapsed_seconds());
    }

    const Index leaving_slot = choose_leaving_row();
    if (leaving_slot < 0) {
      // PRIMAL FEASIBLE ON FRESH FACTORS OR NOT AT ALL - the same rule the primal loop
      // applies to its optimality claim, for the same reason: the basic values were
      // computed through whatever eta file is in play.
      if (m_ > 0 && lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        logger_.verbose(
            "iteration {}: primal feasible through the eta file; re-checking on "
            "fresh factors",
            iterations);
        continue;
      }
      if (any_artificial_bound_active()) {
        return hand_over("optimal for the boxed problem with an artificial bound active");
      }
      // OPTIMAL FOR THE PERTURBED COSTS IS NOT OPTIMAL, exactly as the primal loop says of
      // its perturbed bounds. The point is primal feasible, and with the exact costs back
      // it is a phase-2 start for the primal loop: usually a handful of pivots.
      if (cost_perturbed_) {
        return hand_over("optimal under cost perturbation; exact costs restored");
      }
      remove_artificial_bounds();
      return finish(SolveStatus::kOptimal, {}, iterations, timer.elapsed_seconds());
    }

    const Index leaving = basis_[static_cast<std::size_t>(leaving_slot)];
    const auto l = static_cast<std::size_t>(leaving);
    const bool to_upper = is_finite_bound(upper_[l]) &&
                          x_basic_[static_cast<std::size_t>(leaving_slot)] > upper_[l];

    compute_pivot_row(leaving_slot);
    const DualRatioResult ratio = dual_ratio_test(leaving_slot, to_upper);

    if (ratio.dual_unbounded) {
      // INFEASIBILITY IS CLAIMED ON FRESH FACTORS, and only about the caller's model. With
      // an artificial bound anywhere the problem being solved is not the caller's, and a
      // box can be what makes it infeasible; the primal loop settles it instead.
      if (lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        continue;
      }
      if (any_artificial_bound()) {
        return hand_over("no dual ratio-test candidate while artificial bounds are in play");
      }
      remove_cost_perturbation();
      // THE SAME CAUTION THE PRIMAL APPLIES TO ITS PHASE-1 STALL. "No candidate" is a Farkas
      // certificate in exact arithmetic; in floating point it ignores every column whose
      // pivot-row entry is below kPivotTolerance, and on a violation within a few orders of
      // the feasibility tolerance one of those could still close it. A marginal violation
      // is therefore handed to the primal loop, whose phase 1 decides between a numerical
      // stall and a proof on its own terms; only a violation far above tolerance is claimed
      // here, and the message says so.
      const double violation =
          to_upper ? x_basic_[static_cast<std::size_t>(leaving_slot)] - upper_[l]
                   : lower_[l] - x_basic_[static_cast<std::size_t>(leaving_slot)];
      if (violation <= kInfeasibilityProofFactor * primal_tolerance_) {
        return hand_over(fmt::format(
            "no dual ratio-test candidate on a marginal violation of {:.3e}", violation));
      }
      return finish(SolveStatus::kInfeasible,
                    fmt::format("dual simplex: basic variable {} is outside its bounds by "
                                "{:.3e}, far above the {:.1e} feasibility tolerance, and no "
                                "nonbasic column can move it (the dual is unbounded), at "
                                "iteration {}",
                                leaving, violation, primal_tolerance_, iterations),
                    iterations, timer.elapsed_seconds());
    }

    for (const Index k : ratio.flips) {
      const auto u = static_cast<std::size_t>(k);
      if (status_[u] == BasisStatus::kAtLower) {
        status_[u] = BasisStatus::kAtUpper;
        nonbasic_value_[u] = upper_[u];
      } else {
        status_[u] = BasisStatus::kAtLower;
        nonbasic_value_[u] = lower_[u];
      }
      ++bound_flips_;
    }
    if (!ratio.flips.empty()) compute_basic_values();

    const double x_now = x_basic_[static_cast<std::size_t>(leaving_slot)];
    const double target = to_upper ? upper_[l] : lower_[l];
    const double residual = to_upper ? x_now - target : target - x_now;
    // A LIMIT EXIT STILL CARRIES A BOUND. The basis is dual feasible at every iteration of
    // this loop, so the objective of its (primal infeasible) basic solution is the dual
    // objective: a valid bound on the LP optimum - as long as no artificial bound is in
    // play, since those bound a different problem. Strong branching (#69) caps its probes
    // at a few dozen iterations and reads exactly this; finish() itself only ever states a
    // bound for an optimal exit.
    const auto stop_at_limit = [&](SolveStatus status, const std::string& why) {
      // Exact costs first: the bound below is the objective of the basic solution, and a
      // perturbed cost vector would put a perturbation-sized error into a number that
      // branch and bound prunes against.
      remove_cost_perturbation();
      const bool bound_is_valid = !any_artificial_bound();
      const double bound = minimization_objective();
      remove_artificial_bounds();
      compute_reduced_costs(false);
      Solution stopped = finish(status, why, iterations, timer.elapsed_seconds());
      if (bound_is_valid) stopped.dual_bound = bound;
      return stopped;
    };
    const auto count_iteration = [&]() -> std::optional<Solution> {
      ++iterations;
      ++dual_iterations_;
      if (iteration_limit_ >= 0 && iterations >= iteration_limit_) {
        return stop_at_limit(SolveStatus::kIterationLimit,
                             fmt::format("iteration limit {} reached", iteration_limit_));
      }
      const double elapsed = timer.elapsed_seconds();
      if (elapsed > time_limit_) {
        return stop_at_limit(SolveStatus::kTimeLimit,
                             fmt::format("time limit {:g}s reached", time_limit_));
      }
      return std::nullopt;
    };
    if (residual <= primal_tolerance_ || ratio.entering < 0) {
      // The flips alone brought the row inside its bounds; no basis change this iteration.
      if (std::optional<Solution> stopped = count_iteration()) return stopped;
      continue;
    }

    const Index entering = ratio.entering;
    const auto e = static_cast<std::size_t>(entering);
    ftran_entering_column(entering);
    const double pivot = alpha_[static_cast<std::size_t>(leaving_slot)];
    const double pivot_by_row = pivot_row_[e];

    // THE PIVOT IS COMPUTED TWICE AND MUST AGREE. alpha_rq along the row came through the
    // BTRAN of e_r, along the column through the FTRAN of a_q; they are the same entry of
    // B^-1 a_q. A disagreement is the factors talking, not the model: refactorize and take
    // the iteration again. If fresh factors still disagree, or the pivot is negligible, the
    // basis is beyond what this loop can do with it and the primal loop takes over.
    const bool agree = std::fabs(pivot - pivot_by_row) <=
                       kPivotAgreement * std::max(1.0, std::fabs(pivot_by_row));
    if (!agree || std::fabs(pivot) <= tol::kPivotTolerance) {
      if (lu_.eta_count() > 0) {
        if (!refresh()) return singular();
        logger_.verbose(
            "iteration {}: pivot row/column disagreement {:.3e} vs {:.3e} through "
            "the eta file; refactorized",
            iterations, pivot_by_row, pivot);
        continue;
      }
      return hand_over(
          fmt::format("pivot {:.3e} along the column against {:.3e} along the "
                      "row on fresh factors",
                      pivot, pivot_by_row));
    }

    // Dual step length, for the stall counter only: the duals themselves are recomputed
    // from the new basis below rather than updated.
    const double dual_step = reduced_cost_[e] / pivot_by_row;
    if (std::fabs(dual_step) <= tol::kRatioTestFeasibility) {
      ++degenerate_run;
      // PERTURB BEFORE HANDING OVER. The primal loop is the last resort, and on dfl001 it
      // cost 15,000 phase-1 iterations; breaking the ties costs one pass over the costs.
      // The pivot in hand is discarded and the iteration taken again from the perturbed
      // reduced costs, which the ratio test now sees as distinct.
      if (!cost_perturbed_ && degenerate_run > kPerturbationTrigger) {
        logger_.verbose(
            "{} consecutive dual-degenerate iterations at iteration {}: perturbing costs",
            degenerate_run, iterations);
        perturb_costs();
        degenerate_run = 0;
        continue;
      }
      if (degenerate_run > kStallLimit) {
        return hand_over(
            fmt::format("{} consecutive dual-degenerate iterations", degenerate_run));
      }
    } else {
      degenerate_run = 0;
    }

    // Before the basis changes: the weight update reads alpha_ (already computed) and the
    // leaving slot's weight.
    update_dual_weights(leaving_slot, pivot);

    // The leaving variable lands exactly on the bound it violated; the entering variable's
    // value follows from the row: x_r - delta * alpha_rq = bound, so delta = (x_r - bound) /
    // alpha_rq is the change in x_q.
    basis_position_[l] = -1;
    nonbasic_value_[l] = target;
    status_[l] = to_upper ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
    if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;

    const double delta = (x_now - target) / pivot;
    basis_[static_cast<std::size_t>(leaving_slot)] = entering;
    basis_position_[e] = leaving_slot;
    status_[e] = BasisStatus::kBasic;
    nonbasic_value_[e] += delta;

    // The same update-or-refactorize policy as the primal loop, for the same reasons; see
    // the primal loop for the evidence behind each trigger.
    const bool trust_update = !basis_needed_stricter_threshold_;
    const bool updated = trust_update && lu_.update(leaving_slot, alpha_.data());
    if (trust_update && !updated) ++rejected_updates_;
    eta_work_since_refactor_ += static_cast<double>(lu_.eta_nonzeros());
    const bool past_break_even =
        eta_work_since_refactor_ >
        refactor_work_ratio_ * std::max(1.0, static_cast<double>(lu_.factor_nonzeros()));
    if (!updated || past_break_even || lu_.should_refactorize()) {
      if (!refactorize()) return singular();
      ++refactorizations_;
    }
    compute_basic_values();
    compute_reduced_costs(false);

    if (std::optional<Solution> stopped = count_iteration()) return stopped;
  }
}

void Simplex::perturb_costs() {
  if (cost_perturbed_) return;
  unperturbed_cost_ = cost_;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    // The per-variable factor in (0.25, 1] that perturbation_for() encodes, rescaled from
    // the bound perturbation's size to the cost one's, times the cost's own magnitude.
    const double factor = perturbation_for(k) / kPerturbationSize;
    const double shift = kDualCostPerturbation * factor * std::max(1.0, std::fabs(cost_[u]));
    if (status_[u] == BasisStatus::kAtLower) {
      cost_[u] += shift;  // d_j = c_j - a_j^T y grows: further inside dual feasibility
    } else if (status_[u] == BasisStatus::kAtUpper) {
      cost_[u] -= shift;  // d_j shrinks: further inside on the upper side
    }
    // Fixed and free nonbasic columns are left alone: a fixed one has no sign condition
    // to protect, and a free one must keep its reduced cost at zero.
  }
  cost_perturbed_ = true;
  ++cost_perturbations_;
  compute_reduced_costs(false);
}

void Simplex::remove_cost_perturbation() {
  if (!cost_perturbed_) return;
  cost_ = unperturbed_cost_;
  cost_perturbed_ = false;
  compute_reduced_costs(false);
}

Solution Simplex::run_dual(const WarmStart* warm) {
  Timer timer;
  algorithm_name_ = "simplex-dual";
  if (std::optional<Solution> early = prepare(warm, timer)) return *early;

  logger_.info("Dual simplex: {} rows, {} columns, {} nonzeros{}", m_, n_,
               model_.num_nonzeros(), warm_started_ ? ", warm start" : "");
  logger_.begin_iteration_table();

  Count iterations = 0;
  std::optional<Solution> done = dual_loop(timer, &iterations);
  if (bound_flips_ > 0 || dual_iterations_ > 0) {
    logger_.verbose(
        "dual simplex: {} iterations, {} bound flips, {} weight resets, {} cost "
        "perturbation(s)",
        dual_iterations_, bound_flips_, dual_weight_resets_, cost_perturbations_);
  }
  if (done) return *done;
  algorithm_name_ = "simplex-dual+primal";
  return primal_loop(timer, &iterations);
}

}  // namespace sankhya::detail
