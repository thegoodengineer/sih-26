// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the primal push of a crossover (#343): from an interior point on the optimal
// face to a vertex of it, one superbasic variable at a time.
//
// THE PROBLEM THE PUSH SOLVES. A basis guessed off an interior point (crossover.cpp) sends
// every nonbasic variable to a bound at once, and a point that had thousands of variables
// strictly inside their bounds lands far from feasibility; the dual simplex then rebuilds
// feasibility with as many pivots as a cold start (41,872 against 34,071 on the 5,000-row
// staircase model). The push keeps the point FEASIBLE throughout instead: every nonbasic
// variable starts at the interior point's value - "superbasic", nonbasic but not at a bound
// - and is moved to its nearer bound along a direction that keeps A x = b, with a ratio test
// against the basic variables. When a basic variable would leave its bounds first, it leaves
// the basis and the pushed variable enters: one pivot. When nothing blocks, the variable
// simply arrives at its bound and the basis is unchanged. After the last push every nonbasic
// variable is at a bound, the point is a primal feasible vertex on (or near) the optimal face,
// and the ordinary primal loop finishes from there in the few pivots the residual
// degeneracy needs.
//
// Bixby, "Solving real-world linear programs: a decade and more of progress", Operations
// Research 50 (2002), sec. 4; Andersen & Ye, "Combining interior-point and pivoting
// algorithms for constrained linear programs", Management Science 42 (1996); Megiddo, "On
// finding primal- and dual-optimal bases", ORSA J. Computing 3 (1991) for the argument that
// the push terminates at a vertex of the optimal face.

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "primal_simplex.hpp"
#include "sankhya/tolerances.hpp"
#include "simplex_core.hpp"

namespace sankhya::detail {

Solution Simplex::run_push(const WarmStart& warm, const std::vector<double>& interior_x,
                           const std::vector<double>& interior_activity) {
  Timer timer;
  arm_deadline(timer);
  if (std::optional<Solution> early = prepare(&warm, timer)) return *early;
  logger_.info("Crossover push: {} rows, {} columns, {} nonzeros{}", m_, n_,
               model_.num_nonzeros(),
               warm_started_ ? ", basis guess installed" : ", slack basis");

  // Every nonbasic variable that the interior point held strictly inside its bounds starts
  // there. The working problem's logical for row i carries the row's activity, so the same
  // rule applies to rows. A fixed variable has nowhere to go.
  std::vector<Index> superbasic;
  const bool have_point = static_cast<Index>(interior_x.size()) == n_ &&
                          static_cast<Index>(interior_activity.size()) == m_;
  if (have_point) {
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (basis_position_[u] >= 0 || status_[u] == BasisStatus::kFixed) continue;
      const double v =
          k < n_ ? interior_x[u] : interior_activity[static_cast<std::size_t>(k - n_)];
      if (!std::isfinite(v)) continue;
      const double scale = std::max(1.0, std::fabs(v));
      const bool inside_lower =
          !is_finite_bound(lower_[u]) || v - lower_[u] > primal_tolerance_ * scale;
      const bool inside_upper =
          !is_finite_bound(upper_[u]) || upper_[u] - v > primal_tolerance_ * scale;
      if (!inside_lower || !inside_upper) continue;
      nonbasic_value_[u] = v;
      superbasic.push_back(k);
    }
  }
  compute_basic_values();
  // THE PUSH RESTS ON THE POINT BEING FEASIBLE. x_B is recomputed from the superbasic
  // values through the guessed basis, and on an ill-conditioned model (wood1p) that basis
  // can amplify the interior point's residual into a point that is not feasible at all;
  // pushing from there, and then asking the primal loop to repair 1e-2 of infeasibility on
  // a massively degenerate model, ran 683,000 pivots to a time limit. A point that is not
  // feasible to a loose multiple of the tolerance is handed back as "not solved", and the
  // caller keeps the interior point's answer.
  const double loose = 1e3 * primal_tolerance_;
  if (max_infeasibility() > loose) {
    return finish(SolveStatus::kNotSolved,
                  fmt::format("crossover push declined: the basis guess reproduces the "
                              "interior point to an infeasibility of {:.3e}",
                              max_infeasibility()),
                  0, timer.elapsed_seconds());
  }

  Count pivots = 0;
  Count arrived = 0;
  Count refactorizations = 0;
  Count since_refactor = 0;
  bool deadline_hit = false;
  for (const Index k : superbasic) {
    if (deadline_ && (pivots & 63) == 0 && deadline_()) {
      deadline_hit = true;
      break;
    }
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;  // entered the basis while another was pushed
    const double value = nonbasic_value_[u];
    // The bound to push towards: the nearer finite one; a free variable goes to zero, the
    // value the simplex holds a nonbasic free variable at.
    double target;
    BasisStatus target_status;
    const bool has_lower = is_finite_bound(lower_[u]);
    const bool has_upper = is_finite_bound(upper_[u]);
    if (has_lower && has_upper) {
      const bool nearer_upper = upper_[u] - value < value - lower_[u];
      target = nearer_upper ? upper_[u] : lower_[u];
      target_status = nearer_upper ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
    } else if (has_lower) {
      target = lower_[u];
      target_status = BasisStatus::kAtLower;
    } else if (has_upper) {
      target = upper_[u];
      target_status = BasisStatus::kAtUpper;
    } else {
      target = 0.0;
      target_status = BasisStatus::kNonbasicFree;
    }
    const int direction = target >= value ? 1 : -1;
    const double distance = std::fabs(target - value);
    if (distance == 0.0) {
      status_[u] = target_status;
      ++arrived;
      continue;
    }

    // alpha = B^-1 a_k: how the basic variables move per unit of the push.
    ftran_entering_column(k);
    double step = distance;
    Index leaving_slot = -1;
    bool leaving_to_upper = false;
    for (Index slot = 0; slot < m_; ++slot) {
      const double a = alpha_[static_cast<std::size_t>(slot)];
      if (std::fabs(a) <= tol::kPivotTolerance) continue;
      // x_B changes by -direction * a per unit step.
      const double rate = -static_cast<double>(direction) * a;
      const auto i = static_cast<std::size_t>(basis_[static_cast<std::size_t>(slot)]);
      const double xb = x_basic_[static_cast<std::size_t>(slot)];
      double room;
      if (rate < 0.0) {
        if (!is_finite_bound(lower_[i])) continue;
        room = xb - lower_[i];
      } else {
        if (!is_finite_bound(upper_[i])) continue;
        room = upper_[i] - xb;
      }
      const double t = std::max(0.0, room) / std::fabs(rate);
      if (t < step) {
        step = t;
        leaving_slot = slot;
        leaving_to_upper = rate > 0.0;
      }
    }

    for (Index slot = 0; slot < m_; ++slot) {
      x_basic_[static_cast<std::size_t>(slot)] -=
          static_cast<double>(direction) * step * alpha_[static_cast<std::size_t>(slot)];
    }
    if (leaving_slot < 0) {
      nonbasic_value_[u] = target;
      status_[u] = target_status;
      ++arrived;
      continue;
    }

    // A basic variable blocks: it leaves at the bound it reached, the pushed variable
    // enters at the value the step carried it to. The same swap the primal loop makes.
    const auto slot = static_cast<std::size_t>(leaving_slot);
    const Index leaving = basis_[slot];
    const auto l = static_cast<std::size_t>(leaving);
    basis_position_[l] = -1;
    if (leaving_to_upper) {
      nonbasic_value_[l] = upper_[l];
      status_[l] = BasisStatus::kAtUpper;
    } else {
      nonbasic_value_[l] = lower_[l];
      status_[l] = BasisStatus::kAtLower;
    }
    if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;
    basis_[slot] = k;
    basis_position_[u] = leaving_slot;
    status_[u] = BasisStatus::kBasic;
    x_basic_[slot] = value + static_cast<double>(direction) * step;
    nonbasic_value_[u] = x_basic_[slot];
    ++pivots;
    ++since_refactor;
    const bool updated = lu_.update(leaving_slot, alpha_.data());
    if (!updated || since_refactor >= 100) {
      if (!refactorize()) {
        if (factors_abandoned_) return factorization_failed(pivots, timer);
        return finish(SolveStatus::kNumericalError,
                      "the basis went singular during the crossover push", pivots,
                      timer.elapsed_seconds());
      }
      ++refactorizations;
      since_refactor = 0;
      compute_basic_values();
    }
  }
  if (!refactorize()) {
    if (factors_abandoned_) return factorization_failed(pivots, timer);
    return finish(SolveStatus::kNumericalError,
                  "the basis went singular after the crossover push", pivots,
                  timer.elapsed_seconds());
  }
  compute_basic_values();
  reset_devex();
  logger_.info(
      "Crossover push: {} superbasic variable(s), {} arrived at a bound without a pivot, {} "
      "basis changes, {} refactorizations, max infeasibility {:.3e}{}",
      superbasic.size(), arrived, pivots, refactorizations, max_infeasibility(),
      deadline_hit ? " (time limit reached inside the push)" : "");
  if (max_infeasibility() > loose) {
    return finish(SolveStatus::kNotSolved,
                  fmt::format("crossover push declined: the pushed point is {:.3e} infeasible, "
                              "beyond what the primal loop should be asked to repair",
                              max_infeasibility()),
                  pivots, timer.elapsed_seconds());
  }

  // The vertex is primal feasible (up to the push's arithmetic); the primal loop prices the
  // reduced costs and finishes. Its iteration count continues from the push's pivots.
  Count iterations = pivots;
  Solution solution = primal_loop(timer, &iterations);
  const std::string note = fmt::format("crossover push: {} superbasic, {} pivots in the push",
                                       superbasic.size(), pivots);
  solution.message = solution.message.empty() ? note : solution.message + "; " + note;
  return solution;
}

}  // namespace sankhya::detail
