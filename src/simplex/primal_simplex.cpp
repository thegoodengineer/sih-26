// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bounded-variable revised primal simplex.
//
// References
//   Dantzig, "Linear Programming and Extensions" (Princeton, 1963) - the method.
//   Chvatal, "Linear Programming" (Freeman, 1983), ch. 3 and 8 - the bounded-variable form
//     and Bland's anti-cycling rule.
//   Maros, "Computational Techniques of the Simplex Method" (Kluwer, 2003), ch. 9 - the
//     piecewise-linear (composite) phase 1 used below.
//
// FORMULATION. Every row gets a logical variable, so the working system is
//
//     [ A | -I ] [ x ; s ] = 0,     row_lower <= s <= row_upper,
//                                   col_lower <= x <= col_upper
//
// with n structural variables indexed [0, n) and m logical variables indexed [n, n + m).
// A basis is m of those columns. The all-logical basis B = -I is available for free, is
// always nonsingular, and is where every solve starts.
//
// WHY NO BIG-M. The obvious phase 1 adds an artificial variable per row with a large cost
// M. It is easy to write and it is a numerical trap: M has to dominate the real objective
// to force feasibility first, so the cost vector spans M and the original coefficients at
// once, which is precisely how you manufacture an ill-conditioned pricing step. Too small
// an M silently returns an infeasible point as optimal. There is no value of M that is
// right for every model, and the problem statement specifically asks about ill-conditioned
// instances.
//
// Instead phase 1 minimises the piecewise-linear sum of bound violations of the basic
// variables directly. The starting basis is the slack basis, no artificial columns are
// introduced at all, and the phase-1 gradient is exactly -1 / 0 / +1 per basic variable.
// It is bounded below by zero by construction, so "phase 1 stalls with no improving
// column" is a proof of infeasibility rather than an inconclusive result.
//
// SCOPE. Dantzig pricing with a Bland fallback, a textbook ratio test, and a full dense
// refactorization every iteration. Devex pricing, the Harris two-pass ratio test, bound
// flipping, perturbation and the sparse LU all arrive in Phase 6. Adding them now would
// mean debugging five interacting approximations at once against a simplex that has never
// been shown to be right, which is the opposite of the order this project needs.

#include "primal_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "dense_lu.hpp"

namespace sankhya {
namespace {

/// Consecutive zero-length steps tolerated before the solve is declared stalled. Twenty
/// times the Bland switch threshold: long enough that no honest degenerate plateau trips
/// it, short enough that a genuine cycle is reported in under a second.
constexpr int kStallLimit = 20 * tol::kBlandSwitchIterations;

/// How a basic variable sits relative to its own bounds. Phase 1 exists to empty the two
/// outer categories.
enum class Position { kBelowLower, kFeasible, kAboveUpper };

/// Outcome of the ratio test.
struct RatioResult {
  double step = 0.0;
  Index leaving_position = -1;  ///< index into basis_, or -1 for a bound flip
  bool leaving_to_upper = false;
  bool unbounded = false;
};

class PrimalSimplex {
 public:
  PrimalSimplex(const Model& model, const Options& options, Logger& logger)
      : model_(model), options_(options), logger_(logger) {}

  Solution run();

 private:
  void build_working_problem();
  void set_initial_basis();

  /// Rebuild and refactorize the basis matrix from scratch. Returns false when singular.
  [[nodiscard]] bool refactorize();

  /// x_B = B^{-1} (-N x_N). Recomputed from the bounds every iteration rather than updated,
  /// so no round-off accumulates across pivots.
  void compute_basic_values();

  [[nodiscard]] Position position_of(Index basic_slot) const;
  [[nodiscard]] double total_infeasibility() const;

  /// Fill cost_basic_ with the phase-1 gradient or the phase-2 costs, then BTRAN for y and
  /// price every nonbasic column into reduced_cost_.
  void compute_reduced_costs(bool phase_one);

  /// Choose an entering column. Returns -1 when none is eligible.
  [[nodiscard]] Index price(bool bland, int* direction) const;

  void ftran_entering_column(Index entering);
  [[nodiscard]] RatioResult ratio_test(Index entering, int direction, bool phase_one) const;

  /// Iterate over the entries of column k of [A | -I].
  template <typename Fn>
  void for_each_entry(Index k, Fn&& fn) const {
    if (k < n_) {
      const ColumnView column = model_.matrix.column(k);
      for (Index p = 0; p < column.size; ++p) fn(column.rows[p], column.values[p]);
    } else {
      fn(k - n_, -1.0);
    }
  }

  [[nodiscard]] double variable_value(Index k) const {
    const Index slot = basis_position_[static_cast<std::size_t>(k)];
    return slot >= 0 ? x_basic_[static_cast<std::size_t>(slot)]
                     : nonbasic_value_[static_cast<std::size_t>(k)];
  }

  [[nodiscard]] double minimization_objective() const;
  Solution finish(SolveStatus status, const std::string& message, Count iterations,
                  double seconds);

  const Model& model_;
  const Options& options_;
  Logger& logger_;

  Index n_ = 0;
  Index m_ = 0;
  Index total_ = 0;

  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<double> cost_;  ///< always in MINIMIZATION sense

  std::vector<Index> basis_;
  std::vector<Index> basis_position_;  ///< -1 when nonbasic
  std::vector<BasisStatus> status_;
  std::vector<double> nonbasic_value_;

  DenseLu lu_;
  std::vector<double> x_basic_;
  std::vector<double> cost_basic_;
  std::vector<double> y_;
  std::vector<double> reduced_cost_;
  std::vector<double> alpha_;
  double primal_tolerance_ = tol::kPrimalFeasibility;
  double dual_tolerance_ = tol::kDualFeasibility;
};

// -----------------------------------------------------------------------------------------
// Set-up
// -----------------------------------------------------------------------------------------

void PrimalSimplex::build_working_problem() {
  n_ = model_.num_cols();
  m_ = model_.num_rows();
  total_ = n_ + m_;

  const double sense = model_.sense_multiplier();
  lower_.resize(static_cast<std::size_t>(total_));
  upper_.resize(static_cast<std::size_t>(total_));
  cost_.assign(static_cast<std::size_t>(total_), 0.0);

  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    lower_[u] = model_.col_lower[u];
    upper_[u] = model_.col_upper[u];
    cost_[u] = sense * model_.col_cost[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(n_ + i);
    const auto r = static_cast<std::size_t>(i);
    lower_[u] = model_.row_lower[r];
    upper_[u] = model_.row_upper[r];
  }

  x_basic_.assign(static_cast<std::size_t>(m_), 0.0);
  cost_basic_.assign(static_cast<std::size_t>(m_), 0.0);
  y_.assign(static_cast<std::size_t>(m_), 0.0);
  alpha_.assign(static_cast<std::size_t>(m_), 0.0);
  reduced_cost_.assign(static_cast<std::size_t>(total_), 0.0);
}

void PrimalSimplex::set_initial_basis() {
  basis_.resize(static_cast<std::size_t>(m_));
  basis_position_.assign(static_cast<std::size_t>(total_), -1);
  status_.assign(static_cast<std::size_t>(total_), BasisStatus::kUnknown);
  nonbasic_value_.assign(static_cast<std::size_t>(total_), 0.0);

  for (Index i = 0; i < m_; ++i) {
    const Index logical = n_ + i;
    basis_[static_cast<std::size_t>(i)] = logical;
    basis_position_[static_cast<std::size_t>(logical)] = i;
    status_[static_cast<std::size_t>(logical)] = BasisStatus::kBasic;
  }

  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double lo = lower_[u];
    const double hi = upper_[u];
    if (lo == hi) {
      status_[u] = BasisStatus::kFixed;
      nonbasic_value_[u] = lo;
    } else if (is_finite_bound(lo)) {
      status_[u] = BasisStatus::kAtLower;
      nonbasic_value_[u] = lo;
    } else if (is_finite_bound(hi)) {
      status_[u] = BasisStatus::kAtUpper;
      nonbasic_value_[u] = hi;
    } else {
      status_[u] = BasisStatus::kNonbasicFree;
      nonbasic_value_[u] = 0.0;
    }
  }
}

bool PrimalSimplex::refactorize() {
  std::vector<double> columns(static_cast<std::size_t>(m_) * static_cast<std::size_t>(m_), 0.0);
  for (Index slot = 0; slot < m_; ++slot) {
    const Index k = basis_[static_cast<std::size_t>(slot)];
    const std::size_t base = static_cast<std::size_t>(slot) * static_cast<std::size_t>(m_);
    for_each_entry(k, [&](Index row, double value) {
      columns[base + static_cast<std::size_t>(row)] += value;
    });
  }
  return lu_.factorize(std::move(columns), m_, tol::kPivotTolerance);
}

void PrimalSimplex::compute_basic_values() {
  // [A | -I][x ; s] = 0, so B x_B = -N x_N.
  std::vector<double> rhs(static_cast<std::size_t>(m_), 0.0);
  for (Index k = 0; k < total_; ++k) {
    if (basis_position_[static_cast<std::size_t>(k)] >= 0) continue;
    const double value = nonbasic_value_[static_cast<std::size_t>(k)];
    if (value == 0.0) continue;
    for_each_entry(k, [&](Index row, double coefficient) {
      rhs[static_cast<std::size_t>(row)] -= coefficient * value;
    });
  }
  lu_.solve(rhs.data());
  x_basic_ = std::move(rhs);
}

// -----------------------------------------------------------------------------------------
// Phase classification and pricing
// -----------------------------------------------------------------------------------------

Position PrimalSimplex::position_of(Index basic_slot) const {
  const Index k = basis_[static_cast<std::size_t>(basic_slot)];
  const double x = x_basic_[static_cast<std::size_t>(basic_slot)];
  const double lo = lower_[static_cast<std::size_t>(k)];
  const double hi = upper_[static_cast<std::size_t>(k)];
  if (is_finite_bound(lo) && x < lo - primal_tolerance_) return Position::kBelowLower;
  if (is_finite_bound(hi) && x > hi + primal_tolerance_) return Position::kAboveUpper;
  return Position::kFeasible;
}

double PrimalSimplex::total_infeasibility() const {
  double sum = 0.0;
  for (Index slot = 0; slot < m_; ++slot) {
    const Index k = basis_[static_cast<std::size_t>(slot)];
    const double x = x_basic_[static_cast<std::size_t>(slot)];
    const double lo = lower_[static_cast<std::size_t>(k)];
    const double hi = upper_[static_cast<std::size_t>(k)];
    if (is_finite_bound(lo) && x < lo) sum += lo - x;
    if (is_finite_bound(hi) && x > hi) sum += x - hi;
  }
  return sum;
}

void PrimalSimplex::compute_reduced_costs(bool phase_one) {
  if (phase_one) {
    // Gradient of sum of bound violations with respect to each basic variable. Nonbasic
    // variables sit exactly on a bound and contribute nothing, so their phase-1 cost is 0.
    for (Index slot = 0; slot < m_; ++slot) {
      switch (position_of(slot)) {
        case Position::kBelowLower: cost_basic_[static_cast<std::size_t>(slot)] = -1.0; break;
        case Position::kAboveUpper: cost_basic_[static_cast<std::size_t>(slot)] = 1.0; break;
        case Position::kFeasible: cost_basic_[static_cast<std::size_t>(slot)] = 0.0; break;
      }
    }
  } else {
    for (Index slot = 0; slot < m_; ++slot) {
      cost_basic_[static_cast<std::size_t>(slot)] =
          cost_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(slot)])];
    }
  }

  y_ = cost_basic_;
  lu_.solve_transpose(y_.data());

  for (Index k = 0; k < total_; ++k) {
    if (basis_position_[static_cast<std::size_t>(k)] >= 0) {
      reduced_cost_[static_cast<std::size_t>(k)] = 0.0;
      continue;
    }
    double dot = 0.0;
    for_each_entry(k, [&](Index row, double coefficient) {
      dot += y_[static_cast<std::size_t>(row)] * coefficient;
    });
    const double own_cost = phase_one ? 0.0 : cost_[static_cast<std::size_t>(k)];
    reduced_cost_[static_cast<std::size_t>(k)] = own_cost - dot;
  }
}

Index PrimalSimplex::price(bool bland, int* direction) const {
  Index best = -1;
  double best_magnitude = dual_tolerance_;

  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kFixed) continue;

    const double d = reduced_cost_[u];
    int candidate_direction = 0;
    if (status_[u] == BasisStatus::kAtLower) {
      if (d < -dual_tolerance_) candidate_direction = 1;
    } else if (status_[u] == BasisStatus::kAtUpper) {
      if (d > dual_tolerance_) candidate_direction = -1;
    } else {  // free, held at zero: either direction is available
      if (d < -dual_tolerance_) {
        candidate_direction = 1;
      } else if (d > dual_tolerance_) {
        candidate_direction = -1;
      }
    }
    if (candidate_direction == 0) continue;

    if (bland) {
      // Bland's rule: the lowest index that prices out. Provably terminates, prices badly,
      // which is why it is only reached after a run of degenerate iterations.
      *direction = candidate_direction;
      return k;
    }
    const double magnitude = std::fabs(d);
    if (magnitude > best_magnitude) {
      best_magnitude = magnitude;
      best = k;
      *direction = candidate_direction;
    }
  }
  return best;
}

void PrimalSimplex::ftran_entering_column(Index entering) {
  std::fill(alpha_.begin(), alpha_.end(), 0.0);
  for_each_entry(entering, [&](Index row, double value) {
    alpha_[static_cast<std::size_t>(row)] += value;
  });
  lu_.solve(alpha_.data());
}

// -----------------------------------------------------------------------------------------
// Ratio test
// -----------------------------------------------------------------------------------------

RatioResult PrimalSimplex::ratio_test(Index entering, int direction, bool phase_one) const {
  RatioResult result;
  const double sign = static_cast<double>(direction);

  // The entering variable's own range limits the step even when nothing blocks: moving from
  // one finite bound to the other is a bound flip and leaves the basis untouched.
  double best_step = std::numeric_limits<double>::infinity();
  const auto e = static_cast<std::size_t>(entering);
  if (is_finite_bound(lower_[e]) && is_finite_bound(upper_[e])) {
    best_step = upper_[e] - lower_[e];
  }
  Index best_slot = -1;
  bool best_to_upper = false;
  double best_pivot_magnitude = 0.0;

  for (Index slot = 0; slot < m_; ++slot) {
    const double a = alpha_[static_cast<std::size_t>(slot)];
    // rate = d(x_B[slot]) / dt as the entering variable moves in `direction`.
    const double rate = -sign * a;
    if (std::fabs(rate) <= tol::kPivotTolerance) continue;

    const Index k = basis_[static_cast<std::size_t>(slot)];
    const double x = x_basic_[static_cast<std::size_t>(slot)];
    const double lo = lower_[static_cast<std::size_t>(k)];
    const double hi = upper_[static_cast<std::size_t>(k)];

    double step = std::numeric_limits<double>::infinity();
    bool to_upper = false;

    const Position position = phase_one ? position_of(slot) : Position::kFeasible;
    switch (position) {
      case Position::kBelowLower:
        // Infeasible below its lower bound. Moving up, it becomes feasible exactly at the
        // bound and we stop there. The full piecewise-linear ratio test would be allowed to
        // step past this breakpoint - the phase-1 slope only changes there, it does not
        // reverse - and would take longer steps. Stopping is correct, just less efficient,
        // and it keeps the test a single comparison. Phase 6 revisits this.
        if (rate > 0.0 && is_finite_bound(lo)) step = (lo - x) / rate;
        break;
      case Position::kAboveUpper:
        if (rate < 0.0 && is_finite_bound(hi)) {
          step = (hi - x) / rate;
          to_upper = true;
        }
        break;
      case Position::kFeasible:
        if (rate > 0.0 && is_finite_bound(hi)) {
          step = (hi - x) / rate;
          to_upper = true;
        } else if (rate < 0.0 && is_finite_bound(lo)) {
          step = (lo - x) / rate;
        }
        break;
    }

    if (!(step < std::numeric_limits<double>::infinity())) continue;
    // A basic variable already a hair outside its bound would otherwise yield a small
    // negative step and drive the iterate backwards.
    if (step < 0.0) step = 0.0;

    const double pivot_magnitude = std::fabs(a);
    const bool strictly_shorter = step < best_step - tol::kRatioTestFeasibility;
    const bool tied_but_stabler = std::fabs(step - best_step) <= tol::kRatioTestFeasibility &&
                                  pivot_magnitude > best_pivot_magnitude;
    if (strictly_shorter || tied_but_stabler) {
      best_step = step;
      best_slot = slot;
      best_to_upper = to_upper;
      best_pivot_magnitude = pivot_magnitude;
    }
  }

  if (!(best_step < std::numeric_limits<double>::infinity())) {
    result.unbounded = true;
    return result;
  }
  result.step = best_step;
  result.leaving_position = best_slot;
  result.leaving_to_upper = best_to_upper;
  return result;
}

// -----------------------------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------------------------

double PrimalSimplex::minimization_objective() const {
  double sum = 0.0;
  for (Index k = 0; k < n_; ++k) sum += cost_[static_cast<std::size_t>(k)] * variable_value(k);
  // cost_ is stored in minimization sense; undo that and add the offset so the number in
  // the iteration table is the same quantity the final line and the .sol file report.
  return model_.sense_multiplier() * sum + model_.objective_offset;
}

Solution PrimalSimplex::finish(SolveStatus status, const std::string& message, Count iterations,
                               double seconds) {
  Solution solution;
  solution.allocate_for(model_);
  solution.status = status;
  solution.algorithm = "simplex-primal";
  solution.message = message;
  solution.iterations = iterations;
  solution.solve_seconds = seconds;

  const bool have_point = status == SolveStatus::kOptimal || status == SolveStatus::kFeasible ||
                          status == SolveStatus::kIterationLimit ||
                          status == SolveStatus::kTimeLimit;
  if (!have_point) {
    solution.recompute_quality(model_);
    solution.dual_bound = status == SolveStatus::kInfeasible ? kInfinity : -kInfinity;
    return solution;
  }

  const double sense = model_.sense_multiplier();
  for (Index j = 0; j < n_; ++j) {
    solution.col_value[static_cast<std::size_t>(j)] = normalize_zero(variable_value(j));
    solution.col_dual[static_cast<std::size_t>(j)] =
        normalize_zero(sense * reduced_cost_[static_cast<std::size_t>(j)]);
    solution.col_status[static_cast<std::size_t>(j)] = status_[static_cast<std::size_t>(j)];
  }
  for (Index i = 0; i < m_; ++i) {
    // The reduced cost of logical i is 0 - y^T(-e_i) = y_i, so the row dual IS y_i. The
    // sense multiplier converts it back into the units of the file the user handed us.
    solution.row_dual[static_cast<std::size_t>(i)] =
        normalize_zero(sense * y_[static_cast<std::size_t>(i)]);
    solution.row_status[static_cast<std::size_t>(i)] =
        status_[static_cast<std::size_t>(n_ + i)];
  }

  solution.recompute_quality(model_);
  solution.dual_bound = solution.objective;
  return solution;
}

// -----------------------------------------------------------------------------------------
// The iteration loop
// -----------------------------------------------------------------------------------------

Solution PrimalSimplex::run() {
  Timer timer;
  primal_tolerance_ = options_.get_double("primal_feasibility_tolerance");
  dual_tolerance_ = options_.get_double("dual_feasibility_tolerance");
  if (!(primal_tolerance_ > 0.0)) primal_tolerance_ = tol::kPrimalFeasibility;
  if (!(dual_tolerance_ > 0.0)) dual_tolerance_ = tol::kDualFeasibility;

  const double time_limit = options_.get_double("time_limit");
  const std::int64_t iteration_limit = options_.get_int("iteration_limit");

  build_working_problem();
  set_initial_basis();

  if (m_ > 400) {
    logger_.warning(
        "{} rows: this Phase 2 simplex refactorizes a DENSE basis every iteration, which is "
        "O(m^3) per pivot. Expect it to be slow; the sparse LU lands in Phase 6.",
        m_);
  }

  if (!refactorize()) {
    return finish(SolveStatus::kNumericalError, "the initial slack basis is singular", 0,
                  timer.elapsed_seconds());
  }
  compute_basic_values();

  logger_.info("Primal simplex: {} rows, {} columns, {} nonzeros", m_, n_,
               model_.num_nonzeros());
  logger_.begin_iteration_table();

  Count iterations = 0;
  int degenerate_run = 0;
  bool bland = false;
  bool was_phase_one = true;

  for (;;) {
    const double infeasibility = total_infeasibility();
    const bool phase_one = infeasibility > primal_tolerance_;
    if (was_phase_one && !phase_one) {
      logger_.info("Phase 1 complete after {} iterations: primal feasible", iterations);
      degenerate_run = 0;
      bland = false;
    }
    was_phase_one = phase_one;

    compute_reduced_costs(phase_one);

    if (iterations % 20 == 0) {
      logger_.iteration(iterations, minimization_objective(), infeasibility, -1.0,
                        timer.elapsed_seconds());
    }

    int direction = 0;
    const Index entering = price(bland, &direction);

    if (entering < 0) {
      if (phase_one) {
        // Phase 1 is bounded below by zero, so a stall with residual infeasibility is a
        // proof that no feasible point exists, not an inconclusive stop.
        return finish(SolveStatus::kInfeasible,
                      fmt::format("phase 1 terminated with total infeasibility {:.3e}",
                                  infeasibility),
                      iterations, timer.elapsed_seconds());
      }
      return finish(SolveStatus::kOptimal, {}, iterations, timer.elapsed_seconds());
    }

    ftran_entering_column(entering);
    const RatioResult ratio = ratio_test(entering, direction, phase_one);

    if (ratio.unbounded) {
      if (phase_one) {
        return finish(SolveStatus::kNumericalError,
                      "phase 1 ratio test found no blocking variable, which cannot happen "
                      "for an objective bounded below by zero",
                      iterations, timer.elapsed_seconds());
      }
      return finish(SolveStatus::kUnbounded, {}, iterations, timer.elapsed_seconds());
    }

    const double step = ratio.step;
    if (step <= tol::kRatioTestFeasibility) {
      ++degenerate_run;
      if (!bland && degenerate_run > tol::kBlandSwitchIterations) {
        logger_.verbose("{} consecutive degenerate iterations: switching to Bland's rule",
                        degenerate_run);
        bland = true;
      }
      // Bland's rule is proved non-cycling for the classical simplex. The composite phase-1
      // objective redefines itself whenever the infeasible set changes, so that proof does
      // not carry over untouched, and a stall must be reported rather than spun on.
      if (degenerate_run > kStallLimit) {
        compute_reduced_costs(false);
        return finish(SolveStatus::kNumericalError,
                      fmt::format("stalled: {} consecutive degenerate iterations under "
                                  "Bland's rule", degenerate_run),
                      iterations, timer.elapsed_seconds());
      }
    } else {
      degenerate_run = 0;
      bland = false;
    }

    const auto e = static_cast<std::size_t>(entering);
    const double entering_value = nonbasic_value_[e] + static_cast<double>(direction) * step;

    if (ratio.leaving_position < 0) {
      // Bound flip: the entering variable travels its whole range and the basis is unchanged.
      nonbasic_value_[e] = (direction > 0) ? upper_[e] : lower_[e];
      status_[e] = (direction > 0) ? BasisStatus::kAtUpper : BasisStatus::kAtLower;
      compute_basic_values();
    } else {
      const auto slot = static_cast<std::size_t>(ratio.leaving_position);
      const Index leaving = basis_[slot];
      const auto l = static_cast<std::size_t>(leaving);

      basis_position_[l] = -1;
      // Snap the departing variable exactly onto the bound it hit. Leaving it at the
      // computed value would let a 1e-16 residual accumulate into a genuine bound violation
      // over hundreds of pivots.
      if (ratio.leaving_to_upper) {
        nonbasic_value_[l] = upper_[l];
        status_[l] = BasisStatus::kAtUpper;
      } else {
        nonbasic_value_[l] = lower_[l];
        status_[l] = BasisStatus::kAtLower;
      }
      if (lower_[l] == upper_[l]) status_[l] = BasisStatus::kFixed;

      basis_[slot] = entering;
      basis_position_[e] = ratio.leaving_position;
      status_[e] = BasisStatus::kBasic;
      nonbasic_value_[e] = entering_value;

      if (!refactorize()) {
        return finish(SolveStatus::kNumericalError,
                      fmt::format("basis became singular at iteration {}", iterations),
                      iterations, timer.elapsed_seconds());
      }
      compute_basic_values();
    }

    ++iterations;

    if (iteration_limit >= 0 && iterations >= iteration_limit) {
      compute_reduced_costs(false);
      return finish(SolveStatus::kIterationLimit,
                    fmt::format("iteration limit {} reached", iteration_limit), iterations,
                    timer.elapsed_seconds());
    }
    const double elapsed = timer.elapsed_seconds();
    if (elapsed > time_limit) {
      compute_reduced_costs(false);
      return finish(SolveStatus::kTimeLimit,
                    fmt::format("time limit {:g}s reached", time_limit), iterations, elapsed);
    }
  }
}

}  // namespace

Solution solve_primal_simplex(const Model& model, const Options& options, Logger& logger) {
  PrimalSimplex simplex(model, options, logger);
  return simplex.run();
}

}  // namespace sankhya
