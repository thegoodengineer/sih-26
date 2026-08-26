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
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "../la/lu.hpp"
#include "../la/scaling.hpp"

namespace sankhya {
namespace {

/// Consecutive zero-length steps tolerated before the solve is declared stalled. Twenty
/// times the Bland switch threshold: long enough that no honest degenerate plateau trips
/// it, short enough that a genuine cycle is reported in under a second.
constexpr int kStallLimit = 20 * tol::kBlandSwitchIterations;

/// How far above the feasibility tolerance a phase-1 stall has to sit before it is reported
/// as a genuine infeasibility rather than a numerical stall.
///
/// There is no principled value: the honest position is that a floating-point stall proves
/// nothing either way, and this only decides which of two imperfect answers is less
/// misleading. 1e3 keeps kInfeasible for residuals that are large in absolute terms while
/// refusing to make a definitive claim about a point that is nearly feasible. Netlib grow15
/// and grow22 stalled at 1.06e-07 and 1.31e-07 against a 1e-07 tolerance - a factor of 1.3 -
/// and both have published optima.
constexpr double kInfeasibilityProofFactor = 1e3;

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
  /// Largest single bound violation over the basic variables. THIS is the feasibility test.
  ///
  /// The two must not be confused. kPrimalFeasibility is documented as "max allowed
  /// row/column bound violation" and Solution::recompute_quality() measures exactly that, so
  /// comparing the SUM against it silently demands a per-row violation of tolerance/m: the
  /// larger the model, the stricter the requirement. On Netlib grow15 (300 rows) that made a
  /// point with a max violation of 0.000e+00 - feasible by the project's own measurement -
  /// fail a test reading 1.062e-07, and phase 1 then reported the model INFEASIBLE.
  [[nodiscard]] double max_infeasibility() const;

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

  SparseLu lu_;

  /// Reused across refactorizations so the hot path allocates nothing. Structural columns
  /// point straight into the model's CSC arrays - no copy at all - while logical columns are
  /// the single entry -1 in their own row, served from the two buffers below.
  std::vector<LuColumn> basis_columns_;
  std::vector<Index> logical_rows_;
  std::vector<double> logical_values_;

  /// Reported once per solve, not once per refactorization.
  bool warned_about_threshold_ = false;

  /// Set by refactorize() when the Markowitz ladder had to climb past its default. Used to
  /// switch the basis update off for the rest of the solve; see the pivot loop.
  bool basis_needed_stricter_threshold_ = false;

  /// Effort counters for the solve log. rejected_updates_ is the interesting one: a basis
  /// that keeps producing unsafe pivots is badly conditioned, and that is worth seeing.
  Count refactorizations_ = 0;
  Count rejected_updates_ = 0;
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
  // Phase 2 materialised a dense m x m array here and threw it away again on every pivot:
  // O(m^2) of memory traffic and O(m^3) of arithmetic to factorize a matrix that is better
  // than 99% structural zeros at any realistic size. Nothing is materialised now. A
  // structural column is handed to the factorization as a pointer into the model's own CSC
  // storage, and a logical column is the single entry -1.
  if (logical_rows_.empty() && m_ > 0) {
    logical_rows_.resize(static_cast<std::size_t>(m_));
    logical_values_.assign(static_cast<std::size_t>(m_), -1.0);
    for (Index i = 0; i < m_; ++i) logical_rows_[static_cast<std::size_t>(i)] = i;
  }

  basis_columns_.assign(static_cast<std::size_t>(m_), LuColumn{});
  for (Index slot = 0; slot < m_; ++slot) {
    const Index k = basis_[static_cast<std::size_t>(slot)];
    LuColumn& target = basis_columns_[static_cast<std::size_t>(slot)];
    if (k < n_) {
      const ColumnView column = model_.matrix.column(k);
      target.rows = column.rows;
      target.values = column.values;
      target.size = column.size;
    } else {
      const auto row = static_cast<std::size_t>(k - n_);
      target.rows = logical_rows_.data() + row;
      target.values = logical_values_.data() + row;
      target.size = 1;
    }
  }
  // Markowitz trades stability for fill: at tau = 0.01 a pivot may be a hundred times
  // smaller than the largest entry in its column, and on a badly scaled basis that choice
  // can leave a later step with nothing above the pivot tolerance at all. The dense
  // factorization never had this failure mode because partial pivoting always takes the
  // largest entry, i.e. it is this same algorithm at tau = 1.
  //
  // Netlib `blend` is a real instance that fails at 0.01 and succeeds at a stricter
  // threshold. So a failure is not reported as a singular basis until the ordering has been
  // retried with progressively more stability, ending at full partial pivoting - more fill,
  // slower, and still enormously better than a dense refactorization. Only a basis that is
  // singular under partial pivoting is genuinely singular.
  static constexpr double kThresholdLadder[] = {tol::kMarkowitzThreshold, 0.1, 0.5, 1.0};
  for (std::size_t attempt = 0; attempt < std::size(kThresholdLadder); ++attempt) {
    if (lu_.factorize(basis_columns_, m_, tol::kPivotTolerance, kThresholdLadder[attempt])) {
      // A basis that needed a stricter threshold than the default is poorly conditioned, and
      // that is the signal used below to stop trusting the product-form update on this model.
      // It LATCHES: a model that has produced one ill-conditioned basis will produce more,
      // and assignment rather than latching would clear the flag on the very next basis that
      // happened to factorize cleanly - leaving the update switched on for most of the solve.
      if (attempt > 0) basis_needed_stricter_threshold_ = true;
      if (attempt > 0 && !warned_about_threshold_) {
        warned_about_threshold_ = true;
        logger_.warning(
            "basis factorization needed a Markowitz threshold of {:g} rather than {:g}; "
            "the basis is poorly scaled and the factors will carry more fill",
            kThresholdLadder[attempt], tol::kMarkowitzThreshold);
      }
      return true;
    }
  }
  return false;
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

double PrimalSimplex::max_infeasibility() const {
  double worst = 0.0;
  for (Index slot = 0; slot < m_; ++slot) {
    const Index k = basis_[static_cast<std::size_t>(slot)];
    const double x = x_basic_[static_cast<std::size_t>(slot)];
    const double lo = lower_[static_cast<std::size_t>(k)];
    const double hi = upper_[static_cast<std::size_t>(k)];
    if (is_finite_bound(lo) && x < lo) worst = std::max(worst, lo - x);
    if (is_finite_bound(hi) && x > hi) worst = std::max(worst, x - hi);
  }
  return worst;
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
  // The ratio of refactorizations to iterations is the cheapest available read on how well
  // the basis update is holding up: a run that refactorizes on most pivots has gained
  // nothing, and a high rejection count means the bases being produced are ill conditioned.
  logger_.info("Basis: {} refactorizations over {} iterations, {} update(s) declined as unsafe",
               refactorizations_, iterations, rejected_updates_);

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

  // The dual bound must be stated BEFORE recompute_quality(), because that is what derives
  // absolute_gap and relative_gap from it. Setting it afterwards left both gaps measured
  // against a bound of zero, so every proven-optimal LP reported relative_gap = 1.
  //
  // And only an OPTIMAL basis proves a bound. On an iteration or time limit the point in
  // hand is an incumbent, not a proof; claiming the objective as a dual bound there asserts
  // an optimality that was never established, which a Phase 5 branch-and-bound would then
  // happily prune against. An unknown bound is the infinity on the unexplored side of the
  // objective, and yields an infinite gap rather than a fake zero.
  if (status == SolveStatus::kOptimal) {
    solution.dual_bound = model_.evaluate_objective(solution.col_value.data());
  } else {
    solution.dual_bound = model_.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model_);
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
    // The phase decision is made on the largest single violation, not on their sum. The sum
    // is still computed for the iteration log, where it is the objective being minimised.
    const double infeasibility = max_infeasibility();
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
        // Phase 1 is bounded below by zero, so its true minimum being positive would indeed
        // prove that no feasible point exists. What is observed here is weaker: no column
        // PRICES as improving to within the dual tolerance. On a badly scaled basis that
        // happens while an improving direction still exists, so a stall is evidence, not a
        // proof, and the strength of the evidence depends on how far from feasible we are.
        //
        // A residual within a couple of orders of magnitude of the feasibility tolerance is a
        // numerical stall and is reported as one. Claiming kInfeasible there tells a planner
        // their model has no solution when it has one, which is the least checkable and most
        // damaging answer this solver can give.
        if (infeasibility <= kInfeasibilityProofFactor * primal_tolerance_) {
          return finish(SolveStatus::kNumericalError,
                        fmt::format("phase 1 stalled at max bound violation {:.3e}, only just "
                                    "above the {:.1e} feasibility tolerance; no column prices "
                                    "as improving, but this is a numerical stall rather than "
                                    "a proof that the model is infeasible",
                                    infeasibility, primal_tolerance_),
                        iterations, timer.elapsed_seconds());
        }
        return finish(
            SolveStatus::kInfeasible,
            fmt::format("phase 1 terminated with max bound violation {:.3e}, far above the "
                        "{:.1e} feasibility tolerance",
                        infeasibility, primal_tolerance_),
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
                                  "Bland's rule",
                                  degenerate_run),
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

      // A pivot changes ONE column of the basis, so the factorization is updated rather than
      // rebuilt. alpha_ already holds B^-1 a for the entering column - the ratio test needed
      // it - so the update is free of any extra solve.
      //
      // Refactorize when the update declines the pivot as numerically unsafe, or when the
      // eta file has grown enough that it costs more per solve than fresh factors would.
      // Both paths matter: refactorizing every iteration was slow but had no accumulated
      // update error, and that property is only preserved by taking the trigger seriously.
      // The update is NOT used once the factorization has told us the basis is poorly
      // conditioned. Netlib d6cube is why. It needed a stricter Markowitz threshold, and
      // carrying an eta file on top of such a basis degraded the pivot path badly: phase 1
      // went from 1947 iterations to 38634, a twentyfold increase, for the same final status.
      // The cost is not per-iteration arithmetic, it is that slightly drifted reduced costs
      // pick different entering columns and the simplex wanders.
      //
      // Refactorizing every pivot is exactly the behaviour that had no accumulated update
      // error, so falling back to it on an ill-conditioned model is the conservative choice
      // rather than a special case: the update is an optimization, and it is switched off
      // where the evidence says it does not pay.
      const bool trust_update = !basis_needed_stricter_threshold_;
      const bool updated = trust_update && lu_.update(ratio.leaving_position, alpha_.data());
      if (trust_update && !updated) ++rejected_updates_;
      if (!updated || lu_.should_refactorize()) {
        if (!refactorize()) {
          return finish(SolveStatus::kNumericalError,
                        fmt::format("basis became singular at iteration {}", iterations),
                        iterations, timer.elapsed_seconds());
        }
        ++refactorizations_;
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

/// Number of Ruiz equilibration passes. Ruiz proves geometric convergence of the row and
/// column infinity norms toward 1, so a handful of passes captures nearly all of the
/// available improvement; PDLP section 4.1 uses ten and reports the tail as negligible.
constexpr int kRuizIterations = 10;

Solution solve_primal_simplex(const Model& model, const Options& options, Logger& logger) {
  // WHY THE SIMPLEX IS SCALED. It was assumed for a long time that it need not be - a
  // simplex pivots on ratios, so a uniform rescaling of a row cancels. That reasoning is
  // correct about the ALGEBRA and wrong about the ARITHMETIC, and the Netlib medium tier
  // said so: 18 of its 24 failures were the identical message "basis became singular", on
  // the known badly scaled corner of the set (fit1d, fit2d, israel, pilot4, e226, ...).
  // fit1d failed after 23 iterations, far too early for accumulated drift. The bases were
  // ill-conditioned from the start because the model was.
  //
  // Markowitz threshold pivoting (issue #22) helped, but it only chooses among the pivots
  // available; scaling changes which pivots exist at all. See issue #49.
  if (!options.get_bool("scaling")) {
    PrimalSimplex simplex(model, options, logger);
    return simplex.run();
  }

  // Cost is passed in the ORIGINAL sense, not minimise space. build_scaling only multiplies
  // it by the column multipliers, and the multipliers themselves come from matrix norms, so
  // the sense never enters; folding it in here would mean unfolding it again below.
  const Scaling scaling = build_scaling(model, model.col_cost, kRuizIterations);

  Model scaled = model;
  scaled.matrix = scaling.matrix;
  scaled.col_cost = scaling.cost;
  scaled.col_lower = scaling.col_lower;
  scaled.col_upper = scaling.col_upper;
  scaled.row_lower = scaling.row_lower;
  scaled.row_upper = scaling.row_upper;
  // sense, objective_offset, col_type and the names are carried unchanged: a diagonal change
  // of variable leaves the objective VALUE alone, so no offset correction is needed.

  logger.debug("Scaling: entries {:.3e} to {:.3e} after {} Ruiz passes and one Pock-Chambolle",
               scaling.min_abs, scaling.max_abs, kRuizIterations);

  PrimalSimplex simplex(scaled, options, logger);
  Solution solution = simplex.run();

  // UNSCALE, AND UNSCALE EVERYTHING. A diagonal change of variable that is undone for the
  // primal point but not for the duals produces a point that is feasible, an objective that
  // is right, and reduced costs that are silently wrong - which passes every check the
  // solver makes about itself and fails only against an independent verifier. The mapping is
  // stated in src/la/scaling.hpp and derived there:
  //
  //     x = Dc xhat        y = Dr yhat        d = Dc^-1 dhat
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    if (u < solution.col_value.size()) solution.col_value[u] *= dc;
    if (u < solution.col_dual.size()) solution.col_dual[u] /= dc;
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (u < solution.row_dual.size()) solution.row_dual[u] *= scaling.row[u];
  }

  // Re-measure against the ORIGINAL model. This is the second half of the correctness
  // argument and it is not optional: tolerances in tolerances.hpp are ABSOLUTE and stated in
  // the original problem's units, so a point judged feasible in scaled space says nothing
  // about the answer we return. recompute_quality rebuilds the row activities from the
  // original matrix and recomputes the objective, so everything reported below is measured
  // where the caller lives.
  solution.recompute_quality(model);

  // FALL BACK WHEN SCALING DOES NOT PAY. Equilibration is a heuristic: it rescues models the
  // unscaled simplex cannot factorize at all, and on a handful of models it costs more
  // accuracy than it buys. Measured on the Netlib medium tier, scaling alone took 26/50 to
  // 35/50 but broke two instances that had been passing - degen2 stalled under Bland's rule
  // and scsd6 went singular - and degraded the round-trip on our own ill_conditioned case
  // study from 5e-20 to 1.2e-07, just over the tolerance.
  //
  // Rather than pick one path and lose the other's wins, take the union: if the scaled solve
  // did not produce a point that is feasible IN ORIGINAL UNITS, solve again unscaled and
  // keep that instead. The second solve costs nothing on the models where scaling already
  // worked, because it never runs.
  const double primal_tolerance = options.get_double("primal_feasibility_tolerance");
  const bool usable =
      (solution.status == SolveStatus::kOptimal || solution.status == SolveStatus::kFeasible) &&
      solution.primal_infeasibility <= primal_tolerance;
  if (usable) return solution;

  logger.info("Scaled solve returned {} (primal infeasibility {:.3e}); retrying unscaled",
              to_string(solution.status), solution.primal_infeasibility);
  PrimalSimplex unscaled_simplex(model, options, logger);
  Solution unscaled = unscaled_simplex.run();
  const bool unscaled_usable =
      (unscaled.status == SolveStatus::kOptimal || unscaled.status == SolveStatus::kFeasible) &&
      unscaled.primal_infeasibility <= primal_tolerance;
  if (unscaled_usable) {
    logger.info("Unscaled solve succeeded where the scaled one did not");
    return unscaled;
  }

  // Neither worked. Report the one that came closer to feasibility, so the message the user
  // sees describes the better of the two attempts rather than whichever ran last.
  return unscaled.primal_infeasibility < solution.primal_infeasibility ? unscaled : solution;
}

}  // namespace sankhya
