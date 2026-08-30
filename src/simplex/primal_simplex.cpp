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

/// Largest amount a bound is relaxed by while escaping a degenerate stall (#51).
///
/// MUST STAY WELL UNDER kPrimalFeasibility. Relaxing a bound outward means a point feasible
/// for the perturbed problem can violate the TRUE bound by up to this much, and the
/// perturbation is removed before optimality is reported - so what matters is that the
/// leftover violation is below the tolerance the answer is judged against. At 1e-9 against a
/// 1e-7 feasibility tolerance there are two orders of margin, and phase 1 does not even
/// re-engage on the restored bounds.
constexpr double kPerturbationSize = 1e-9;
constexpr int kPerturbationTrigger = kStallLimit / 2;

/// Deterministic per-variable shift in (0, kPerturbationSize].
///
/// Deterministic and not random: CLAUDE.md's evidence rules are worth nothing if a rerun of
/// the same commit on the same instance can take a different path. A fixed hash of the index
/// gives every variable a DIFFERENT shift, which is the property that actually breaks the
/// ties, without making the run irreproducible.
[[nodiscard]] double perturbation_for(Index k) noexcept {
  const auto mixed = static_cast<std::uint64_t>(k) * 2654435761ULL + 1013904223ULL;
  const double unit = static_cast<double>(mixed % 1000003ULL) / 1000003.0;
  return kPerturbationSize * (0.25 + 0.75 * unit);
}

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

/// How often the updated factorization is checked against the basis it claims to represent,
/// and how much relative residual is tolerated before it is rebuilt.
///
/// The check costs one sparse mat-vec over the basis columns, which is the same order as the
/// FTRAN it verifies, so it is amortised over an interval rather than run every pivot.
constexpr Count kAccuracyCheckInterval = 16;

/// Restart the devex reference framework once any weight passes this.
///
/// The weights approximate steepest-edge norms measured from the basis the framework was
/// last reset in, and they only grow. Large values mean the approximation has drifted far
/// from what it is approximating, not that the column is genuinely bad, so continuing to
/// price on them re-creates the problem devex exists to solve. Forrest and Goldfarb restart
/// on this test; 1e6 is their suggested order and is where a reset costs one sweep of ones
/// against pivots that are no longer being chosen on meaningful information.
constexpr double kDevexResetThreshold = 1e6;

/// Reset the reference framework when the entering column's weight has drifted this far from
/// its exact steepest-edge norm.
///
/// DEVEX'S WEIGHTS ARE A LOWER BOUND: w_j <= gamma_j = 1 + ||B^-1 a_j||^2 always holds if
/// the update is sound, and the approximation is only useful while it stays near gamma.
/// Nothing was checking that. The absolute cap above fires at 1e6, which says nothing about
/// accuracy - a weight of 1e5 is fine beside a gamma of 1e5 and catastrophic beside a gamma
/// of 1.
///
/// MEASURED, by recomputing gamma exactly for every nonbasic column on scsd8 (#66). The
/// ratio w/gamma starts inside [0.89, 0.96] and climbs to 2.3e+02 by iteration 200 and
/// 3.7e+03 in the following solve. An inflated weight makes d^2/w rank a good column as a
/// bad one, so pricing steadily loses the information it is supposed to be using, and the
/// bases it then chooses are the ones that decay - which is the singular basis #66 was
/// chasing.
///
/// The test is nearly free: the entering column's alpha = B^-1 a_q is already computed for
/// the ratio test, so gamma_q costs one dot product, and it is checked on one column per
/// iteration rather than all of them. A factor of 4 is loose enough not to thrash the
/// A factor of 1.5 is what the measurement chose: at 2.0 and above scsd8 still fails,
/// at 1.5 and 1.05 it solves, and the committed small set costs 828 iterations either
/// way against 830 without the check - so the tighter test is free on healthy models.
constexpr double kDevexAccuracyFactor = 1.5;

/// Tied to kPrimalFeasibility rather than chosen independently, because that is the quantity
/// this check ultimately protects: factors whose residual is below the feasibility tolerance
/// cannot corrupt a feasibility judgement made at that tolerance.
///
/// It was 1e-9 when this check was written, which is inside the ordinary accumulated rounding
/// of a sequence of triangular solves rather than evidence that the factors have stopped
/// representing the basis. At 1e-9 it fired twice on grow22, and because a forced
/// refactorization changes which row wins the ratio test, those two rebuilds moved the final
/// vertex from a primal infeasibility of 5.652e-08 to 6.244e-07 - across the reported
/// tolerance, turning a published optimum into a numerical_error. A factorization that has
/// genuinely lost its basis misses by orders of magnitude more than this, so the looser
/// threshold keeps every case the check exists to catch.
constexpr double kUpdateAccuracyTolerance = tol::kPrimalFeasibility;

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

  /// Relax every finite bound slightly, so a degenerate vertex stops being degenerate.
  void perturb_bounds();

  /// Put the true bounds back. Called before optimality can be reported.
  void remove_perturbation();

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

  /// Reset every reference weight to 1, restarting the reference framework.
  void reset_devex();

  /// Fold this pivot into the reference weights. Needs the leaving ROW of B^-1 A, which is
  /// one BTRAN plus a pass over the nonbasic columns - the same shape as the reduced-cost
  /// computation, and the price devex pays for not doing a solve per candidate.
  void update_devex_weights(Index entering, Index leaving_row, double pivot);

  void ftran_entering_column(Index entering);

  /// Relative residual of the claimed FTRAN result: || B alpha - a ||_inf / || a ||_inf,
  /// with B taken from the CURRENT basis columns and alpha from the updated factors.
  ///
  /// This is the direct measurement of the thing that actually matters - whether the factors
  /// plus their eta file still represent the basis - and it replaces inferring conditioning
  /// from whether the Markowitz ladder happened to fire.
  [[nodiscard]] double ftran_residual(Index entering) const;
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

  /// PERTURBATION FOR DEGENERACY (#51; Maros, "Computational Techniques of the Simplex
  /// Method", ch. 9, and the bound-shifting scheme every production code uses).
  ///
  /// A degenerate vertex has more active constraints than dimensions, so the ratio test ties
  /// and the step is zero. Anti-cycling rules ARBITRATE those ties; perturbation REMOVES
  /// them, by moving each bound a different tiny amount so no two can be active at once.
  ///
  /// Measured on tuff, which is why this exists: it does not terminate under Bland's rule at
  /// 1000, 10000, 50000 or 200001 consecutive degenerate iterations. Implementing Bland
  /// correctly - lowest index on the LEAVING variable too, which our ratio test does not do -
  /// breaks the cycle and produces a singular basis instead, and takes wood1p down with it.
  /// The two properties a tie-break must supply, termination and conditioning, want opposite
  /// things from the same choice. Perturbation sidesteps the conflict.
  bool perturbed_ = false;
  Count perturbations_ = 0;

  SparseLu lu_;

  /// Reused across refactorizations so the hot path allocates nothing. Structural columns
  /// point straight into the model's CSC arrays - no copy at all - while logical columns are
  /// the single entry -1 in their own row, served from the two buffers below.
  std::vector<LuColumn> basis_columns_;
  std::vector<Index> logical_rows_;
  std::vector<double> logical_values_;

  /// Reported once per solve, not once per refactorization.
  bool warned_about_threshold_ = false;

  /// Latched by refactorize() when the Markowitz ladder climbed past its default.
  bool basis_needed_stricter_threshold_ = false;

  /// Effort counters for the solve log. rejected_updates_ is the interesting one: a basis
  /// that keeps producing unsafe pivots is badly conditioned, and that is worth seeing.
  Count refactorizations_ = 0;
  double worst_basis_pivot_ = 0.0;  ///< smallest pivot over every factorization
  Count iterations_seen_ = 0;       ///< for the per-refactorization log line only
  Count rejected_updates_ = 0;
  Count accuracy_refactorizations_ = 0;
  std::vector<double> x_basic_;
  std::vector<double> cost_basic_;
  std::vector<double> y_;
  std::vector<double> reduced_cost_;
  std::vector<double> alpha_;

  // ---- Devex pricing -------------------------------------------------------------------
  //
  // Forrest, J.J. and Goldfarb, D. (1992), "Steepest-edge simplex algorithms for linear
  // programming", Mathematical Programming 57, 341-374; the approximation itself is Harris,
  // P.M.J. (1973), "Pivot selection methods of the Devex LP code", Mathematical Programming
  // 5, 1-28.
  //
  // WHAT DANTZIG GETS WRONG. Pricing on |d_j| alone asks which column improves the objective
  // fastest PER UNIT STEP IN THAT VARIABLE, but the step actually taken is set by the ratio
  // test, and that is governed by the size of the FTRAN'd column B^-1 a_j. A column with a
  // large reduced cost and a large ||B^-1 a_j|| buys almost nothing per pivot, and Dantzig
  // picks it again and again. Steepest edge divides by that norm exactly, which costs a
  // solve per candidate. Devex approximates the norm with reference weights updated in O(m)
  // from vectors this iteration already computes, and prices on d_j^2 / w_j.
  //
  // Measured before this landed: 1147 simplex iterations against HiGHS's 531 across the
  // committed Netlib set, worst 4.21x on blend. See issue #66 for the table.
  bool devex_ = false;  ///< opt-in until #67 lands; see the option description and #66
  std::vector<double> devex_weight_;
  std::vector<double> rho_;  ///< B^-T e_r, scratch: rho . a_j gives the leaving row's alpha_rj
  Count devex_resets_ = 0;
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
  devex_weight_.assign(static_cast<std::size_t>(total_), 1.0);
  rho_.assign(static_cast<std::size_t>(m_), 0.0);
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
      // Latches: a model that produced one ill-conditioned basis will produce more, and
      // assignment would clear this on the next basis that happened to factorize cleanly.
      if (attempt > 0) basis_needed_stricter_threshold_ = true;
      if (attempt > 0 && !warned_about_threshold_) {
        warned_about_threshold_ = true;
        logger_.warning(
            "basis factorization needed a Markowitz threshold of {:g} rather than {:g}; "
            "the basis is poorly scaled and the factors will carry more fill",
            kThresholdLadder[attempt], tol::kMarkowitzThreshold);
      }
      // BASIS CONDITIONING, RECORDED RATHER THAN INFERRED. The smallest pivot of a fresh
      // factorization is the cheapest honest read on how close a basis is to singular. The
      // factorization already computes it; nothing was asking for it.
      //
      // The refactorization count does not answer the same question. It confounds
      // conditioning with FILL - a basis can be perfectly well conditioned and still trigger
      // the eta-fill rule every other pivot - so a run that refactorizes constantly and one
      // whose basis is decaying look identical from the outside, and they need opposite
      // responses.
      //
      // Added because #66 could not be settled without it: devex drives grow22 and scsd8 to
      // "basis became singular" while Dantzig does not, and the ratio test, accumulated
      // update error and small committed pivots had each been eliminated by experiment.
      // With this line the answer took one run - the smallest pivot falls to 6.1e-08 and
      // 1.6e-09 under devex against 1.0e-03 and 4.0e-03 under Dantzig, so the basis really
      // is decaying rather than failing suddenly.
      const double pivot = lu_.smallest_pivot();
      if (pivot > 0.0 && (worst_basis_pivot_ == 0.0 || pivot < worst_basis_pivot_)) {
        worst_basis_pivot_ = pivot;
      }
      logger_.verbose("refactorized at iteration {}: smallest pivot {:.3e}", iterations_seen_,
                      pivot);
      return true;
    }
  }
  return false;
}

void PrimalSimplex::perturb_bounds() {
  if (perturbed_) return;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    // A FIXED variable is left exactly alone. Widening lower == upper would turn a variable
    // the model pins to one value into one with a range, which is a different problem rather
    // than a nudged one.
    if (lower_[u] == upper_[u]) continue;
    const double shift = perturbation_for(k);
    if (is_finite_bound(lower_[u])) lower_[u] -= shift;
    if (is_finite_bound(upper_[u])) upper_[u] += shift;
  }
  perturbed_ = true;
  ++perturbations_;
}

void PrimalSimplex::remove_perturbation() {
  if (!perturbed_) return;
  for (Index k = 0; k < n_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    lower_[u] = model_.col_lower[u];
    upper_[u] = model_.col_upper[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(n_ + i);
    const auto r = static_cast<std::size_t>(i);
    lower_[u] = model_.row_lower[r];
    upper_[u] = model_.row_upper[r];
  }
  // A nonbasic variable was parked on a RELAXED bound and must be moved back onto the true
  // one, or the basic values recomputed from it are wrong by the shift.
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (status_[u] == BasisStatus::kAtLower && is_finite_bound(lower_[u])) {
      nonbasic_value_[u] = lower_[u];
    } else if (status_[u] == BasisStatus::kAtUpper && is_finite_bound(upper_[u])) {
      nonbasic_value_[u] = upper_[u];
    }
  }
  perturbed_ = false;
  compute_basic_values();
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
  // Seeded at zero, not at the dual tolerance: eligibility is now tested explicitly against
  // dual_tolerance_ below, because in devex mode this variable holds d^2 / w and comparing
  // that against a tolerance on |d| would be comparing two different quantities.
  double best_score = 0.0;

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
    // Dantzig compares |d|; devex compares d^2 / w, which is |d| divided by an approximate
    // edge norm. Both are scored against `best_magnitude`, seeded at the dual tolerance, so
    // eligibility is decided by |d| in BOTH modes - the weight changes which eligible column
    // wins, never whether a column is eligible at all. Mixing those two jobs would let a
    // large weight silently suppress a column that genuinely prices out, which is a
    // termination bug rather than a slow pivot.
    const double magnitude = std::fabs(d);
    if (magnitude <= dual_tolerance_) continue;
    const double score = devex_ ? (magnitude * magnitude) / devex_weight_[u] : magnitude;
    if (score > best_score) {
      best_score = score;
      best = k;
      *direction = candidate_direction;
    }
  }
  return best;
}

void PrimalSimplex::reset_devex() {
  std::fill(devex_weight_.begin(), devex_weight_.end(), 1.0);
  ++devex_resets_;
}

void PrimalSimplex::update_devex_weights(Index entering, Index leaving_row, double pivot) {
  if (!devex_ || std::fabs(pivot) < tol::kZeroDrop) return;

  const auto q = static_cast<std::size_t>(entering);
  const double weight_q = devex_weight_[q];

  // rho = B^-T e_r, so that rho . a_j gives alpha_rj, the entry of the leaving row under
  // column j. One BTRAN, then one dot product per nonbasic column.
  std::fill(rho_.begin(), rho_.end(), 0.0);
  rho_[static_cast<std::size_t>(leaving_row)] = 1.0;
  lu_.solve_transpose(rho_.data());

  const double inverse_pivot = 1.0 / pivot;
  const double scaled_weight_q = weight_q * inverse_pivot * inverse_pivot;

  double largest = 1.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (basis_position_[u] >= 0) continue;
    if (k == entering) continue;

    double alpha_rk = 0.0;
    for_each_entry(k, [&](Index row, double coefficient) {
      alpha_rk += rho_[static_cast<std::size_t>(row)] * coefficient;
    });
    if (alpha_rk == 0.0) continue;

    // w_j <- max(w_j, (alpha_rj / alpha_rq)^2 * w_q). The weights only ever GROW inside a
    // reference framework; that monotonicity is what makes the approximation safe to reuse
    // across pivots, and it is also why the framework has to be reset once they blow up.
    const double candidate = alpha_rk * alpha_rk * scaled_weight_q;
    if (candidate > devex_weight_[u]) devex_weight_[u] = candidate;
    if (devex_weight_[u] > largest) largest = devex_weight_[u];
  }

  // The variable that just left the basis becomes nonbasic and needs a weight of its own.
  const double leaving_weight = std::max(scaled_weight_q, 1.0);
  devex_weight_[static_cast<std::size_t>(basis_[static_cast<std::size_t>(leaving_row)])] =
      leaving_weight;
  if (leaving_weight > largest) largest = leaving_weight;

  // A reference weight is an approximation to a steepest-edge norm measured from the
  // framework the weights were last reset in. The further the basis travels from it the
  // worse the approximation, and unbounded growth is the symptom. Restarting costs one
  // sweep and buys back the accuracy; Forrest and Goldfarb restart on exactly this test.
  if (largest > kDevexResetThreshold) reset_devex();
}

void PrimalSimplex::ftran_entering_column(Index entering) {
  std::fill(alpha_.begin(), alpha_.end(), 0.0);
  for_each_entry(entering, [&](Index row, double value) {
    alpha_[static_cast<std::size_t>(row)] += value;
  });
  lu_.solve(alpha_.data());
}

double PrimalSimplex::ftran_residual(Index entering) const {
  // B alpha, accumulated straight from the basis columns.
  std::vector<double> product(static_cast<std::size_t>(m_), 0.0);
  for (Index slot = 0; slot < m_; ++slot) {
    const double weight = alpha_[static_cast<std::size_t>(slot)];
    if (weight == 0.0) continue;
    for_each_entry(basis_[static_cast<std::size_t>(slot)], [&](Index row, double value) {
      product[static_cast<std::size_t>(row)] += value * weight;
    });
  }

  // ... which must reproduce the entering column.
  double worst = 0.0;
  double scale = 1.0;
  for_each_entry(entering, [&](Index row, double value) {
    product[static_cast<std::size_t>(row)] -= value;
    scale = std::max(scale, std::fabs(value));
  });
  for (Index i = 0; i < m_; ++i) {
    worst = std::max(worst, std::fabs(product[static_cast<std::size_t>(i)]));
  }
  return worst / scale;
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
  logger_.info(
      "Basis: {} refactorizations over {} iterations, {} declined as unsafe, {} forced "
      "by the accuracy check; smallest pivot over all factorizations {}",
      refactorizations_, iterations, rejected_updates_, accuracy_refactorizations_,
      // "n/a" and NOT 0.000e+00 when nothing was recorded. A model with no rows factorizes
      // nothing, and printing a zero there says "the basis was singular" - the strongest
      // possible claim about conditioning - when what happened is that the question never
      // arose. This whole line exists to make basis health legible; a plausible-looking
      // number standing in for absent data is the one way it could mislead.
      worst_basis_pivot_ > 0.0 ? fmt::format("{:.3e}", worst_basis_pivot_)
                               : std::string("n/a (nothing was factorized)"));

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

  // Dantzig is kept reachable so the before/after in issue #66 can be REGENERATED rather
  // than quoted from a commit message, and so a suspected pricing bug can be bisected
  // against the rule this replaced without checking out an old tree.
  // DEVEX IS NOT THE DEFAULT YET, and the reason is measured rather than cautious. It cuts
  // iterations substantially - 1147 to 830 on the committed small set, and 52250 to 31618
  // across the Netlib medium set once d6cube (which fails under both rules) is set aside -
  // but it also turns grow22 and scsd8 from `optimal` into "basis became singular", taking
  // the medium pass rate from 41 to 39.
  //
  // The two facts are the same fact. Devex chooses a different entering column, and nothing
  // in the CURRENT ratio test defends the pivot magnitude of that choice: it takes the
  // tightest bound and accepts whatever pivot comes with it. Dantzig happened to pick
  // columns whose pivots were survivable; devex does not, and the basis degrades until it
  // is singular. Both regressions fail EARLIER than the Dantzig run succeeded, which is the
  // signature of conditioning rather than of a longer search.
  //
  // The Harris two-pass ratio test (#67) is what makes this safe: it spends a relaxed bound
  // tolerance to buy a larger pivot, which is exactly the degree of freedom missing here.
  // #67 was implemented once, measured as null under Dantzig, and not shipped for that
  // reason - its value only appears once the pricing rule stops picking safe columns by
  // accident. Landing devex on by default before it would trade a headline iteration count
  // for two correct answers, which CLAUDE.md settles: a wrong answer scores zero.
  const std::string pricing = options_.get_string("pricing");
  devex_ = pricing == "devex";
  if (pricing != "devex" && pricing != "dantzig" && !pricing.empty()) {
    logger_.warning("pricing '{}' is not recognised; using dantzig", pricing);
    devex_ = false;
  }

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
      // The composite phase-1 objective is a different function from the phase-2 one, so
      // weights accumulated against the first approximate edge norms for an objective that
      // no longer exists. Carrying them across is not a slow start, it is wrong information.
      reset_devex();
    }
    was_phase_one = phase_one;

    iterations_seen_ = iterations;
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
      // OPTIMAL FOR THE PERTURBED PROBLEM IS NOT OPTIMAL. The bounds were relaxed to break
      // a stall; reporting this point would answer a question nobody asked, and the answer
      // would be feasible-looking and slightly wrong. Restore the true bounds and keep
      // going - the basis is retained, so the clean finish is usually a handful of pivots.
      if (perturbed_) {
        logger_.verbose("optimal under perturbation; restoring exact bounds at iteration {}",
                        iterations);
        remove_perturbation();
        bland = false;
        degenerate_run = 0;
        continue;
      }
      return finish(SolveStatus::kOptimal, {}, iterations, timer.elapsed_seconds());
    }

    ftran_entering_column(entering);

    // DOES THE WEIGHT STILL APPROXIMATE ANYTHING? alpha is B^-1 a_q, so the exact
    // steepest-edge norm of the column just chosen is one dot product away, and devex
    // guarantees w_q <= gamma_q. A weight that has climbed above gamma is not a slightly
    // stale estimate, it is wrong in the direction that makes pricing avoid good columns.
    // Checking the entering column alone, once per iteration, is enough to notice: it is
    // the column whose weight the ranking just acted on.
    if (devex_) {
      double gamma = 1.0;
      for (Index slot = 0; slot < m_; ++slot) {
        const double v = alpha_[static_cast<std::size_t>(slot)];
        gamma += v * v;
      }
      const double weight = devex_weight_[static_cast<std::size_t>(entering)];
      if (weight > kDevexAccuracyFactor * gamma) reset_devex();
    }

    // Periodically ask whether the updated factors still represent the basis, and rebuild
    // them when they do not. This measures the property that matters rather than guessing at
    // it: an earlier version inferred trouble from the Markowitz threshold ladder having
    // fired, which is a proxy for conditioning and not for accuracy, and it left d6cube 1.8x
    // slower than never updating at all.
    if (lu_.eta_count() > 0 && iterations % kAccuracyCheckInterval == 0) {
      const double residual = ftran_residual(entering);
      if (residual > kUpdateAccuracyTolerance) {
        ++accuracy_refactorizations_;
        if (!refactorize()) {
          return finish(SolveStatus::kNumericalError,
                        fmt::format("basis became singular at iteration {}", iterations),
                        iterations, timer.elapsed_seconds());
        }
        ++refactorizations_;
        ftran_entering_column(entering);
      }
    }

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
      // PERTURB BEFORE FALLING BACK TO BLAND. Bland is a termination guarantee bought with
      // arithmetic quality; perturbation removes the ties that caused the stall instead of
      // arbitrating them, and costs nothing when it works. Bland remains below as the
      // last resort for a stall perturbation did not clear.
      if (!perturbed_ && degenerate_run > kPerturbationTrigger) {
        logger_.verbose("{} consecutive degenerate iterations: perturbing bounds by up to {:g}",
                        degenerate_run, kPerturbationSize);
        perturb_bounds();
        compute_basic_values();
        degenerate_run = 0;
        continue;
      }

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

      // BEFORE the basis changes, and before the factors are updated. The weight update
      // needs rho = B^-T e_r under the basis this pivot is leaving, and it reads
      // basis_[leaving_row] to find the departing variable - both are about to be
      // overwritten. A bound flip never reaches here, which is correct: nothing leaves the
      // basis, so no edge changes and no weight is stale.
      update_devex_weights(entering, ratio.leaving_position,
                           alpha_[static_cast<std::size_t>(ratio.leaving_position)]);

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
      // Two independent controls, and they answer different questions.
      //
      // The accuracy check above asks whether the factors still represent the basis. On
      // d6cube it fires essentially never - the product form stays accurate to better than
      // 1e-9 relative residual for thousands of pivots - so drift is NOT what goes wrong
      // there.
      //
      // What goes wrong is the pivot path. Even with faithful factors, alpha computed
      // through base-plus-etas differs from alpha computed through fresh factors in the last
      // bits, and on a massively degenerate model those bits decide which row wins the ratio
      // test. d6cube then takes 38634 pivots to reach the same singular basis it reaches in
      // 1947 without the update. Neither run produces an answer; one just wastes twenty times
      // as long failing.
      //
      // That is not fixable by controlling accuracy, because accuracy is not the problem -
      // the real remedy is anti-degeneracy machinery (Harris ratio test, perturbation, #67).
      // Until then the update is switched off on a basis the factorization has already
      // flagged as poorly conditioned, which is where its benefit is least reliable and where
      // this behaviour shows up. It is containment, not a fix, and is described as such.
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
