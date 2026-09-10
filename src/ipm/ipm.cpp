// SPDX-License-Identifier: Apache-2.0
// SANKHYA - Mehrotra's predictor-corrector interior-point method for LP (#56).
//
// References:
//   Mehrotra, S., "On the implementation of a primal-dual interior point method", SIAM J.
//     Optimization 2 (1992), 575-601 - the predictor-corrector and the centering rule.
//   Wright, S.J., "Primal-Dual Interior-Point Methods", SIAM (1997), ch. 10-11 - the
//     bounded-variable form, the normal equations, the starting point and the step rule.
//   Altman & Gondzio (1999), see la/ldl.hpp - primal and dual regularization.
//
// THE FORM SOLVED. The same one the simplex uses:  [A | -I] [x; s] = 0  with bounds on every
// component, l <= x <= u for the structurals and row_lower <= s <= row_upper for the
// logicals. A finite lower bound contributes a slack s_l = x - l >= 0 with multiplier
// z_l >= 0, a finite upper bound s_u = u - x >= 0 with z_u >= 0, and the dual constraint
// reads c - Abar^T y - z_l + z_u = 0. Free variables have neither and are held by the
// primal regularization alone.
//
// Every Newton system reduces to the normal equations  (Abar Theta Abar^T + delta I) dy = r
// with Theta diagonal, which is SPD and, because Abar carries -I, at least as well
// conditioned as Theta_s on the logical block. The sparse LDL^T of la/ldl.hpp is analyzed
// once and refactorized every iteration.
//
// WHAT THIS METHOD DOES NOT DO. It produces no basis, so it cannot warm-start branch and
// bound and the simplex remains the node engine. It does not certify infeasibility or
// unboundedness: when it fails to converge it says so and hands back a numerical error,
// never a claim it cannot prove. Both are stated in the option's description.

#include "sankhya/ipm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "la/ldl.hpp"
#include "la/scaling.hpp"
#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::ipm {
namespace {

/// Convergence: relative primal and dual infeasibility and relative complementarity all
/// below this. Tighter than the reported tolerances (1e-7) so the point survives being
/// re-measured against the original model in recompute_quality().
constexpr double kIpmTolerance = 1e-8;
/// The largest single complementarity product s_k z_k at convergence, relative to the
/// variable's magnitude. The average (mu) is what the method drives down; the status guard
/// judges every product on its own against the primal scale, so an iterate whose average
/// is 1e-8 and whose worst product is 1e-7 would be downgraded to `feasible` after
/// converging. One or two more Mehrotra steps close that gap.
///
/// 1e-8, a decade inside the guard's 1e-7, and not tighter: measured on a KKT instance
/// (test_ipm.cpp, trial 109 of the oracle sweep) the worst relative product reaches 2.3e-9
/// at iteration 12 and then plateaus at 1.05e-9 while the normal equations, by then
/// dominated by Theta entries of 1e+8, hand the factorization pivots at its floor. A
/// threshold of 1e-9 turned a point optimal to ten digits into eleven stalled iterations
/// and a `feasible` verdict; 1e-8 accepts it at iteration 12 and still clears the guard.
constexpr double kIpmComplementarity = 1e-8;
/// The relative duality gap at convergence. 1e-8, the same as the feasibility measures,
/// and MEASURED: 1e-9 and 1e-10 were tried against the verifier's strong-duality threshold
/// and stalled one or two of 150 KKT instances a hair short with the objective already
/// exact, while buying nothing on the instances the verifier rejects (pilot4, scagr25):
/// those rejections arise in original units after postsolve, where an absolute
/// complementarity product of 2e-6 is 1e-9 in the scaled units this loop converges in.
/// A tolerance in one space cannot be met by tightening a tolerance in the other; the
/// verifier's verdict on those instances is reported as it stands.
constexpr double kIpmGap = 1e-8;
/// Primal regularization rho added to every Theta^-1 (Altman & Gondzio): holds free
/// variables and keeps Theta finite as a slack goes to zero.
constexpr double kPrimalRegularization = 1e-8;
/// Dual regularization delta on the diagonal of the normal equations, and the pivot floor
/// the factorization enforces.
constexpr double kDualRegularization = 1e-10;
/// Fraction of the way to the boundary a step may go (Mehrotra's eta).
constexpr double kStepToBoundary = 0.995;
constexpr int kMaxIterations = 300;
/// Iterative refinement steps on each normal-equations solve.
constexpr int kRefinementSteps = 2;
constexpr int kRuizIterations = 10;

class InteriorPoint {
 public:
  InteriorPoint(const Model& model, const Options& options, Logger& logger)
      : model_(model), options_(options), logger_(logger) {}

  Solution run();

 private:
  void build();
  void residuals();
  [[nodiscard]] bool factorize();

  /// The deadline handed down to the linear algebra, so a time limit is not defeated by one
  /// very long ordering or factorization (#193). Empty when there is no limit.
  SparseLdl::ShouldStop should_stop_;
  void newton_direction();
  [[nodiscard]] double step_length(const std::vector<double>& s, const std::vector<double>& ds,
                                   const std::vector<double>& t,
                                   const std::vector<double>& dt) const;
  Solution finish(SolveStatus status, const std::string& message, Count iterations,
                  double seconds);

  /// c_j - a_j^T y for a fixed structural column, whose multipliers do not exist.
  [[nodiscard]] double fixed_reduced_cost(Index j) const {
    const ColumnView column = model_.matrix.column(j);
    double dot = 0.0;
    for (Index p = 0; p < column.size; ++p) {
      dot += column.values[p] * y_[static_cast<std::size_t>(column.rows[p])];
    }
    return cost_[static_cast<std::size_t>(j)] - dot;
  }

  /// Abar v for v over all n + m variables: A v_x - v_s.
  void constraint_times(const std::vector<double>& v, std::vector<double>* out) const;
  /// Abar^T w, over all n + m variables.
  void constraint_transpose_times(const std::vector<double>& w, std::vector<double>* out) const;

  const Model& model_;
  const Options& options_;
  Logger& logger_;

  Index n_ = 0;
  Index m_ = 0;
  Index total_ = 0;
  std::vector<double> cost_;   ///< minimisation sense, over the n structurals (0 for logicals)
  std::vector<double> lower_;  ///< over all total_ variables
  std::vector<double> upper_;
  std::vector<bool> has_lower_;
  std::vector<bool> has_upper_;
  /// l == u: a constant, not a variable. No slack, no multiplier, no dual condition, Theta 0.
  std::vector<bool> fixed_;
  Index bound_count_ = 0;

  // Iterates.
  std::vector<double> x_, y_, sl_, zl_, su_, zu_;
  // Residuals.
  std::vector<double> r_b_, r_c_, r_l_, r_u_;
  double primal_infeasibility_ = 0.0;
  double dual_infeasibility_ = 0.0;
  double mu_ = 0.0;
  double max_product_ = 0.0;
  double objective_ = 0.0;

  // Normal equations.
  std::vector<double> theta_;
  SparseMatrix normal_lower_;
  SparseLdl ldl_;
  bool analyzed_ = false;

  // Directions.
  std::vector<double> dx_, dy_, dsl_, dzl_, dsu_, dzu_;
  std::vector<double> r_mu_l_, r_mu_u_;
  Count factorizations_ = 0;
  Count regularized_pivots_ = 0;

  // THE BEST ITERATE IS KEPT. Near the optimum the normal equations lose conditioning and
  // an iteration can drift; when the loop then stalls or hits a limit, the point returned
  // is the best one seen by the convergence measures, not the last one computed.
  struct Snapshot {
    double merit = std::numeric_limits<double>::infinity();
    std::vector<double> x, y, zl, zu;
  } best_;
  void remember_if_best(double merit);
  void restore_best();
};

void InteriorPoint::remember_if_best(double merit) {
  if (!(merit < best_.merit)) return;
  best_.merit = merit;
  best_.x = x_;
  best_.y = y_;
  best_.zl = zl_;
  best_.zu = zu_;
}

void InteriorPoint::restore_best() {
  if (best_.x.empty()) return;
  x_ = best_.x;
  y_ = best_.y;
  zl_ = best_.zl;
  zu_ = best_.zu;
}

void InteriorPoint::build() {
  n_ = model_.num_cols();
  m_ = model_.num_rows();
  total_ = n_ + m_;
  const double sense = model_.sense_multiplier();
  cost_.assign(static_cast<std::size_t>(total_), 0.0);
  lower_.resize(static_cast<std::size_t>(total_));
  upper_.resize(static_cast<std::size_t>(total_));
  has_lower_.assign(static_cast<std::size_t>(total_), false);
  has_upper_.assign(static_cast<std::size_t>(total_), false);
  fixed_.assign(static_cast<std::size_t>(total_), false);
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    cost_[u] = sense * model_.col_cost[u];
    lower_[u] = model_.col_lower[u];
    upper_[u] = model_.col_upper[u];
  }
  for (Index i = 0; i < m_; ++i) {
    const auto u = static_cast<std::size_t>(n_ + i);
    lower_[u] = model_.row_lower[static_cast<std::size_t>(i)];
    upper_[u] = model_.row_upper[static_cast<std::size_t>(i)];
  }
  bound_count_ = 0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    // A FIXED VARIABLE IS A CONSTANT. Every equality row's logical is one, and the
    // bounded form would give it two slacks that both have to vanish: no interior, Theta
    // driven to zero, complementarity unreachable. It is pinned, carries no bound pair,
    // and drops out of the normal equations; its value still enters Abar x.
    if (lower_[u] == upper_[u] && is_finite_bound(lower_[u])) {
      fixed_[u] = true;
      continue;
    }
    has_lower_[u] = is_finite_bound(lower_[u]);
    has_upper_[u] = is_finite_bound(upper_[u]);
    bound_count_ += (has_lower_[u] ? 1 : 0) + (has_upper_[u] ? 1 : 0);
  }

  // STARTING POINT (Wright, sec. 11.3, simplified): every variable inside its box with a
  // margin, every slack at least 1 so the first iterate is comfortably interior, every
  // multiplier 1. The residuals r_l, r_u absorb any gap between x and its slacks.
  x_.assign(static_cast<std::size_t>(total_), 0.0);
  sl_.assign(static_cast<std::size_t>(total_), 0.0);
  su_.assign(static_cast<std::size_t>(total_), 0.0);
  zl_.assign(static_cast<std::size_t>(total_), 0.0);
  zu_.assign(static_cast<std::size_t>(total_), 0.0);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) {
      x_[u] = lower_[u];
      continue;
    }
    if (has_lower_[u] && has_upper_[u]) {
      x_[u] = 0.5 * (lower_[u] + upper_[u]);
    } else if (has_lower_[u]) {
      x_[u] = lower_[u] + 1.0;
    } else if (has_upper_[u]) {
      x_[u] = upper_[u] - 1.0;
    } else {
      x_[u] = 0.0;
    }
  }
  // THE LOGICALS START CONSISTENT WITH THE STRUCTURALS: s = A x exactly, so the constraint
  // residual r_b is zero at the first iterate and the bound residuals r_l, r_u carry the
  // whole infeasibility. Starting every logical at "bound + 1" instead left r_b of the
  // size of A x, and the affine direction's attempt to close it in one step sent mu from
  // 1 to 1e+9 on israel and stocfor1 before the method could recover - or not.
  if (m_ > 0) {
    std::vector<double> activity(static_cast<std::size_t>(m_), 0.0);
    model_.matrix.multiply_add(x_.data(), activity.data());
    for (Index i = 0; i < m_; ++i) {
      const auto u = static_cast<std::size_t>(n_ + i);
      if (!fixed_[u]) x_[u] = activity[static_cast<std::size_t>(i)];
    }
  }
  // THE DUALS START CONSISTENT WITH THE COSTS, for the same reason: with y = 0 the dual
  // residual is c - z_l + z_u, and z_l = max(c, 1), z_u = max(-c, 1) makes it vanish
  // wherever |c| >= 1 and small elsewhere. Starting every multiplier at 1 instead left a
  // dual residual the size of the scaled cost vector, and the first dual step took z to
  // that size at once: on stocfor1 (scaled) mu went 1 -> 3e6 in four iterations and the
  // run never recovered. The primal slacks are floored at 1 so the first iterate is
  // comfortably interior; r_l and r_u absorb whatever that costs in consistency.
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (fixed_[u]) continue;
    if (has_lower_[u]) {
      sl_[u] = std::max(x_[u] - lower_[u], 1.0);
      zl_[u] = std::max(cost_[u], 1.0);
    }
    if (has_upper_[u]) {
      su_[u] = std::max(upper_[u] - x_[u], 1.0);
      zu_[u] = std::max(-cost_[u], 1.0);
    }
  }
  y_.assign(static_cast<std::size_t>(m_), 0.0);
  const auto alloc = [&](std::vector<double>& v, Index size) {
    v.assign(static_cast<std::size_t>(size), 0.0);
  };
  alloc(r_b_, m_);
  alloc(r_c_, total_);
  alloc(r_l_, total_);
  alloc(r_u_, total_);
  alloc(theta_, total_);
  alloc(dx_, total_);
  alloc(dy_, m_);
  alloc(dsl_, total_);
  alloc(dzl_, total_);
  alloc(dsu_, total_);
  alloc(dzu_, total_);
  alloc(r_mu_l_, total_);
  alloc(r_mu_u_, total_);
}

void InteriorPoint::constraint_times(const std::vector<double>& v,
                                     std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.multiply_add(v.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(i)] -= v[static_cast<std::size_t>(n_ + i)];
  }
}

void InteriorPoint::constraint_transpose_times(const std::vector<double>& w,
                                               std::vector<double>* out) const {
  std::fill(out->begin(), out->end(), 0.0);
  if (m_ == 0) return;
  model_.matrix.transpose_multiply_add(w.data(), out->data());
  for (Index i = 0; i < m_; ++i) {
    (*out)[static_cast<std::size_t>(n_ + i)] = -w[static_cast<std::size_t>(i)];
  }
}

void InteriorPoint::residuals() {
  // r_b = -(Abar x); r_c = c - Abar^T y - z_l + z_u; r_l = x - l - s_l; r_u = u - x - s_u.
  constraint_times(x_, &r_b_);
  double x_norm = 0.0;
  for (Index i = 0; i < m_; ++i)
    r_b_[static_cast<std::size_t>(i)] = -r_b_[static_cast<std::size_t>(i)];
  constraint_transpose_times(y_, &r_c_);
  double c_norm = 0.0;
  double complementarity = 0.0;
  max_product_ = 0.0;
  objective_ = 0.0;
  primal_infeasibility_ = 0.0;
  dual_infeasibility_ = 0.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    r_c_[u] = fixed_[u] ? 0.0 : cost_[u] - r_c_[u] - zl_[u] + zu_[u];
    objective_ += cost_[u] * x_[u];
    c_norm = std::max(c_norm, std::fabs(cost_[u]));
    x_norm = std::max(x_norm, std::fabs(x_[u]));
    dual_infeasibility_ = std::max(dual_infeasibility_, std::fabs(r_c_[u]));
    if (has_lower_[u]) {
      r_l_[u] = x_[u] - lower_[u] - sl_[u];
      complementarity += sl_[u] * zl_[u];
      max_product_ = std::max(max_product_, sl_[u] * zl_[u] / (1.0 + std::fabs(x_[u])));
      primal_infeasibility_ = std::max(primal_infeasibility_, std::fabs(r_l_[u]));
    }
    if (has_upper_[u]) {
      r_u_[u] = upper_[u] - x_[u] - su_[u];
      complementarity += su_[u] * zu_[u];
      max_product_ = std::max(max_product_, su_[u] * zu_[u] / (1.0 + std::fabs(x_[u])));
      primal_infeasibility_ = std::max(primal_infeasibility_, std::fabs(r_u_[u]));
    }
  }
  for (Index i = 0; i < m_; ++i) {
    primal_infeasibility_ =
        std::max(primal_infeasibility_, std::fabs(r_b_[static_cast<std::size_t>(i)]));
  }
  primal_infeasibility_ /= 1.0 + x_norm;
  dual_infeasibility_ /= 1.0 + c_norm;
  mu_ = bound_count_ > 0 ? complementarity / static_cast<double>(bound_count_) : 0.0;
}

bool InteriorPoint::factorize() {
  // Theta = (z_l/s_l + z_u/s_u + rho)^-1 over every variable; the logical block of
  // Abar Theta Abar^T is the diagonal Theta_s, added as a row shift.
  std::vector<double> theta_x(static_cast<std::size_t>(n_));
  std::vector<double> row_shift(static_cast<std::size_t>(m_));
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    double inverse = kPrimalRegularization;
    if (has_lower_[u]) inverse += zl_[u] / sl_[u];
    if (has_upper_[u]) inverse += zu_[u] / su_[u];
    theta_[u] = fixed_[u] ? 0.0 : 1.0 / inverse;
    if (k < n_) {
      theta_x[u] = theta_[u];
    } else {
      row_shift[static_cast<std::size_t>(k - n_)] = theta_[u];
    }
  }
  normal_equations_lower(model_.matrix, theta_x, row_shift, kDualRegularization,
                         &normal_lower_);
  if (!analyzed_) {
    if (!ldl_.analyze(normal_lower_, should_stop_)) return false;
    analyzed_ = true;
  }
  if (!ldl_.factorize(normal_lower_, kDualRegularization, should_stop_)) return false;
  ++factorizations_;
  regularized_pivots_ += ldl_.regularized_pivots();
  return true;
}

/// One Newton direction for the current r_mu terms: solves the normal equations for dy,
/// then recovers dx, ds, dz. The factorization in ldl_ is the current one.
void InteriorPoint::newton_direction() {
  // g_k = r_c - r_mu_l/s_l + z_l r_l/s_l + r_mu_u/s_u - z_u r_u/s_u; rhs = r_b + Abar Theta g.
  std::vector<double> g(static_cast<std::size_t>(total_));
  std::vector<double> theta_g(static_cast<std::size_t>(total_));
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    double value = r_c_[u];
    if (has_lower_[u]) value += (-r_mu_l_[u] + zl_[u] * r_l_[u]) / sl_[u];
    if (has_upper_[u]) value += (r_mu_u_[u] - zu_[u] * r_u_[u]) / su_[u];
    g[u] = value;
    theta_g[u] = theta_[u] * value;
  }
  std::vector<double> rhs(static_cast<std::size_t>(m_));
  constraint_times(theta_g, &rhs);
  for (Index i = 0; i < m_; ++i)
    rhs[static_cast<std::size_t>(i)] += r_b_[static_cast<std::size_t>(i)];

  // Solve with iterative refinement against the matrix actually built (the factors carry
  // the regularization; the residual is measured against the unregularized-by-pivot M).
  dy_ = rhs;
  ldl_.solve(dy_.data());
  const auto multiply_normal = [&](const std::vector<double>& v, std::vector<double>* out) {
    // M is stored as its lower triangle: M v = L v + L^T v - diag v.
    std::fill(out->begin(), out->end(), 0.0);
    for (Index j = 0; j < m_; ++j) {
      const ColumnView column = normal_lower_.column(j);
      const double vj = v[static_cast<std::size_t>(j)];
      double dot = 0.0;
      for (Index p = 0; p < column.size; ++p) {
        const Index i = column.rows[p];
        const double a = column.values[p];
        (*out)[static_cast<std::size_t>(i)] += a * vj;
        if (i != j) dot += a * v[static_cast<std::size_t>(i)];
      }
      (*out)[static_cast<std::size_t>(j)] += dot;
    }
  };
  std::vector<double> residual(static_cast<std::size_t>(m_));
  for (int step = 0; step < kRefinementSteps; ++step) {
    multiply_normal(dy_, &residual);
    for (Index i = 0; i < m_; ++i) {
      residual[static_cast<std::size_t>(i)] =
          rhs[static_cast<std::size_t>(i)] - residual[static_cast<std::size_t>(i)];
    }
    ldl_.solve(residual.data());
    for (Index i = 0; i < m_; ++i)
      dy_[static_cast<std::size_t>(i)] += residual[static_cast<std::size_t>(i)];
  }

  // dx = Theta (Abar^T dy - g); ds_l = dx + r_l; ds_u = -dx + r_u;
  // dz_l = (r_mu_l - z_l ds_l)/s_l; dz_u = (r_mu_u - z_u ds_u)/s_u.
  constraint_transpose_times(dy_, &dx_);
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    dx_[u] = theta_[u] * (dx_[u] - g[u]);
    if (has_lower_[u]) {
      dsl_[u] = dx_[u] + r_l_[u];
      dzl_[u] = (r_mu_l_[u] - zl_[u] * dsl_[u]) / sl_[u];
    }
    if (has_upper_[u]) {
      dsu_[u] = -dx_[u] + r_u_[u];
      dzu_[u] = (r_mu_u_[u] - zu_[u] * dsu_[u]) / su_[u];
    }
  }
}

double InteriorPoint::step_length(const std::vector<double>& s, const std::vector<double>& ds,
                                  const std::vector<double>& t,
                                  const std::vector<double>& dt) const {
  double alpha = 1.0;
  for (Index k = 0; k < total_; ++k) {
    const auto u = static_cast<std::size_t>(k);
    if (has_lower_[u] && ds[u] < 0.0) alpha = std::min(alpha, -s[u] / ds[u]);
    if (has_upper_[u] && dt[u] < 0.0) alpha = std::min(alpha, -t[u] / dt[u]);
  }
  return alpha;
}

Solution InteriorPoint::finish(SolveStatus status, const std::string& message, Count iterations,
                               double seconds) {
  Solution solution;
  solution.allocate_for(model_);
  solution.status = status;
  solution.algorithm = "ipm";
  solution.message = message;
  solution.iterations = iterations;
  solution.solve_seconds = seconds;
  logger_.info("IPM: {} iterations, {} factorizations, {} regularized pivot(s) in total",
               iterations, factorizations_, regularized_pivots_);
  const bool have_point = status == SolveStatus::kOptimal || status == SolveStatus::kFeasible ||
                          status == SolveStatus::kIterationLimit ||
                          status == SolveStatus::kTimeLimit;
  if (!have_point) {
    solution.recompute_quality(model_);
    solution.dual_bound = model_.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
    return solution;
  }
  const double sense = model_.sense_multiplier();
  for (Index j = 0; j < n_; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_value[u] = x_[u];
    // The reduced cost is what the dual constraint says it is at convergence: z_l - z_u;
    // for a fixed column, c - a^T y, which no sign condition constrains.
    solution.col_dual[u] = sense * (fixed_[u] ? fixed_reduced_cost(j) : zl_[u] - zu_[u]);
  }
  for (Index i = 0; i < m_; ++i) {
    solution.row_dual[static_cast<std::size_t>(i)] = sense * y_[static_cast<std::size_t>(i)];
  }
  // No basis: the statuses stay as allocate_for() left them, and the header says why.
  solution.col_status.clear();
  solution.row_status.clear();
  if (status == SolveStatus::kOptimal) {
    solution.dual_bound = model_.evaluate_objective(solution.col_value.data());
  } else {
    solution.dual_bound = model_.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model_);
  return solution;
}

Solution InteriorPoint::run() {
  Timer timer;
  const double time_limit = options_.get_double("time_limit");
  // Handed to the linear algebra so the clock is not only consulted between iterations.
  // Captured by reference to the local timer, which outlives every call that uses it.
  if (time_limit > 0.0 && std::isfinite(time_limit)) {
    should_stop_ = [&timer, time_limit] { return timer.elapsed_seconds() > time_limit; };
  }
  const std::int64_t iteration_limit = options_.get_int("iteration_limit");
  build();
  logger_.info("Interior point: {} rows, {} columns, {} nonzeros", m_, n_,
               model_.num_nonzeros());
  logger_.begin_iteration_table();

  Count iterations = 0;
  double previous_mu = std::numeric_limits<double>::infinity();
  int stalled = 0;
  for (;; ++iterations) {
    residuals();
    logger_.iteration(iterations,
                      model_.sense_multiplier() * objective_ + model_.objective_offset,
                      primal_infeasibility_, dual_infeasibility_, timer.elapsed_seconds());
    const double relative_gap =
        mu_ * static_cast<double>(bound_count_) / (1.0 + std::fabs(objective_));
    remember_if_best(
        std::max({primal_infeasibility_, dual_infeasibility_, relative_gap, max_product_}));
    logger_.verbose(
        "ipm iteration {}: mu {:.2e}, relative gap {:.2e}, worst relative product {:.2e}, "
        "regularized pivots so far {}",
        iterations, mu_, relative_gap, max_product_, regularized_pivots_);
    if (primal_infeasibility_ <= kIpmTolerance && dual_infeasibility_ <= kIpmTolerance &&
        relative_gap <= kIpmGap && max_product_ <= kIpmComplementarity) {
      return finish(SolveStatus::kOptimal, {}, iterations, timer.elapsed_seconds());
    }
    if (iterations >= kMaxIterations ||
        (iteration_limit >= 0 && iterations >= iteration_limit)) {
      restore_best();
      return finish(
          SolveStatus::kIterationLimit,
          fmt::format("iteration limit reached after {} iterations; relative "
                      "infeasibility {:.1e} / {:.1e}, gap {:.1e}",
                      iterations, primal_infeasibility_, dual_infeasibility_, relative_gap),
          iterations, timer.elapsed_seconds());
    }
    if (timer.elapsed_seconds() > time_limit) {
      restore_best();
      return finish(SolveStatus::kTimeLimit,
                    fmt::format("time limit {:g}s reached", time_limit), iterations,
                    timer.elapsed_seconds());
    }
    if (!factorize()) {
      // Told to stop rather than unable to: the difference matters to a reader, and to the
      // status guard. Neither is a point, but only one of them is a failure.
      if (ldl_.stopped_early()) {
        restore_best();
        return finish(SolveStatus::kTimeLimit,
                      fmt::format("time limit {:g}s reached inside the factorization, which "
                                  "was abandoned",
                                  time_limit),
                      iterations, timer.elapsed_seconds());
      }
      return finish(SolveStatus::kNumericalError,
                    "the normal equations could not be factorized", iterations,
                    timer.elapsed_seconds());
    }

    // PREDICTOR: the affine-scaling direction (mu-terms = -s z).
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      r_mu_l_[u] = has_lower_[u] ? -sl_[u] * zl_[u] : 0.0;
      r_mu_u_[u] = has_upper_[u] ? -su_[u] * zu_[u] : 0.0;
    }
    newton_direction();
    const double alpha_p_aff = step_length(sl_, dsl_, su_, dsu_);
    const double alpha_d_aff = step_length(zl_, dzl_, zu_, dzu_);
    double mu_aff = 0.0;
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (has_lower_[u])
        mu_aff += (sl_[u] + alpha_p_aff * dsl_[u]) * (zl_[u] + alpha_d_aff * dzl_[u]);
      if (has_upper_[u])
        mu_aff += (su_[u] + alpha_p_aff * dsu_[u]) * (zu_[u] + alpha_d_aff * dzu_[u]);
    }
    mu_aff = bound_count_ > 0 ? mu_aff / static_cast<double>(bound_count_) : 0.0;
    const double ratio = mu_ > 0.0 ? mu_aff / mu_ : 0.0;
    const double sigma = std::min(1.0, ratio * ratio * ratio);

    // CORRECTOR: centering plus the second-order term from the predictor.
    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      if (has_lower_[u]) r_mu_l_[u] = sigma * mu_ - sl_[u] * zl_[u] - dsl_[u] * dzl_[u];
      if (has_upper_[u]) r_mu_u_[u] = sigma * mu_ - su_[u] * zu_[u] - dsu_[u] * dzu_[u];
    }
    newton_direction();
    const double alpha_p = std::min(1.0, kStepToBoundary * step_length(sl_, dsl_, su_, dsu_));
    const double alpha_d = std::min(1.0, kStepToBoundary * step_length(zl_, dzl_, zu_, dzu_));

    for (Index k = 0; k < total_; ++k) {
      const auto u = static_cast<std::size_t>(k);
      x_[u] += alpha_p * dx_[u];
      if (has_lower_[u]) {
        sl_[u] += alpha_p * dsl_[u];
        zl_[u] += alpha_d * dzl_[u];
      }
      if (has_upper_[u]) {
        su_[u] += alpha_p * dsu_[u];
        zu_[u] += alpha_d * dzu_[u];
      }
    }
    for (Index i = 0; i < m_; ++i)
      y_[static_cast<std::size_t>(i)] += alpha_d * dy_[static_cast<std::size_t>(i)];

    // A method that stops moving is not converging; say so rather than spin to the limit.
    if (mu_ >= 0.999 * previous_mu && alpha_p < 1e-6 && alpha_d < 1e-6) {
      if (++stalled >= 5) {
        // The best iterate is returned as a FEASIBLE point when it met the feasibility
        // tolerances, for the status guard to judge; otherwise as the numerical failure it
        // is. Never as optimal: the convergence test above is the only thing that says so.
        restore_best();
        residuals();
        const bool usable = best_.merit < std::numeric_limits<double>::infinity() &&
                            primal_infeasibility_ <= 1e-6 && dual_infeasibility_ <= 1e-6;
        return finish(usable ? SolveStatus::kFeasible : SolveStatus::kNumericalError,
                      fmt::format("the interior-point iteration stalled after {} iterations "
                                  "(steps {:.1e} / {:.1e}); {}",
                                  iterations, alpha_p, alpha_d,
                                  usable ? "the best iterate is reported as a feasible point"
                                         : "the model may be infeasible or unbounded, which "
                                           "this method does not certify"),
                      iterations, timer.elapsed_seconds());
      }
    } else {
      stalled = 0;
    }
    previous_mu = mu_;
  }
}

/// Scaled solve: Ruiz + Pock-Chambolle equilibration, exactly as the simplex entry point
/// applies it, then the point, the row duals and the reduced costs are mapped back:
/// x = Dc xhat, y = Dr yhat, d = Dc^-1 dhat (la/scaling.hpp).
Solution solve_scaled(const Model& model, const Options& options, Logger& logger) {
  if (!options.get_bool("scaling")) {
    InteriorPoint engine(model, options, logger);
    return engine.run();
  }
  const Scaling scaling = build_scaling(model, model.col_cost, kRuizIterations);
  Model scaled = model;
  scaled.matrix = scaling.matrix;
  scaled.col_cost = scaling.cost;
  const Index n = model.num_cols();
  const Index m = model.num_rows();
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double dc = scaling.column[u];
    scaled.col_lower[u] =
        is_finite_bound(model.col_lower[u]) ? model.col_lower[u] / dc : model.col_lower[u];
    scaled.col_upper[u] =
        is_finite_bound(model.col_upper[u]) ? model.col_upper[u] / dc : model.col_upper[u];
  }
  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double dr = scaling.row[u];
    scaled.row_lower[u] =
        is_finite_bound(model.row_lower[u]) ? model.row_lower[u] * dr : model.row_lower[u];
    scaled.row_upper[u] =
        is_finite_bound(model.row_upper[u]) ? model.row_upper[u] * dr : model.row_upper[u];
  }
  InteriorPoint engine(scaled, options, logger);
  Solution solution = engine.run();
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
  if (solution.status == SolveStatus::kOptimal) {
    solution.dual_bound = model.evaluate_objective(solution.col_value.data());
  }
  solution.recompute_quality(model);
  return solution;
}

}  // namespace

Solution solve_ipm(const Model& model, const Options& options, Logger& logger) {
  return solve_scaled(model, options, logger);
}

}  // namespace sankhya::ipm
