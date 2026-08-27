// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convex QP by a primal-dual proximal method with a forward step.
//
// References
//   Condat, "A primal-dual splitting method for convex optimization involving Lipschitzian,
//     proximable and linear composite terms", J. Optimization Theory and Applications 158
//     (2013).
//   Vu, "A splitting algorithm for dual monotone inclusions involving cocoercive operators",
//     Advances in Computational Mathematics 38 (2013).
//   Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//     applications to imaging", JMIV 40 (2011) - the Q = 0 special case, which is what
//     src/pdhg/ implements for LP.
//
// WHY THIS SHAPE. The LP engine already solves
//
//     min_x max_y   c'x + y'Ax - sigma_C(y),      x in box
//
// by alternating a projected gradient step in x with a proximal step in y. A convex QP adds
// a SMOOTH term, 0.5 x'Qx, whose gradient Qx can simply be added to the primal step - that is
// exactly the Condat-Vu extension of Chambolle-Pock, and it is why the issue recommended this
// route over an active-set or interior-point QP: the projection and Moreau machinery carry
// over unchanged and only the gradient and the step-size rule change.
//
//     x_{k+1} = proj_box( x_k - tau (c + Q x_k + A' y_k) )
//     xbar    = 2 x_{k+1} - x_k
//     y_{k+1} = prox_{sigma sigma_C}( y_k + sigma A xbar )
//
// CONVERGENCE CONDITION. Condat-Vu requires
//
//     1/tau  -  sigma ||A||^2  >=  L / 2,        L = ||Q||_2
//
// Chambolle-Pock's LP condition is the L = 0 case. Both norms are estimated by power
// iteration rather than bounded by a Frobenius or row-sum estimate: a loose bound on ||A||
// forces a small tau and costs iterations directly, and the estimate is computed once.
//
// THIS IS A SEPARATE FILE FROM src/pdhg/ ON PURPOSE. That engine is tuned and verified
// against the whole Netlib set; folding a quadratic term through its adaptive step size and
// restart logic would put the LP path at risk to save duplication in a first QP engine.
// CLAUDE.md is explicit that a wrong answer scores zero, and the LP path is the thing most
// of the project's evidence rests on.

#include "sankhya/qp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "convexity.hpp"

namespace sankhya::qp {
namespace {

/// Componentwise projection onto [lower, upper], tolerant of infinite sides.
[[nodiscard]] double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

/// Power iteration for the largest singular value of A, i.e. ||A||_2.
///
/// Twenty iterations from a fixed seed. This is used only to pick a step size, so a few
/// percent of underestimate would break the convergence condition - the result is therefore
/// inflated by a small margin below rather than used raw.
[[nodiscard]] double spectral_norm(const SparseMatrix& matrix, Index rows, Index cols) {
  if (rows == 0 || cols == 0 || matrix.num_nonzeros() == 0) return 0.0;
  std::mt19937 rng(26119);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);

  std::vector<double> v(static_cast<std::size_t>(cols));
  for (double& value : v) value = unit(rng);
  std::vector<double> av(static_cast<std::size_t>(rows), 0.0);

  double estimate = 0.0;
  for (int iteration = 0; iteration < 20; ++iteration) {
    double norm = 0.0;
    for (const double value : v) norm += value * value;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return 0.0;
    for (double& value : v) value /= norm;

    std::fill(av.begin(), av.end(), 0.0);
    matrix.multiply_add(v.data(), av.data());
    std::fill(v.begin(), v.end(), 0.0);
    matrix.transpose_multiply_add(av.data(), v.data());

    double next = 0.0;
    for (const double value : v) next += value * value;
    estimate = std::sqrt(std::sqrt(next));  // ||A^T A||^{1/2} converges to ||A||
  }
  return estimate;
}

/// Largest eigenvalue of the symmetric Q held as a lower triangle, by the same route.
[[nodiscard]] double hessian_norm(const Model& model) {
  const Index n = model.num_cols();
  if (model.hessian.num_nonzeros() == 0 || n == 0) return 0.0;

  std::mt19937 rng(20260826);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> v(static_cast<std::size_t>(n));
  for (double& value : v) value = unit(rng);
  std::vector<double> qv(static_cast<std::size_t>(n), 0.0);

  double estimate = 0.0;
  for (int iteration = 0; iteration < 20; ++iteration) {
    double norm = 0.0;
    for (const double value : v) norm += value * value;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return 0.0;
    for (double& value : v) value /= norm;

    // qv = Q v, from the lower triangle: the stored off-diagonal entry serves both halves.
    std::fill(qv.begin(), qv.end(), 0.0);
    for (Index j = 0; j < model.hessian.num_cols(); ++j) {
      const ColumnView column = model.hessian.column(j);
      const auto uj = static_cast<std::size_t>(j);
      for (Index k = 0; k < column.size; ++k) {
        const auto ui = static_cast<std::size_t>(column.rows[k]);
        const double value = column.values[k];
        qv[ui] += value * v[uj];
        if (column.rows[k] != j) qv[uj] += value * v[ui];
      }
    }

    double next = 0.0;
    for (std::size_t i = 0; i < qv.size(); ++i) next += qv[i] * v[i];
    estimate = std::fabs(next);
    v = qv;
  }
  return estimate;
}

/// Q x, from the stored lower triangle.
void hessian_multiply(const Model& model, const std::vector<double>& x,
                      std::vector<double>* out) {
  std::fill(out->begin(), out->end(), 0.0);
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    const auto uj = static_cast<std::size_t>(j);
    for (Index k = 0; k < column.size; ++k) {
      const auto ui = static_cast<std::size_t>(column.rows[k]);
      const double value = column.values[k];
      (*out)[ui] += value * x[uj];
      if (column.rows[k] != j) (*out)[uj] += value * x[ui];
    }
  }
}

}  // namespace

Solution solve_convex_qp(const Model& model, const Options& options, Logger& logger) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "qp-condat-vu";

  // ---- convexity, before any arithmetic ---------------------------------------------------
  const ConvexityResult convexity = check_convexity(model);
  if (convexity.verdict != Convexity::kConvex) {
    solution.status = SolveStatus::kModelError;
    solution.message =
        convexity.verdict == Convexity::kIndefinite
            ? fmt::format(
                  "the quadratic objective is not convex: {}. A non-convex QP has "
                  "local minima and saddle points, and this engine would return one "
                  "of them labelled optimal, so it is refused instead",
                  convexity.detail)
            : fmt::format("convexity could not be established: {}", convexity.detail);
    logger.warning("{}", solution.message);
    solution.solve_seconds = timer.elapsed_seconds();
    return solution;
  }

  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const auto un = static_cast<std::size_t>(n);
  const auto um = static_cast<std::size_t>(m);
  const double sense = model.sense_multiplier();

  // ---- step sizes -------------------------------------------------------------------------
  const double norm_a = spectral_norm(model.matrix, m, n);
  const double norm_q = hessian_norm(model);

  // Condat-Vu needs 1/tau - sigma ||A||^2 >= L/2. Take sigma ||A||^2 = 1/(2 tau) and solve,
  // with a 5% margin because both norms are power-iteration ESTIMATES and an underestimate
  // would violate the condition rather than merely slow convergence.
  const double margin = 1.05;
  const double l = margin * norm_q;
  const double a2 = margin * margin * std::max(norm_a * norm_a, 1e-12);
  const double tau = 1.0 / (l / 2.0 + std::sqrt(a2) + 1e-12);
  const double sigma = (m > 0) ? (1.0 / tau - l / 2.0) / (2.0 * a2) : 0.0;

  logger.verbose("QP step sizes: ||A|| ~ {:.4g}, ||Q|| ~ {:.4g}, tau {:.4g}, sigma {:.4g}",
                 norm_a, norm_q, tau, sigma);

  // ---- iterate ----------------------------------------------------------------------------
  std::vector<double> x(un, 0.0);
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    x[u] = project(0.0, model.col_lower[u], model.col_upper[u]);
  }
  std::vector<double> y(um, 0.0);
  std::vector<double> x_next(un, 0.0);
  std::vector<double> extrapolated(un, 0.0);
  std::vector<double> qx(un, 0.0);
  std::vector<double> at_y(un, 0.0);
  std::vector<double> ax(um, 0.0);

  const double tolerance = options.get_double("qp_tolerance");
  const Count iteration_limit =
      options.get_int("iteration_limit") < 0 ? 1000000 : options.get_int("iteration_limit");
  const double time_limit = options.get_double("time_limit");

  Count iterations = 0;
  std::string message;
  SolveStatus status = SolveStatus::kIterationLimit;

  for (; iterations < iteration_limit; ++iterations) {
    // Primal: x' = proj_box( x - tau (c + Qx + A'y) ). The Qx term is the whole difference
    // from the LP engine; everything else is Chambolle-Pock unchanged.
    hessian_multiply(model, x, &qx);
    std::fill(at_y.begin(), at_y.end(), 0.0);
    if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = sense * model.col_cost[u] + sense * qx[u] + at_y[u];
      x_next[u] = project(x[u] - tau * gradient, model.col_lower[u], model.col_upper[u]);
      extrapolated[u] = 2.0 * x_next[u] - x[u];
    }

    // Dual: y' = prox_{sigma sigma_C}(y + sigma A xbar) = v - sigma proj_C(v / sigma).
    if (m > 0) {
      model.matrix.multiply(extrapolated.data(), ax.data());
      for (Index i = 0; i < m; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double v = y[u] + sigma * ax[u];
        y[u] = v - sigma * project(v / sigma, model.row_lower[u], model.row_upper[u]);
      }
    }
    x.swap(x_next);

    // ---- termination, every 50 iterations -------------------------------------------------
    if (iterations % 50 != 0) continue;

    if (time_limit < std::numeric_limits<double>::max() &&
        timer.elapsed_seconds() > time_limit) {
      status = SolveStatus::kTimeLimit;
      message = fmt::format("time limit {:.3g}s reached", time_limit);
      break;
    }

    // Primal residual: the worst row-bound violation.
    double primal_residual = 0.0;
    if (m > 0) {
      model.matrix.multiply(x.data(), ax.data());
      for (Index i = 0; i < m; ++i) {
        const auto u = static_cast<std::size_t>(i);
        const double projected = project(ax[u], model.row_lower[u], model.row_upper[u]);
        primal_residual = std::max(primal_residual, std::fabs(ax[u] - projected));
      }
    }

    // Dual residual: the projected-gradient stationarity measure. At an optimum, taking a
    // unit gradient step and projecting back changes nothing.
    hessian_multiply(model, x, &qx);
    std::fill(at_y.begin(), at_y.end(), 0.0);
    if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
    double dual_residual = 0.0;
    for (Index j = 0; j < n; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = sense * model.col_cost[u] + sense * qx[u] + at_y[u];
      const double stepped = project(x[u] - gradient, model.col_lower[u], model.col_upper[u]);
      dual_residual = std::max(dual_residual, std::fabs(x[u] - stepped));
    }

    if (primal_residual <= tolerance && dual_residual <= tolerance) {
      status = SolveStatus::kOptimal;
      break;
    }
    if (iterations % 5000 == 0) {
      logger.iteration(iterations, model.evaluate_objective(x.data()), primal_residual,
                       dual_residual, timer.elapsed_seconds());
    }
  }

  if (status == SolveStatus::kIterationLimit && message.empty()) {
    message = fmt::format("iteration limit {} reached", iteration_limit);
  }

  solution.status = status;
  solution.message = message;
  solution.col_value.assign(x.begin(), x.end());
  for (Index i = 0; i < m; ++i) {
    solution.row_dual[static_cast<std::size_t>(i)] = -sense * y[static_cast<std::size_t>(i)];
  }
  hessian_multiply(model, x, &qx);
  std::fill(at_y.begin(), at_y.end(), 0.0);
  if (m > 0) model.matrix.transpose_multiply_add(y.data(), at_y.data());
  for (Index j = 0; j < n; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_dual[u] = model.col_cost[u] + qx[u] + sense * at_y[u];
  }
  solution.iterations = iterations;
  solution.solve_seconds = timer.elapsed_seconds();
  // An unfinished run has proven nothing about the bound, so it must not present its
  // incumbent as one - the same rule the LP engines follow.
  solution.dual_bound = status == SolveStatus::kOptimal
                            ? model.evaluate_objective(x.data())
                            : (model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity);
  solution.recompute_quality(model);
  return solution;
}

}  // namespace sankhya::qp
