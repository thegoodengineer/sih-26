// SPDX-License-Identifier: Apache-2.0
// SANKHYA - restarted primal-dual hybrid gradient for LP.
//
// References, all written from the papers. Per CLAUDE.md the source of PDLP, cuPDLP,
// cuPDLP-C, cuPDLPx, OR-Tools and HiGHS was NOT consulted.
//   [CP11]  Chambolle & Pock, "A first-order primal-dual algorithm for convex problems with
//           applications to imaging", JMIV 40(1), 2011. Algorithm 1 is the iteration below.
//   [PDLP]  Applegate, Diaz, Hinder, Lu, Lubin, O'Donoghue, Schudy, "Practical Large-Scale
//           Linear Programming using Primal-Dual Hybrid Gradient", NeurIPS 2021.
//           Section 3.1 adaptive step size, 3.2 primal weight, 4.3 restarts.
//   [cuPDLP] Lu & Yang, "cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid
//           Gradient for Linear Programming in Julia", arXiv:2311.12180.
//
// WHY THIS ENGINE EXISTS. The revised simplex is sequential: every pivot depends on the one
// before it, so it does not parallelise onto a GPU and we will not claim it does. PDHG
// replaces factorization with repeated sparse matrix-vector products, has no serial
// dependency inside an iteration, and is therefore the engine that can go on a GPU. That is
// the whole GPU story, and it is the same conclusion the field reached after 2023.
//
// FORMULATION. Everything is kept in the two-sided form the Model already carries, with no
// row splitting and no slack variables:
//
//     min_{x in X} max_y   c'x + y'Ax - sigma_C(y),    X = [l, u],  C = [rl, ru]
//
// where sigma_C is the support function of the row-bound box. The y update is then a
// proximal step on sigma_C, and Moreau's identity turns that into a projection onto C:
//
//     prox_{s sigma_C}(v) = v - s * proj_C(v / s)
//
// which handles equality rows, one-sided rows, range rows and free rows through one
// expression. Splitting a range row into two inequalities would have doubled the matrix and
// added a transformation to get wrong.
//
// Note the sign convention: y here is the NEGATIVE of the multiplier the simplex reports,
// because the Lagrangian above adds y'Ax rather than subtracting it. unscale_and_report()
// flips it back, and test_pdhg.cpp pins the two engines against each other so the flip
// cannot silently invert.

#include "sankhya/pdhg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "../la/scaling.hpp"

namespace sankhya::pdhg {
namespace {

constexpr int kRuizIterations = 10;
constexpr int kPowerIterations = 30;
/// How often the (relatively expensive, unscaled) convergence test runs.
constexpr Count kEvaluationInterval = 40;

/// Componentwise projection onto [lower, upper], tolerant of infinite sides.
double project(double value, double lower, double upper) {
  if (is_finite_bound(lower) && value < lower) return lower;
  if (is_finite_bound(upper) && value > upper) return upper;
  return value;
}

double squared_norm(const std::vector<double>& v) {
  double total = 0.0;
  for (const double value : v) total += value * value;
  return total;
}

double euclidean_norm(const std::vector<double>& v) {
  return std::sqrt(squared_norm(v));
}

/// Convergence measured in the ORIGINAL problem, never the scaled one. Reporting residuals
/// from the scaled problem would let a well-chosen preconditioner flatter the result.
struct Residuals {
  double primal = 0.0;  ///< relative primal infeasibility
  double dual = 0.0;    ///< relative dual infeasibility
  double gap = 0.0;     ///< relative primal-dual objective gap
  double primal_objective = 0.0;
  double dual_objective = 0.0;

  /// The same violations UNSCALED. A relative residual divides by (1 + ||bounds||), so on a
  /// model whose right-hand sides run to 1e4 a relative 1e-8 still permits an absolute
  /// violation around 1e-4. That is standard and fine as a stopping rule - it is what the
  /// PDLP literature uses - but it is NOT the standard the rest of this project reports
  /// against, and conflating the two is how the solver ends up stamping "optimal" on a point
  /// tools/verify_solution.py then rejects.
  double absolute_primal = 0.0;
  double absolute_dual = 0.0;

  /// The duality gap normalised the way tools/verify_solution.py normalises it, by
  /// max(1, |primal objective|), rather than the PDLP convention of 1 + |primal| + |dual|.
  /// The two differ by roughly a factor of two, which is more than enough for this engine to
  /// pass its own optimality test and fail the verifier's on the same point. The PDLP form
  /// stays as the stopping rule because that is the literature convention; the claim is
  /// judged by the verifier's form because that is what will be checked.
  double gap_as_verified = 0.0;

  /// max |multiplier| * slack over rows and columns - the same product form
  /// tools/verify_solution.py uses. The duality gap implies this only in the limit, so a
  /// point can show a tiny gap and still price a constraint it is not sitting on.
  double complementarity = 0.0;

  [[nodiscard]] double worst() const { return std::max({primal, dual, gap}); }

  /// Has the run met the tolerance the CALLER asked for, AND is the point actually feasible
  /// in absolute terms? Both are required to stop.
  ///
  /// The absolute half is not pedantry. kFeasible in sankhya::Solution asserts that a
  /// feasible point is being reported, so stopping on a relative residual alone would let
  /// this engine claim feasibility for a point that misses the project's own primal
  /// tolerance - a weaker claim than kOptimal, but still one the verifier rejects.
  [[nodiscard]] bool meets_request(double tolerance) const {
    return primal <= tolerance && dual <= tolerance && gap <= tolerance &&
           absolute_primal <= tol::kPrimalFeasibility;
  }

  /// Would this point survive independent verification? These are the project's own
  /// tolerances from include/sankhya/tolerances.hpp, the same ones the .sol file is judged
  /// against, and meeting them is the ONLY basis on which this engine claims kOptimal.
  [[nodiscard]] bool meets_project_standard() const {
    return absolute_primal <= tol::kPrimalFeasibility &&
           absolute_dual <= tol::kDualFeasibility && gap_as_verified <= tol::kDualityGap &&
           complementarity <= 1e-6;
  }
};

/// The unscaled problem, held once so the convergence test does not rebuild it.
struct Problem {
  const Model* model = nullptr;
  std::vector<double> cost;  ///< minimise-space objective
  double bound_norm = 0.0;   ///< ||finite row bounds||, for the relative primal residual
  double cost_norm = 0.0;    ///< ||c||, for the relative dual residual
};

/// Compute the unscaled residuals for a candidate (x, y).
Residuals evaluate(const Problem& problem, const std::vector<double>& x,
                   const std::vector<double>& y, std::vector<double>* activity,
                   std::vector<double>* reduced) {
  const Model& model = *problem.model;
  const Index rows = model.num_rows();
  const Index cols = model.num_cols();

  Residuals r;

  // ---- Primal: how far Ax falls outside the row bounds. x is projected every iteration,
  // so the column bounds hold by construction and contribute nothing.
  activity->assign(static_cast<std::size_t>(rows), 0.0);
  if (rows > 0) model.matrix.multiply(x.data(), activity->data());
  double primal_violation = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double a = (*activity)[u];
    double violation = 0.0;
    if (is_finite_bound(model.row_lower[u])) {
      violation = std::max(violation, model.row_lower[u] - a);
    }
    if (is_finite_bound(model.row_upper[u])) {
      violation = std::max(violation, a - model.row_upper[u]);
    }
    primal_violation += violation * violation;
  }
  r.absolute_primal = std::sqrt(primal_violation);
  r.primal = r.absolute_primal / (1.0 + problem.bound_norm);

  // ---- Dual: d = c + A'y. A component of d is only a violation where no bound can absorb
  // it, i.e. a positive reduced cost on a variable with no lower bound, or a negative one
  // on a variable with no upper bound.
  reduced->assign(static_cast<std::size_t>(cols), 0.0);
  for (Index j = 0; j < cols; ++j) {
    (*reduced)[static_cast<std::size_t>(j)] = problem.cost[static_cast<std::size_t>(j)];
  }
  if (rows > 0) model.matrix.transpose_multiply_add(y.data(), reduced->data());

  double dual_violation = 0.0;
  double bound_contribution = 0.0;
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    const double d = (*reduced)[u];
    if (d > 0.0) {
      if (is_finite_bound(model.col_lower[u])) {
        bound_contribution += d * model.col_lower[u];
      } else {
        dual_violation += d * d;
      }
    } else if (d < 0.0) {
      if (is_finite_bound(model.col_upper[u])) {
        bound_contribution += d * model.col_upper[u];
      } else {
        dual_violation += d * d;
      }
    }
  }
  // The dual residual is assigned once, below, after the row multipliers have had
  // their chance to contribute a violation too.

  // ---- Objectives. The dual objective is the Lagrangian bound:
  //   sum_j (d_j > 0 ? d_j l_j : d_j u_j)  -  sigma_C(y)
  double support = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double yi = y[u];
    if (yi > 0.0) {
      if (is_finite_bound(model.row_upper[u])) {
        support += yi * model.row_upper[u];
      } else {
        dual_violation += yi * yi;  // no bound to price against: the dual is infeasible
      }
    } else if (yi < 0.0) {
      if (is_finite_bound(model.row_lower[u])) {
        support += yi * model.row_lower[u];
      } else {
        dual_violation += yi * yi;
      }
    }
  }
  r.absolute_dual = std::sqrt(dual_violation);
  r.dual = r.absolute_dual / (1.0 + problem.cost_norm);

  // Complementary slackness, in the product form the verifier uses.
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (model.row_lower[u] == model.row_upper[u]) continue;  // equality: always tight
    const double lower_slack = is_finite_bound(model.row_lower[u])
                                   ? (*activity)[u] - model.row_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.row_upper[u])
                                   ? model.row_upper[u] - (*activity)[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity =
        std::max(r.complementarity, std::fabs(y[u]) * std::min(lower_slack, upper_slack));
  }
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    if (model.col_lower[u] == model.col_upper[u]) continue;  // fixed column
    const double lower_slack = is_finite_bound(model.col_lower[u])
                                   ? x[u] - model.col_lower[u]
                                   : std::numeric_limits<double>::infinity();
    const double upper_slack = is_finite_bound(model.col_upper[u])
                                   ? model.col_upper[u] - x[u]
                                   : std::numeric_limits<double>::infinity();
    r.complementarity = std::max(r.complementarity,
                                 std::fabs((*reduced)[u]) * std::min(lower_slack, upper_slack));
  }

  double primal_objective = 0.0;
  for (Index j = 0; j < cols; ++j) {
    primal_objective +=
        problem.cost[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
  }
  r.primal_objective = primal_objective;
  r.dual_objective = bound_contribution - support;
  const double absolute_gap = std::fabs(r.primal_objective - r.dual_objective);
  r.gap = absolute_gap / (1.0 + std::fabs(r.primal_objective) + std::fabs(r.dual_objective));
  r.gap_as_verified = absolute_gap / std::max(1.0, std::fabs(r.primal_objective));
  return r;
}

}  // namespace

Solution solve_pdhg(const Model& model, const Options& options, Logger& logger) {
  Timer timer;
  Solution solution;
  solution.allocate_for(model);
  solution.algorithm = "pdhg-cpu";

  const std::string problem_text = model.validate();
  if (!problem_text.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem_text;
    return solution;
  }

  const Index rows = model.num_rows();
  const Index cols = model.num_cols();
  const double sense = model.sense_multiplier();

  // ---- The unscaled problem, in minimise space ------------------------------------------
  Problem problem;
  problem.model = &model;
  problem.cost.resize(static_cast<std::size_t>(cols));
  for (Index j = 0; j < cols; ++j) {
    problem.cost[static_cast<std::size_t>(j)] =
        sense * model.col_cost[static_cast<std::size_t>(j)];
  }
  problem.cost_norm = euclidean_norm(problem.cost);
  double bound_square = 0.0;
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double b = is_finite_bound(model.row_lower[u])
                         ? model.row_lower[u]
                         : (is_finite_bound(model.row_upper[u]) ? model.row_upper[u] : 0.0);
    bound_square += b * b;
  }
  problem.bound_norm = std::sqrt(bound_square);

  // ---- Preconditioning -------------------------------------------------------------------
  const Scaling scaling = build_scaling(model, problem.cost, kRuizIterations);
  const double spectral_norm =
      estimate_spectral_norm(scaling.matrix, kPowerIterations,
                             static_cast<unsigned>(options.get_int("random_seed")) + 1u);

  const double tolerance = options.get_double("pdhg_tolerance");
  const double time_limit = options.get_double("time_limit");
  const std::int64_t iteration_option = options.get_int("iteration_limit");
  const Count iteration_limit =
      iteration_option < 0 ? 1000000 : static_cast<Count>(iteration_option);
  const bool use_restarts = options.get_bool("pdhg_restart");

  logger.info("Solving LP with restarted PDHG: {} rows, {} columns, {} nonzeros", rows, cols,
              model.num_nonzeros());
  logger.info("Scaled matrix entries in [{:.3e}, {:.3e}], estimated ||A||_2 = {:.4e}",
              scaling.min_abs, scaling.max_abs, spectral_norm);
  logger.info("Target relative tolerance {:.1e}, restarts {}", tolerance,
              use_restarts ? "on" : "off");

  // ---- Iterates, in SCALED space ----------------------------------------------------------
  const auto n = static_cast<std::size_t>(cols);
  const auto m = static_cast<std::size_t>(rows);
  std::vector<double> x(n, 0.0);
  std::vector<double> y(m, 0.0);
  for (Index j = 0; j < cols; ++j) {
    // Start at the projection of zero, which is the closest feasible point to the origin.
    x[static_cast<std::size_t>(j)] =
        project(0.0, scaling.col_lower[static_cast<std::size_t>(j)],
                scaling.col_upper[static_cast<std::size_t>(j)]);
  }

  std::vector<double> x_next(n, 0.0);
  std::vector<double> y_next(m, 0.0);
  std::vector<double> extrapolated(n, 0.0);
  std::vector<double> at_y(n, 0.0);
  std::vector<double> a_x(m, 0.0);

  // Running average since the last restart. PDLP restarts to whichever of the average and
  // the current iterate has the better KKT error.
  std::vector<double> x_sum(n, 0.0);
  std::vector<double> y_sum(m, 0.0);
  Count averaged = 0;

  std::vector<double> x_restart = x;
  std::vector<double> y_restart = y;

  // Unscaled scratch for the convergence test.
  std::vector<double> x_unscaled(n, 0.0);
  std::vector<double> y_unscaled(m, 0.0);
  std::vector<double> activity(m, 0.0);
  std::vector<double> reduced(n, 0.0);

  const auto unscale = [&](const std::vector<double>& xs, const std::vector<double>& ys) {
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      x_unscaled[u] = xs[u] * scaling.column[u];
    }
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      y_unscaled[u] = ys[u] * scaling.row[u];
    }
  };

  double eta = spectral_norm > 0.0 ? 1.0 / spectral_norm : 1.0;
  double omega = 1.0;  // primal weight
  Count iteration = 0;
  Count restarts = 0;
  Count last_restart = 0;
  double restart_kkt = std::numeric_limits<double>::infinity();

  Residuals best;
  best.primal = best.dual = best.gap = std::numeric_limits<double>::infinity();
  std::vector<double> best_x = x;
  std::vector<double> best_y = y;

  bool converged = false;
  bool logged_table = false;

  while (true) {
    if (iteration >= iteration_limit) break;
    if (timer.elapsed_seconds() > time_limit) break;

    // ---- One PDHG step, [CP11] Algorithm 1 with step sizes tau = eta/omega, sigma =
    // eta*omega.
    const double tau = eta / omega;
    const double sigma = eta * omega;

    // Primal: x' = proj_X( x - tau (c + A'y) )
    for (Index j = 0; j < cols; ++j) at_y[static_cast<std::size_t>(j)] = 0.0;
    if (rows > 0) scaling.matrix.transpose_multiply(y.data(), at_y.data());
    for (Index j = 0; j < cols; ++j) {
      const auto u = static_cast<std::size_t>(j);
      const double gradient = scaling.cost[u] + at_y[u];
      x_next[u] = project(x[u] - tau * gradient, scaling.col_lower[u], scaling.col_upper[u]);
      extrapolated[u] = 2.0 * x_next[u] - x[u];  // the [CP11] extrapolation
    }

    // Dual: y' = prox_{sigma sigma_C}( y + sigma A xbar ) = v - sigma proj_C(v / sigma)
    if (rows > 0) scaling.matrix.multiply(extrapolated.data(), a_x.data());
    for (Index i = 0; i < rows; ++i) {
      const auto u = static_cast<std::size_t>(i);
      const double v = y[u] + sigma * a_x[u];
      y_next[u] = v - sigma * project(v / sigma, scaling.row_lower[u], scaling.row_upper[u]);
    }

    // ---- Adaptive step size, [PDLP] section 3.1 ------------------------------------------
    // The step is admissible while eta <= (movement) / (interaction). Both are measured on
    // the step just taken, so a rejected step costs one matvec and is retried smaller.
    double movement = 0.0;
    for (Index j = 0; j < cols; ++j) {
      const double d = x_next[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(j)];
      movement += 0.5 * omega * d * d;
    }
    for (Index i = 0; i < rows; ++i) {
      const double d = y_next[static_cast<std::size_t>(i)] - y[static_cast<std::size_t>(i)];
      movement += 0.5 * d * d / omega;
    }

    double interaction = 0.0;
    if (rows > 0) {
      // (y' - y)' A (x' - x)
      std::vector<double> dx(n);
      for (Index j = 0; j < cols; ++j) {
        dx[static_cast<std::size_t>(j)] =
            x_next[static_cast<std::size_t>(j)] - x[static_cast<std::size_t>(j)];
      }
      std::vector<double> adx(m, 0.0);
      scaling.matrix.multiply(dx.data(), adx.data());
      for (Index i = 0; i < rows; ++i) {
        const auto u = static_cast<std::size_t>(i);
        interaction += (y_next[u] - y[u]) * adx[u];
      }
      interaction = std::fabs(interaction);
    }

    // Zero interaction means the step carried NO information about how large eta may safely
    // be. That is not an exotic case: it happens whenever one of the two blocks does not
    // move, which is the normal transient while the primal is still pinned against a bound
    // at start-up, and again once the iterates converge.
    //
    // Neither of the obvious readings works. Treating it as an infinite limit lets eta grow
    // by the `grow` factor every iteration, so on a problem that converges in a few steps
    // eta overflows to infinity and the objective comes back NaN. Setting limit = eta does
    // not hold eta either, because the proposal is min(shrink * limit, grow * eta) and
    // shrink < 1, so eta HALVES on every such iteration and collapses to the floor - after
    // which nothing can move at all.
    //
    // The step is trivially admissible when there is no interaction, so the honest response
    // is to accept it and leave eta exactly where it was.
    const bool no_information = interaction <= 0.0;
    const double limit =
        no_information ? std::numeric_limits<double>::infinity() : movement / interaction;
    const double exponent = static_cast<double>(iteration + 1);
    const double shrink = 1.0 - std::pow(exponent, -0.3);
    const double grow = 1.0 + std::pow(exponent, -0.6);
    const double proposed = std::min(shrink * limit, grow * eta);

    if (eta <= limit) {
      // Accept.
      x.swap(x_next);
      y.swap(y_next);
      for (Index j = 0; j < cols; ++j) {
        x_sum[static_cast<std::size_t>(j)] += x[static_cast<std::size_t>(j)];
      }
      for (Index i = 0; i < rows; ++i) {
        y_sum[static_cast<std::size_t>(i)] += y[static_cast<std::size_t>(i)];
      }
      ++averaged;
      ++iteration;
    }
    // Whether accepted or not, the step size moves to the proposal. A rejected step is
    // therefore always retried smaller, which is what makes the rule terminate. The upper
    // clamp is a backstop against unbounded growth: the vanilla method needs
    // eta <= 1/||A||_2, and the adaptive rule may exceed that safely, but never by orders
    // of magnitude.
    const double eta_ceiling = 1.0e3 / std::max(spectral_norm, 1e-12);
    if (!no_information) eta = std::clamp(proposed, 1e-12, eta_ceiling);

    // ---- Convergence and restart -----------------------------------------------------------
    // Evaluate on the periodic tick, and ALSO the moment the iterates stop moving: a small
    // problem can converge in fewer steps than the tick interval, and would otherwise spin
    // to the iteration limit having already found the answer.
    if (iteration == 0) continue;
    if (iteration % kEvaluationInterval != 0 && !no_information) continue;

    unscale(x, y);
    std::vector<double> current_x = x_unscaled;
    std::vector<double> current_y = y_unscaled;
    const Residuals current = evaluate(problem, current_x, current_y, &activity, &reduced);

    // PDLP restarts to whichever of the running average and the current iterate has the
    // better KKT error, so both are evaluated and the better one is carried forward.
    const Residuals* chosen = &current;
    const std::vector<double>* chosen_x = &current_x;
    const std::vector<double>* chosen_y = &current_y;

    Residuals average;
    std::vector<double> average_x;
    std::vector<double> average_y;
    if (averaged > 0) {
      std::vector<double> x_avg(n);
      std::vector<double> y_avg(m);
      const auto count = static_cast<double>(averaged);
      for (Index j = 0; j < cols; ++j) {
        x_avg[static_cast<std::size_t>(j)] = x_sum[static_cast<std::size_t>(j)] / count;
      }
      for (Index i = 0; i < rows; ++i) {
        y_avg[static_cast<std::size_t>(i)] = y_sum[static_cast<std::size_t>(i)] / count;
      }
      unscale(x_avg, y_avg);
      average_x = x_unscaled;
      average_y = y_unscaled;
      average = evaluate(problem, average_x, average_y, &activity, &reduced);
      if (average.worst() < current.worst()) {
        chosen = &average;
        chosen_x = &average_x;
        chosen_y = &average_y;
      }
    }

    const Residuals& better = *chosen;
    if (better.worst() < best.worst()) {
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
    }

    if (!logged_table) {
      logger.begin_iteration_table();
      logged_table = true;
    }
    // `+ model.objective_offset`, for the same reason PrimalSimplex::minimization_objective()
    // adds it: the number in the iteration table and in the --progress-out stream has to be
    // the same quantity the final line and the .sol file report. Without it this logged the
    // objective of whatever model the engine was handed, and presolve hands it a model whose
    // offset absorbs every fixed and empty column it eliminated (presolve.cpp:397). The
    // final Solution added the offset back at line 660 below; the live stream did not, so a
    // solve watched through `tail -f` converged to a number the result never reached - off
    // by exactly the offset, 365 on the demo's 5000x5000 instance. Both are correct with
    // presolve disabled, which is what made it look like noise rather than a missing term.
    logger.iteration(iteration, sense * better.primal_objective + model.objective_offset,
                     better.primal, better.dual, timer.elapsed_seconds());

    if (better.meets_request(tolerance)) {
      // Report the point that PASSED, not whichever earlier iterate happened to have the
      // smallest relative residual. best_x tracks worst(), which is a relative measure, so
      // an earlier iterate can hold that title while being less feasible in absolute terms -
      // and reporting it would hand back a point that never satisfied the stopping test.
      best = better;
      best_x = *chosen_x;
      best_y = *chosen_y;
      converged = true;
      break;
    }

    if (use_restarts) {
      // [PDLP] section 4.3. The exact normalised duality gap needs a trust-region
      // subproblem per candidate; the KKT error is the practical proxy the paper describes,
      // and it is what is used here. Said plainly so the log is not mistaken for the
      // theoretical criterion.
      const double kkt = better.worst();
      const Count since = iteration - last_restart;
      const bool sufficient = kkt <= 0.2 * restart_kkt;
      const bool artificial =
          since >= std::max<Count>(kEvaluationInterval,
                                   static_cast<Count>(0.36 * static_cast<double>(iteration)));
      if (sufficient || artificial) {
        // Restart at the better candidate, and move the primal weight towards the observed
        // ratio of dual to primal movement ([PDLP] section 3.2, theta = 0.5).
        std::vector<double> dx(n);
        std::vector<double> dy(m);
        for (Index j = 0; j < cols; ++j) {
          dx[static_cast<std::size_t>(j)] =
              x[static_cast<std::size_t>(j)] - x_restart[static_cast<std::size_t>(j)];
        }
        for (Index i = 0; i < rows; ++i) {
          dy[static_cast<std::size_t>(i)] =
              y[static_cast<std::size_t>(i)] - y_restart[static_cast<std::size_t>(i)];
        }
        const double dx_norm = euclidean_norm(dx);
        const double dy_norm = euclidean_norm(dy);
        if (dx_norm > 1e-12 && dy_norm > 1e-12) {
          const double theta = 0.5;
          omega =
              std::exp(theta * std::log(dy_norm / dx_norm) + (1.0 - theta) * std::log(omega));
          omega = std::clamp(omega, 1e-6, 1e6);
        }

        std::fill(x_sum.begin(), x_sum.end(), 0.0);
        std::fill(y_sum.begin(), y_sum.end(), 0.0);
        averaged = 0;
        x_restart = x;
        y_restart = y;
        restart_kkt = kkt;
        last_restart = iteration;
        ++restarts;
        logger.verbose("restart {} at iteration {}: KKT {:.3e}, primal weight {:.3e}", restarts,
                       iteration, kkt, omega);
      }
    }
  }

  // ---- Report ----------------------------------------------------------------------------
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_value[u] = best_x.empty() ? 0.0 : best_x[u];
  }
  // Recompute the reduced costs at the reported point so the .sol file is self-consistent.
  const Residuals final_residuals = evaluate(problem, best_x, best_y, &activity, &reduced);
  for (Index j = 0; j < cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    solution.col_dual[u] = sense * reduced[u];
  }
  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    // The Lagrangian above adds y'Ax, so the reported multiplier is the negation.
    solution.row_dual[u] = sense * (-best_y[u]);
  }

  solution.iterations = iteration;
  solution.solve_seconds = timer.elapsed_seconds();

  // kOptimal is a claim that this point would survive tools/verify_solution.py, which
  // measures ABSOLUTE feasibility against the tolerances in tolerances.hpp. Meeting the
  // caller's RELATIVE tolerance is a different and weaker statement: on a model whose
  // right-hand sides run to 1e4, a relative 1e-8 leaves an absolute violation around 1e-4.
  //
  // Claiming optimality on the weaker test is how this engine came to stamp "optimal" on
  // points the verifier rejected. A point that stops on the caller's tolerance but misses
  // the project standard is a usable answer with no optimality claim attached - which is
  // exactly what kFeasible means, and it is what gets reported now.
  const bool verifiable = converged && final_residuals.meets_project_standard();

  if (verifiable) {
    solution.status = SolveStatus::kOptimal;
    solution.message = fmt::format(
        "converged after {} iterations and {} restarts; absolute primal {:.3e}, dual {:.3e}, "
        "relative gap {:.3e}",
        iteration, restarts, final_residuals.absolute_primal, final_residuals.absolute_dual,
        final_residuals.gap_as_verified);
  } else if (converged) {
    solution.status = SolveStatus::kFeasible;
    solution.message = fmt::format(
        "met the requested relative tolerance {:.1e} after {} iterations, but NOT the "
        "absolute standard this project verifies against (primal {:.3e} vs {:.1e}, dual "
        "{:.3e} vs {:.1e}, relative gap {:.3e} vs {:.1e}). Reported as feasible, not "
        "optimal. Tighten --option pdhg_tolerance to close it",
        tolerance, iteration, final_residuals.absolute_primal, tol::kPrimalFeasibility,
        final_residuals.absolute_dual, tol::kDualFeasibility, final_residuals.gap_as_verified,
        tol::kDualityGap);
  } else {
    // PDHG stopping short is the normal case, not an exception. Report the residuals it
    // actually reached rather than implying the point is optimal.
    solution.status = timer.elapsed_seconds() > time_limit ? SolveStatus::kTimeLimit
                                                           : SolveStatus::kIterationLimit;
    solution.message = fmt::format(
        "stopped at relative primal {:.3e}, dual {:.3e}, gap {:.3e} after {} iterations "
        "and {} restarts (target {:.1e})",
        final_residuals.primal, final_residuals.dual, final_residuals.gap, iteration, restarts,
        tolerance);
  }

  // Only a verifiable point carries a dual bound. Anything else leaves it unknown, which is
  // the infinity on the unexplored side of the objective.
  if (verifiable) {
    solution.dual_bound = sense * final_residuals.dual_objective + model.objective_offset;
  } else {
    solution.dual_bound = model.sense == ObjSense::kMaximize ? kInfinity : -kInfinity;
  }
  solution.recompute_quality(model);

  logger.info("");
  logger.info("Status: {}   objective {:.10e}   iterations {}   restarts {}   time {:.3f}s",
              to_string(solution.status), solution.objective, solution.iterations, restarts,
              solution.solve_seconds);
  logger.info("Relative residuals: primal {:.3e}, dual {:.3e}, gap {:.3e}",
              final_residuals.primal, final_residuals.dual, final_residuals.gap);
  if (!solution.message.empty()) logger.info("{}", solution.message);
  return solution;
}

}  // namespace sankhya::pdhg
