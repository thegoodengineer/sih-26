// SPDX-License-Identifier: Apache-2.0
// SANKHYA - a convex NLP engine (#226). The method and its guarantees are on the declaration.

#include "nlp/convex_nlp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "sankhya/timer.hpp"

namespace sankhya::nlp {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double project(double v, double lo, double hi) {
  if (is_finite_bound(lo) && v < lo) return lo;
  if (is_finite_bound(hi) && v > hi) return hi;
  return v;
}

/// The objective in MINIMISE space, f(x) = sense * (base objective + expression), with the
/// exact gradient. The base part is c'x + 0.5 x'Qx + offset, Q held as a lower triangle.
class Objective {
 public:
  explicit Objective(const NonlinearModel& model)
      : model_(model), sense_(model.base.sense_multiplier()) {}

  /// False when x is outside the expression's domain (or the value overflowed).
  bool value(const std::vector<double>& x, double* out, std::string* why) const {
    double v = model_.base.evaluate_objective(x.data());
    if (model_.objective != kNoExpr) {
      const Evaluation e = model_.graph.evaluate(model_.objective, x);
      if (!e.ok()) {
        if (why != nullptr) *why = e.message;
        return false;
      }
      v += e.value;
    }
    if (!std::isfinite(v)) return false;
    *out = sense_ * v;
    return true;
  }

  bool gradient(const std::vector<double>& x, std::vector<double>* g) const {
    const auto n = x.size();
    g->assign(n, 0.0);
    for (std::size_t j = 0; j < n; ++j) (*g)[j] = model_.base.col_cost[j];
    const SparseMatrix& q = model_.base.hessian;
    for (Index j = 0; j < q.num_cols(); ++j) {
      const ColumnView col = q.column(j);
      for (Index k = 0; k < col.size; ++k) {
        const auto i = static_cast<std::size_t>(col.rows[k]);
        const auto u = static_cast<std::size_t>(j);
        (*g)[i] += col.values[k] * x[u];
        if (i != u) (*g)[u] += col.values[k] * x[i];
      }
    }
    if (model_.objective != kNoExpr) {
      SparseEntries grad;
      Evaluation error;
      if (!model_.graph.gradient(model_.objective, x, &grad, &error)) return false;
      for (const auto& [j, v] : grad) (*g)[static_cast<std::size_t>(j)] += v;
    }
    for (double& v : *g) {
      v *= sense_;
      if (!std::isfinite(v)) return false;
    }
    return true;
  }

 private:
  const NonlinearModel& model_;
  double sense_;
};

/// ||A||_2 by power iteration on A'A from a seeded start - an ESTIMATE, which is why the step
/// rule below carries a margin.
double spectral_norm(const SparseMatrix& a, unsigned seed) {
  if (a.num_rows() == 0 || a.num_nonzeros() == 0) return 0.0;
  std::mt19937 rng(seed + 226u);
  std::uniform_real_distribution<double> unit(-1.0, 1.0);
  std::vector<double> v(static_cast<std::size_t>(a.num_cols()));
  for (double& x : v) x = unit(rng);
  std::vector<double> av(static_cast<std::size_t>(a.num_rows()));
  double estimate = 0.0;
  for (int it = 0; it < 30; ++it) {
    double norm = 0.0;
    for (const double x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm <= 0.0) return 0.0;
    for (double& x : v) x /= norm;
    std::fill(av.begin(), av.end(), 0.0);
    a.multiply_add(v.data(), av.data());
    std::fill(v.begin(), v.end(), 0.0);
    a.transpose_multiply_add(av.data(), v.data());
    double next = 0.0;
    for (const double x : v) next += x * x;
    estimate = std::sqrt(std::sqrt(next));
  }
  return estimate;
}

/// Where the iteration starts: inside every finite box, a unit in from a one-sided bound.
std::vector<double> starting_point(const Model& m) {
  std::vector<double> x(static_cast<std::size_t>(m.num_cols()), 0.0);
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double lo = m.col_lower[j];
    const double hi = m.col_upper[j];
    if (is_finite_bound(lo) && is_finite_bound(hi)) {
      x[j] = 0.5 * (lo + hi);
    } else if (is_finite_bound(lo)) {
      x[j] = lo + 1.0;
    } else if (is_finite_bound(hi)) {
      x[j] = hi - 1.0;
    }
  }
  return x;
}

struct Kkt {
  double primal = 0.0;        ///< worst bound or row violation
  double stationarity = 0.0;  ///< || x - proj(x - (grad + A'y)) ||_inf
  double complementarity = 0.0;
  double scale = 1.0;        ///< 1 + ||grad||_inf, for the relative stationarity test
  double bound_scale = 1.0;  ///< 1 + the largest finite |bound|, for primal feasibility
  double dual_scale = 1.0;   ///< 1 + ||y||_inf
};

Kkt measure(const Model& m, const Objective& f, const std::vector<double>& x,
            const std::vector<double>& y, bool* defined) {
  Kkt k;
  const auto n = x.size();
  const auto rows = static_cast<std::size_t>(m.num_rows());
  std::vector<double> g;
  *defined = f.gradient(x, &g);
  if (!*defined) return k;
  std::vector<double> ax(rows, 0.0);
  m.matrix.multiply_add(x.data(), ax.data());
  std::vector<double> aty(n, 0.0);
  m.matrix.transpose_multiply_add(y.data(), aty.data());
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(m.col_lower[j])) k.primal = std::max(k.primal, m.col_lower[j] - x[j]);
    if (is_finite_bound(m.col_upper[j])) k.primal = std::max(k.primal, x[j] - m.col_upper[j]);
    const double step = x[j] - project(x[j] - (g[j] + aty[j]), m.col_lower[j], m.col_upper[j]);
    k.stationarity = std::max(k.stationarity, std::fabs(step));
    k.scale = std::max(k.scale, 1.0 + std::fabs(g[j]));
  }
  for (std::size_t j = 0; j < n; ++j) {
    if (is_finite_bound(m.col_lower[j]))
      k.bound_scale = std::max(k.bound_scale, 1.0 + std::fabs(m.col_lower[j]));
    if (is_finite_bound(m.col_upper[j]))
      k.bound_scale = std::max(k.bound_scale, 1.0 + std::fabs(m.col_upper[j]));
  }
  for (std::size_t i = 0; i < rows; ++i) {
    const double lo = m.row_lower[i];
    const double hi = m.row_upper[i];
    if (is_finite_bound(lo)) k.bound_scale = std::max(k.bound_scale, 1.0 + std::fabs(lo));
    if (is_finite_bound(hi)) k.bound_scale = std::max(k.bound_scale, 1.0 + std::fabs(hi));
    k.dual_scale = std::max(k.dual_scale, 1.0 + std::fabs(y[i]));
    if (is_finite_bound(lo)) k.primal = std::max(k.primal, lo - ax[i]);
    if (is_finite_bound(hi)) k.primal = std::max(k.primal, ax[i] - hi);
    // A positive multiplier prices the upper side, a negative one the lower: a multiplier on
    // a side the row is away from is a complementarity violation, measured as |y| times the
    // distance.
    if (y[i] > 0.0) {
      k.complementarity =
          std::max(k.complementarity, y[i] * (is_finite_bound(hi) ? hi - ax[i] : kInf));
    } else if (y[i] < 0.0) {
      k.complementarity =
          std::max(k.complementarity, -y[i] * (is_finite_bound(lo) ? ax[i] - lo : kInf));
    }
  }
  return k;
}

Solution refuse(Solution solution, std::string why, Logger& logger) {
  solution.status = SolveStatus::kNotSolved;
  solution.algorithm = "nlp-condat-vu";
  solution.message = std::move(why);
  logger.warning("{}", solution.message);
  return solution;
}

}  // namespace

Solution solve_convex_nlp(const NonlinearModel& model, const Options& options) {
  Logger logger(options.get_bool("log_to_console") ? stdout : nullptr);
  return solve_convex_nlp(model, options, logger);
}

Solution solve_convex_nlp(const NonlinearModel& model, const Options& options, Logger& logger) {
  const Timer timer;
  const Model& m = model.base;
  Solution solution;
  solution.allocate_for(m);

  // ---- What this engine does not take, refused with the reason ----------------------------
  if (const std::string problem = model.validate(); !problem.empty()) {
    solution.status = SolveStatus::kModelError;
    solution.message = problem;
    return solution;
  }
  if (m.has_integrality()) {
    return refuse(std::move(solution),
                  "the model has integer columns; this engine solves the continuous convex NLP "
                  "only, and a MINLP needs a branch and bound over it that does not exist yet",
                  logger);
  }
  if (!model.constraints.empty()) {
    return refuse(
        std::move(solution),
        fmt::format("the model has {} nonlinear constraints; this engine takes linear "
                    "constraints only (#226's scope)",
                    model.constraints.size()),
        logger);
  }
  const ConvexityReport convex = model.convexity();
  const bool asserted = options.get_bool("nlp_assume_convex");
  if (!convex.convex && !asserted) {
    return refuse(std::move(solution),
                  fmt::format("convexity could not be proved ({}); a local method on a "
                              "non-convex problem returns a local point, so it is refused. "
                              "nlp_assume_convex=true runs it on the caller's word",
                              convex.reasons.empty() ? std::string("no reason given")
                                                     : convex.reasons.front()),
                  logger);
  }

  const Objective f(model);
  const auto n = static_cast<std::size_t>(m.num_cols());
  const auto rows = static_cast<std::size_t>(m.num_rows());
  std::vector<double> x = starting_point(m);
  double fx = 0.0;
  std::string why;
  if (!f.value(x, &fx, &why)) {
    return refuse(std::move(solution),
                  fmt::format("the objective is not defined at the starting point ({}); give "
                              "the columns bounds that keep it inside its domain",
                              why),
                  logger);
  }

  // ---- Condat-Vu with a backtracked, only-increasing Lipschitz constant -------------------
  const double norm_a =
      spectral_norm(m.matrix, static_cast<unsigned>(options.get_int("random_seed")));
  const double a2 = 1.1025 * std::max(norm_a * norm_a, 1e-12);  // (1.05 ||A||)^2, a margin
  double lipschitz = 1.0;
  double tau = 0.0;
  double sigma = 0.0;
  const auto set_steps = [&] {
    tau = 1.0 / (lipschitz / 2.0 + std::sqrt(a2) + 1e-12);
    sigma = rows > 0 ? (1.0 / tau - lipschitz / 2.0) / (2.0 * a2) : 0.0;
  };
  set_steps();

  const double tolerance = options.get_double("nlp_tolerance");
  const std::int64_t limit = options.get_int("iteration_limit");
  const Count max_iterations = limit >= 0 ? limit : 200000;
  const double time_limit = options.get_double("time_limit");

  std::vector<double> y(rows, 0.0);
  std::vector<double> g;
  std::vector<double> aty(n);
  std::vector<double> x_next(n);
  std::vector<double> ax(rows);
  Count iterations = 0;
  Count backtracks = 0;
  std::string stop_reason;
  bool converged = false;
  while (iterations < max_iterations) {
    if (timer.elapsed_seconds() > time_limit) {
      stop_reason = "time_limit";
      break;
    }
    ++iterations;
    if (!f.gradient(x, &g)) {
      solution.status = SolveStatus::kNumericalError;
      solution.message = "the gradient is not defined at an accepted iterate";
      break;
    }
    std::fill(aty.begin(), aty.end(), 0.0);
    if (rows > 0) m.matrix.transpose_multiply_add(y.data(), aty.data());

    // Backtrack on L until the descent lemma holds and x+ is inside f's domain.
    double f_next = 0.0;
    for (int attempt = 0;; ++attempt) {
      for (std::size_t j = 0; j < n; ++j) {
        x_next[j] = project(x[j] - tau * (g[j] + aty[j]), m.col_lower[j], m.col_upper[j]);
      }
      double model_value = fx;
      double step_sq = 0.0;
      for (std::size_t j = 0; j < n; ++j) {
        const double d = x_next[j] - x[j];
        model_value += g[j] * d;
        step_sq += d * d;
      }
      model_value += 0.5 * lipschitz * step_sq;
      if (f.value(x_next, &f_next, nullptr) &&
          f_next <= model_value + 1e-12 * (1.0 + std::fabs(fx))) {
        break;
      }
      if (attempt >= 60) {
        solution.status = SolveStatus::kNumericalError;
        solution.message = "the step could not be made to satisfy the descent lemma";
        break;
      }
      lipschitz *= 2.0;
      ++backtracks;
      set_steps();
    }
    if (solution.status == SolveStatus::kNumericalError) break;

    // Dual step on the extrapolated point.
    double dual_move = 0.0;
    if (rows > 0) {
      std::vector<double> extrapolated(n);
      for (std::size_t j = 0; j < n; ++j) extrapolated[j] = 2.0 * x_next[j] - x[j];
      std::fill(ax.begin(), ax.end(), 0.0);
      m.matrix.multiply_add(extrapolated.data(), ax.data());
      for (std::size_t i = 0; i < rows; ++i) {
        const double v = y[i] + sigma * ax[i];
        const double next = v - sigma * project(v / sigma, m.row_lower[i], m.row_upper[i]);
        dual_move = std::max(dual_move, std::fabs(next - y[i]));
        y[i] = next;
      }
    }
    double primal_move = 0.0;
    for (std::size_t j = 0; j < n; ++j)
      primal_move = std::max(primal_move, std::fabs(x_next[j] - x[j]));
    x.swap(x_next);
    fx = f_next;

    // The iteration's own stopping test: the fixed-point residual, scaled by the steps. The
    // STATUS is decided by the independent KKT check below, not by this.
    if (primal_move / tau <= tolerance && (rows == 0 || dual_move / sigma <= tolerance)) {
      converged = true;
      break;
    }
  }

  // ---- The status is decided here, by measurement ----------------------------------------
  bool defined = false;
  const Kkt kkt = measure(m, f, x, y, &defined);
  solution.col_value = x;
  solution.row_dual = y;
  solution.iterations = iterations;
  solution.algorithm = "nlp-condat-vu";
  double reported = 0.0;
  (void)f.value(x, &reported, nullptr);
  solution.objective = m.sense_multiplier() * reported;
  std::vector<double> activity(rows, 0.0);
  m.matrix.multiply_add(x.data(), activity.data());
  solution.row_activity = activity;
  solution.primal_infeasibility = kkt.primal;
  solution.dual_infeasibility = kkt.stationarity;
  solution.complementarity_violation = kkt.complementarity;
  solution.solve_seconds = timer.elapsed_seconds();

  if (solution.status == SolveStatus::kNumericalError) return solution;
  // ONE tolerance, three measures, each relative to its own scale: primal feasibility to
  // the bounds', stationarity to the gradient's, complementarity to the product of the
  // multipliers' and the bounds'.
  const bool primal_ok = kkt.primal <= tolerance * kkt.bound_scale;
  const bool stationary = defined && kkt.stationarity <= tolerance * kkt.scale;
  const bool complementary =
      kkt.complementarity <= tolerance * kkt.dual_scale * kkt.bound_scale;
  const std::string basis =
      convex.convex
          ? std::string("convexity proved by the composition rules")
          : std::string("convexity ASSERTED by the caller (nlp_assume_convex), not proved");
  if (primal_ok && stationary && complementary) {
    solution.status = SolveStatus::kOptimal;
  } else if (primal_ok) {
    solution.status = SolveStatus::kFeasible;
  } else if (!stop_reason.empty()) {
    solution.status = SolveStatus::kTimeLimit;
  } else {
    solution.status = SolveStatus::kIterationLimit;
  }
  solution.message = fmt::format(
      "{} after {} iterations ({} step-size halvings, L = {:.3g}); KKT: primal {:.2e}, "
      "stationarity {:.2e} (scale {:.3g}), complementarity {:.2e}; {}{}",
      converged ? "converged" : "stopped", iterations, backtracks, lipschitz, kkt.primal,
      kkt.stationarity, kkt.scale, kkt.complementarity, basis,
      solution.status == SolveStatus::kOptimal ? "" : "; not certified optimal");
  logger.info("Convex NLP: {} {}", to_string(solution.status), solution.message);
  return solution;
}

}  // namespace sankhya::nlp
