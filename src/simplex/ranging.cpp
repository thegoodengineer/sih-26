// SPDX-License-Identifier: Apache-2.0
// SANKHYA - LP sensitivity ranging.
//
// Reference: Chvatal, "Linear Programming", ch. 10 (W. H. Freeman, 1983).
//
// Cost ranging (column j):
//   Nonbasic at lower (d_j >= 0): basis stays optimal if c_j decreases by at most d_j or
//     increases without limit.  delta_lo = d_j, delta_hi = +inf.
//   Nonbasic at upper (d_j <= 0): delta_lo = +inf, delta_hi = |d_j|.
//   Basic at position p: BTRAN(e_p) gives w; for each nonbasic k, alpha_k = w^T a_k;
//     the ratio test over {d_k / alpha_k} yields delta_lo and delta_hi.
//
// RHS ranging (row i):
//   FTRAN(e_i) gives v = B^{-1} e_i: the direction basic variables move when the active
//   bound of row i increases by one unit.  The standard ratio test on those variables'
//   bounds yields delta_lo and delta_hi.

#include "ranging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"
#include "util/profiler.hpp"

#include "../la/lu.hpp"

namespace sankhya::detail {

namespace {

using Sz = std::size_t;

// Ratio-test helper: update (lo, hi) given that the basic variable changes by v * delta
// and must remain in [lb, ub].
void ratio(double v, double current, double lb, double ub, double& lo, double& hi) {
  if (std::abs(v) < tol::kPivotTolerance) return;
  if (v > 0.0) {
    if (ub < kInfinity) hi = std::min(hi, (ub - current) / v);
    if (lb > -kInfinity) lo = std::min(lo, (current - lb) / v);
  } else {
    if (lb > -kInfinity) hi = std::min(hi, (current - lb) / (-v));
    if (ub < kInfinity) lo = std::min(lo, (ub - current) / (-v));
  }
}

}  // namespace

void compute_ranging(const Model& model, const Options& options, Logger& logger,
                     Solution& solution) {
  if (!options.get_bool("ranging")) return;
  if (solution.status != SolveStatus::kOptimal) {
    logger.warning(
        "ranging: skipped, the status is {} and ranges are defined at an optimal basis only",
        to_string(solution.status));
    return;
  }
  if (solution.col_status.empty() || solution.row_status.empty()) {
    logger.warning(
        "ranging: skipped, the engine produced no basis (the interior point and PDHG do "
        "not; use the simplex)");
    return;
  }
  ProfileScope timed(logger.profiler(), "ranging");  // #285, after the reasons not to run

  const Index n = model.num_cols();
  const Index m = model.num_rows();
  const Sz sn = static_cast<Sz>(n);
  const Sz sm = static_cast<Sz>(m);

  // --- 1. Reconstruct the basis layout ---
  // basis_col[p]: index of the column occupying basis position p.
  //   0..n-1  = structural column j
  //   n..n+m-1 = logical column for row (j-n) (single entry -1.0 at that row)
  std::vector<Index> basis_col(sm, -1);
  // pos_of[col]: basis position of structural or logical column, -1 if nonbasic.
  std::vector<Index> pos_of(sn + sm, -1);

  {
    Index p = 0;
    for (Index j = 0; j < n; ++j) {
      if (solution.col_status[static_cast<Sz>(j)] == BasisStatus::kBasic) {
        if (p < m) {
          basis_col[static_cast<Sz>(p)] = j;
          pos_of[static_cast<Sz>(j)] = p;
        }
        ++p;
      }
    }
    for (Index i = 0; i < m; ++i) {
      if (solution.row_status[static_cast<Sz>(i)] == BasisStatus::kBasic) {
        if (p < m) {
          basis_col[static_cast<Sz>(p)] = n + i;
          pos_of[static_cast<Sz>(n + i)] = p;
        }
        ++p;
      }
    }
    if (p != m) {
      logger.warning("ranging: skipped, the reported basis has {} basic variables for {} rows",
                     p, m);
      return;
    }
  }

  // --- 2. Build LU columns and factorize the basis ---
  // Logical columns need stable storage for their single (row, value) pair.
  std::vector<Index> log_row(sm);
  std::vector<double> log_val(sm, -1.0);
  std::vector<LuColumn> lu_cols(sm);
  for (Index p = 0; p < m; ++p) {
    const Sz pp = static_cast<Sz>(p);
    const Index col = basis_col[pp];
    if (col < n) {
      const ColumnView cv = model.matrix.column(col);
      lu_cols[pp] = {cv.rows, cv.values, cv.size};
    } else {
      log_row[pp] = col - n;
      lu_cols[pp] = {&log_row[pp], &log_val[pp], 1};
    }
  }

  SparseLu lu;
  if (!lu.factorize(lu_cols, m, tol::kPivotTolerance, tol::kMarkowitzThreshold)) {
    logger.warning("ranging: skipped, the reported basis is singular to working precision");
    return;
  }

  // DEGENERACY, STATED. A basic variable on one of its bounds means the vertex has more
  // than one basis, and the ranges below belong to the one in hand; a planner reading
  // "the plan holds until the price moves by x" must know when x is a property of the
  // tie-break rather than of the plan (#220, item 4).
  Index degenerate = 0;
  for (Index p = 0; p < m; ++p) {
    const Index col = basis_col[static_cast<Sz>(p)];
    double lb, ub, cur;
    if (col < n) {
      const Sz cj = static_cast<Sz>(col);
      lb = model.col_lower[cj];
      ub = model.col_upper[cj];
      cur = solution.col_value[cj];
    } else {
      const Sz ri = static_cast<Sz>(col - n);
      lb = model.row_lower[ri];
      ub = model.row_upper[ri];
      cur = solution.row_activity[ri];
    }
    const double room = tol::kPrimalFeasibility * std::max(1.0, std::abs(cur));
    if ((lb > -kInfinity && cur - lb <= room) || (ub < kInfinity && ub - cur <= room)) {
      ++degenerate;
    }
  }
  solution.ranging_basis_degenerate = degenerate > 0;
  if (degenerate > 0) {
    logger.warning(
        "ranging: the optimal basis is degenerate ({} basic variable(s) on a bound); the "
        "ranges are those of this basis, not of the unique optimum",
        degenerate);
  }

  // --- 3. Allocate output ---
  solution.col_ranging_lower.assign(sn, kInfinity);
  solution.col_ranging_upper.assign(sn, kInfinity);
  solution.row_ranging_lower.assign(sm, kInfinity);
  solution.row_ranging_upper.assign(sm, kInfinity);

  // Reduced costs in minimization sign convention (solution.col_dual uses model sign).
  const double sense = model.sense_multiplier();
  std::vector<double> d(sn);
  for (Index j = 0; j < n; ++j)
    d[static_cast<Sz>(j)] = sense * solution.col_dual[static_cast<Sz>(j)];
  // Row dual in minimization sign (reduced cost of the logical column = y_i).
  std::vector<double> y(sm);
  for (Index i = 0; i < m; ++i)
    y[static_cast<Sz>(i)] = sense * solution.row_dual[static_cast<Sz>(i)];

  std::vector<double> work(sm);

  // --- 4. Cost ranging for each structural column ---
  for (Index j = 0; j < n; ++j) {
    const Sz jj = static_cast<Sz>(j);
    const BasisStatus st = solution.col_status[jj];
    const double dj = d[jj];

    if (st == BasisStatus::kFixed) {
      // A fixed column never enters whatever its cost: both sides are unbounded.
      continue;
    }
    if (st == BasisStatus::kAtLower) {
      // dj >= 0: decrease by at most dj, increase freely.
      solution.col_ranging_lower[jj] = dj;
      solution.col_ranging_upper[jj] = kInfinity;
      continue;
    }
    if (st == BasisStatus::kAtUpper) {
      // dj <= 0: increase by at most |dj|, decrease freely.
      solution.col_ranging_lower[jj] = kInfinity;
      solution.col_ranging_upper[jj] = -dj;
      continue;
    }
    if (st == BasisStatus::kNonbasicFree) {
      // dj = 0: no room to move in either direction.
      solution.col_ranging_lower[jj] = 0.0;
      solution.col_ranging_upper[jj] = 0.0;
      continue;
    }
    if (st != BasisStatus::kBasic) continue;

    // Basic: BTRAN(e_p) -> w, then ratio test over nonbasics.
    // The ratio test determines: for what range of perturbation delta to c_j does every
    // nonbasic reduced cost keep its sign?
    //   d_k(delta) = d_k - alpha_k * delta  where alpha_k = w^T a_k.
    // For k nonbasic at lower (d_k >= 0): d_k - alpha_k * delta >= 0
    //   alpha_k > 0 => delta <= d_k / alpha_k  (limits increase)
    //   alpha_k < 0 => delta >= d_k / alpha_k  (limits decrease)
    // For k nonbasic at upper (d_k <= 0): d_k - alpha_k * delta <= 0
    //   alpha_k > 0 => delta >= d_k / alpha_k  (limits decrease)
    //   alpha_k < 0 => delta <= d_k / alpha_k  (limits increase)

    const Index pos = pos_of[jj];
    std::fill(work.begin(), work.end(), 0.0);
    work[static_cast<Sz>(pos)] = 1.0;
    lu.solve_transpose(work.data());

    double lo = kInfinity;
    double hi = kInfinity;

    // Structural nonbasic columns.
    for (Index k = 0; k < n; ++k) {
      const Sz kk = static_cast<Sz>(k);
      const BasisStatus sk = solution.col_status[kk];
      // A fixed column cannot enter, so its reduced cost's sign never binds the range.
      if (sk == BasisStatus::kBasic || sk == BasisStatus::kFixed) continue;

      double alpha = 0.0;
      const ColumnView cv = model.matrix.column(k);
      for (Index q = 0; q < cv.size; ++q)
        alpha += work[static_cast<Sz>(cv.rows[q])] * cv.values[q];
      if (std::abs(alpha) < tol::kPivotTolerance) continue;

      const double dk = d[kk];
      if (sk == BasisStatus::kAtLower) {
        if (alpha > 0.0)
          hi = std::min(hi, dk / alpha);
        else
          lo = std::min(lo, dk / (-alpha));
      } else if (sk == BasisStatus::kAtUpper) {
        if (alpha > 0.0)
          lo = std::min(lo, (-dk) / alpha);
        else
          hi = std::min(hi, (-dk) / (-alpha));
      }
    }

    // Logical nonbasic columns (row slacks): column of [A|-I] is -e_i, so w^T(-e_i) = -w[i].
    for (Index i = 0; i < m; ++i) {
      const Sz ii = static_cast<Sz>(i);
      const BasisStatus si = solution.row_status[ii];
      // An equality row's logical is fixed and never enters; its dual's sign is free.
      if (si == BasisStatus::kBasic || si == BasisStatus::kFixed) continue;

      const double alpha = -work[ii];
      if (std::abs(alpha) < tol::kPivotTolerance) continue;

      const double dk = y[ii];
      if (si == BasisStatus::kAtLower) {
        if (alpha > 0.0)
          hi = std::min(hi, dk / alpha);
        else
          lo = std::min(lo, dk / (-alpha));
      } else if (si == BasisStatus::kAtUpper) {
        if (alpha > 0.0)
          lo = std::min(lo, (-dk) / alpha);
        else
          hi = std::min(hi, (-dk) / (-alpha));
      }
    }

    solution.col_ranging_lower[jj] = std::max(lo, 0.0);
    solution.col_ranging_upper[jj] = std::max(hi, 0.0);
  }

  // THE MODEL'S OWN SENSE. Everything above is a range on the minimization-space cost
  // c' = sense * c. For a maximize model c' = -c, so "c' may fall by x" is "c may rise by
  // x": the two sides change places. Measured before this swap on the demo blend
  // (maximize): Bonny Light at its cap was reported as "decrease inf, increase 0.34" and
  // re-solving showed the basis change on a DEcrease of 0.37 and survive an increase.
  if (model.sense == ObjSense::kMaximize) {
    std::swap(solution.col_ranging_lower, solution.col_ranging_upper);
  }

  // --- 5. RHS ranging for each row ---
  // FTRAN(e_i) -> v = B^{-1} e_i.  Basic variable at position p changes by v[p]*delta
  // when the active bound of row i increases by delta.  Ratio test on those variables.
  //
  // For a row whose logical is BASIC - a constraint that is not binding - there is no
  // active bound and B^{-1} e_i is minus the logical's own unit vector, so the ratio test
  // reduces to the slack on each side: row_ranging_lower is how far the UPPER bound can
  // fall and row_ranging_upper how far the LOWER bound can rise before the row binds and
  // the basis changes. Independent of the objective sense.
  for (Index i = 0; i < m; ++i) {
    const Sz ii = static_cast<Sz>(i);
    std::fill(work.begin(), work.end(), 0.0);
    work[ii] = 1.0;
    lu.solve(work.data());

    double lo = kInfinity;
    double hi = kInfinity;

    for (Index p = 0; p < m; ++p) {
      const Sz pp = static_cast<Sz>(p);
      const double v = work[pp];
      const Index col = basis_col[pp];
      double lb, ub, cur;
      if (col < n) {
        const Sz cj = static_cast<Sz>(col);
        lb = model.col_lower[cj];
        ub = model.col_upper[cj];
        cur = solution.col_value[cj];
      } else {
        const Sz ri = static_cast<Sz>(col - n);
        lb = model.row_lower[ri];
        ub = model.row_upper[ri];
        cur = solution.row_activity[ri];
      }
      ratio(v, cur, lb, ub, lo, hi);
    }

    solution.row_ranging_lower[ii] = std::max(lo, 0.0);
    solution.row_ranging_upper[ii] = std::max(hi, 0.0);
  }
}

}  // namespace sankhya::detail
