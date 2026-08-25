// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact simplex over the rationals. TESTS ONLY.
//
// Reference: Chvatal, "Linear Programming" (1983), ch. 2-3 - the tableau method and Bland's
// anti-cycling rule. Written from the textbook formulation, deliberately unoptimised.

#include "oracles/rational_simplex.hpp"

#include <algorithm>
#include <limits>
#include <sstream>

namespace sankhya::oracle {
namespace {

/// The standard form the tableau consumes:  min c'x  s.t.  Ax = b,  x >= 0,  b >= 0.
struct StandardForm {
  Index rows = 0;
  Index cols = 0;
  std::vector<std::vector<Rational>> a;
  std::vector<Rational> b;
  std::vector<Rational> c;
  /// Position of each original column within the standard-form variable vector.
  std::vector<Index> original_column;
};

/// Convert  min c'x  s.t.  Ax >= b, 0 <= x <= u  into standard form.
///
/// Two additions, each obvious on inspection:
///   * every row gets a surplus:   sum_j a_ij x_j - s_i = b_i,  s_i >= 0
///   * every finite upper bound gets a slack:  x_j + t_j = u_j,  t_j >= 0
/// Rows with a negative right-hand side are negated so that b >= 0, which is what lets the
/// phase-1 basis be the artificials.
StandardForm to_standard_form(const GeneratedLp& lp) {
  std::vector<Index> bounded;
  for (Index j = 0; j < lp.num_cols; ++j) {
    if (lp.upper[static_cast<std::size_t>(j)] != kNoUpperBound) bounded.push_back(j);
  }

  StandardForm sf;
  sf.rows = lp.num_rows + static_cast<Index>(bounded.size());
  sf.cols = lp.num_cols + lp.num_rows + static_cast<Index>(bounded.size());
  sf.a.assign(static_cast<std::size_t>(sf.rows),
              std::vector<Rational>(static_cast<std::size_t>(sf.cols), Rational(0)));
  sf.b.assign(static_cast<std::size_t>(sf.rows), Rational(0));
  sf.c.assign(static_cast<std::size_t>(sf.cols), Rational(0));

  sf.original_column.resize(static_cast<std::size_t>(lp.num_cols));
  for (Index j = 0; j < lp.num_cols; ++j) {
    sf.original_column[static_cast<std::size_t>(j)] = j;
    sf.c[static_cast<std::size_t>(j)] = Rational(lp.c[static_cast<std::size_t>(j)]);
  }

  for (Index i = 0; i < lp.num_rows; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    for (Index j = 0; j < lp.num_cols; ++j) {
      sf.a[ui][static_cast<std::size_t>(j)] = Rational(lp.a[ui][static_cast<std::size_t>(j)]);
    }
    // Surplus variable for row i.
    sf.a[ui][static_cast<std::size_t>(lp.num_cols + i)] = Rational(-1);
    sf.b[ui] = Rational(lp.b[ui]);
  }

  for (std::size_t k = 0; k < bounded.size(); ++k) {
    const Index j = bounded[k];
    const auto row = static_cast<std::size_t>(lp.num_rows) + k;
    sf.a[row][static_cast<std::size_t>(j)] = Rational(1);
    sf.a[row][static_cast<std::size_t>(lp.num_cols + lp.num_rows) + k] = Rational(1);
    sf.b[row] = Rational(lp.upper[static_cast<std::size_t>(j)]);
  }

  // Normalise to b >= 0.
  for (std::size_t i = 0; i < static_cast<std::size_t>(sf.rows); ++i) {
    if (sf.b[i].is_negative()) {
      sf.b[i] = -sf.b[i];
      for (Rational& value : sf.a[i]) value = -value;
    }
  }
  return sf;
}

/// Dense tableau kept in canonical form: the basis columns form an identity.
class Tableau {
 public:
  Tableau(const StandardForm& sf, Index artificial_count)
      : rows_(sf.rows), cols_(sf.cols + artificial_count) {
    t_.assign(static_cast<std::size_t>(rows_),
              std::vector<Rational>(static_cast<std::size_t>(cols_) + 1, Rational(0)));
    for (Index i = 0; i < rows_; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      for (Index j = 0; j < sf.cols; ++j) {
        t_[ui][static_cast<std::size_t>(j)] = sf.a[ui][static_cast<std::size_t>(j)];
      }
      if (artificial_count > 0) {
        t_[ui][static_cast<std::size_t>(sf.cols + i)] = Rational(1);
      }
      t_[ui][static_cast<std::size_t>(cols_)] = sf.b[ui];
    }
  }

  [[nodiscard]] Index rows() const noexcept { return rows_; }
  [[nodiscard]] Index cols() const noexcept { return cols_; }

  Rational& at(Index i, Index j) {
    return t_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  }
  [[nodiscard]] const Rational& at(Index i, Index j) const {
    return t_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
  }
  Rational& rhs(Index i) {
    return t_[static_cast<std::size_t>(i)][static_cast<std::size_t>(cols_)];
  }
  [[nodiscard]] const Rational& rhs(Index i) const {
    return t_[static_cast<std::size_t>(i)][static_cast<std::size_t>(cols_)];
  }

  /// Gauss-Jordan on (row, column): scale the pivot row to 1 and clear the column elsewhere.
  void pivot(Index row, Index column) {
    const Rational inverse = Rational(1) / at(row, column);
    for (Index j = 0; j <= cols_; ++j) at(row, j) *= inverse;
    at(row, column) = Rational(1);  // exact by construction; assign to avoid a stray 1/1

    for (Index i = 0; i < rows_; ++i) {
      if (i == row) continue;
      const Rational factor = at(i, column);
      if (factor.is_zero()) continue;
      for (Index j = 0; j <= cols_; ++j) {
        at(i, j) -= factor * at(row, j);
      }
      at(i, column) = Rational(0);
    }
  }

 private:
  Index rows_;
  Index cols_;
  std::vector<std::vector<Rational>> t_;
};

/// One simplex phase under Bland's rule. Returns false when the objective is unbounded.
/// `price_limit` is the number of leading columns pricing may consider. Phase 1 prices
/// everything; phase 2 passes the count of REAL columns so the artificials are excluded.
/// Excluding them is not an optimisation - an artificial can price attractively in phase 2
/// and re-enter, which reintroduces the infeasibility phase 1 just removed.
bool run_phase(Tableau& tableau, const std::vector<Rational>& cost, std::vector<Index>& basis,
               Index price_limit, std::int64_t* iterations, std::int64_t iteration_cap) {
  const Index rows = tableau.rows();
  const Index cols = price_limit;

  while (true) {
    if (*iterations >= iteration_cap) return true;  // caller reports the limit

    // Reduced costs, recomputed from scratch. Exact arithmetic means there is no drift to
    // accumulate, so the cheap incremental update buys nothing and could hide a mistake.
    Index entering = -1;
    for (Index j = 0; j < cols; ++j) {
      Rational reduced = cost[static_cast<std::size_t>(j)];
      for (Index i = 0; i < rows; ++i) {
        const Rational& basic_cost =
            cost[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])];
        if (basic_cost.is_zero()) continue;
        reduced -= basic_cost * tableau.at(i, j);
      }
      if (reduced.is_negative()) {
        entering = j;  // Bland: the FIRST improving column, which is what forbids cycling
        break;
      }
    }
    if (entering < 0) return true;  // optimal for this phase

    // Ratio test, exact. Ties break on the smallest basis index, again per Bland.
    Index leaving = -1;
    Rational best_ratio;
    for (Index i = 0; i < rows; ++i) {
      const Rational& alpha = tableau.at(i, entering);
      if (!alpha.is_positive()) continue;
      const Rational ratio = tableau.rhs(i) / alpha;
      if (leaving < 0 || ratio < best_ratio ||
          (ratio == best_ratio &&
           basis[static_cast<std::size_t>(i)] < basis[static_cast<std::size_t>(leaving)])) {
        best_ratio = ratio;
        leaving = i;
      }
    }
    if (leaving < 0) return false;  // unbounded

    tableau.pivot(leaving, entering);
    basis[static_cast<std::size_t>(leaving)] = entering;
    ++(*iterations);
  }
}

}  // namespace

const char* to_string(OracleStatus status) noexcept {
  switch (status) {
    case OracleStatus::kOptimal: return "optimal";
    case OracleStatus::kInfeasible: return "infeasible";
    case OracleStatus::kUnbounded: return "unbounded";
    case OracleStatus::kOverflow: return "overflow";
    case OracleStatus::kIterationLimit: return "iteration_limit";
  }
  return "unknown";
}

std::string GeneratedLp::to_text() const {
  std::ostringstream out;
  out << "minimize";
  for (Index j = 0; j < num_cols; ++j) {
    out << "  " << c[static_cast<std::size_t>(j)] << "*x" << j;
  }
  out << "\nsubject to\n";
  for (Index i = 0; i < num_rows; ++i) {
    for (Index j = 0; j < num_cols; ++j) {
      out << "  " << a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] << "*x" << j;
    }
    out << "  >=  " << b[static_cast<std::size_t>(i)] << "\n";
  }
  out << "bounds\n";
  for (Index j = 0; j < num_cols; ++j) {
    out << "  0 <= x" << j << " <= ";
    if (upper[static_cast<std::size_t>(j)] == kNoUpperBound) {
      out << "inf\n";
    } else {
      out << upper[static_cast<std::size_t>(j)] << "\n";
    }
  }
  return out.str();
}

Model to_model(const GeneratedLp& lp) {
  Model model;
  model.name = "generated";
  model.sense = ObjSense::kMinimize;
  model.col_cost.resize(static_cast<std::size_t>(lp.num_cols));
  model.col_lower.assign(static_cast<std::size_t>(lp.num_cols), 0.0);
  model.col_upper.assign(static_cast<std::size_t>(lp.num_cols), kInfinity);
  model.col_type.assign(static_cast<std::size_t>(lp.num_cols), VarType::kContinuous);

  for (Index j = 0; j < lp.num_cols; ++j) {
    const auto u = static_cast<std::size_t>(j);
    model.col_cost[u] = static_cast<double>(lp.c[u]);
    if (lp.upper[u] != kNoUpperBound) model.col_upper[u] = static_cast<double>(lp.upper[u]);
  }

  model.row_lower.resize(static_cast<std::size_t>(lp.num_rows));
  model.row_upper.assign(static_cast<std::size_t>(lp.num_rows), kInfinity);
  model.matrix.reset(lp.num_rows, lp.num_cols);
  for (Index i = 0; i < lp.num_rows; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    model.row_lower[ui] = static_cast<double>(lp.b[ui]);
    for (Index j = 0; j < lp.num_cols; ++j) {
      const std::int64_t value = lp.a[ui][static_cast<std::size_t>(j)];
      if (value != 0) model.matrix.add_entry(i, j, static_cast<double>(value));
    }
  }
  model.matrix.finalize();
  return model;
}

OracleResult solve_exact(const GeneratedLp& lp) {
  OracleResult result;
  // A generous cap. Bland's rule guarantees termination, so hitting this means the instance
  // is simply larger than the oracle is meant for - reported, never mistaken for optimality.
  const std::int64_t iteration_cap = 200000;

  try {
    const StandardForm sf = to_standard_form(lp);
    const Index artificials = sf.rows;
    Tableau tableau(sf, artificials);

    std::vector<Index> basis(static_cast<std::size_t>(sf.rows));
    for (Index i = 0; i < sf.rows; ++i) {
      basis[static_cast<std::size_t>(i)] = sf.cols + i;
    }

    // ---- Phase 1: minimise the sum of the artificials ----------------------------------
    std::vector<Rational> phase_one_cost(static_cast<std::size_t>(tableau.cols()), Rational(0));
    for (Index i = 0; i < artificials; ++i) {
      phase_one_cost[static_cast<std::size_t>(sf.cols + i)] = Rational(1);
    }
    if (!run_phase(tableau, phase_one_cost, basis, tableau.cols(), &result.iterations,
                   iteration_cap)) {
      // The phase-1 objective is bounded below by zero, so this cannot happen.
      result.status = OracleStatus::kOverflow;
      return result;
    }
    if (result.iterations >= iteration_cap) {
      result.status = OracleStatus::kIterationLimit;
      return result;
    }

    Rational infeasibility(0);
    for (Index i = 0; i < sf.rows; ++i) {
      if (basis[static_cast<std::size_t>(i)] >= sf.cols) infeasibility += tableau.rhs(i);
    }
    if (!infeasibility.is_zero()) {
      result.status = OracleStatus::kInfeasible;
      return result;
    }

    // ---- Drive the artificials out of the basis -----------------------------------------
    // A remaining artificial sits at zero. Pivot it out on any nonzero structural entry; if
    // the whole structural part of its row is zero the row is redundant and is left alone,
    // since the artificial is then pinned at zero and cannot re-enter with a negative cost.
    for (Index i = 0; i < sf.rows; ++i) {
      if (basis[static_cast<std::size_t>(i)] < sf.cols) continue;
      for (Index j = 0; j < sf.cols; ++j) {
        if (!tableau.at(i, j).is_zero()) {
          tableau.pivot(i, j);
          basis[static_cast<std::size_t>(i)] = j;
          break;
        }
      }
    }

    // ---- Phase 2 -------------------------------------------------------------------------
    std::vector<Rational> cost(static_cast<std::size_t>(tableau.cols()), Rational(0));
    for (Index j = 0; j < sf.cols; ++j) {
      cost[static_cast<std::size_t>(j)] = sf.c[static_cast<std::size_t>(j)];
    }
    // Phase 2 prices only the first sf.cols columns, so no artificial can enter. Any
    // artificial still basic sits on a redundant row whose structural part is entirely
    // zero; such a row is unaffected by pivots on structural columns, so it stays at zero.
    if (!run_phase(tableau, cost, basis, sf.cols, &result.iterations, iteration_cap)) {
      result.status = OracleStatus::kUnbounded;
      return result;
    }
    if (result.iterations >= iteration_cap) {
      result.status = OracleStatus::kIterationLimit;
      return result;
    }

    // ---- Read off the answer -------------------------------------------------------------
    std::vector<Rational> full(static_cast<std::size_t>(tableau.cols()), Rational(0));
    for (Index i = 0; i < sf.rows; ++i) {
      full[static_cast<std::size_t>(basis[static_cast<std::size_t>(i)])] = tableau.rhs(i);
    }
    result.x.assign(static_cast<std::size_t>(lp.num_cols), Rational(0));
    Rational objective(0);
    for (Index j = 0; j < lp.num_cols; ++j) {
      const Rational value = full[static_cast<std::size_t>(j)];
      result.x[static_cast<std::size_t>(j)] = value;
      objective += Rational(lp.c[static_cast<std::size_t>(j)]) * value;
    }
    result.objective = objective;
    result.status = OracleStatus::kOptimal;
    return result;
  } catch (const RationalOverflow&) {
    result.status = OracleStatus::kOverflow;
    return result;
  }
}

}  // namespace sankhya::oracle
