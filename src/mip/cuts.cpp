// SPDX-License-Identifier: Apache-2.0
// SANKHYA - cutting planes. See cuts.hpp for the correctness obligations this file carries.

#include "cuts.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

#include "la/lu.hpp"
#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {
namespace {

// =========================================================================================
// Shared numeric helper
// =========================================================================================

/// Is `value` an integer, to within the integrality tolerance?
///
/// The tolerance is deliberately the same one the solver uses to decide whether a variable is
/// integral. A coefficient this routine wrongly accepts as an integer makes the row's
/// activity non-integral in truth, and the rounding below then removes feasible points - the
/// exact failure cuts.hpp exists to prevent. Anything the solver would not call an integer,
/// this must not either.
[[nodiscard]] bool is_integral(double value) noexcept {
  return std::fabs(value - std::round(value)) <= tol::kIntegrality;
}

// =========================================================================================
// Exact 0/1 knapsack solver (used exclusively by the sequential lifting procedure)
// =========================================================================================

/// Solve the 0/1 knapsack problem EXACTLY using standard dynamic programming.
///
///   Maximise    sum(profit[k] * x[k])
///   subject to  sum(weight[k] * x[k]) <= capacity
///               x[k] in {0, 1}
///
/// All weights and profits are non-negative integers (passed as double but guaranteed integral
/// by the caller). Capacity is a non-negative integer (also guaranteed integral by the caller).
///
/// CORRECTNESS OBLIGATION: every Z_k this function returns feeds a lifting coefficient
///   alpha_k = (|C| - 1) - Z_k.
/// A value of Z_k that is too LARGE makes alpha_k too SMALL (weaker cut, valid but weak).
/// A value of Z_k that is too SMALL makes alpha_k too LARGE (invalid cut).
/// The DP is exact, so neither error can occur.
///
/// COMPLEXITY: O(n * capacity) time and O(capacity) space (1-D rolling array).
/// The caller checks the product against kKnapsackDpLimit before calling.
[[nodiscard]] double exact_01_knapsack(const std::vector<double>& weight,
                                       const std::vector<double>& profit,
                                       double capacity) noexcept {
  const std::size_t n = weight.size();
  const auto cap = static_cast<std::size_t>(std::llround(capacity));

  // dp[c] = best achievable profit using items considered so far with total weight <= c.
  // 1-D rolling array: traverse capacity right-to-left to enforce "each item at most once".
  std::vector<double> dp(cap + 1, 0.0);

  for (std::size_t k = 0; k < n; ++k) {
    const auto w = static_cast<std::size_t>(std::llround(weight[k]));
    const double p = profit[k];
    if (w == 0) continue;
    for (std::size_t c = cap; c >= w; --c) {
      const double candidate = dp[c - w] + p;
      if (candidate > dp[c]) dp[c] = candidate;
    }
  }

  return dp[cap];
}

/// Maximum DP table size we are willing to allocate per lifting sub-problem.
/// 10M doubles = 80 MB.
inline constexpr std::size_t kKnapsackDpLimit = 10'000'000;

// =========================================================================================
// Knapsack cover cut generator (internal types and helpers)
// =========================================================================================

/// A single variable participating in the candidate row.
struct KnapsackVar {
  Index col;  ///< original model column index
  double a;   ///< coefficient (positive, integral, checked binary)
};

/// Classify row `row` of `model` against the supported knapsack class.
///
/// Supported class:
///   - row_lower == -infinity  (pure <= constraint)
///   - row_upper is finite and positive
///   - every nonzero in the row has a_j > 0, is_integral(a_j)
///   - every nonzero variable is kInteger with bounds exactly [0, 1]
///   - is_integral(row_upper)
///
/// Returns variables sorted by coefficient DESCENDING, ties broken by column index ASCENDING.
/// This ordering drives the greedy cover construction and is the source of determinism.
/// Returns empty if anything in the row is unsupported.
[[nodiscard]] std::vector<KnapsackVar> classify_row(const Model& model, Index row) {
  const auto r = static_cast<std::size_t>(row);

  // Must be a pure <= (lower = -inf, upper finite positive).
  if (is_finite_bound(model.row_lower[r])) return {};
  if (!is_finite_bound(model.row_upper[r])) return {};
  const double b = model.row_upper[r];
  if (b <= 0.0) return {};
  if (!is_integral(b)) return {};

  std::vector<KnapsackVar> vars;
  const Index ncols = model.num_cols();

  for (Index j = 0; j < ncols; ++j) {
    const ColumnView col = model.matrix.column(j);
    for (Index k = 0; k < col.size; ++k) {
      if (col.rows[k] != row) continue;
      const double a = col.values[k];

      // Continuous variable: unsupported class.
      if (model.col_type[static_cast<std::size_t>(j)] != VarType::kInteger) return {};

      // Non-positive coefficient: unsupported class.
      if (a <= 0.0) return {};

      // Non-integral coefficient: DP would not be exact.
      if (!is_integral(a)) return {};

      // Variable bounds must be exactly [0, 1] (binary).
      const double lb = model.col_lower[static_cast<std::size_t>(j)];
      const double ub = model.col_upper[static_cast<std::size_t>(j)];
      if (!is_integral(lb) || !is_integral(ub)) return {};
      if (static_cast<int>(std::llround(lb)) != 0) return {};
      if (static_cast<int>(std::llround(ub)) != 1) return {};

      vars.push_back({j, a});
    }
  }

  if (vars.empty()) return {};

  // Deterministic ordering: descending coefficient, ascending column index on ties.
  std::sort(vars.begin(), vars.end(), [](const KnapsackVar& u, const KnapsackVar& v) {
    if (u.a != v.a) return u.a > v.a;
    return u.col < v.col;
  });

  return vars;
}

/// Construct a minimal cover for the sorted variable list with capacity b.
///
/// ALGORITHM (greedy descending + minimality trim):
///   1. Accumulate variables in sorted order until the running sum exceeds b. This always
///      produces a cover if one exists, because we are taking the heaviest items first.
///   2. Trim: as long as removing the last-added member leaves the sum still exceeding b,
///      remove it. The result is minimal because:
///        - after trim, removing ANY member drops the sum to <= b (by the stopping rule).
///      The trim starts from the back (lightest member added so far) to avoid O(n^2).
///
/// DETERMINISM: since vars is already in a fixed order (descending a_j, ascending col),
/// the greedy prefix and the trim order are both deterministic.
///
/// Returns indices INTO `vars` forming the minimal cover, in ascending order (0,1,...).
/// Returns empty when no cover exists (total weight <= b).
[[nodiscard]] std::vector<std::size_t> find_minimal_cover(const std::vector<KnapsackVar>& vars,
                                                          double b) {
  double sum = 0.0;
  std::vector<std::size_t> cover;
  for (std::size_t k = 0; k < vars.size(); ++k) {
    sum += vars[k].a;
    cover.push_back(k);
    if (sum > b) break;
  }
  if (sum <= b) return {};  // even all variables together do not exceed b

  // Minimality trim: drop the last-added (smallest) member while the rest still exceed b.
  while (cover.size() > 1) {
    const std::size_t back = cover.back();
    if (sum - vars[back].a > b) {
      sum -= vars[back].a;
      cover.pop_back();
    } else {
      break;
    }
  }

  // The cover indices are in ascending order (0, 1, 2, ...) because we pushed in order.
  return cover;
}

}  // namespace

// =========================================================================================
// Public API: generate_knapsack_cover_cut
// =========================================================================================

std::optional<KnapsackCoverCut> generate_knapsack_cover_cut(const Model& model, Index row) {
  if (row < 0 || row >= model.num_rows()) return std::nullopt;
  if (model.num_cols() == 0) return std::nullopt;

  // Step 1. Classify row.
  const std::vector<KnapsackVar> vars = classify_row(model, row);
  if (vars.empty()) return std::nullopt;

  const double b = model.row_upper[static_cast<std::size_t>(row)];

  // Step 2. Find a minimal cover.
  const std::vector<std::size_t> cover_indices = find_minimal_cover(vars, b);
  if (cover_indices.empty()) return std::nullopt;

  const double cover_size = static_cast<double>(cover_indices.size());
  const double base_rhs = cover_size - 1.0;

  // Step 3. Initialise the cut coefficients.
  // We build a map from position in `vars` to its current cut coefficient.
  // Cover members start at 1.0 (base cover inequality).
  // Non-cover variables start at 0.0 (no coefficient until lifted).
  std::vector<double> coeff(vars.size(), 0.0);
  for (const std::size_t idx : cover_indices) {
    coeff[idx] = 1.0;
  }

  // Mark which positions are in the cover.
  std::vector<bool> in_cover(vars.size(), false);
  for (const std::size_t idx : cover_indices) in_cover[idx] = true;

  // Step 4. Sequential lifting for every non-cover variable.
  //
  // For each variable k outside the cover (in the order they appear in vars, which is
  // deterministic), compute:
  //
  //   Z_k = max { sum(alpha_j * x_j) : sum(a_j * x_j) <= b - a_k,
  //                                     x_j in {0,1},  j in C U L }
  //
  // where C is the original cover and L is the set of already-lifted variables with their
  // computed lifting coefficients. The lifting coefficient is:
  //
  //   alpha_k = (|C| - 1) - Z_k
  //
  // Valid range: alpha_k >= 0. (A negative value would mean the cut is already implied at
  // the LP relaxation level, which cannot happen for a valid minimal cover. If it occurs due
  // to floating-point drift, the cut is rejected.)
  //
  // The profit vector for the auxiliary knapsack is the current cut coefficient vector
  // restricted to items in C union L (because non-lifted, non-cover items have coeff 0,
  // including them at zero profit does not change Z_k but wastes DP work).

  // Accumulated list of (vars index, current coefficient) for items already placed in the
  // auxiliary knapsack (cover members + previously lifted).
  // We keep parallel weight/profit arrays for the DP.
  std::vector<double> dp_weight;
  std::vector<double> dp_profit;
  dp_weight.reserve(vars.size());
  dp_profit.reserve(vars.size());

  // Initialise dp_weight/dp_profit with the cover items.
  for (const std::size_t idx : cover_indices) {
    dp_weight.push_back(vars[idx].a);
    dp_profit.push_back(1.0);  // base cover coefficient
  }

  for (std::size_t k = 0; k < vars.size(); ++k) {
    if (in_cover[k]) continue;  // cover items are not lifted

    const double a_k = vars[k].a;

    // Reduced capacity for the auxiliary problem.
    const double aux_cap = b - a_k;
    if (aux_cap < 0.0) {
      // a_k > b means x_k = 0 is forced in any feasible solution (the constraint
      // sum(a_j x_j) <= b forces x_k = 0 if a_k > b). Lifting coefficient = |C| - 1.
      // This is a valid upper bound: any solution with x_k = 1 is infeasible.
      // However, since x_k = 0 at all feasible points, alpha_k can be anything >= 0.
      // We conservatively set alpha_k = 0 (adds no strengthening but is always valid).
      coeff[k] = 0.0;
      continue;
    }

    // Guard against DP table blowup.
    const auto dp_cap = static_cast<std::size_t>(std::llround(aux_cap));
    if (dp_weight.size() * dp_cap > kKnapsackDpLimit) {
      // Auxiliary problem is too large. Reject this lifted coefficient; emit no lifting
      // for this variable (coeff stays 0.0, which is valid: a zero coefficient simply
      // means the cut is weaker for this variable, never invalid).
      continue;
    }

    const double z_k = exact_01_knapsack(dp_weight, dp_profit, aux_cap);
    const double alpha_k = base_rhs - z_k;

    if (alpha_k < 0.0) {
      // Numerical drift: reject the whole cut rather than emit an invalid coefficient.
      // This should not occur for a valid minimal cover, but safety first.
      return std::nullopt;
    }

    coeff[k] = alpha_k;

    if (alpha_k > 0.0) {
      // Add this lifted variable to the auxiliary knapsack for subsequent liftings.
      dp_weight.push_back(a_k);
      dp_profit.push_back(alpha_k);
    }
    // If alpha_k == 0.0, adding it to the DP contributes zero profit and can be skipped.
  }

  // Step 5. Build the output cut.
  // col_index is sorted ascending (requirement from the struct's invariant).
  // We sort by column index.
  std::vector<std::pair<Index, double>> entries;
  entries.reserve(vars.size());
  for (std::size_t k = 0; k < vars.size(); ++k) {
    if (coeff[k] > 0.0 || in_cover[k]) {
      // Include cover members (coeff == 1.0) and any lifted variable with positive coeff.
      // Zero-coefficient non-cover variables are omitted for sparsity.
      entries.push_back({vars[k].col, coeff[k]});
    }
  }

  // Sort by column index ascending.
  std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b_pair) { return a.first < b_pair.first; });

  KnapsackCoverCut cut;
  cut.rhs = base_rhs;
  cut.col_index.reserve(entries.size());
  cut.coeff.reserve(entries.size());
  for (const auto& [col, c] : entries) {
    cut.col_index.push_back(col);
    cut.coeff.push_back(c);
  }

  return cut;
}

// =========================================================================================
// tighten_integral_rows â€” unchanged from baseline
// =========================================================================================

RowTightening tighten_integral_rows(Model* model, Logger& logger) {
  RowTightening result;
  if (model == nullptr) return result;

  const Index rows = model->num_rows();
  const Index cols = model->num_cols();
  if (rows == 0 || cols == 0 || !model->has_integrality()) return result;

  // Walk the matrix column-wise, since that is how it is stored, and accumulate per row:
  // whether every entry so far belongs to an integer column with an integral coefficient.
  // Rows with no entries stay `true` and are skipped afterwards - an empty row's activity is
  // zero, which is integral, but rounding its bounds changes nothing and presolve has
  // already removed it.
  std::vector<char> eligible(static_cast<std::size_t>(rows), 1);
  std::vector<char> has_entry(static_cast<std::size_t>(rows), 0);

  for (Index j = 0; j < cols; ++j) {
    const ColumnView column = model->matrix.column(j);
    const bool integer_column =
        model->col_type[static_cast<std::size_t>(j)] == VarType::kInteger;
    for (Index k = 0; k < column.size; ++k) {
      const auto row = static_cast<std::size_t>(column.rows[k]);
      has_entry[row] = 1;
      if (!integer_column || !is_integral(column.values[k])) eligible[row] = 0;
    }
  }

  for (Index i = 0; i < rows; ++i) {
    const auto u = static_cast<std::size_t>(i);
    if (!eligible[u] || has_entry[u] == 0) continue;

    double& lower = model->row_lower[u];
    double& upper = model->row_upper[u];
    bool moved = false;

    // Round INWARD on both sides. Outward would relax the row, which is not what this is for
    // and would be a silent correctness change in the other direction.
    //
    // The tolerance is applied before rounding so that a bound already sitting on an integer,
    // give or take floating-point noise, is not pushed a whole unit by its own representation
    // error: ceil(3.0000000001) is 4, and that would cut off the perfectly feasible point
    // where the row is tight at 3.
    if (is_finite_bound(lower)) {
      const double rounded = std::ceil(lower - tol::kIntegrality);
      if (rounded > lower) {
        lower = rounded;
        ++result.bounds_moved;
        moved = true;
      }
    }
    if (is_finite_bound(upper)) {
      const double rounded = std::floor(upper + tol::kIntegrality);
      if (rounded < upper) {
        upper = rounded;
        ++result.bounds_moved;
        moved = true;
      }
    }

    if (moved) ++result.rows_tightened;

    // Rounding can only make an infeasible row visibly infeasible; it cannot create
    // infeasibility that was not already there, because every integer-feasible point
    // satisfies the rounded bounds. Saying so is still worth a line, because "the model
    // became infeasible after cuts" is the first thing anyone will suspect.
    if (is_finite_bound(lower) && is_finite_bound(upper) && lower > upper) {
      logger.verbose(
          "row {} has no integral activity in [{:g}, {:g}]; it was already infeasible for "
          "integer points and the rounding has made that explicit",
          i, model->row_lower[u], model->row_upper[u]);
    }
  }

  if (result.rows_tightened > 0) {
    logger.info("Integer rounding: tightened {} row bound(s) on {} row(s)", result.bounds_moved,
                result.rows_tightened);
  }
  return result;
}

namespace detail {

std::optional<ReconstructedTableauRow> get_tableau_for_testing(const Model& model,
                                                               const Solution& solution,
                                                               Index basis_row) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<std::size_t>(n) != solution.col_status.size() ||
      static_cast<std::size_t>(m) != solution.row_status.size()) {
    return std::nullopt;
  }

  Index basic_count = 0;
  for (BasisStatus st : solution.col_status) {
    if (st == BasisStatus::kBasic) ++basic_count;
  }
  for (BasisStatus st : solution.row_status) {
    if (st == BasisStatus::kBasic) ++basic_count;
  }
  if (basic_count != m) return std::nullopt;

  std::vector<bool> slot_is_structural(static_cast<std::size_t>(m));
  std::vector<Index> slot_original_index(static_cast<std::size_t>(m));
  std::vector<LuColumn> columns;
  columns.reserve(static_cast<std::size_t>(m));

  std::vector<double> logical_values(static_cast<std::size_t>(m), -1.0);
  std::vector<Index> logical_rows(static_cast<std::size_t>(m));

  Index slot = 0;

  for (Index j = 0; j < n; ++j) {
    if (solution.col_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic) {
      slot_is_structural[static_cast<std::size_t>(slot)] = true;
      slot_original_index[static_cast<std::size_t>(slot)] = j;
      const ColumnView cview = model.matrix.column(j);
      LuColumn lucol;
      lucol.size = cview.size;
      lucol.rows = const_cast<Index*>(cview.rows);
      lucol.values = const_cast<double*>(cview.values);
      columns.push_back(lucol);
      ++slot;
    }
  }

  for (Index i = 0; i < m; ++i) {
    if (solution.row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic) {
      slot_is_structural[static_cast<std::size_t>(slot)] = false;
      slot_original_index[static_cast<std::size_t>(slot)] = i;
      logical_rows[static_cast<std::size_t>(slot)] = i;
      LuColumn lucol;
      lucol.size = 1;
      lucol.rows = &logical_rows[static_cast<std::size_t>(slot)];
      lucol.values = &logical_values[static_cast<std::size_t>(slot)];
      columns.push_back(lucol);
      ++slot;
    }
  }

  SparseLu lu;
  if (!lu.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold))
    return std::nullopt;

  std::vector<double> z(static_cast<std::size_t>(m), 0.0);
  z[static_cast<std::size_t>(basis_row)] = 1.0;
  lu.solve_transpose(z.data());

  ReconstructedTableauRow row;
  row.basis_row = basis_row;
  row.basic_is_structural = slot_is_structural[static_cast<std::size_t>(basis_row)];
  row.basic_index = slot_original_index[static_cast<std::size_t>(basis_row)];

  if (row.basic_is_structural) {
    row.rhs = solution.col_value[static_cast<std::size_t>(row.basic_index)];
  } else {
    row.rhs = solution.row_activity[static_cast<std::size_t>(row.basic_index)];
  }

  row.structural_coefs.resize(static_cast<std::size_t>(n), 0.0);
  row.logical_coefs.resize(static_cast<std::size_t>(m), 0.0);
  row.structural_status = solution.col_status;
  row.logical_status = solution.row_status;

  for (Index j = 0; j < n; ++j) {
    const ColumnView cview = model.matrix.column(j);
    double alpha = 0.0;
    for (Index k = 0; k < cview.size; ++k) {
      alpha += cview.values[k] * z[static_cast<std::size_t>(cview.rows[k])];
    }
    row.structural_coefs[static_cast<std::size_t>(j)] = alpha;
  }

  for (Index i = 0; i < m; ++i) {
    row.logical_coefs[static_cast<std::size_t>(i)] = -z[static_cast<std::size_t>(i)];
  }

  return row;
}

}  // namespace detail

namespace {

std::optional<Cut> compute_gmi_from_tableau(const Model& model,
                                            const detail::ReconstructedTableauRow& tableau) {
  if (!tableau.basic_is_structural) return std::nullopt;
  if (model.col_type[static_cast<std::size_t>(tableau.basic_index)] != VarType::kInteger) {
    return std::nullopt;
  }

  double f0 = tableau.rhs - std::floor(tableau.rhs);
  if (f0 <= tol::kIntegrality || f0 >= 1.0 - tol::kIntegrality) {
    return std::nullopt;
  }

  const Index n = model.num_cols();
  const Index m = model.num_rows();

  std::vector<double> gamma_j(static_cast<std::size_t>(n), 0.0);
  std::vector<double> gamma_s(static_cast<std::size_t>(m), 0.0);
  double beta_raw = 1.0;
  bool has_useful_pi = false;

  for (Index j = 0; j < n; ++j) {
    if (tableau.structural_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic) continue;

    const double alpha = tableau.structural_coefs[static_cast<std::size_t>(j)];
    const double L = model.col_lower[static_cast<std::size_t>(j)];
    const double U = model.col_upper[static_cast<std::size_t>(j)];
    const BasisStatus status = tableau.structural_status[static_cast<std::size_t>(j)];

    double alpha_prime = 0.0;
    double bound_val = 0.0;
    int sign = 1;

    if (status == BasisStatus::kFixed) continue;

    if (status == BasisStatus::kAtLower) {
      alpha_prime = alpha;
      bound_val = L;
      sign = 1;
    } else if (status == BasisStatus::kAtUpper) {
      alpha_prime = -alpha;
      bound_val = U;
      sign = -1;
    } else if (status == BasisStatus::kNonbasicFree) {
      if (std::abs(alpha) > tol::kZeroDrop) return std::nullopt;
      continue;
    } else {
      return std::nullopt;
    }

    if (std::abs(alpha_prime) <= tol::kZeroDrop) continue;

    double pi = 0.0;
    bool treat_as_integer = (model.col_type[static_cast<std::size_t>(j)] == VarType::kInteger);
    if (treat_as_integer) {
      const double f_bound = bound_val - std::floor(bound_val);
      if (f_bound > tol::kIntegrality && f_bound < 1.0 - tol::kIntegrality) {
        treat_as_integer = false;
      }
    }

    if (treat_as_integer) {
      const double fj = alpha_prime - std::floor(alpha_prime);
      if ((fj > 0.0 && fj < tol::kIntegrality) || (fj > 1.0 - tol::kIntegrality && fj < 1.0)) {
        return std::nullopt;
      }
      if (fj == 0.0) {
        pi = 0.0;
      } else {
        pi = std::min(fj / f0, (1.0 - fj) / (1.0 - f0));
      }
    } else {
      if (alpha_prime > 0.0) {
        pi = alpha_prime / f0;
      } else if (alpha_prime < 0.0) {
        pi = -alpha_prime / (1.0 - f0);
      }
    }

    if (pi > tol::kZeroDrop) {
      has_useful_pi = true;
      gamma_j[static_cast<std::size_t>(j)] = sign * pi;
      if (sign == 1) {
        beta_raw += pi * bound_val;
      } else {
        beta_raw -= pi * bound_val;
      }
    }
  }

  for (Index i = 0; i < m; ++i) {
    if (tableau.logical_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic) continue;

    const double alpha_s = tableau.logical_coefs[static_cast<std::size_t>(i)];
    const double L = model.row_lower[static_cast<std::size_t>(i)];
    const double U = model.row_upper[static_cast<std::size_t>(i)];
    const BasisStatus status = tableau.logical_status[static_cast<std::size_t>(i)];

    double alpha_prime = 0.0;
    double bound_val = 0.0;
    int sign = 1;

    if (status == BasisStatus::kFixed) continue;

    if (status == BasisStatus::kAtLower) {
      alpha_prime = alpha_s;
      bound_val = L;
      sign = 1;
    } else if (status == BasisStatus::kAtUpper) {
      alpha_prime = -alpha_s;
      bound_val = U;
      sign = -1;
    } else if (status == BasisStatus::kNonbasicFree) {
      if (std::abs(alpha_s) > tol::kZeroDrop) return std::nullopt;
      continue;
    } else {
      return std::nullopt;
    }

    if (std::abs(alpha_prime) <= tol::kZeroDrop) continue;

    double pi_s = 0.0;
    if (alpha_prime > 0.0) {
      pi_s = alpha_prime / f0;
    } else if (alpha_prime < 0.0) {
      pi_s = -alpha_prime / (1.0 - f0);
    }

    if (pi_s > tol::kZeroDrop) {
      has_useful_pi = true;
      gamma_s[static_cast<std::size_t>(i)] = sign * pi_s;
      if (sign == 1) {
        beta_raw += pi_s * bound_val;
      } else {
        beta_raw -= pi_s * bound_val;
      }
    }
  }

  if (!has_useful_pi) return std::nullopt;

  std::vector<double> final_gamma = gamma_j;
  for (Index j = 0; j < n; ++j) {
    const ColumnView cview = model.matrix.column(j);
    for (Index k = 0; k < cview.size; ++k) {
      const Index i = cview.rows[k];
      const double A_ij = cview.values[k];
      final_gamma[static_cast<std::size_t>(j)] += gamma_s[static_cast<std::size_t>(i)] * A_ij;
    }
  }

  bool has_nonzero_coeff = false;
  Cut cut;
  cut.coeff.resize(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    cut.coeff[static_cast<std::size_t>(j)] = -final_gamma[static_cast<std::size_t>(j)];
    if (std::abs(cut.coeff[static_cast<std::size_t>(j)]) > tol::kZeroDrop) {
      has_nonzero_coeff = true;
    }
  }
  cut.rhs = -beta_raw;

  if (!has_nonzero_coeff) return std::nullopt;
  return cut;
}

}  // namespace

std::optional<Cut> generate_gmi_cut(const Model& model, const Solution& solution,
                                    Index basis_row) {
  auto tableau_opt = detail::get_tableau_for_testing(model, solution, basis_row);
  if (!tableau_opt) return std::nullopt;
  return compute_gmi_from_tableau(model, *tableau_opt);
}

RootGmiContext::RootGmiContext(const Model& model, const Solution& solution) {
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<std::size_t>(n) != solution.col_status.size() ||
      static_cast<std::size_t>(m) != solution.row_status.size()) {
    return;
  }

  Index basic_count = 0;
  for (BasisStatus st : solution.col_status) {
    if (st == BasisStatus::kBasic) ++basic_count;
  }
  for (BasisStatus st : solution.row_status) {
    if (st == BasisStatus::kBasic) ++basic_count;
  }
  if (basic_count != m) return;

  slot_is_structural.resize(static_cast<std::size_t>(m));
  slot_original_index.resize(static_cast<std::size_t>(m));
  columns.reserve(static_cast<std::size_t>(m));

  logical_values.assign(static_cast<std::size_t>(m), -1.0);
  logical_rows.resize(static_cast<std::size_t>(m));

  Index slot = 0;

  for (Index j = 0; j < n; ++j) {
    if (solution.col_status[static_cast<std::size_t>(j)] == BasisStatus::kBasic) {
      slot_is_structural[static_cast<std::size_t>(slot)] = 1;
      slot_original_index[static_cast<std::size_t>(slot)] = j;
      const ColumnView cview = model.matrix.column(j);
      LuColumn lucol;
      lucol.size = cview.size;
      lucol.rows = const_cast<Index*>(cview.rows);
      lucol.values = const_cast<double*>(cview.values);
      columns.push_back(lucol);
      ++slot;
    }
  }

  for (Index i = 0; i < m; ++i) {
    if (solution.row_status[static_cast<std::size_t>(i)] == BasisStatus::kBasic) {
      slot_is_structural[static_cast<std::size_t>(slot)] = 0;
      slot_original_index[static_cast<std::size_t>(slot)] = i;
      logical_rows[static_cast<std::size_t>(slot)] = i;
      LuColumn lucol;
      lucol.size = 1;
      lucol.rows = &logical_rows[static_cast<std::size_t>(slot)];
      lucol.values = &logical_values[static_cast<std::size_t>(slot)];
      columns.push_back(lucol);
      ++slot;
    }
  }

  if (!lu.factorize(columns, m, tol::kPivotTolerance, tol::kMarkowitzThreshold)) {
    return;
  }

  is_valid = true;
}

std::optional<detail::ReconstructedTableauRow> RootGmiContext::tableau_row(
    const Model& model, const Solution& solution, Index basis_row) const {
  if (!is_valid) return std::nullopt;
  const Index m = model.num_rows();
  const Index n = model.num_cols();

  std::vector<double> z(static_cast<std::size_t>(m), 0.0);
  z[static_cast<std::size_t>(basis_row)] = 1.0;

  const_cast<SparseLu&>(lu).solve_transpose(z.data());

  detail::ReconstructedTableauRow row;
  row.basis_row = basis_row;
  row.basic_is_structural = slot_is_structural[static_cast<std::size_t>(basis_row)] != 0;
  row.basic_index = slot_original_index[static_cast<std::size_t>(basis_row)];

  if (row.basic_is_structural) {
    row.rhs = solution.col_value[static_cast<std::size_t>(row.basic_index)];
  } else {
    row.rhs = solution.row_activity[static_cast<std::size_t>(row.basic_index)];
  }

  row.structural_coefs.resize(static_cast<std::size_t>(n), 0.0);
  row.logical_coefs.resize(static_cast<std::size_t>(m), 0.0);
  row.structural_status = solution.col_status;
  row.logical_status = solution.row_status;

  for (Index j = 0; j < n; ++j) {
    const ColumnView cview = model.matrix.column(j);
    double alpha = 0.0;
    for (Index k = 0; k < cview.size; ++k) {
      alpha += cview.values[k] * z[static_cast<std::size_t>(cview.rows[k])];
    }
    row.structural_coefs[static_cast<std::size_t>(j)] = alpha;
  }

  for (Index i = 0; i < m; ++i) {
    row.logical_coefs[static_cast<std::size_t>(i)] = -z[static_cast<std::size_t>(i)];
  }

  return row;
}

std::vector<Cut> generate_gmi_cuts(const Model& model, const Solution& solution) {
  std::vector<Cut> cuts;
  RootGmiContext context(model, solution);
  if (!context.is_valid) return cuts;

  const Index m = model.num_rows();
  for (Index slot = 0; slot < m; ++slot) {
    if (!context.slot_is_structural[static_cast<std::size_t>(slot)]) continue;

    Index basic_index = context.slot_original_index[static_cast<std::size_t>(slot)];
    if (model.col_type[static_cast<std::size_t>(basic_index)] != VarType::kInteger) continue;

    double val = solution.col_value[static_cast<std::size_t>(basic_index)];
    double f0 = val - std::floor(val);
    if (f0 <= tol::kIntegrality || f0 >= 1.0 - tol::kIntegrality) continue;

    auto row_opt = context.tableau_row(model, solution, slot);
    if (!row_opt) continue;

    auto cut_opt = compute_gmi_from_tableau(model, *row_opt);
    if (cut_opt) cuts.push_back(*cut_opt);
  }

  return cuts;
}

// =========================================================================================
// Cut Filtering and Deduplication
// =========================================================================================

std::vector<FilteredCut> filter_and_deduplicate_cuts(const Model& model,
                                                     const Solution& root_solution,
                                                     const std::vector<Cut>& candidates) {
  std::vector<FilteredCut> results;
  results.reserve(candidates.size());

  const Index n = model.num_cols();
  if (n < 0) return results;  // Safety check

  for (const Cut& c : candidates) {
    FilteredCut fc;
    fc.cut = c;  // Copy candidate (filtering never modifies coefficients or RHS)
    fc.reason = CutFilterReason::kAccepted;

    // 1. Finite-value check
    if (!std::isfinite(c.rhs)) {
      fc.reason = CutFilterReason::kNonfinite;
      results.push_back(fc);
      continue;
    }
    bool is_finite = true;
    for (double val : c.coeff) {
      if (!std::isfinite(val)) {
        is_finite = false;
        break;
      }
    }
    if (!is_finite) {
      fc.reason = CutFilterReason::kNonfinite;
      results.push_back(fc);
      continue;
    }

    // 2. Support filter, Density, Coefficient Ratio
    Index nonzero_count = 0;
    double max_abs = 0.0;
    double min_abs = std::numeric_limits<double>::infinity();

    for (double val : c.coeff) {
      double abs_val = std::abs(val);
      if (abs_val > tol::kZeroDrop) {
        ++nonzero_count;
        if (abs_val > max_abs) max_abs = abs_val;
        if (abs_val < min_abs) min_abs = abs_val;
      }
    }

    if (nonzero_count == 0) {
      fc.reason = CutFilterReason::kEmptySupport;
      results.push_back(fc);
      continue;
    }

    // Density filter
    double density = 0.0;
    if (n > 0) {
      density = static_cast<double>(nonzero_count) / static_cast<double>(n);
    }
    if (density > tol::kCutMaxDensity) {
      fc.reason = CutFilterReason::kTooDense;
      results.push_back(fc);
      continue;
    }

    // Coefficient-ratio filter
    double ratio = max_abs / min_abs;
    if (ratio > tol::kCutMaxCoefficientRatio) {
      fc.reason = CutFilterReason::kCoefficientRatio;
      results.push_back(fc);
      continue;
    }

    // Root-LP violation filter
    double lhs = 0.0;
    for (Index j = 0; j < n; ++j) {
      lhs += c.coeff[static_cast<std::size_t>(j)] *
             root_solution.col_value[static_cast<std::size_t>(j)];
    }
    double violation = lhs - c.rhs;
    if (violation <= tol::kCutViolationTolerance) {
      fc.reason = CutFilterReason::kInsufficientViolation;
      results.push_back(fc);
      continue;
    }

    // Deduplication
    bool is_duplicate = false;
    for (const FilteredCut& accepted_fc : results) {
      if (accepted_fc.reason != CutFilterReason::kAccepted) continue;
      const Cut& accepted_c = accepted_fc.cut;

      // Find first materially nonzero coefficient in accepted_c (A)
      Index first_nonzero = -1;
      for (Index j = 0; j < n; ++j) {
        if (std::abs(accepted_c.coeff[static_cast<std::size_t>(j)]) > tol::kZeroDrop) {
          first_nonzero = j;
          break;
        }
      }

      if (first_nonzero == -1) continue;  // Should not happen for accepted cuts

      double A_j = accepted_c.coeff[static_cast<std::size_t>(first_nonzero)];
      double C_j = c.coeff[static_cast<std::size_t>(first_nonzero)];

      // Check sign match
      if ((A_j > 0.0 && C_j <= 0.0) || (A_j < 0.0 && C_j >= 0.0)) {
        continue;
      }

      double lambda = C_j / A_j;
      if (lambda <= 0.0) continue;

      // Compare all coefficients and RHS
      bool identical = true;
      for (Index j = 0; j < n; ++j) {
        double a = c.coeff[static_cast<std::size_t>(j)];
        double b = lambda * accepted_c.coeff[static_cast<std::size_t>(j)];
        if (std::abs(a - b) > tol::kZeroDrop * std::max({1.0, std::abs(a), std::abs(b)})) {
          identical = false;
          break;
        }
      }

      if (identical) {
        double a_rhs = c.rhs;
        double b_rhs = lambda * accepted_c.rhs;
        if (std::abs(a_rhs - b_rhs) <=
            tol::kZeroDrop * std::max({1.0, std::abs(a_rhs), std::abs(b_rhs)})) {
          is_duplicate = true;
          break;
        }
      }
    }

    if (is_duplicate) {
      fc.reason = CutFilterReason::kDuplicate;
      results.push_back(fc);
      continue;
    }

    results.push_back(fc);
  }

  return results;
}

}  // namespace sankhya::mip
