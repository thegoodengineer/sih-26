// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity test. See convexity.hpp for why a refusal is the right default.

#include "convexity.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "sankhya/tolerances.hpp"

namespace sankhya::qp {
namespace {

/// Above this the dense O(n^2) working set stops being reasonable and the test reports
/// kUnverified rather than guessing. A sparse LDL^T (#70) lifts this; until it exists,
/// refusing a large QP is the honest answer and refusing is what kUnverified means.
constexpr Index kDenseLimit = 2000;

/// A pivot may go slightly negative on a genuinely semidefinite matrix purely through
/// rounding. Higham's analysis bounds that perturbation by a small multiple of eps times the
/// largest diagonal entry, so the test scales its threshold by that rather than using an
/// absolute number - an indefinite direction in a badly scaled Q would otherwise hide under a
/// fixed tolerance.
constexpr double kPivotSlackFactor = 1e-10;

}  // namespace

ConvexityResult check_convexity(const Model& model) {
  ConvexityResult result;
  const Index n = model.num_cols();

  if (model.hessian.num_nonzeros() == 0) {
    result.verdict = Convexity::kConvex;
    result.detail = "the objective has no quadratic term";
    return result;
  }
  if (n > kDenseLimit) {
    result.detail = fmt::format(
        "{} columns exceeds the {} the dense convexity test can decide; a sparse LDL^T is "
        "needed before a QP this size can be accepted",
        n, kDenseLimit);
    return result;  // kUnverified
  }

  // Densify the symmetric matrix from its stored lower triangle, IN MINIMIZATION SENSE.
  //
  // The engines minimise sense * (c'x + 0.5 x'Qx), so the Hessian they actually see is
  // sense * Q. What has to be positive semidefinite is that, not Q. For a MAXIMIZATION model
  // the requirement is therefore that Q be NEGATIVE semidefinite - a concave objective - and
  // testing Q itself would reject every well posed concave maximisation while accepting the
  // convex ones, which are exactly the unbounded-above cases that must be refused.
  const double sense = model.sense_multiplier();

  // The stored entries are Q itself (the 0.5 lives in the objective, not in the storage), so
  // an off-diagonal entry appears twice.
  const auto un = static_cast<std::size_t>(n);
  std::vector<double> q(un * un, 0.0);
  double largest_diagonal = 0.0;
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      const double value = sense * column.values[k];
      q[static_cast<std::size_t>(i) * un + static_cast<std::size_t>(j)] = value;
      q[static_cast<std::size_t>(j) * un + static_cast<std::size_t>(i)] = value;
      if (i == j) largest_diagonal = std::max(largest_diagonal, std::fabs(value));
    }
  }

  // A negative diagonal entry is an immediate certificate: e_i^T Q e_i < 0.
  for (Index i = 0; i < n; ++i) {
    const double d = q[static_cast<std::size_t>(i) * un + static_cast<std::size_t>(i)];
    if (d < -kPivotSlackFactor * std::max(1.0, largest_diagonal)) {
      result.verdict = Convexity::kIndefinite;
      result.detail = fmt::format(
          "the Hessian diagonal for column {} is {:.6g} in minimization sense; e^T Q e < 0 "
          "makes the objective non-convex along that column alone",
          i, d);
      return result;
    }
  }

  // LDL^T without row interchanges. For a semidefinite matrix a zero pivot means the
  // remaining column is already in the span of the previous ones, so it contributes nothing
  // and is skipped; for an indefinite one a negative pivot appears instead, and that is the
  // certificate being looked for.
  const double slack = kPivotSlackFactor * std::max(1.0, largest_diagonal);
  std::vector<double> l(un * un, 0.0);
  std::vector<double> d(un, 0.0);

  for (Index j = 0; j < n; ++j) {
    const auto uj = static_cast<std::size_t>(j);
    double pivot = q[uj * un + uj];
    for (Index k = 0; k < j; ++k) {
      const auto uk = static_cast<std::size_t>(k);
      pivot -= l[uj * un + uk] * l[uj * un + uk] * d[uk];
    }

    if (pivot < -slack) {
      result.verdict = Convexity::kIndefinite;
      result.detail = fmt::format(
          "LDL^T reached a pivot of {:.6g} at column {}; a negative pivot exhibits a "
          "direction in which the objective curves downward, so the model is non-convex",
          pivot, j);
      return result;
    }
    if (pivot <= slack) {
      // Semidefinite but singular in this direction. Legitimate - a rank-deficient Q is
      // still convex - so the column is skipped rather than divided through.
      d[uj] = 0.0;
      for (Index i = j + 1; i < n; ++i) l[static_cast<std::size_t>(i) * un + uj] = 0.0;
      continue;
    }

    d[uj] = pivot;
    for (Index i = j + 1; i < n; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      double sum = q[ui * un + uj];
      for (Index k = 0; k < j; ++k) {
        const auto uk = static_cast<std::size_t>(k);
        sum -= l[ui * un + uk] * l[uj * un + uk] * d[uk];
      }
      l[ui * un + uj] = sum / pivot;
    }
  }

  result.verdict = Convexity::kConvex;
  result.detail = "LDL^T completed with every pivot non-negative";
  return result;
}

}  // namespace sankhya::qp
