// SPDX-License-Identifier: Apache-2.0
// SANKHYA - convexity test. See convexity.hpp for why a refusal is the right default, and for
// why there are two implementations of the same decision.

#include "convexity.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "la/ldl.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::qp {
namespace {

/// The dense reference is O(n^2) memory and O(n^3) time, so it is only ever run on a matrix
/// small enough for both to be free. Production goes through the sparse test at every size;
/// this bound exists so the reference stays a reference.
constexpr Index kDenseReferenceLimit = 2000;

/// A pivot may go slightly negative on a genuinely semidefinite matrix purely through
/// rounding. Higham's analysis bounds that perturbation by a small multiple of eps times the
/// largest diagonal entry, so the test scales its threshold by that rather than using an
/// absolute number - an indefinite direction in a badly scaled Q would otherwise hide under a
/// fixed tolerance.
constexpr double kPivotSlackFactor = 1e-10;

/// Q in MINIMIZATION sense, as the lower triangle of a symmetric matrix.
///
/// The engines minimise sense * (c'x + 0.5 x'Qx), so the Hessian they actually see is
/// sense * Q. What has to be positive semidefinite is that, not Q. For a MAXIMIZATION model
/// the requirement is therefore that Q be NEGATIVE semidefinite - a concave objective - and
/// testing Q itself would reject every well posed concave maximisation while accepting the
/// convex ones, which are exactly the unbounded-above cases that must be refused.
///
/// Model documents the Hessian as lower-triangular and both the readers and the C API
/// normalise to (max, min); an entry that arrives above the diagonal anyway is folded onto
/// its mirror rather than ignored, because ignoring it would test a different matrix.
[[nodiscard]] SparseMatrix minimization_lower_triangle(const Model& model) {
  const Index n = model.num_cols();
  const double sense = model.sense_multiplier();
  SparseMatrix lower(n, n);
  lower.reserve(static_cast<std::size_t>(model.hessian.num_nonzeros()));
  for (Index j = 0; j < model.hessian.num_cols(); ++j) {
    const ColumnView column = model.hessian.column(j);
    for (Index k = 0; k < column.size; ++k) {
      const Index i = column.rows[k];
      lower.add_entry(std::max(i, j), std::min(i, j), sense * column.values[k]);
    }
  }
  lower.finalize();
  return lower;
}

}  // namespace

ConvexityResult check_convexity_dense(const Model& model) {
  ConvexityResult result;
  const Index n = model.num_cols();

  if (model.hessian.num_nonzeros() == 0) {
    result.verdict = Convexity::kConvex;
    result.detail = "the objective has no quadratic term";
    return result;
  }
  if (n > kDenseReferenceLimit) {
    result.detail = fmt::format(
        "{} columns exceeds the {} the dense reference test holds in "
        "memory; check_convexity() decides this size sparsely",
        n, kDenseReferenceLimit);
    return result;  // kUnverified
  }

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
      // Semidefinite but singular in this direction - legitimate, a rank-deficient Q is
      // still convex - PROVIDED the rest of the column vanishes with it. For a positive
      // semidefinite matrix it must (Higham 1990): a zero on the diagonal forces the whole
      // row and column to zero. Skipping the column without checking that is how
      // Q = [[0, 1], [1, 0]] used to be reported convex, and 2*x0*x1 is a saddle.
      d[uj] = 0.0;
      for (Index i = j + 1; i < n; ++i) {
        const auto ui = static_cast<std::size_t>(i);
        double sum = q[ui * un + uj];
        for (Index k = 0; k < j; ++k) {
          const auto uk = static_cast<std::size_t>(k);
          sum -= l[ui * un + uk] * l[uj * un + uk] * d[uk];
        }
        if (std::fabs(sum) > slack) {
          result.verdict = Convexity::kIndefinite;
          result.detail = fmt::format(
              "column {} has a zero pivot but entry ({}, {}) of the remaining Schur "
              "complement is {:.6g}; a positive semidefinite matrix cannot carry a nonzero "
              "beside a zero pivot, so the objective has a direction of negative curvature",
              j, i, j, sum);
          return result;
        }
        l[ui * un + uj] = 0.0;
      }
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

ConvexityResult check_convexity(const Model& model) {
  ConvexityResult result;

  if (model.hessian.num_nonzeros() == 0) {
    result.verdict = Convexity::kConvex;
    result.detail = "the objective has no quadratic term";
    return result;
  }

  const SparseMatrix lower = minimization_lower_triangle(model);
  SparseLdl ldl;
  const SemidefiniteReport report = ldl.check_semidefinite(lower, kPivotSlackFactor);
  switch (report.verdict) {
    case SemidefiniteReport::Verdict::kPositiveSemidefinite:
      result.verdict = Convexity::kConvex;
      result.detail =
          fmt::format("sparse LDL^T over {} nonzeros completed with every pivot non-negative",
                      lower.num_nonzeros());
      return result;
    case SemidefiniteReport::Verdict::kIndefinite:
      result.verdict = Convexity::kIndefinite;
      result.detail = fmt::format(
          "sparse LDL^T reached {:.6g} at column {} in minimization sense; that exhibits a "
          "direction in which the objective curves downward, so the model is non-convex",
          report.pivot, report.column);
      return result;
    case SemidefiniteReport::Verdict::kUndecided: break;
  }

  result.detail = "the Hessian could not be ordered and factorized, so convexity is unproven";
  return result;  // kUnverified
}

}  // namespace sankhya::qp
