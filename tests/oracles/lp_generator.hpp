// SPDX-License-Identifier: Apache-2.0
// SANKHYA - random LP generators for the oracle fuzz. TESTS ONLY.
//
// Three families, because they fail the solver in different ways:
//
//   random_lp      unstructured. Mostly infeasible or unbounded, which is the point: those
//                  two statuses are as easy to get wrong as an objective, and nothing else
//                  in the suite exercises them in bulk.
//
//   degenerate_lp  many constraints made tight at one common point. Degeneracy is where a
//                  simplex cycles or stalls, and it is the failure the anti-cycling rule
//                  exists for, so it needs deliberate coverage rather than luck.
//
//   kkt_lp         built backwards from a primal-dual pair chosen in advance, so the optimal
//                  objective is known ANALYTICALLY before anything is solved. This is the
//                  strongest generator in the file: it needs no oracle to say what the answer
//                  is, so it cannot be fooled by the oracle and the solver sharing a mistake.
#pragma once

#include <cstdint>
#include <random>

#include "oracles/rational_simplex.hpp"

namespace sankhya::oracle {

/// Shape and coefficient range for a generated instance.
struct GeneratorConfig {
  Index min_rows = 2;
  Index max_rows = 8;
  Index min_cols = 2;
  Index max_cols = 8;
  /// Probability that any given matrix entry is nonzero.
  double density = 0.6;
  /// Coefficients are drawn uniformly from [-magnitude, magnitude], excluding nothing.
  std::int64_t magnitude = 6;
  /// Probability that a column receives a finite upper bound.
  double bounded_column_probability = 0.35;
};

/// An unstructured instance. No feasibility or boundedness is promised.
[[nodiscard]] GeneratedLp random_lp(std::mt19937_64& rng, const GeneratorConfig& config);

/// An instance with many constraints active simultaneously at one point, which is what makes
/// a vertex degenerate.
[[nodiscard]] GeneratedLp degenerate_lp(std::mt19937_64& rng, const GeneratorConfig& config);

/// An instance built from a chosen KKT point, together with the objective value that point
/// achieves. Construction (for  min c'x  s.t.  Ax >= b, x >= 0):
///
///   pick x* >= 0 and a set of active rows
///   pick y* >= 0 with y*_i = 0 on inactive rows
///   set  c = A^T y* + d  with d >= 0 and d_j = 0 wherever x*_j > 0
///   set  b_i = A_i x*  on active rows, and strictly below it on inactive rows
///
/// Then primal feasibility, dual feasibility and complementary slackness all hold by
/// construction, so x* is optimal and the optimum is exactly c'x*.
struct KktInstance {
  GeneratedLp lp;
  /// The optimal objective, exact. Integer because every ingredient above is an integer.
  std::int64_t optimal_objective = 0;
  std::vector<std::int64_t> optimal_x;
};

[[nodiscard]] KktInstance kkt_lp(std::mt19937_64& rng, const GeneratorConfig& config);

}  // namespace sankhya::oracle
