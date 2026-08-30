// SPDX-License-Identifier: Apache-2.0
// SANKHYA - bounded-variable revised primal simplex.
#pragma once

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

#include "la/scaling.hpp"

namespace sankhya {

/// Solve a continuous LP with the revised primal simplex. Integrality is IGNORED: this is
/// the node solver branch-and-cut will call in Phase 5, and it is the caller's job to know
/// whether it wanted a relaxation. solve() in src/core refuses a MILP for exactly this
/// reason rather than quietly returning a fractional point labelled optimal.
[[nodiscard]] Solution solve_primal_simplex(const Model& model, const Options& options,
                                            Logger& logger);

/// The equilibration a repeated caller can compute once and hand back on every solve.
///
/// Branch and bound calls the simplex once per node over ONE working model whose constraint
/// matrix never changes - nodes differ only in variable bounds. The row and column
/// multipliers are therefore identical at every node, and recomputing them per node is ten
/// Ruiz passes plus a Pock-Chambolle pass over a full copy of the matrix, thrown away and
/// done again at the next node (#76).
///
/// Build this once with `build_node_scaling`, then pass it to the overload below. The
/// multipliers and the scaled matrix are reused; the BOUNDS are rescaled per call, because
/// those are exactly what branching changes.
struct NodeScaling {
  Scaling scaling;
  bool valid = false;
};

/// Compute the reusable part once. Returns an invalid cache when scaling is switched off, so
/// a caller can build it unconditionally and let the solve decide.
[[nodiscard]] NodeScaling build_node_scaling(const Model& model, const Options& options);

/// Solve using a precomputed equilibration.
///
/// `cache` must have been built from a model with the SAME constraint matrix, cost vector and
/// row bounds as `model` - in practice, the same working model earlier in the same search.
/// Column bounds may differ freely and are rescaled here.
///
/// An invalid cache is not an error: it falls through to the unscaled path, which is what
/// `scaling=false` wants anyway.
[[nodiscard]] Solution solve_primal_simplex(const Model& model, const Options& options,
                                            Logger& logger, const NodeScaling& cache);

}  // namespace sankhya
