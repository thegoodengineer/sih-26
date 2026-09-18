// SPDX-License-Identifier: Apache-2.0
// SANKHYA - crossover: from the interior point's answer to a vertex (#219).
//
// The interior point stops strictly inside the feasible region, accurate to its tolerance,
// with no basis: fine for an objective value, useless for everything downstream - a warm
// start, sensitivity ranging, or a planner who wants to read "these units run at capacity,
// these are off" rather than a point where every variable is slightly on. Crossover is the
// standard bridge (Bixby, "Solving real-world linear programs: a decade and more of
// progress", Operations Research 50 (2002), sec. 4; Andersen & Ye, "Combining interior
// point and pivoting algorithms for constrained linear programs", Management Science 42
// (1996)): identify from the interior point which variables sit at bounds, guess a basis
// from the rest, and let the simplex pivot from that basis to an optimal vertex. Because the
// starting point is already optimal to 1e-8, the pivots are few.
//
// WHAT IS GUESSED AND WHAT IS PROVED. The classification here is a heuristic: a column is
// called nonbasic-at-lower when it is within a tolerance of its lower bound and its reduced
// cost points there, basic otherwise, and the m entries with the largest slack from their
// bounds become the basis guess. Nothing about the answer rests on that guess being right:
// the simplex installs it, repairs it if it is singular, pivots to optimality and reports its
// own vertex, which the status guard and the independent verifier then judge exactly as they
// judge any simplex answer. When the simplex does not reach an optimum inside what is left
// of the time limit, the interior point's answer stands and the message says so.
#pragma once

#include <vector>

#include "sankhya/logging.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"
#include "sankhya/solve_control.hpp"
#include "sankhya/timer.hpp"

namespace sankhya {

/// The basis guess crossover builds from an interior point, exposed for the test that checks
/// the guess describes exactly m basic entries with every nonbasic one on a bound it has.
struct CrossoverGuess {
  std::vector<BasisStatus> col_status;
  std::vector<BasisStatus> row_status;
  Index basic = 0;     ///< entries marked basic (m for a well-formed model with a point)
  Index interior = 0;  ///< entries the interior point held strictly inside their bounds
  Index repaired = 0;  ///< structurals evicted for row logicals to make the guess full rank
};

/// Classify an interior point (`col_value`, `row_activity`, `col_dual` in the model's sense)
/// into a basis guess with exactly m basic entries.
[[nodiscard]] CrossoverGuess crossover_guess(const Model& model, const Solution& interior);

/// Push `interior` (an optimal interior-point answer to `model`) to an optimal vertex with the
/// dual simplex warm-started from crossover_guess(). Returns the simplex's vertex when it
/// reaches `optimal`, with the pivot count in the message and `iterations` summed; otherwise
/// returns `interior` unchanged apart from a note saying why the vertex was not reached.
/// `timer` is the solve's clock: the pivots get what the time limit has left.
[[nodiscard]] Solution crossover_to_vertex(const Model& model, Solution interior,
                                           const Options& options, Logger& logger,
                                           SolveControl* control, const Timer& timer);

}  // namespace sankhya
