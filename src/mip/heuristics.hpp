// SPDX-License-Identifier: Apache-2.0
// SANKHYA - MILP primal heuristics (#290).
//
// A primal heuristic PROPOSES a point; it never decides one. Every candidate these functions
// return goes through BranchAndBound::offer_incumbent(), which re-checks integrality, the
// original bounds and every original row before it can become the incumbent. A heuristic with
// a bug therefore costs time, never an answer - which is the property that lets them be
// aggressive.
//
// They are pure functions of a Model and a point, with no access to the search, so each can
// be tested on its own model. The search owns the scheduling (when, how often, with what
// budget) and the statistics.
//
//   lock rounding    Achterberg, "Constraint Integer Programming", thesis, TU Berlin 2007,
//                    sec. 9.1 (simple rounding by locks): a column no row can be violated by
//                    moving down is rounded down, and symmetrically; the rest to nearest.
//   repair           a greedy one-unit shift of integer columns in the most violated row,
//                    choosing the shift that removes the most violation and, among equals,
//                    costs the objective least - a bounded local search in the spirit of
//                    Berthold's shift-and-propagate (2014), without the propagation.
//   RINS             Danna, Rothberg and Le Pape, "Exploring relaxation induced
//                    neighborhoods to improve MIP solutions", Math. Programming 102 (2005):
//                    fix the integer columns on which the incumbent and the node relaxation
//                    agree, and search what is left as a small sub-MIP.
//   feasibility pump Fischetti, Glover and Lodi, "The feasibility pump", Math. Programming
//                    104 (2005): alternate rounding the LP point and projecting the rounding
//                    back onto the LP polytope in the L1 distance, until they meet. The
//                    distance is linear only for a column rounded to one of its bounds; a
//                    general integer rounded strictly inside its bounds needs an auxiliary
//                    column (Bertacco, Fischetti and Lodi 2007) and is left out of the
//                    distance here, which the PR says.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::mip {

/// What one heuristic did over a search.
struct HeuristicStats {
  std::string name;
  Count calls = 0;
  Count found = 0;     ///< candidates proposed to offer_incumbent()
  Count improved = 0;  ///< of those, the ones it accepted: feasible, integral and better
  Count work = 0;      ///< LP solves, sub-MIP nodes or repair moves, whichever it spends
  double seconds = 0.0;
};

/// Row locks (Achterberg 2007, sec. 9.1): up[j] counts the rows that moving column j UP could
/// violate, down[j] the rows moving it down could.
struct Locks {
  std::vector<int> up;
  std::vector<int> down;
};
[[nodiscard]] Locks compute_locks(const Model& model);

/// x with every integer column rounded in the direction its locks allow, the others to the
/// nearest integer; clamped to the column bounds. Continuous columns are left as they are.
[[nodiscard]] std::vector<double> lock_round(const Model& model, const Locks& locks,
                                             const std::vector<Index>& integer_columns,
                                             const std::vector<double>& x);

/// Starting from a point whose integer columns are integral, shift integer columns one unit
/// at a time to remove row violations. Returns true when every row is satisfied to
/// `tolerance` within `max_moves` shifts; `moves` reports how many were made either way.
bool repair(const Model& model, const std::vector<Index>& integer_columns,
            std::vector<double>* x, int max_moves, double tolerance, Count* moves);

/// The RINS sub-model: `model` with every integer column fixed where `relaxation` and
/// `incumbent` agree to `tolerance`. False, leaving `out` alone, when fewer than
/// `min_fixed_fraction` of the integer columns agree - a neighbourhood that large is not a
/// neighbourhood. `fixed` reports how many were fixed.
bool rins_submodel(const Model& model, const std::vector<Index>& integer_columns,
                   const std::vector<double>& relaxation, const std::vector<double>& incumbent,
                   double min_fixed_fraction, double tolerance, Model* out, Count* fixed);

/// The feasibility pump from an LP point `start`. Returns the first point whose integer
/// columns are integral and which the pump's own LP says is feasible, or an empty vector.
/// `lp_solves` reports the LP projections spent. `lp_options` is what each projection is
/// solved with; the caller sets its limits.
[[nodiscard]] std::vector<double> feasibility_pump(const Model& model,
                                                   const std::vector<Index>& integer_columns,
                                                   const std::vector<double>& start,
                                                   const Options& lp_options, int max_rounds,
                                                   double integrality_tolerance,
                                                   Count* lp_solves);

}  // namespace sankhya::mip
