// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solution pool: the good integer points the tree found, not only the best
// (#225).
//
// References, written from the literature:
//   Danna, Fenelon, Gu & Wunderling, "Generating multiple solutions for mixed integer
//     programming problems", IPCO 2007, LNCS 4513 - the pool as a by-product of the search,
//     and continuing the search past the optimum to fill it ("populate")
//   Glover, Lokketangen & Woodruff, "Scatter search to generate diverse MIP solutions", in
//     Laguna & Gonzalez-Velarde (eds.), Computing Tools for Modeling, Optimization and
//     Simulation (2000) - Hamming distance on the integer assignment as the diversity measure
//
// WHY A PLANNER WANTS THIS. The model never knows everything: the second-best plan is often
// the one that survives the constraint nobody wrote down ("that unit is down in March"). The
// tree already produces these points on its way to the optimum and used to throw away all but
// the last; keeping them is bookkeeping, not search.
//
// WHAT IS KEPT. Every point offered here has already passed BranchAndBound::offer_incumbent's
// feasibility test against the original model - integrality, column bounds, rows - so the
// pool never holds a point the incumbent would have refused. Two points with the same integer
// assignment are one plan, whatever their continuous values, and only the better is kept.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sankhya/types.hpp"

namespace sankhya::mip {

class SolutionPool {
 public:
  struct Entry {
    double objective = 0.0;  ///< minimise space, excluding the offset, as the search sees it
    std::vector<double> x;
    std::vector<std::int64_t> key;  ///< the integer assignment, rounded
  };

  SolutionPool() = default;
  /// `capacity` 0 keeps nothing. With `diversity` a full pool evicts the member closest to
  /// the others in Hamming distance instead of the worst one (never the best).
  SolutionPool(std::vector<Index> integer_columns, std::size_t capacity, bool diversity);

  /// Offer an integer-feasible point. The caller has already checked feasibility.
  void offer(double objective, const std::vector<double>& x);

  [[nodiscard]] bool enabled() const { return capacity_ > 0; }
  [[nodiscard]] bool full() const { return capacity_ > 0 && entries_.size() >= capacity_; }
  [[nodiscard]] std::size_t size() const { return entries_.size(); }

  /// The worst objective held when the pool is full and ordered by objective alone, else
  /// +infinity. A node whose bound is no better than this cannot contribute a member, which
  /// is what lets the complete search (`pool_complete`) prune at all. Always +infinity with
  /// diversity on: a diverse pool may evict a better point for a more different one, so no
  /// objective value bounds what it could still accept.
  [[nodiscard]] double cutoff() const;

  /// The pool as reported: the incumbent first, then the rest best first, keeping only
  /// members within `relative_gap` of the incumbent (relative to max(1, |incumbent|)).
  [[nodiscard]] std::vector<Entry> finish(double incumbent_objective,
                                          const std::vector<double>& incumbent_x,
                                          double relative_gap) const;

  /// Number of integer columns on which two assignments differ.
  [[nodiscard]] static std::size_t hamming(const std::vector<std::int64_t>& a,
                                           const std::vector<std::int64_t>& b);

 private:
  [[nodiscard]] std::vector<std::int64_t> key_of(const std::vector<double>& x) const;
  void evict_one();

  std::vector<Index> integer_columns_;
  std::size_t capacity_ = 0;
  bool diversity_ = false;
  std::vector<Entry> entries_;  ///< ascending objective; ties keep the order they arrived in
};

}  // namespace sankhya::mip
