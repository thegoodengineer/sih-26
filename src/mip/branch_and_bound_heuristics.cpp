// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch and bound: scheduling the primal heuristics and accounting for them
// (#290). The heuristics themselves are pure functions in heuristics.cpp; this file decides
// when each runs, with what budget, and offers what they propose to offer_incumbent(), which
// is the only way anything becomes the incumbent.

#include "branch_and_bound_internal.hpp"

#include <algorithm>
#include <cmath>

#include <fmt/format.h>

#include "sankhya/timer.hpp"

namespace sankhya::mip {

namespace {
/// Where each heuristic's row lives in heuristic_stats_.
enum Slot : std::size_t { kRounding, kLockRounding, kRepair, kDive, kPump, kRins, kSlots };
constexpr const char* kNames[kSlots] = {"rounding", "lock rounding",    "repair",
                                        "diving",   "feasibility pump", "RINS"};
}  // namespace

void BranchAndBound::init_heuristics() {
  heuristic_stats_.assign(kSlots, HeuristicStats{});
  for (std::size_t k = 0; k < kSlots; ++k) heuristic_stats_[k].name = kNames[k];
  heuristics_on_ = options_.get_bool("mip_heuristics");
  rins_frequency_ = options_.get_int("mip_rins_frequency");
  rins_nodes_ = options_.get_int("mip_rins_nodes");
  pump_rounds_ = static_cast<int>(options_.get_int("mip_pump_rounds"));
  if (heuristics_on_) locks_ = compute_locks(original_);
}

bool BranchAndBound::offer_from(std::size_t slot, const std::vector<double>& x) {
  // offer_incumbent() accepts only a point that is integral, feasible against the ORIGINAL
  // model AND better than the incumbent, so "proposed" and "improved" are the two numbers
  // that can differ: what the heuristic produced, and what survived the check.
  HeuristicStats& s = heuristic_stats_[slot];
  ++s.found;
  if (!offer_incumbent(x)) return false;
  ++s.improved;
  return true;
}

void BranchAndBound::run_node_heuristics(Index node_index, const Solution& relaxation) {
  const std::vector<double>& x = relaxation.col_value;
  {
    // The rounding every node always had, now counted.
    HeuristicStats& s = heuristic_stats_[kRounding];
    const Timer clock;
    ++s.calls;
    const double before = incumbent_internal_;
    const bool had = have_incumbent_;
    try_rounding(x);
    ++s.found;  // try_rounding always proposes its rounding
    if (have_incumbent_ && (!had || incumbent_internal_ < before)) ++s.improved;
    s.seconds += clock.elapsed_seconds();
  }
  if (!heuristics_on_) return;

  // LOCK ROUNDING, every node: O(nonzeros), no LP, and it succeeds exactly where nearest
  // rounding fails for the reason locks see - a covering row rounded the wrong way.
  {
    HeuristicStats& s = heuristic_stats_[kLockRounding];
    const Timer clock;
    ++s.calls;
    (void)offer_from(kLockRounding, lock_round(original_, locks_, integer_columns_, x));
    s.seconds += clock.elapsed_seconds();
  }

  // REPAIR, at the root only: it is the most expensive of the no-LP heuristics, and its
  // value is an early incumbent, which only matters before the tree has found one.
  if (node_index == 0 && !have_incumbent_) {
    HeuristicStats& s = heuristic_stats_[kRepair];
    const Timer clock;
    ++s.calls;
    std::vector<double> candidate = lock_round(original_, locks_, integer_columns_, x);
    Count moves = 0;
    const int budget = static_cast<int>(std::min<std::size_t>(4 * integer_columns_.size() + 10,
                                                              static_cast<std::size_t>(10000)));
    if (repair(original_, integer_columns_, &candidate, budget, tol::kPrimalFeasibility,
               &moves)) {
      (void)offer_from(kRepair, candidate);
    }
    s.work += moves;
    s.seconds += clock.elapsed_seconds();
  }

  // RINS, every mip_rins_frequency nodes once there is an incumbent to compare against, with
  // TWO budgets - a heuristic that out-spends the search it serves is not one. Its sub-MIP
  // nodes are capped in total at a tenth of the main search's (with a floor), and its TIME at
  // a tenth of the solve's. The node cap alone was not enough: on pk1 a sub-MIP node cost
  // seven main nodes, and 601 of them - inside the node cap - took 8.8 s of a 20 s solve.
  if (rins_frequency_ > 0 && have_incumbent_ && node_index > 0 &&
      nodes_explored_ % rins_frequency_ == 0) {
    HeuristicStats& s = heuristic_stats_[kRins];
    const Count budget = std::max<Count>(rins_nodes_, nodes_explored_ / 10);
    const double time_budget = 0.1 * std::max(1.0, timer_.elapsed_seconds());
    if (s.work < budget && s.seconds < time_budget) {
      const Timer clock;
      ++s.calls;
      Model neighbourhood;
      Count fixed = 0;
      if (rins_submodel(original_, integer_columns_, x, incumbent_x_, 0.5,
                        integrality_tolerance_, &neighbourhood, &fixed)) {
        Options sub = options_;
        sub.set_bool("log_to_console", false);
        sub.set_bool("mip_heuristics", false);  // no RINS inside RINS
        sub.set_int("node_limit", std::min<Count>(rins_nodes_, budget - s.work));
        sub.set_bool("pool_complete", false);
        sub.set_int("pool_size", 1);
        // The search's checkpoint is the search's (#287): a sub-MIP stopped at its node cap
        // would otherwise write its own tree over the file, and after a resume every sub-MIP
        // would try to load the search's file and be refused as a different model.
        sub.set_string("checkpoint", "");
        sub.set_string("resume", "");
        // The same for the two other file-valued options (#368, #223): a sub-MIP's profile
        // and progress stream would otherwise be written over the search's.
        sub.set_string("profile_out", "");
        sub.set_string("progress_out", "");
        if (limits_.has_time_limit()) {
          sub.set_double(
              "time_limit",
              std::max(0.0, 0.1 * limits_.remaining_seconds(timer_.elapsed_seconds())));
        }
        Logger quiet(nullptr);
        const Solution found = solve_branch_and_bound(neighbourhood, sub, quiet, control_);
        s.work += found.nodes;
        if (claims_a_point(found.status) && !found.col_value.empty()) {
          (void)offer_from(kRins, found.col_value);
        }
      }
      s.seconds += clock.elapsed_seconds();
    }
  }
}

void BranchAndBound::run_root_dive(const std::vector<double>& x) {
  // The dive (#25) predates this file; it is counted here so the report covers every
  // heuristic the search runs, not only the new ones. It offers its own point.
  HeuristicStats& s = heuristic_stats_[kDive];
  const Timer clock;
  ++s.calls;
  const double before = incumbent_internal_;
  const bool had = have_incumbent_;
  dive_from_root(x);
  if (have_incumbent_ && (!had || incumbent_internal_ < before)) {
    ++s.found;
    ++s.improved;
  }
  s.seconds += clock.elapsed_seconds();
}

void BranchAndBound::run_root_pump(const Solution& relaxation) {
  if (!heuristics_on_ || pump_rounds_ <= 0 || have_incumbent_ || quadratic_) return;
  HeuristicStats& s = heuristic_stats_[kPump];
  const Timer clock;
  ++s.calls;
  Options lp = node_options_;
  lp.set_bool("log_to_console", false);
  lp.set_bool("presolve", false);
  if (limits_.has_time_limit()) {
    lp.set_double("time_limit",
                  std::max(0.0, 0.2 * limits_.remaining_seconds(timer_.elapsed_seconds())));
  }
  Count solves = 0;
  const std::vector<double> x =
      feasibility_pump(original_, integer_columns_, relaxation.col_value, lp, pump_rounds_,
                       integrality_tolerance_, &solves);
  s.work += solves;
  if (!x.empty()) (void)offer_from(kPump, x);
  s.seconds += clock.elapsed_seconds();
}

void BranchAndBound::report_heuristics() {
  if (heuristic_stats_.empty()) return;
  for (const HeuristicStats& s : heuristic_stats_) {
    if (s.calls == 0) continue;
    logger_.info(
        "Heuristic {:<17} calls {:>7}  proposed {:>6}  improved {:>4}  work {:>7}  {:.3f}s",
        s.name, s.calls, s.found, s.improved, s.work, s.seconds);
  }
}

}  // namespace sankhya::mip
