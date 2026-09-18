// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the option registry and its string parser.
//
// THE REGISTRY IS THE ONLY PLACE AN OPTION IS DECLARED. The CLI, the C API and the Python
// bindings all read it. Adding a knob is one row here.
//
// Defaults are taken from include/sankhya/tolerances.hpp so that there is exactly one
// numerical source of truth, per ENGINEERING_RULES.md.

#include "sankhya/options.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <unordered_map>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "sankhya/tolerances.hpp"
#include "sankhya/types.hpp"

namespace sankhya {
namespace {

constexpr double kNoLimit = std::numeric_limits<double>::max();

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(std::string_view s) {
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) --e;
  return std::string(s.substr(b, e - b));
}

/// Index of each option name in the registry, built once.
const std::unordered_map<std::string, std::size_t>& name_index() {
  static const std::unordered_map<std::string, std::size_t> index = [] {
    std::unordered_map<std::string, std::size_t> m;
    const std::vector<OptionSpec>& specs = Options::registry();
    for (std::size_t i = 0; i < specs.size(); ++i) m.emplace(specs[i].name, i);
    return m;
  }();
  return index;
}

/// A typed accessor naming an option that is not in the registry is a programming error in
/// our own code, not bad user input. Fail loudly and identically in Release and Debug: an
/// `assert` alone vanishes under NDEBUG and leaves a genuine out-of-range dereference in the
/// shipped binary, which is exactly the silent-wrong-answer failure mode ENGINEERING_RULES.md
/// warns about. Marking the failure path [[noreturn]] also tells the optimizer the iterator is
/// dereferenceable, which is what clears -Wnull-dereference on GCC 16.
[[noreturn]] void unknown_option_name(const std::string& name) {
  fmt::print(stderr, "sankhya: internal error - unknown option name '{}' in a typed accessor\n",
             name);
  std::abort();
}

std::size_t require_index(const std::string& name) {
  const std::unordered_map<std::string, std::size_t>& index = name_index();
  const auto it = index.find(name);
  if (it == index.end()) unknown_option_name(name);
  return it->second;
}

}  // namespace

const std::vector<OptionSpec>& Options::registry() {
  static const std::vector<OptionSpec> specs = [] {
    std::vector<OptionSpec> s;

    // ---- Termination -------------------------------------------------------------------
    s.push_back({"time_limit",
                 OptionType::Double,
                 kNoLimit,
                 "Wall-clock limit in seconds, measured from the start of solve() and so "
                 "covering presolve, the engine and postsolve, but not reading the model or "
                 "writing the answer. The default is the no-limit sentinel; ZERO IS A BUDGET "
                 "OF ZERO SECONDS, not an absent limit, and stops the solve at its first safe "
                 "point. A solve is stopped between iterations or between nodes, so a "
                 "factorization already running finishes first (#289).",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back({"refactor_work_ratio",
                 OptionType::Double,
                 128.0,
                 "Simplex refactorizes once the eta-file nonzeros summed over iterations since "
                 "the last refactorization exceed this multiple of the base factor size. "
                 "Deterministic; calibrated from measurements on Netlib d2q06c and greenbea "
                 "(#68). 0 refactorizes every iteration.",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back({"iteration_limit",
                 OptionType::Int,
                 std::int64_t{-1},
                 "Simplex/IPM/PDHG iteration limit; -1 for no limit. N means AT MOST N "
                 "iterations, zero included. With pdhg_polish the interior-point finish has "
                 "polish_iteration_limit of its own and the reported count is the sum of "
                 "both phases (#229), which is the one case where it can exceed this (#289).",
                 -1.0,
                 kNoLimit,
                 {}});
    s.push_back({"node_limit",
                 OptionType::Int,
                 std::int64_t{-1},
                 "Branch-and-cut node limit; -1 for no limit. N means AT MOST N nodes, zero "
                 "included. A search stopped this way keeps its incumbent, its bound and its "
                 "gap; one stopped before it found an integer point reports no point and an "
                 "infinite gap rather than a zero one (#289).",
                 -1.0,
                 kNoLimit,
                 {}});

    // ---- Engine selection --------------------------------------------------------------
    s.push_back({"algorithm",
                 OptionType::String,
                 std::string("auto"),
                 "LP engine: auto (a rule-based selection from the model's shape, #284: "
                 "the dual simplex below 20,000 rows and 100,000 nonzeros, where it passes "
                 "80 of 89 Netlib instances; the interior point with crossover above either; "
                 "PDHG from 100,000 rows; a starting basis always means the dual simplex; "
                 "the answer's engine_rule/engine_reason say which rule fired and why), "
                 "simplex (the primal), dual-simplex, pdhg, or ipm (#56: Mehrotra "
                 "predictor-corrector on the normal equations with a sparse LDL^T; produces "
                 "no basis and does not certify infeasibility or unboundedness).",
                 0.0,
                 0.0,
                 {"auto", "simplex", "dual-simplex", "pdhg", "ipm"}});
    s.push_back({"mip_branching",
                 OptionType::String,
                 std::string("reliability"),
                 "Branching rule: reliability (default; #69 - pseudocosts once a column has "
                 "been branched on kPseudocostReliability times in a direction, strong "
                 "branching with capped warm-started dual solves until then, product "
                 "score) or most-fractional (the rule this replaced, kept for comparison).",
                 0.0,
                 0.0,
                 {"reliability", "most-fractional"}});
    s.push_back({"mip_node_selection",
                 OptionType::String,
                 std::string("hybrid"),
                 "Which open node branch and bound takes next (#293): hybrid (default; dive "
                 "to a leaf, then best-bound), best-bound (fewest nodes to a proof, widest "
                 "tree), depth-first (narrow tree, early incumbent, slow bound), or "
                 "best-estimate (the pseudocost guess at where a good incumbent is). Order "
                 "only: every policy explores the same tree and proves the same optimum.",
                 0.0,
                 0.0,
                 {"hybrid", "best-bound", "depth-first", "best-estimate"}});
    s.push_back({"mip_node_engine",
                 OptionType::String,
                 std::string("dual"),
                 "LP engine for branch-and-bound nodes below the root: dual (default) "
                 "warm-starts each child from its parent's optimal basis with the dual "
                 "simplex, which is dual feasible there and typically a few pivots from "
                 "the child's optimum; primal re-solves every node from the slack basis, "
                 "kept so the two can be compared (#65).",
                 0.0,
                 0.0,
                 {"dual", "primal"}});
    s.push_back({"pricing",
                 OptionType::String,
                 std::string("devex"),
                 "Simplex entering-variable rule: devex (default) or dantzig. Devex was "
                 "opt-in while it drove two Netlib medium instances to a singular basis; "
                 "that failure class was removed by #144 and #147, and re-measured on the "
                 "medium tier devex solves the same 49 instances in a third fewer "
                 "iterations and a third less time (#66). Dantzig is kept so the "
                 "comparison can be regenerated.",
                 0.0,
                 0.0,
                 {"devex", "dantzig"}});
    s.push_back({"basis_update",
                 OptionType::String,
                 std::string("product-form"),
                 "How the simplex absorbs a pivot into the basis factorization: product-form "
                 "(default; an eta per pivot, read in full by every later solve) or "
                 "forrest-tomlin (#279; the new column is folded into U and only a row eta "
                 "is kept, Forrest & Tomlin 1972). Both refactorize under the same "
                 "break-even rule. forrest-tomlin is selectable so the A/B on #279 can be "
                 "measured; it becomes the default only when that measurement says so.",
                 0.0,
                 0.0,
                 {"product-form", "forrest-tomlin"}});
    s.push_back({"ratio_test",
                 OptionType::String,
                 std::string("textbook"),
                 "Simplex leaving-variable rule: textbook (default) or harris. Harris (#67) "
                 "relaxes bounds by a controlled amount to pick a larger, more stable pivot "
                 "and adds long-step bound flipping, but measured on the Netlib medium tier "
                 "under Dantzig pricing it trades one instance (grow22) for no reduction in "
                 "singular-basis failures, so it is not the default; see the citation in "
                 "primal_simplex.cpp for the numbers.",
                 0.0,
                 0.0,
                 {"harris", "textbook"}});
    s.push_back({"mps_format",
                 OptionType::String,
                 std::string("auto"),
                 "MPS dialect: auto, free, fixed. auto reads with the whitespace tokenizer "
                 "and retries in fixed columns only if that fails.",
                 0.0,
                 0.0,
                 {"auto", "free", "fixed"}});
    // Default TRUE. Measured: equilibration took the Netlib medium tier from 26/50 to
    // 39/50 by itself. Exposed as an option because turning it off is how a scaling bug
    // gets localised, not because it is optional.
    s.push_back({"scaling",
                 OptionType::Bool,
                 true,
                 "Equilibrate the constraint matrix (Ruiz + Pock-Chambolle) before solving.",
                 0.0,
                 0.0,
                 {}});
    // Implemented as of #43, so there is no planned_for marker any more. Default TRUE for
    // the same reason as scaling: the postsolve round-trip is asserted against the exact
    // rational oracle and re-measured against the ORIGINAL model on every solve, so
    // defaulting it off would mean shipping a deliberately slower solver to dodge a risk
    // the tests already cover.
    s.push_back({"presolve",
                 OptionType::Bool,
                 true,
                 "Run presolve reductions before solving, and postsolve the answer back.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"tree_cut_depth",
                 OptionType::Int,
                 std::int64_t{0},
                 "Deepest tree node at which a cut round runs (#221); 0 keeps cuts at the "
                 "root only. Only read when enable_root_cuts is set. Tree rounds add MIR "
                 "cuts built on the global bounds, so every cut is valid for the whole "
                 "tree and stays as a row; rows slack for 50 node solves are freed.",
                 0.0,
                 1000.0,
                 {}});
    s.push_back({"tree_cut_rows_per_round",
                 OptionType::Int,
                 std::int64_t{20},
                 "Most cut rows one tree cut round may add (#221).",
                 1.0,
                 100000.0,
                 {}});
    s.push_back({"checkpoint",
                 OptionType::String,
                 std::string(""),
                 "Write the branch-and-bound search to this file when a limit stops it, and "
                 "every checkpoint_nodes nodes if that is set (#287). Written atomically: to "
                 "<file>.tmp, then moved over <file>, so a failed write leaves the previous "
                 "checkpoint intact. Empty writes none.",
                 0.0,
                 0.0,
                 {},
                 /*planned_for=*/std::string(""),
                 /*case_sensitive=*/true});
    s.push_back({"checkpoint_nodes",
                 OptionType::Int,
                 std::int64_t{0},
                 "Also write the checkpoint every this many nodes (#287); 0 writes only when a "
                 "limit stops the search.",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back(
        {"resume",
         OptionType::String,
         std::string(""),
         "Continue the branch-and-bound search saved in this checkpoint (#287). Refused, "
         "before anything runs, when it was written for a different model (the model "
         "fingerprint, which presolve settings change too), another format version, "
         "another integrality tolerance, or its checksum does not match. The saved "
         "incumbent is re-verified against the model; node_limit counts the nodes "
         "explored before the checkpoint too.",
         0.0,
         0.0,
         {},
         /*planned_for=*/std::string(""),
         /*case_sensitive=*/true});
    s.push_back({"nlp_tolerance",
                 OptionType::Double,
                 1e-6,
                 "The convex NLP engine's KKT tolerance (#226): primal feasibility relative to "
                 "the bounds, stationarity relative to the gradient, complementarity relative "
                 "to the multipliers and bounds. `optimal` means all three were MEASURED below "
                 "it at the returned point.",
                 1e-12,
                 1e-2,
                 {}});
    s.push_back({"nlp_assume_convex",
                 OptionType::Bool,
                 false,
                 "Run the convex NLP engine on an objective the composition rules cannot prove "
                 "convex (x log x, cross terms), on the caller's word (#226). The answer's "
                 "message then says the convexity was asserted, not proved.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"enable_mir_cuts",
                 OptionType::Bool,
                 true,
                 "Include mixed-integer rounding cuts from the model's own rows in the root "
                 "cut round (#221). Only read when enable_root_cuts is set; exists so the "
                 "MIR family can be measured on its own against the Gomory and cover cuts.",
                 0.0,
                 0.0,
                 {}});
    s.push_back(
        {"enable_root_cuts",
         OptionType::Bool,
         false,
         "Enable root-node cutting planes (Gomory mixed-integer, lifted knapsack cover and, "
         "since #221, mixed-integer rounding; tree_cut_depth adds rounds below the root). "
         "OFF by default, and that is a measurement, not caution: on the 30-instance "
         "MIPLIB set at a 60 s limit at 5e78399 the root round proves the same 9 "
         "instances, takes the node count to 0.896x (0.835x with tree rounds at depth 4), "
         "and costs one published match at the limit, neos-3611689-kaihu (119 without "
         "cuts, 120 with them, both unproved at 60 s). A round that saves nodes without "
         "proving anything more does not earn the default. See "
         "bench/results/miplib-cuts-{off,on,tree}.csv and docs/BENCHMARKS.md section 2.",
         0.0,
         0.0,
         {}});
    s.push_back({"gpu",
                 OptionType::Bool,
                 false,
                 "Use the CUDA backend where one is compiled in (the CLI spells it --gpu); "
                 "otherwise warn once and run on the CPU. No build carries the backend yet "
                 "(#16-#19).",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"ranging",
                 OptionType::Bool,
                 false,
                 "Compute LP sensitivity ranges at optimality and write them to the .sol file "
                 "(the CLI spells it --ranging). For each column, the interval over which its "
                 "cost coefficient can move before the optimal basis changes; for each row, "
                 "the interval over which the active bound can move before the basis becomes "
                 "primal infeasible. Requires the primal simplex to produce a basis.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"threads",
                 OptionType::Int,
                 std::int64_t{1},
                 "Worker threads for the column loops of simplex pricing, the dual's pivot "
                 "row and the sparse transpose product (#57); 0 means one per hardware "
                 "core. Every parallel loop is a gather with no cross-thread reduction, so "
                 "the answer is identical at any thread count. Ignored, with a note in the "
                 "log, in a build without OpenMP.",
                 0.0,
                 1024.0,
                 {}});
    // Registered as planned since Phase 7 and implemented in #288. Its default was `true`
    // while nothing read it; it is `false` now that something does, because the mode REFUSES
    // a wall-clock time_limit and a caller who asked for one must keep getting it.
    s.push_back({"deterministic",
                 OptionType::Bool,
                 false,
                 "Make the solve reproducible: no decision the solver takes is allowed to "
                 "depend on the clock (#288). A wall-clock time_limit is refused rather than "
                 "silently ignored, the interior-point polish is bounded by "
                 "polish_max_factor_nonzeros instead of by seconds, and the column loops run "
                 "on one thread unless threads was set explicitly. Two runs of the same model "
                 "with the same options and the same build then return the same numbers; "
                 "solve_seconds and the log timings still differ, and nothing is claimed "
                 "across compilers or machines. Bound the search with iteration_limit or "
                 "node_limit, which count the same everywhere.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"random_seed",
                 OptionType::Int,
                 std::int64_t{0},
                 "Seed for every randomised decision in the solver. There are three, all of "
                 "them the starting vector of a power iteration that estimates a norm for a "
                 "step size: PDHG's preconditioner (src/pdhg/pdhg.cpp) and the two in the "
                 "convex QP (src/qp/qp_condat_vu.cpp). Nothing in the solver seeds from the "
                 "clock, so a run is reproducible at the default (#288).",
                 0.0,
                 kNoLimit,
                 {}});

    // ---- Tolerances. Defaults come from tolerances.hpp, never from a literal here. -----
    s.push_back({"primal_feasibility_tolerance",
                 OptionType::Double,
                 tol::kPrimalFeasibility,
                 "Max allowed row/column bound violation.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back(
        {"scaled_share",
         OptionType::Double,
         0.5,
         "Fraction of time_limit the scaled simplex attempt may use before the unscaled "
         "retry gets the rest. 0.5 guarantees the retry a real share when the first "
         "attempt fails outright; 1.0 gives the scaled attempt the whole budget, so a "
         "retry runs only on what an early failure leaves and never after a time limit "
         "(#244 measures the two).",
         0.05,
         1.0,
         {}});
    s.push_back({"crossover",
                 OptionType::Bool,
                 true,
                 "After an interior-point solve (algorithm=ipm) reaches optimal, push the "
                 "answer to a vertex with the dual simplex warm-started from a basis guessed "
                 "off the interior point (#219): the result then carries a basis for warm "
                 "starts and sensitivity ranging and every nonbasic variable sits on a bound. "
                 "The pivots get what the time limit has left; if they do not reach an optimum "
                 "the interior point's answer stands. false keeps the interior point's answer "
                 "as it is, for a caller who wants only the objective on a very large model.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"dual_feasibility_tolerance",
                 OptionType::Double,
                 tol::kDualFeasibility,
                 "Max allowed reduced-cost sign violation.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back({"integrality_tolerance",
                 OptionType::Double,
                 tol::kIntegrality,
                 "Max distance from an integer still counted as integral.",
                 1e-12,
                 1e-3,
                 {}});
    s.push_back({"mip_relative_gap",
                 OptionType::Double,
                 tol::kMipRelativeGap,
                 "Stop when the relative MIP gap falls below this.",
                 0.0,
                 1.0,
                 {}});
    s.push_back({"mip_absolute_gap",
                 OptionType::Double,
                 tol::kMipAbsoluteGap,
                 "Stop when the absolute MIP gap falls below this.",
                 0.0,
                 kNoLimit,
                 {}});
    // ---- Solution pool (#225) ----
    s.push_back({"pool_size",
                 OptionType::Int,
                 std::int64_t{10},
                 "MILP/MIQP: how many integer-feasible solutions with different integer "
                 "assignments to keep and report, best first; 0 keeps none.",
                 0.0,
                 1e6,
                 {}});
    s.push_back({"pool_gap",
                 OptionType::Double,
                 kNoLimit,
                 "MILP/MIQP: report only pool members whose objective is within this relative "
                 "gap of the solution (relative to max(1, |objective|)); default keeps all.",
                 0.0,
                 kNoLimit,
                 {}});
    s.push_back({"pool_diversity",
                 OptionType::Bool,
                 false,
                 "MILP/MIQP: when the pool is full, drop the member nearest the others in "
                 "Hamming distance on the integer assignment instead of the worst one.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"pool_complete",
                 OptionType::Bool,
                 false,
                 "MILP/MIQP: keep searching past the optimum until the pool provably holds "
                 "the pool_size best assignments (within pool_gap). Costs nodes; off by "
                 "default, which reports only what the ordinary search found. With "
                 "pool_diversity on it cannot prune and enumerates every integer-feasible "
                 "assignment, so combine the two only on small models or with pool_gap.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"pool_write_all_columns",
                 OptionType::Bool,
                 false,
                 "MILP/MIQP: write every column of each pool member to the .sol file, not "
                 "only the integer ones, so tools/verify_solution.py can check every row of "
                 "every member exactly. Off by default because on a large model the pool "
                 "section would repeat the whole point pool_size times.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"pdhg_restart",
                 OptionType::Bool,
                 true,
                 "Restart PDHG on the KKT-error criterion; off is for evidence runs.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"qp_tolerance",
                 OptionType::Double,
                 1e-8,
                 "Convex QP primal/dual residual termination tolerance.",
                 1e-14,
                 0.1,
                 {}});
    s.push_back({"pdhg_polish",
                 OptionType::Bool,
                 true,
                 "Finish PDHG's point with the interior point when PDHG stops short of the "
                 "standard (#229): its point, row duals and reduced costs become the interior "
                 "point's starting point, with polish_iteration_limit iterations and whatever "
                 "the time limit has left - PDHG is given 70% of a finite time_limit so that "
                 "there is some. The answer's iteration count is the SUM of both phases, its "
                 "algorithm reads pdhg+ipm, and polish_iterations records the second phase. "
                 "false is the unpolished first-order answer.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"polish_iteration_limit",
                 OptionType::Int,
                 std::int64_t{50},
                 "Interior-point iterations the polish may spend (#229).",
                 1.0,
                 kNoLimit,
                 {}});
    s.push_back({"polish_start_margin",
                 OptionType::Double,
                 1e-1,
                 "The interior point's warm start floors every slack and multiplier at this "
                 "value (in the scaled model's units) so its first iterate is interior; the "
                 "residuals absorb the difference and the method removes it. A cold start "
                 "uses 1. Measured on the staircase family at 1,000 and 5,000 rows: 0.1 "
                 "polishes in 7 and 12 iterations against a cold 18 and 24; 0.01 and below "
                 "start too close to the boundary and break the factorization near the end.",
                 1e-8,
                 1.0,
                 {}});
    s.push_back({"polish_max_seconds",
                 OptionType::Double,
                 30.0,
                 "Wall-clock the polish may spend, ordering included; the smaller of this and "
                 "what time_limit has left. On the random scale family at 20,000 rows the "
                 "ordering alone ran for 1,200 s before it could size the factor, for a PDHG "
                 "phase of 1.8 s: past this many seconds the polish is declined and the "
                 "first-order answer stands.",
                 0.1,
                 kNoLimit,
                 {}});
    s.push_back({"polish_max_factor_nonzeros",
                 OptionType::Int,
                 std::int64_t{50000000},
                 "The polish declines when the interior point's factor would hold more "
                 "nonzeros than this - the ordering says so before anything is built; 5e7 "
                 "doubles is 400 MB - and PDHG's answer stands, with the reason in the "
                 "message. -1 for no cap.",
                 -1.0,
                 kNoLimit,
                 {}});
    s.push_back({"ipm_setup_share",
                 OptionType::Double,
                 0.2,
                 "Share of the time limit the interior point may spend on its set-up - the "
                 "ordering and the first factorization of the normal equations - before it "
                 "declines (#357): a set-up that has not finished by then is abandoned with "
                 "status not_solved, not time_limit, so that under algorithm=auto another "
                 "engine runs on the rest of the budget. 1 means no separate budget; no "
                 "effect without a time limit. Measured on the random 20,000-row scale "
                 "shape, where the first factorization (57 million nonzeros) alone ran out "
                 "the whole 120 s with 0 iterations.",
                 0.0,
                 1.0,
                 {}});
    s.push_back({"ipm_max_ordering_entries",
                 OptionType::Int,
                 std::int64_t{100000000},
                 "The interior point gives up the ordering of its normal equations when the "
                 "quotient graph holds more than this many list entries (4 bytes each; 1e8 is "
                 "400 MB), reporting a numerical error with the number rather than running "
                 "the machine out of memory: on the 100,000-row random scale model the "
                 "ordering's fill grew until a std::bad_alloc killed the process 170 s past "
                 "its time limit (#246). -1 for no cap.",
                 -1.0,
                 kNoLimit,
                 {}});
    s.push_back({"ipm_max_factor_nonzeros",
                 OptionType::Int,
                 std::int64_t{100000000},
                 "The interior point declines when the ordering says its factor would hold "
                 "more nonzeros than this (1e8 is 800 MB of values and 400 MB of pattern), "
                 "before any of it is allocated; the message carries both numbers. The polish "
                 "of a first-order answer has its own, tighter cap in "
                 "polish_max_factor_nonzeros. -1 for no cap.",
                 -1.0,
                 kNoLimit,
                 {}});
    s.push_back({"pdhg_tolerance",
                 OptionType::Double,
                 tol::kPdhgLoose,
                 "PDHG relative KKT termination tolerance. By default the loop also runs on "
                 "until the point meets the project's ABSOLUTE tolerances, so a request looser "
                 "than those changes nothing (#180); see pdhg_stop_at_request.",
                 1e-14,
                 1e-1,
                 {}});
    s.push_back(
        {"pdhg_stop_at_request",
         OptionType::Bool,
         false,
         "Stop PDHG once pdhg_tolerance is met and the point is primal-feasible to the "
         "project's absolute tolerance, without waiting for absolute dual feasibility, the "
         "verified gap or complementarity. Absolute primal feasibility is kept because "
         "`feasible` promises a feasible point, so on a model with large row bounds a loose "
         "request may not stop the loop much earlier. The point is reported as `feasible` "
         "unless it meets the full standard anyway; the switch cannot manufacture an "
         "`optimal`. Opt-in, because the default has to be the answer that can be verified "
         "(#180).",
         0.0,
         0.0,
         {}});

    // ---- Reporting ---------------------------------------------------------------------
    s.push_back({"log_level",
                 OptionType::String,
                 std::string("info"),
                 "off, error, warning, info, verbose, debug.",
                 0.0,
                 0.0,
                 {"off", "error", "warning", "info", "verbose", "debug"}});
    s.push_back({"log_to_console",
                 OptionType::Bool,
                 true,
                 "Write the solver log to stdout.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"numerics_report",
                 OptionType::Bool,
                 false,
                 "Print scaling ranges, basis conditioning and refactorization counts.",
                 0.0,
                 0.0,
                 {},
                 "Phase 9"});
    s.push_back({"profile",
                 OptionType::String,
                 std::string("off"),
                 "Where the solve's time goes (#285): off, basic (the phases - presolve, the "
                 "engine, postsolve, verification - and the headline counters) or detailed "
                 "(also the sub-phases inside an engine: the simplex's pricing, ratio test and "
                 "factorizations, the interior point's ordering and factorization, the "
                 "branch and bound's node LPs and branching). The table is written to the log; "
                 "off costs a null-pointer test per scope.",
                 0.0,
                 0.0,
                 {"off", "basic", "detailed"}});
    s.push_back(
        {"profile_out",
         OptionType::String,
         std::string(""),
         "Also write the profile as JSON to this file; empty writes the log table only.",
         0.0,
         0.0,
         {},
         /*planned_for=*/std::string(""),
         /*case_sensitive=*/true});
    s.push_back({"progress_out",
                 OptionType::String,
                 std::string(""),
                 "Append live solve progress as JSON lines to this file; empty disables it.",
                 0.0,
                 0.0,
                 {},
                 /*planned_for=*/std::string(""),
                 /*case_sensitive=*/true});
    s.push_back({"compute_iis",
                 OptionType::Bool,
                 true,
                 "When the model is infeasible and a Farkas certificate was produced, run the "
                 "Chinneck-Dravnieks deletion filter to find the irreducible infeasible "
                 "subsystem (#217). Disable to skip the extra re-solves.",
                 0.0,
                 0.0,
                 {}});
    s.push_back({"write_presolved",
                 OptionType::String,
                 std::string(""),
                 "Write the presolved model to this path before solving it. Extension "
                 "determines format: .lp -> CPLEX LP, anything else -> free MPS. "
                 "Empty disables the dump. Requires presolve: with presolve=false there "
                 "is no presolved model, nothing is written, and a warning says so.",
                 0.0,
                 0.0,
                 {},
                 /*planned_for=*/std::string(""),
                 /*case_sensitive=*/true});
    return s;
  }();
  return specs;
}

Options::Options() {
  const std::vector<OptionSpec>& specs = registry();
  values_.reserve(specs.size());
  for (const OptionSpec& spec : specs) values_.push_back(spec.default_value);
}

const OptionSpec* Options::find_spec(const std::string& name) {
  const auto it = name_index().find(name);
  if (it == name_index().end()) return nullptr;
  return &registry()[it->second];
}

bool Options::exists(const std::string& name) {
  return find_spec(name) != nullptr;
}

const OptionValue& Options::value_of(const std::string& name) const {
  return values_[require_index(name)];
}

OptionValue& Options::mutable_value_of(const std::string& name) {
  return values_[require_index(name)];
}

bool Options::set_from_string(const std::string& raw_name, const std::string& raw_text,
                              std::string* error) {
  const std::string name = to_lower(trim(raw_name));
  const std::string text = trim(raw_text);
  const auto it = name_index().find(name);
  if (it == name_index().end()) {
    if (error != nullptr) *error = fmt::format("unknown option '{}'", raw_name);
    return false;
  }
  const OptionSpec& spec = registry()[it->second];

  switch (spec.type) {
    case OptionType::Bool: {
      const std::string v = to_lower(text);
      if (v == "1" || v == "true" || v == "on" || v == "yes") {
        values_[it->second] = true;
        return true;
      }
      if (v == "0" || v == "false" || v == "off" || v == "no") {
        values_[it->second] = false;
        return true;
      }
      if (error != nullptr) {
        *error = fmt::format("option '{}' expects a boolean, got '{}'", name, raw_text);
      }
      return false;
    }
    case OptionType::Int: {
      errno = 0;
      char* end = nullptr;
      const long long parsed = std::strtoll(text.c_str(), &end, 10);
      if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' expects an integer, got '{}'", name, raw_text);
        }
        return false;
      }
      const auto v = static_cast<double>(parsed);
      if (v < spec.min_value || v > spec.max_value) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be in [{:g}, {:g}], got {}", name,
                               spec.min_value, spec.max_value, parsed);
        }
        return false;
      }
      values_[it->second] = static_cast<std::int64_t>(parsed);
      return true;
    }
    case OptionType::Double: {
      errno = 0;
      char* end = nullptr;
      const double parsed = std::strtod(text.c_str(), &end);
      if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' expects a number, got '{}'", name, raw_text);
        }
        return false;
      }
      if (parsed < spec.min_value || parsed > spec.max_value) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be in [{:g}, {:g}], got {:g}", name,
                               spec.min_value, spec.max_value, parsed);
        }
        return false;
      }
      values_[it->second] = parsed;
      return true;
    }
    case OptionType::String: {
      const std::string v = spec.case_sensitive ? text : to_lower(text);
      if (!spec.choices.empty() &&
          std::find(spec.choices.begin(), spec.choices.end(), v) == spec.choices.end()) {
        if (error != nullptr) {
          *error = fmt::format("option '{}' must be one of [{}], got '{}'", name,
                               fmt::join(spec.choices, ", "), raw_text);
        }
        return false;
      }
      values_[it->second] = v;
      return true;
    }
  }
  if (error != nullptr) *error = "unreachable option type";
  return false;
}

void Options::set_bool(const std::string& name, bool value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<bool>(slot) && "option is not a bool");
  slot = value;
}

void Options::set_int(const std::string& name, std::int64_t value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::int64_t>(slot) && "option is not an int");
  slot = value;
}

void Options::set_double(const std::string& name, double value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<double>(slot) && "option is not a double");
  slot = value;
}

void Options::set_string(const std::string& name, const std::string& value) {
  OptionValue& slot = mutable_value_of(name);
  assert(std::holds_alternative<std::string>(slot) && "option is not a string");
  const OptionSpec* spec = find_spec(name);
  slot = (spec != nullptr && spec->case_sensitive) ? value : to_lower(value);
}

bool Options::get_bool(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<bool>(slot) && "option is not a bool");
  return std::get<bool>(slot);
}

std::int64_t Options::get_int(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<std::int64_t>(slot) && "option is not an int");
  return std::get<std::int64_t>(slot);
}

double Options::get_double(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<double>(slot) && "option is not a double");
  return std::get<double>(slot);
}

const std::string& Options::get_string(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  assert(std::holds_alternative<std::string>(slot) && "option is not a string");
  return std::get<std::string>(slot);
}

bool Options::is_modified(const std::string& name) const {
  const std::size_t i = require_index(name);
  return values_[i] != registry()[i].default_value;
}

std::vector<std::string> Options::modified_names() const {
  std::vector<std::string> names;
  const std::vector<OptionSpec>& specs = registry();
  for (std::size_t i = 0; i < specs.size(); ++i) {
    if (values_[i] != specs[i].default_value) names.push_back(specs[i].name);
  }
  return names;
}

std::string Options::value_as_string(const std::string& name) const {
  const OptionValue& slot = value_of(name);
  return std::visit(
      [](const auto& v) -> std::string {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) {
          return v ? "true" : "false";
        } else if constexpr (std::is_same_v<T, std::string>) {
          return v;
        } else if constexpr (std::is_same_v<T, double>) {
          return v >= kNoLimit ? std::string("inf") : fmt::format("{:g}", v);
        } else {
          return fmt::format("{}", v);
        }
      },
      slot);
}

}  // namespace sankhya
