// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the bounded-variable revised simplex: shared state for the primal and dual
// methods. PRIVATE to src/simplex; the public entry points are in primal_simplex.hpp.
//
// The class carries everything both methods need - the working problem in [A | -I] form,
// the basis and its LU factors with the product-form update, refactorization with the
// Markowitz ladder and basis repair, scaling, perturbation, reporting - and each method is
// one driver loop over that state. primal_simplex.cpp defines the shared machinery and the
// primal loop; dual_simplex.cpp defines the dual loop (#65). The design notes that
// justify each constant and member are kept with them below, where they were written.
#pragma once

#include "primal_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "sankhya/timer.hpp"
#include "sankhya/tolerances.hpp"

#include "../la/lu.hpp"
#include "../la/scaling.hpp"

namespace sankhya::detail {

/// Consecutive zero-length steps tolerated before the solve is declared stalled. Twenty
/// times the Bland switch threshold: long enough that no honest degenerate plateau trips
/// it, short enough that a genuine cycle is reported in under a second.
constexpr int kStallLimit = 20 * tol::kBlandSwitchIterations;

/// Largest amount a bound is relaxed by while escaping a degenerate stall (#51).
///
/// MUST STAY WELL UNDER kPrimalFeasibility. Relaxing a bound outward means a point feasible
/// for the perturbed problem can violate the TRUE bound by up to this much, and the
/// perturbation is removed before optimality is reported - so what matters is that the
/// leftover violation is below the tolerance the answer is judged against. At 1e-9 against a
/// 1e-7 feasibility tolerance there are two orders of margin, and phase 1 does not even
/// re-engage on the restored bounds.
/// Most basis repairs a single solve may make (#34, #174).
///
/// A repair moves the current point, so it can hand the search a basis that goes singular
/// again a few pivots later, and repairing THAT one costs another move. Measured on pilot4
/// before the size guard existed, the ungated repair fired 202 times and never terminated -
/// a fast clean failure turned into a hang, which is strictly worse than the failure.
///
/// THE NUMBER USED TO BE 8, AND 8 WAS THE WRONG QUESTION. What distinguishes a rescue from a
/// hang is not how many repairs a solve makes but whether it gets anywhere between them.
/// Measured on Mittelmann's qap15 (6,330 x 22,275, bench/results/mittelmann-592aea3.csv):
/// nine singular bases between iterations 12,355 and 13,954, every one of them a single
/// dependent column, separated by 39, 27, 14, 65, 283, 304, 639 and 228 iterations of
/// ordinary progress - a solve being carried by the repair, killed on the ninth because the
/// count had run out. The pilot4 runaway looks nothing like that: it was the same enormous
/// repair over and over with no progress at all, and the size guard below already refuses it.
///
/// So the ceiling is high enough not to end a solve that is still moving, and the stall guard
/// beside it is what actually stops a loop.
constexpr Count kMaxBasisRepairs = 64;

/// Repairs allowed without the solve advancing in between (#174).
///
/// A repair moves the point, so the next iteration should be able to pivot. Two repairs at
/// the same iteration mean it could not, and a third is a loop rather than a rescue - which
/// is the shape the pilot4 hang actually had. This is the guard that bounds the work; the
/// count above is a backstop.
constexpr Count kMaxStalledBasisRepairs = 3;

constexpr double kPerturbationSize = 1e-9;

/// Refinement steps on the final basis (#72). Each is two triangular solves; the residual
/// usually drops by the factor's condition margin per step and stops improving by the third.
constexpr int kMaxRefinementSteps = 3;
constexpr int kPerturbationTrigger = kStallLimit / 2;

/// Deterministic per-variable shift in (0, kPerturbationSize].
///
/// Deterministic and not random: CLAUDE.md's evidence rules are worth nothing if a rerun of
/// the same commit on the same instance can take a different path. A fixed hash of the index
/// gives every variable a DIFFERENT shift, which is the property that actually breaks the
/// ties, without making the run irreproducible.
[[nodiscard]] inline double perturbation_for(Index k) noexcept {
  // UINT64_C and not a ULL suffix: uint64_t is unsigned long on Linux and unsigned long long
  // on Windows, and mixing the two trips -Werror=sign-conversion on one platform only. CI is
  // the authority on -Werror cleanliness, and it duly said so.
  const auto mixed =
      static_cast<std::uint64_t>(k) * UINT64_C(2654435761) + UINT64_C(1013904223);
  const double unit = static_cast<double>(mixed % UINT64_C(1000003)) / 1000003.0;
  return kPerturbationSize * (0.25 + 0.75 * unit);
}

/// How far above the feasibility tolerance a phase-1 stall has to sit before it is reported
/// as a genuine infeasibility rather than a numerical stall.
///
/// There is no principled value: the honest position is that a floating-point stall proves
/// nothing either way, and this only decides which of two imperfect answers is less
/// misleading. 1e3 keeps kInfeasible for residuals that are large in absolute terms while
/// refusing to make a definitive claim about a point that is nearly feasible. Netlib grow15
/// and grow22 stalled at 1.06e-07 and 1.31e-07 against a 1e-07 tolerance - a factor of 1.3 -
/// and both have published optima.
constexpr double kInfeasibilityProofFactor = 1e3;

/// How often the updated factorization is checked against the basis it claims to represent,
/// and how much relative residual is tolerated before it is rebuilt.
///
/// The check costs one sparse mat-vec over the basis columns, which is the same order as the
/// FTRAN it verifies, so it is amortised over an interval rather than run every pivot.
constexpr Count kAccuracyCheckInterval = 16;

/// Restart the devex reference framework once any weight passes this.
///
/// The weights approximate steepest-edge norms measured from the basis the framework was
/// last reset in, and they only grow. Large values mean the approximation has drifted far
/// from what it is approximating, not that the column is genuinely bad, so continuing to
/// price on them re-creates the problem devex exists to solve. Forrest and Goldfarb restart
/// on this test; 1e6 is their suggested order and is where a reset costs one sweep of ones
/// against pivots that are no longer being chosen on meaningful information.
constexpr double kDevexResetThreshold = 1e6;

/// Reset the reference framework when the entering column's weight has drifted this far from
/// its exact steepest-edge norm.
///
/// DEVEX'S WEIGHTS ARE A LOWER BOUND: w_j <= gamma_j = 1 + ||B^-1 a_j||^2 always holds if
/// the update is sound, and the approximation is only useful while it stays near gamma.
/// Nothing was checking that. The absolute cap above fires at 1e6, which says nothing about
/// accuracy - a weight of 1e5 is fine beside a gamma of 1e5 and catastrophic beside a gamma
/// of 1.
///
/// MEASURED, by recomputing gamma exactly for every nonbasic column on scsd8 (#66). The
/// ratio w/gamma starts inside [0.89, 0.96] and climbs to 2.3e+02 by iteration 200 and
/// 3.7e+03 in the following solve. An inflated weight makes d^2/w rank a good column as a
/// bad one, so pricing steadily loses the information it is supposed to be using, and the
/// bases it then chooses are the ones that decay - which is the singular basis #66 was
/// chasing.
///
/// The test is nearly free: the entering column's alpha = B^-1 a_q is already computed for
/// the ratio test, so gamma_q costs one dot product, and it is checked on one column per
/// iteration rather than all of them. A factor of 4 is loose enough not to thrash the
/// A factor of 1.5 is what the measurement chose: at 2.0 and above scsd8 still fails,
/// at 1.5 and 1.05 it solves, and the committed small set costs 828 iterations either
/// way against 830 without the check - so the tighter test is free on healthy models.
constexpr double kDevexAccuracyFactor = 1.5;

/// Tied to kPrimalFeasibility rather than chosen independently, because that is the quantity
/// this check ultimately protects: factors whose residual is below the feasibility tolerance
/// cannot corrupt a feasibility judgement made at that tolerance.
///
/// It was 1e-9 when this check was written, which is inside the ordinary accumulated rounding
/// of a sequence of triangular solves rather than evidence that the factors have stopped
/// representing the basis. At 1e-9 it fired twice on grow22, and because a forced
/// refactorization changes which row wins the ratio test, those two rebuilds moved the final
/// vertex from a primal infeasibility of 5.652e-08 to 6.244e-07 - across the reported
/// tolerance, turning a published optimum into a numerical_error. A factorization that has
/// genuinely lost its basis misses by orders of magnitude more than this, so the looser
/// threshold keeps every case the check exists to catch.
constexpr double kUpdateAccuracyTolerance = tol::kPrimalFeasibility;

/// How a basic variable sits relative to its own bounds. Phase 1 exists to empty the two
/// outer categories.
enum class Position { kBelowLower, kFeasible, kAboveUpper };

/// Outcome of the ratio test.
struct RatioResult {
  double step = 0.0;
  Index leaving_position = -1;  ///< index into basis_, or -1 for a bound flip
  bool leaving_to_upper = false;
  bool unbounded = false;
};

/// What the dual ratio test decided (dual_simplex.cpp).
struct DualRatioResult {
  Index entering = -1;          ///< column to enter, or -1 when the flips alone were the step
  std::vector<Index> flips;     ///< boxed nonbasic columns to move to their other bound first
  bool dual_unbounded = false;  ///< no candidate at all: the primal is infeasible
};

/// Which loop drives the shared state.
enum class Engine { kPrimal, kDual };

class Simplex {
 public:
  Simplex(const Model& model, const Options& options, Logger& logger)
      : model_(model), options_(options), logger_(logger) {}

  /// The primal simplex, from the slack basis or from `warm`.
  Solution run(const WarmStart* warm = nullptr);
  /// The dual simplex (#65), from the slack basis or from `warm`; hands over to the primal
  /// loop when the dual cannot finish honestly (artificial bounds active, or a stall).
  Solution run_dual(const WarmStart* warm = nullptr);

  /// Shift every nonbasic cost in the direction that keeps its reduced cost dual feasible,
  /// so the dual ratio test stops tying. Recomputes the reduced costs.
  void perturb_costs();
  /// Put the exact costs back and recompute the reduced costs. Safe to call when inactive.
  void remove_cost_perturbation();

 private:
  /// Options, the working problem, the starting basis and its first factorization. Returns
  /// a finished Solution when nothing can start (a singular slack basis), else nothing.
  [[nodiscard]] std::optional<Solution> prepare(const WarmStart* warm, const Timer& timer);
  /// Install a basis from statuses. Returns false, having touched nothing, when the
  /// statuses do not describe m basic variables.
  [[nodiscard]] bool seed_basis(const WarmStart& warm);
  /// The primal iteration loop. `iterations` counts on from what the caller has already
  /// spent, so a dual-then-primal solve reports one total.
  Solution primal_loop(Timer& timer, Count* iterations);

  // ---- dual simplex (dual_simplex.cpp) ---------------------------------------------------
  /// The dual iteration loop. Empty when the basis must be handed to primal_loop(): the
  /// artificial bounds have already been removed and the basis is in place.
  [[nodiscard]] std::optional<Solution> dual_loop(Timer& timer, Count* iterations);
  /// Flip or box every nonbasic column whose reduced cost has the wrong sign, so the basis
  /// is dual feasible. Recomputes the basic values.
  void make_dual_feasible();
  /// Restore every artificial bound to the model's own and move the columns parked on one.
  void remove_artificial_bounds();
  [[nodiscard]] bool any_artificial_bound_active() const;
  [[nodiscard]] bool any_artificial_bound() const;
  /// Dual pricing: the basic slot with the largest weighted primal infeasibility, or -1.
  [[nodiscard]] Index choose_leaving_row() const;
  /// rho_ = B^-T e_r and pivot_row_[j] = rho . a_j for every nonbasic column j.
  void compute_pivot_row(Index leaving_slot);
  /// The bounded dual ratio test with bound flipping (Maros ch. 10; Koberstein 2005).
  [[nodiscard]] DualRatioResult dual_ratio_test(Index leaving_slot,
                                                bool leaving_to_upper) const;
  void reset_dual_weights();
  /// Dual devex: fold the pivot into the row weights, from alpha_ (the entering column).
  void update_dual_weights(Index leaving_slot, double pivot);

  void build_working_problem();
  void set_initial_basis();

  /// Rebuild and refactorize the basis matrix from scratch. Returns false when singular.
  [[nodiscard]] bool refactorize();

  /// Whether the direction the ratio test called unbounded is a genuine ray of the feasible
  /// region: [A | -I] d must vanish. Returns the worst residual relative to its terms.
  [[nodiscard]] double unbounded_ray_residual(Index entering, int direction) const;

  /// Replace the linearly dependent basis columns with logicals, making the basis
  /// nonsingular by construction. Returns false when the defect cannot be located.
  [[nodiscard]] bool repair_basis();

  /// Park a variable on whichever of its bounds it should hold while nonbasic. `current`
  /// is its value before it left the basis, used only to pick the nearer of two bounds.
  void make_nonbasic(Index k, double current);

  /// Relax every finite bound slightly, so a degenerate vertex stops being degenerate.
  void perturb_bounds();

  /// Put the true bounds back. Called before optimality can be reported.
  void remove_perturbation();

  /// ITERATIVE REFINEMENT OF THE FINAL BASIS (#72). Wilkinson, "Rounding Errors in
  /// Algebraic Processes" (1963), ch. 4; the residual is accumulated with the error-free
  /// transformations of Ogita, Rump & Oishi, "Accurate sum and dot product", SIAM J. Sci.
  /// Comput. 26 (2005), so it is correct to roughly double working precision, which is what
  /// makes a refinement step recover digits rather than reshuffle them.
  ///
  /// Both systems, from the same factors: the primal B x_B = -N x_N and the dual
  /// B^T y = c_B. Refining one without the other produces a point whose reduced costs no
  /// longer describe it - the primal-only version of this regressed ganges. Stops when the
  /// residual stops shrinking or after kMaxRefinementSteps; never changes a status. Fills
  /// refinement_steps_ and the two residuals for the log and the stats JSON.
  void refine_final_basis();
  Count refinement_steps_ = 0;
  double residual_before_refinement_ = 0.0;
  double residual_after_refinement_ = 0.0;

  /// x_B = B^{-1} (-N x_N). Recomputed from the bounds every iteration rather than updated,
  /// so no round-off accumulates across pivots.
  void compute_basic_values();

  [[nodiscard]] Position position_of(Index basic_slot) const;
  /// Largest single bound violation over the basic variables. THIS is the feasibility test.
  ///
  /// The two must not be confused. kPrimalFeasibility is documented as "max allowed
  /// row/column bound violation" and Solution::recompute_quality() measures exactly that, so
  /// comparing the SUM against it silently demands a per-row violation of tolerance/m: the
  /// larger the model, the stricter the requirement. On Netlib grow15 (300 rows) that made a
  /// point with a max violation of 0.000e+00 - feasible by the project's own measurement -
  /// fail a test reading 1.062e-07, and phase 1 then reported the model INFEASIBLE.
  [[nodiscard]] double max_infeasibility() const;

  /// Fill cost_basic_ with the phase-1 gradient or the phase-2 costs, then BTRAN for y and
  /// price every nonbasic column into reduced_cost_.
  void compute_reduced_costs(bool phase_one);

  /// Choose an entering column. Returns -1 when none is eligible.
  [[nodiscard]] Index price(bool bland, int* direction) const;

  /// Reset every reference weight to 1, restarting the reference framework.
  void reset_devex();

  /// Fold this pivot into the reference weights. Needs the leaving ROW of B^-1 A, which is
  /// one BTRAN plus a pass over the nonbasic columns - the same shape as the reduced-cost
  /// computation, and the price devex pays for not doing a solve per candidate.
  void update_devex_weights(Index entering, Index leaving_row, double pivot);

  void ftran_entering_column(Index entering);

  /// Relative residual of the claimed FTRAN result: || B alpha - a ||_inf / || a ||_inf,
  /// with B taken from the CURRENT basis columns and alpha from the updated factors.
  ///
  /// This is the direct measurement of the thing that actually matters - whether the factors
  /// plus their eta file still represent the basis - and it replaces inferring conditioning
  /// from whether the Markowitz ladder happened to fire.
  [[nodiscard]] double ftran_residual(Index entering) const;

  /// Dispatches to whichever rule `ratio_test_` selects.
  [[nodiscard]] RatioResult ratio_test(Index entering, int direction, bool phase_one) const;
  /// The textbook rule: the single tightest step, tie-broken by pivot magnitude within
  /// kRatioTestFeasibility. Default - see ratio_test_ for why.
  [[nodiscard]] RatioResult ratio_test_textbook(Index entering, int direction,
                                                bool phase_one) const;
  /// Harris's two-pass rule with long-step bound flipping (issue #67). Opt-in - see
  /// ratio_test_ for why.
  [[nodiscard]] RatioResult ratio_test_harris(Index entering, int direction,
                                              bool phase_one) const;

  /// Iterate over the entries of column k of [A | -I].
  template <typename Fn>
  void for_each_entry(Index k, Fn&& fn) const {
    if (k < n_) {
      const ColumnView column = model_.matrix.column(k);
      for (Index p = 0; p < column.size; ++p) fn(column.rows[p], column.values[p]);
    } else {
      fn(k - n_, -1.0);
    }
  }

  [[nodiscard]] double variable_value(Index k) const {
    const Index slot = basis_position_[static_cast<std::size_t>(k)];
    return slot >= 0 ? x_basic_[static_cast<std::size_t>(slot)]
                     : nonbasic_value_[static_cast<std::size_t>(k)];
  }

  /// The ray the ratio test proved unblocked, over the STRUCTURAL columns: the entering
  /// variable moves by `direction` and each basic variable by -direction * alpha. This is the
  /// same vector unbounded_ray_residual() assembles to check itself; #191 keeps it.
  [[nodiscard]] std::vector<double> unbounded_ray(Index entering, int direction) const;

  [[nodiscard]] double minimization_objective() const;
  Solution finish(SolveStatus status, const std::string& message, Count iterations,
                  double seconds);

  const Model& model_;
  const Options& options_;
  Logger& logger_;

  Index n_ = 0;
  Index m_ = 0;
  Index total_ = 0;

  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<double> cost_;  ///< always in MINIMIZATION sense

  std::vector<Index> basis_;
  std::vector<Index> basis_position_;  ///< -1 when nonbasic
  std::vector<BasisStatus> status_;
  std::vector<double> nonbasic_value_;

  /// PERTURBATION FOR DEGENERACY (#51; Maros, "Computational Techniques of the Simplex
  /// Method", ch. 9, and the bound-shifting scheme every production code uses).
  ///
  /// A degenerate vertex has more active constraints than dimensions, so the ratio test ties
  /// and the step is zero. Anti-cycling rules ARBITRATE those ties; perturbation REMOVES
  /// them, by moving each bound a different tiny amount so no two can be active at once.
  ///
  /// Measured on tuff, which is why this exists: it does not terminate under Bland's rule at
  /// 1000, 10000, 50000 or 200001 consecutive degenerate iterations. Implementing Bland
  /// correctly - lowest index on the LEAVING variable too, which our ratio test does not do -
  /// breaks the cycle and produces a singular basis instead, and takes wood1p down with it.
  /// The two properties a tie-break must supply, termination and conditioning, want opposite
  /// things from the same choice. Perturbation sidesteps the conflict.
  bool perturbed_ = false;
  Count perturbations_ = 0;

  /// Cost perturbation in the dual loop (see kDualCostPerturbation in dual_simplex.cpp).
  /// unperturbed_cost_ holds the exact minimisation-sense costs while it is active.
  bool cost_perturbed_ = false;
  std::vector<double> unperturbed_cost_;
  Count cost_perturbations_ = 0;

  /// Number of basis columns swapped for logicals to escape a singular basis (#34).
  Count repaired_columns_ = 0;
  Count repairs_ = 0;
  /// The iteration the last repair happened at, and how many have happened since the solve
  /// last advanced. See kMaxStalledBasisRepairs.
  Count last_repair_iteration_ = 0;
  Count stalled_repairs_ = 0;
  bool has_repaired_ = false;

  /// ADAPTIVE REFACTORIZATION (#68, redirected). Eta-file nonzeros summed over every
  /// iteration since the last refactorization: a deterministic proxy for the extra solve
  /// work the updates have cost. The simplex refactorizes when it exceeds
  /// refactor_work_ratio_ times the size of the base factors - the break-even between "keep
  /// updating" and "start fresh", in operation counts rather than seconds. See the trigger
  /// for why it must not be seconds.
  double eta_work_since_refactor_ = 0.0;
  double refactor_work_ratio_ = 128.0;

  SparseLu lu_;

  /// Reused across refactorizations so the hot path allocates nothing. Structural columns
  /// point straight into the model's CSC arrays - no copy at all - while logical columns are
  /// the single entry -1 in their own row, served from the two buffers below.
  std::vector<LuColumn> basis_columns_;
  std::vector<Index> logical_rows_;
  std::vector<double> logical_values_;

  /// Reported once per solve, not once per refactorization.
  bool warned_about_threshold_ = false;

  /// Latched by refactorize() when the Markowitz ladder climbed past its default.
  bool basis_needed_stricter_threshold_ = false;

  /// Effort counters for the solve log. rejected_updates_ is the interesting one: a basis
  /// that keeps producing unsafe pivots is badly conditioned, and that is worth seeing.
  Count refactorizations_ = 0;
  double worst_basis_pivot_ = 0.0;  ///< smallest pivot over every factorization
  Count iterations_seen_ = 0;       ///< for the per-refactorization log line only
  Count rejected_updates_ = 0;
  Count accuracy_refactorizations_ = 0;
  /// A certificate the loop has just computed, handed to finish() rather than recomputed.
  /// Both are in the units of the model this engine was given, which may be a scaled one;
  /// solve_with_scaling() maps them back the same way it maps the duals and the point (#191).
  std::vector<double> pending_farkas_;
  std::vector<double> pending_ray_;

  std::vector<double> x_basic_;
  std::vector<double> cost_basic_;
  std::vector<double> y_;
  std::vector<double> reduced_cost_;
  std::vector<double> alpha_;

  // ---- Devex pricing -------------------------------------------------------------------
  //
  // Forrest, J.J. and Goldfarb, D. (1992), "Steepest-edge simplex algorithms for linear
  // programming", Mathematical Programming 57, 341-374; the approximation itself is Harris,
  // P.M.J. (1973), "Pivot selection methods of the Devex LP code", Mathematical Programming
  // 5, 1-28.
  //
  // WHAT DANTZIG GETS WRONG. Pricing on |d_j| alone asks which column improves the objective
  // fastest PER UNIT STEP IN THAT VARIABLE, but the step actually taken is set by the ratio
  // test, and that is governed by the size of the FTRAN'd column B^-1 a_j. A column with a
  // large reduced cost and a large ||B^-1 a_j|| buys almost nothing per pivot, and Dantzig
  // picks it again and again. Steepest edge divides by that norm exactly, which costs a
  // solve per candidate. Devex approximates the norm with reference weights updated in O(m)
  // from vectors this iteration already computes, and prices on d_j^2 / w_j.
  //
  // Measured before this landed: 1147 simplex iterations against HiGHS's 531 across the
  // committed Netlib set, worst 4.21x on blend. See issue #66 for the table.
  //
  // DEVEX IS THE DEFAULT, since the basis chain changed under it. It was opt-in while it
  // drove grow22 and scsd8 to "basis became singular" - the weights were audited and one
  // real defect fixed (kDevexAccuracyFactor), and grow22 still failed at iteration 679; #67
  // (Harris) did not rescue it. The failure class itself was then removed by #144 (the
  // Markowitz search no longer declares a basis singular for want of budget) and #147
  // (rank-deficient bases are repaired rather than abandoned), and the comment that stood
  // here said this should be re-measured the moment the conditioning chain changed. It was:
  //
  //   Netlib medium tier, 120 s, dba3a65 (#160 + #162), one option changed per run
  //     dantzig           46/50 PASS, 49 optimal, 0 failures, 51968 iterations, 191.0 s
  //     devex             46/50 PASS, 49 optimal, 0 failures, 34580 iterations, 128.0 s
  //     devex + harris    46/50 PASS, 49 optimal, 0 failures, 38352 iterations, 128.5 s
  //     dantzig + harris  46/50 PASS, 49 optimal, 0 failures, 51547 iterations, 274.8 s
  //   (iterations and time over the 49 instances every rule solves; d6cube times out under
  //   all four; grow22 and scsd8 solve under devex in 1324 and 1790 iterations)
  //
  // Same answers, a third fewer iterations, a third less time, and the extra BTRAN per
  // iteration is paid for on every instance where it matters. Dantzig stays selectable so
  // this table can be regenerated (bench/results/netlib-medium-dba3a65-*.csv).
  bool devex_ = true;  ///< pricing=dantzig selects the old rule; see #66 and ratio_test_ below
  std::vector<double> devex_weight_;
  std::vector<double> rho_;  ///< B^-T e_r, scratch: rho . a_j gives the leaving row's alpha_rj
  Count devex_resets_ = 0;

  // ---- Ratio test (issue #67) ------------------------------------------------------------
  //
  // Harris, P.M.J. (1973), "Pivot selection methods of the Devex LP code", Mathematical
  // Programming 5, 1-28.
  //
  // MEASURED, Netlib medium tier, Dantzig pricing (the default) both ways: the textbook rule
  // passes 41/50; Harris passes 40/50, trading grow22 (was optimal, primal infeasibility
  // 8.904e-07 under Harris - just over kPrimalFeasibility) for no singular-basis win at all.
  // The three singular-basis failures (d6cube, grow15, pilot4) are IDENTICAL under both
  // rules; Harris relaxes ratio-test ties, and none of these three fail on a tie. Tightening
  // kHarrisRelaxation by 10x (0.01 * kPrimalFeasibility instead of 0.1x) does not change
  // grow22's outcome either - the regression comes from pass two choosing a different pivot
  // altogether on this instance, not from the size of the relaxation.
  //
  // This matches what #67's own comment thread already found when Harris was first tried
  // against devex ("measures null") - it is not a devex-specific interaction, it reproduces
  // under plain Dantzig too. The textbook rule stays the default so the medium pass rate does
  // not drop (CLAUDE.md); Harris is implemented, cited, tested and selectable
  // (--option ratio_test=harris) so this can be re-measured the moment something else in the
  // basis-conditioning chain changes, without reimplementing it from scratch.
  bool harris_ratio_test_ = false;
  double primal_tolerance_ = tol::kPrimalFeasibility;
  double dual_tolerance_ = tol::kDualFeasibility;
  double time_limit_ = std::numeric_limits<double>::infinity();
  std::int64_t iteration_limit_ = -1;
  bool warm_started_ = false;
  std::string algorithm_name_ = "simplex-primal";  ///< what finish() reports

  // ---- Dual simplex state (#65) ---------------------------------------------------------
  //
  // Forrest, J.J. and Goldfarb, D. (1992), "Steepest-edge simplex algorithms for linear
  // programming", Mathematical Programming 57, 341-374, sec. 3 - the dual devex weights.
  // Koberstein, A. (2005), "The dual simplex method, techniques for a fast and stable
  // implementation", PhD thesis, Paderborn - the bound-flipping ratio test and the
  // artificial-bound treatment of dual infeasibility.
  //
  // The dual simplex keeps every reduced cost sign-admissible and drives the basic values
  // into their bounds; a column whose reduced cost has the wrong sign and has no other bound
  // to sit at is given an ARTIFICIAL one so that it can. Those bounds are bookkept here and
  // removed before any answer is reported: a point at an artificial bound is a point of a
  // different problem, and the primal loop finishes from that basis instead.
  std::vector<double> dual_weight_;     ///< dual devex weight per basic slot
  std::vector<double> pivot_row_;       ///< row r of B^-1 [A | -I] over every column
  std::vector<char> artificial_lower_;  ///< lower_[k] is an artificial bound
  std::vector<char> artificial_upper_;  ///< upper_[k] is an artificial bound
  Count dual_iterations_ = 0;
  Count bound_flips_ = 0;
  Count dual_weight_resets_ = 0;
};

/// The scaled solve with its unscaled retry, for either engine. Defined in
/// primal_simplex.cpp, where the portfolio logic and its evidence live.
[[nodiscard]] Solution solve_with_scaling(const Model& model, const Options& options,
                                          Logger& logger, const NodeScaling& cache,
                                          Engine engine, const WarmStart* warm);

/// One row's candidate breakpoint, gathered in pass one of the Harris test and re-examined
/// in pass two.
struct RatioCandidate {
  Index slot;
  double exact_step;
  bool to_upper;
  double pivot_magnitude;
};

}  // namespace sankhya::detail
