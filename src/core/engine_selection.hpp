// SPDX-License-Identifier: Apache-2.0
// SANKHYA - which LP engine `algorithm=auto` means (#284): a rule-based selection from
// the model's shape, returned as a structured decision with the rule and the measurement
// behind it, so the choice is explainable and reproducible rather than a threshold buried
// in the dispatcher.
//
// The rules encode MEASUREMENTS, not intuition, and every threshold below names the CSV it
// comes from. Where the evidence does not reach, the rule says so in its reason. No machine
// learning; the first version is deliberately a short table anyone can read.
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {

/// The engines the selection can name. Strings, not an enum, because they are the same
/// words the `algorithm` option takes and the answer's `algorithm` field reports.
struct EngineSelection {
  std::string algorithm;  ///< "dual-simplex", "simplex", "ipm" or "pdhg"
  std::string rule;       ///< short tag of the rule that fired, e.g. "requested", "size:ipm"
  std::string reason;     ///< one sentence naming the statistics and the measurement
  Index rows = 0;
  Index columns = 0;
  Count nonzeros = 0;
  bool warm_start = false;
};

/// Thresholds, each tied to a row of `docs/BENCHMARKS.md`:
///
/// - kDualSimplexRowLimit: below this many rows the dual simplex is the measured default:
///   80 of 89 on the Netlib full set (`netlib-full-b3f1660.csv`), whose largest instance
///   has 6,071 rows; at 20,000 rows on both generated shapes it hits the 120 s limit where
///   the interior point solves the staircase in 19 s (`scale-staircase-e134aeb.csv`).
///   The 5,000-row generated instances sit BELOW this limit and go to the dual simplex,
///   which times out on them while the interior point solves them; the limit is set where
///   the Netlib evidence ends, not where the generated evidence begins, because dfl001
///   (6,071 rows) solves in 47 s on the dual and not at all on the interior point at 120 s.
/// - kIpmNonzeroFloor: a dense model below the row limit goes to the interior point:
///   maros-r7 (3,136 rows, 144,848 nonzeros) times out on the dual simplex and solves in
///   14 s on the interior point with crossover; no Netlib instance under 100,000 nonzeros
///   is measured to prefer it (fit2p, 50,284 nonzeros, times out on it).
/// - kPdhgRowFloor: from 100,000 rows the direct factorization is out of reach
///   (`scale-e134aeb.csv`: the interior point produces no output on the 100,000-row random
///   shape and hits the limit on the staircase; PDHG reaches the optimum to 1e-7 on both),
///   and the 779,640-row refinery year is reached only by PDHG
///   (`scale-refinery-e134aeb.csv`).
inline constexpr Index kDualSimplexRowLimit = 20000;
inline constexpr Count kIpmNonzeroFloor = 100000;
inline constexpr Index kPdhgRowFloor = 100000;

/// Decide the LP engine for `model` under `options`. An explicit `algorithm` other than
/// "auto" is honoured as given (rule "requested"); a starting basis forces a simplex
/// (rule "warm-start"), because only a simplex can use one.
[[nodiscard]] EngineSelection select_engine(const Model& model, const Options& options,
                                            bool warm_start);

}  // namespace sankhya
