// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact simplex over the rationals. TESTS ONLY.
//
// This is the standard the floating-point simplex is judged against, so it is written for
// transparency rather than speed: a full dense tableau, reduced costs recomputed from
// scratch every iteration, and Bland's rule throughout. Bland's rule is slow and it is the
// point - it is the only pivoting rule with a termination PROOF, so the oracle cannot cycle
// and cannot need a tolerance to decide it has finished. There are no tolerances anywhere in
// this file; every comparison is exact.
//
// The instance form is deliberately narrow:
//
//     min c'x   subject to   A x >= b,   0 <= x <= u   (u_j optionally infinite)
//
// Narrow because the conversion to the standard form the tableau needs must be obviously
// correct by inspection: one surplus variable per row, one slack per finite upper bound.
// A richer form would need a transformation with enough moving parts to hide a bug, and a
// buggy oracle is worse than no oracle - it manufactures confidence.
#pragma once

#include <cstdint>
#include <vector>

#include "sankhya/model.hpp"

#include "oracles/rational.hpp"

namespace sankhya::oracle {

/// No upper bound on a column.
inline constexpr std::int64_t kNoUpperBound = std::numeric_limits<std::int64_t>::max();

/// A generated instance. Coefficients are integers so that the exact arithmetic starts from
/// small numbers and overflow stays rare.
struct GeneratedLp {
  Index num_rows = 0;
  Index num_cols = 0;
  /// Row-major, num_rows x num_cols.
  std::vector<std::vector<std::int64_t>> a;
  /// Row lower bounds: A x >= b.
  std::vector<std::int64_t> b;
  /// Objective coefficients, minimized.
  std::vector<std::int64_t> c;
  /// Per-column upper bound, or kNoUpperBound.
  std::vector<std::int64_t> upper;

  /// A human-readable dump, used when the fuzz harness reports a mismatch. A failing
  /// instance nobody can reproduce is not evidence.
  [[nodiscard]] std::string to_text() const;
};

/// Build the sankhya::Model that represents exactly the same instance, so the float solver
/// and the oracle see one problem and not two.
[[nodiscard]] Model to_model(const GeneratedLp& lp);

enum class OracleStatus {
  kOptimal,
  kInfeasible,
  kUnbounded,
  /// Exact arithmetic exceeded __int128. The instance is discarded and counted, never
  /// silently treated as agreement.
  kOverflow,
  kIterationLimit
};

[[nodiscard]] const char* to_string(OracleStatus status) noexcept;

struct OracleResult {
  OracleStatus status = OracleStatus::kOptimal;
  Rational objective;
  std::vector<Rational> x;
  std::int64_t iterations = 0;
};

/// Solve exactly. Never throws: RationalOverflow becomes kOverflow.
[[nodiscard]] OracleResult solve_exact(const GeneratedLp& lp);

}  // namespace sankhya::oracle
