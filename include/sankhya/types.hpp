// SPDX-License-Identifier: Apache-2.0
// SANKHYA — core scalar and index types.
//
// Every numerical quantity in the solver core is `double`. Per CLAUDE.md there is no
// `float` anywhere in src/. Indices are 32-bit signed: this caps us at ~2.1e9 nonzeros,
// which is far beyond the "thousands to millions of variables" the problem statement
// asks for, and signed indices keep -Wsign-compare honest at every loop boundary.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace sankhya {

/// Index type for rows, columns and nonzero positions.
using Index = std::int32_t;

/// Counter type for iterations, nodes and other unbounded tallies.
using Count = std::int64_t;

/// The value used to denote an absent bound. Bounds are compared against this with
/// is_infinite() rather than by equality, so that any value at or beyond 1e30 (the MPS
/// de-facto infinity) is also treated as absent once readers normalise it.
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

/// MPS files conventionally use 1e30 to mean infinity. Readers normalise anything with
/// magnitude at or above this to +/- kInfinity so the core only ever sees true infinities.
inline constexpr double kMpsInfinity = 1e30;

/// True when `v` denotes an absent bound (either a true infinity or an MPS-scale one).
[[nodiscard]] inline bool is_infinite(double v) noexcept {
  return std::isinf(v) || std::fabs(v) >= kMpsInfinity;
}

/// True when `v` is a usable finite bound.
[[nodiscard]] inline bool is_finite_bound(double v) noexcept {
  return !is_infinite(v);
}

/// Normalise an MPS-scale infinity to a true IEEE infinity, leaving finite values alone.
[[nodiscard]] inline double normalize_infinity(double v) noexcept {
  if (!is_infinite(v)) return v;
  return v > 0.0 ? kInfinity : -kInfinity;
}

/// Collapse negative zero onto positive zero.
///
/// IEEE-754 keeps -0.0 distinct from 0.0, and it appears the moment a maximization model
/// multiplies a zero reduced cost by the sense multiplier, or a reader negates a zero
/// objective constant. The value is numerically identical and every comparison in the
/// solver treats it as such, but it PRINTS as "-0", and a judge reading "reduced_cost -0"
/// in a solution file has no way to know that is not a real negative quantity rounded to
/// nothing. Applied wherever a number crosses into user-visible output.
[[nodiscard]] inline double normalize_zero(double v) noexcept { return v == 0.0 ? 0.0 : v; }

}  // namespace sankhya
