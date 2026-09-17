// SPDX-License-Identifier: Apache-2.0
// SANKHYA — core scalar and index types.
//
// Every numerical quantity in the solver core is `double`. Per ENGINEERING_RULES.md there is no
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

/// The most nonzeros any sparse structure here may hold, and the largest offset a column or
/// row start may name (#305).
///
/// WHY A LIMIT RATHER THAN A WIDER INDEX. Every offset, loop bound and allocation size in a
/// sparse structure is an Index, so they all fit exactly when the entry COUNT does. Widening
/// Index to 64 bits would double the memory of every pattern array - `column_starts_`,
/// `row_indices_`, the LU eta file, the LDL^T pattern - on every model, to buy a size no
/// benchmark in this project reaches (the largest generated instance so far is ~5e6
/// nonzeros, three orders of magnitude below this). So the count is CHECKED where it grows
/// instead: a model that would exceed it is refused with a diagnostic rather than wrapping a
/// prefix sum into a negative offset and indexing memory that was never allocated.
///
/// The failure mode this exists to prevent is silent. `static_cast<Index>(values_.size())`
/// on 2^31 entries is implementation-defined, the prefix sum in finalize() is signed
/// overflow and therefore undefined, and what comes out the far side is a matrix whose
/// column starts run backwards - not a crash, a wrong answer.
inline constexpr Index kMaxNonzeros = std::numeric_limits<Index>::max();

/// True when `count` entries can be addressed by Index offsets.
[[nodiscard]] inline bool nonzero_count_fits(std::size_t count) noexcept {
  return count <= static_cast<std::size_t>(kMaxNonzeros);
}

/// True when `a + b` would overflow Index. Both arguments must be non-negative; the
/// subtraction is done on the limit so the sum itself is never formed.
[[nodiscard]] inline bool index_sum_overflows(Index a, Index b) noexcept {
  return a < 0 || b < 0 || b > kMaxNonzeros - a;
}

/// True when `a * b` would overflow Index. Both arguments must be non-negative.
///
/// Used before sizing anything shaped like a product - a dense n x n working set, a buffer of
/// `rows * columns` - where the multiplication is the step that wraps.
[[nodiscard]] inline bool index_product_overflows(Index a, Index b) noexcept {
  if (a < 0 || b < 0) return true;
  if (a == 0 || b == 0) return false;
  return a > kMaxNonzeros / b;
}

/// True when a dense `n x n` array of `element_bytes` would exceed `budget_bytes`.
///
/// The product is formed in std::size_t and guarded against wrapping there too, which is the
/// point: on a 64-bit host `n * n * 8` for n = 50,000 is 2e10, a number that fits size_t and
/// then asks the allocator for 20 GB. Callers use this to refuse before allocating.
[[nodiscard]] inline bool dense_square_exceeds(Index n, std::size_t element_bytes,
                                               std::size_t budget_bytes) noexcept {
  if (n <= 0) return false;
  const auto un = static_cast<std::size_t>(n);
  if (un > (std::numeric_limits<std::size_t>::max() / un)) return true;
  const std::size_t entries = un * un;
  if (element_bytes != 0 &&
      entries > (std::numeric_limits<std::size_t>::max() / element_bytes)) {
    return true;
  }
  return entries * element_bytes > budget_bytes;
}

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
[[nodiscard]] inline double normalize_zero(double v) noexcept {
  return v == 0.0 ? 0.0 : v;
}

}  // namespace sankhya
