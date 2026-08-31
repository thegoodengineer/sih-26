// SPDX-License-Identifier: Apache-2.0
// SANKHYA - exact rational arithmetic for the reference oracle.
//
// TESTS ONLY. Nothing in src/ includes this, and nothing here is fast. Its entire job is to
// be OBVIOUSLY CORRECT, because it is the standard the floating-point simplex is judged
// against, and an oracle with a bug is worse than no oracle at all: it manufactures
// confidence.
//
// Representation is a normalised fraction over __int128 with a strictly positive
// denominator. Every operation checks for overflow and throws RationalOverflow rather than
// wrapping. That distinction matters: a wrapped intermediate would silently turn the oracle
// into a random number generator, and the fuzz harness would then "confirm" whatever the
// float simplex did. Overflow is counted and reported by the harness instead.
#pragma once

#include <cstdint>
#include <exception>
#include <limits>

namespace sankhya::oracle {

/// Thrown when an exact operation cannot be represented in __int128.
struct RationalOverflow : std::exception {
  [[nodiscard]] const char* what() const noexcept override {
    return "exact rational arithmetic overflowed";
  }
};

class Rational {
 public:
#if defined(_MSC_VER) && !defined(__clang__)
  using Int = std::int64_t;
#else
  using Int = __int128;
#endif

  Rational() = default;
  Rational(Int numerator) : numerator_(numerator), denominator_(1) {}  // NOLINT: implicit
  Rational(Int numerator, Int denominator) : numerator_(numerator), denominator_(denominator) {
    normalize();
  }

  [[nodiscard]] Int numerator() const noexcept { return numerator_; }
  [[nodiscard]] Int denominator() const noexcept { return denominator_; }

  [[nodiscard]] bool is_zero() const noexcept { return numerator_ == 0; }
  [[nodiscard]] bool is_positive() const noexcept { return numerator_ > 0; }
  [[nodiscard]] bool is_negative() const noexcept { return numerator_ < 0; }
  [[nodiscard]] int sign() const noexcept {
    return numerator_ > 0 ? 1 : (numerator_ < 0 ? -1 : 0);
  }

  /// Nearest double. Used only to report a result or to compare against the float solver -
  /// never inside the oracle's own arithmetic.
  [[nodiscard]] double to_double() const noexcept {
    return static_cast<double>(numerator_) / static_cast<double>(denominator_);
  }

  Rational operator-() const {
    Rational r;
    r.numerator_ = negate(numerator_);
    r.denominator_ = denominator_;
    return r;
  }

  Rational operator+(const Rational& other) const {
    // a/b + c/d = (a*d + c*b) / (b*d)
    return Rational(
        add(multiply(numerator_, other.denominator_), multiply(other.numerator_, denominator_)),
        multiply(denominator_, other.denominator_));
  }

  Rational operator-(const Rational& other) const { return *this + (-other); }

  Rational operator*(const Rational& other) const {
    return Rational(multiply(numerator_, other.numerator_),
                    multiply(denominator_, other.denominator_));
  }

  Rational operator/(const Rational& other) const {
    if (other.numerator_ == 0) throw RationalOverflow();  // division by zero is a bug here
    return Rational(multiply(numerator_, other.denominator_),
                    multiply(denominator_, other.numerator_));
  }

  Rational& operator+=(const Rational& other) { return *this = *this + other; }
  Rational& operator-=(const Rational& other) { return *this = *this - other; }
  Rational& operator*=(const Rational& other) { return *this = *this * other; }
  Rational& operator/=(const Rational& other) { return *this = *this / other; }

  /// Exact comparison. a/b < c/d with b, d > 0 is a*d < c*b, and the products are checked.
  bool operator<(const Rational& other) const {
    return multiply(numerator_, other.denominator_) < multiply(other.numerator_, denominator_);
  }
  bool operator>(const Rational& other) const { return other < *this; }
  bool operator<=(const Rational& other) const { return !(other < *this); }
  bool operator>=(const Rational& other) const { return !(*this < other); }
  bool operator==(const Rational& other) const {
    // Both sides are normalised, so equality is componentwise and needs no multiplication.
    return numerator_ == other.numerator_ && denominator_ == other.denominator_;
  }
  bool operator!=(const Rational& other) const { return !(*this == other); }

 private:
  static Int absolute(Int v) { return v < 0 ? negate(v) : v; }

  static Int negate(Int v) {
    // The most negative value has no positive counterpart.
    if (v == kMin) throw RationalOverflow();
    return -v;
  }

  static Int add(Int a, Int b) {
#if defined(_MSC_VER) && !defined(__clang__)
    if ((b > 0 && a > INT64_MAX - b) || (b < 0 && a < INT64_MIN - b)) {
      throw RationalOverflow();
    }
    return a + b;
#else
    Int result = 0;
    if (__builtin_add_overflow(a, b, &result)) throw RationalOverflow();
    return result;
#endif
  }

  static Int multiply(Int a, Int b) {
#if defined(_MSC_VER) && !defined(__clang__)
    if (a == 0 || b == 0) return 0;
    if (a > 0) {
      if (b > 0) {
        if (a > INT64_MAX / b) throw RationalOverflow();
      } else {
        if (b < INT64_MIN / a) throw RationalOverflow();
      }
    } else {
      if (b > 0) {
        if (a < INT64_MIN / b) throw RationalOverflow();
      } else {
        if (a == INT64_MIN || b == INT64_MIN || -a > INT64_MAX / (-b)) {
          throw RationalOverflow();
        }
      }
    }
    return a * b;
#else
    Int result = 0;
    if (__builtin_mul_overflow(a, b, &result)) throw RationalOverflow();
    return result;
#endif
  }

  static Int greatest_common_divisor(Int a, Int b) {
    a = absolute(a);
    b = absolute(b);
    while (b != 0) {
      const Int t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  void normalize() {
    if (denominator_ == 0) throw RationalOverflow();  // 1/0 is a bug in the caller
    if (denominator_ < 0) {
      numerator_ = negate(numerator_);
      denominator_ = negate(denominator_);
    }
    if (numerator_ == 0) {
      denominator_ = 1;
      return;
    }
    const Int g = greatest_common_divisor(numerator_, denominator_);
    if (g > 1) {
      numerator_ /= g;
      denominator_ /= g;
    }
  }

#if defined(_MSC_VER) && !defined(__clang__)
  static constexpr Int kMin = INT64_MIN;
#else
  static constexpr Int kMin = static_cast<Int>(1) << 127;
#endif

  Int numerator_ = 0;
  Int denominator_ = 1;
};

}  // namespace sankhya::oracle
