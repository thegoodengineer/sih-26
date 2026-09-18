// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the LP engine selection behind `algorithm=auto` (#284). See the header for
// the thresholds and the measurements each one rests on.

#include "engine_selection.hpp"

#include <fmt/format.h>

namespace sankhya {

EngineSelection select_engine(const Model& model, const Options& options, bool warm_start) {
  EngineSelection s;
  s.rows = model.num_rows();
  s.columns = model.num_cols();
  s.nonzeros = model.num_nonzeros();
  s.warm_start = warm_start;
  const std::string requested = options.get_string("algorithm");
  const std::string shape =
      fmt::format("{} rows, {} columns, {} nonzeros", s.rows, s.columns, s.nonzeros);

  if (requested != "auto") {
    s.algorithm = requested;
    s.rule = "requested";
    s.reason = fmt::format("algorithm={} was asked for ({})", requested, shape);
    return s;
  }
  if (warm_start) {
    // Only a simplex can start from a basis; the dual is the right restart after a bound
    // or right-hand-side edit (#218), which is what a re-solve usually is.
    s.algorithm = "dual-simplex";
    s.rule = "warm-start";
    s.reason =
        fmt::format("a starting basis was given, and only a simplex can use one ({})", shape);
    return s;
  }
  if (s.rows >= kPdhgRowFloor) {
    s.algorithm = "pdhg";
    s.rule = "size:pdhg";
    s.reason = fmt::format(
        "{}: at {} rows and above the direct factorization is out of reach at the time "
        "limit and the first-order method reaches the optimum (scale-e134aeb.csv, "
        "scale-refinery-e134aeb.csv, docs/BENCHMARKS.md 1f)",
        shape, kPdhgRowFloor);
    return s;
  }
  if (s.rows >= kDualSimplexRowLimit) {
    s.algorithm = "ipm";
    s.rule = "size:ipm";
    s.reason = fmt::format(
        "{}: from {} rows the dual simplex hits the time limit on every measured shape and "
        "the interior point solves the structured ones (scale-staircase-e134aeb.csv, "
        "scale-refinery-e134aeb.csv, docs/BENCHMARKS.md 1f.2-1f.3)",
        shape, kDualSimplexRowLimit);
    return s;
  }
  if (s.nonzeros >= kIpmNonzeroFloor) {
    s.algorithm = "ipm";
    s.rule = "density:ipm";
    s.reason = fmt::format(
        "{}: a model this dense times out on the dual simplex and solves on the interior "
        "point with crossover (maros-r7, 144,848 nonzeros: 14 s against the 120 s limit; "
        "netlib-full-b3f1660.csv)",
        shape);
    return s;
  }
  s.algorithm = "dual-simplex";
  s.rule = "default:dual-simplex";
  s.reason = fmt::format(
      "{}: below {} rows and {} nonzeros the dual simplex is the measured default, 80 of "
      "89 on the Netlib full set (netlib-full-b3f1660.csv); it produces the basis the rest "
      "of the pipeline uses",
      shape, kDualSimplexRowLimit, kIpmNonzeroFloor);
  return s;
}

}  // namespace sankhya
