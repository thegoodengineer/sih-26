// SPDX-License-Identifier: Apache-2.0
#include "sankhya/solve_control.hpp"

namespace sankhya {

const char* to_string(SolvePhase phase) noexcept {
  switch (phase) {
    case SolvePhase::kPresolve: return "presolve";
    case SolvePhase::kLp: return "lp";
    case SolvePhase::kTree: return "tree";
  }
  return "unknown";
}

}  // namespace sankhya
