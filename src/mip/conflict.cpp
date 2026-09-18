// SPDX-License-Identifier: Apache-2.0
// SANKHYA - conflict analysis (#292). See conflict.hpp for what a conflict is and why a
// stored one cannot cut off a feasible point.

#include "conflict.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include <fmt/format.h>

#include "sankhya/sparse.hpp"
#include "sankhya/tolerances.hpp"

namespace sankhya::mip {

const char* to_string(ConflictSource source) noexcept {
  return source == ConflictSource::kLp ? "lp" : "propagation";
}

std::vector<ConflictLiteral> canonical(std::vector<ConflictLiteral> literals) {
  // The tightest per (column, direction): x <= 2 and x <= 3 together say x <= 2.
  std::map<std::pair<Index, bool>, double> tightest;
  for (const ConflictLiteral& literal : literals) {
    const auto key = std::make_pair(literal.column, literal.is_upper);
    const auto found = tightest.find(key);
    if (found == tightest.end()) {
      tightest.emplace(key, literal.value);
    } else {
      found->second = literal.is_upper ? std::min(found->second, literal.value)
                                       : std::max(found->second, literal.value);
    }
  }
  std::vector<ConflictLiteral> result;
  result.reserve(tightest.size());
  for (const auto& [key, value] : tightest) {
    result.push_back(ConflictLiteral{key.first, key.second, value});
  }
  return result;  // std::map iterates in key order, so this is sorted
}

ConflictVerdict check_conflict(const std::vector<ConflictLiteral>& literals,
                               const std::vector<double>& lower,
                               const std::vector<double>& upper, double tolerance) {
  ConflictVerdict verdict;
  int open = -1;
  for (std::size_t k = 0; k < literals.size(); ++k) {
    const ConflictLiteral& literal = literals[k];
    if (literal_holds(literal, lower, upper, tolerance)) continue;
    // One literal already false everywhere and the conflict says nothing here.
    if (literal_refuted(literal, lower, upper, tolerance)) return verdict;
    if (open >= 0) return verdict;  // two undecided: nothing follows
    open = static_cast<int>(k);
  }
  if (open < 0) {
    verdict.infeasible = true;
  } else {
    verdict.implied = open;
  }
  return verdict;
}

FarkasProof aggregate_farkas(const Model& model, const std::vector<double>& y) {
  FarkasProof proof;
  const Index m = model.num_rows();
  const Index n = model.num_cols();
  if (static_cast<Index>(y.size()) != m) return proof;

  // Scale-free: the proof is homogeneous in y, the tolerance below is not.
  double largest = 0.0;
  for (const double value : y) {
    if (!std::isfinite(value)) return proof;
    largest = std::max(largest, std::fabs(value));
  }
  if (largest == 0.0) return proof;

  for (Index i = 0; i < m; ++i) {
    const auto u = static_cast<std::size_t>(i);
    const double yi = y[u] / largest;
    if (yi == 0.0) continue;
    // A multiplier may only lean on a bound the row has; the sign says which.
    const double bound = yi > 0.0 ? model.row_lower[u] : model.row_upper[u];
    if (!is_finite_bound(bound)) return proof;
    proof.required += yi * bound;
    proof.weight += std::fabs(yi) * std::max(1.0, std::fabs(bound));
  }

  proof.d.assign(static_cast<std::size_t>(n), 0.0);
  for (Index j = 0; j < n; ++j) {
    const ColumnView column = model.matrix.column(j);
    double sum = 0.0;
    for (Index k = 0; k < column.size; ++k) {
      sum += column.values[k] * y[static_cast<std::size_t>(column.rows[k])] / largest;
    }
    proof.d[static_cast<std::size_t>(j)] = sum;
  }
  proof.usable = true;
  return proof;
}

bool farkas_contradicts(const FarkasProof& proof, const std::vector<double>& lower,
                        const std::vector<double>& upper) {
  if (!proof.usable) return false;
  double reachable = 0.0;
  double magnitude = 0.0;
  for (std::size_t j = 0; j < proof.d.size(); ++j) {
    const double dj = proof.d[j];
    if (dj == 0.0) continue;
    // The true maximum of dj * x over [lower, upper]. No coefficient is rounded to zero: a
    // tiny coefficient on a huge bound is not tiny, and on an infinite one it is fatal.
    const double bound = dj > 0.0 ? upper[j] : lower[j];
    if (!is_finite_bound(bound)) return false;
    reachable += dj * bound;
    magnitude += std::fabs(dj * bound);
  }
  const double margin = proof.required - reachable;
  // A point that meets every row to within the feasibility tolerance can fall short of
  // `required` by up to that tolerance times the weight; the summation itself can be off by
  // rounding in the magnitude. The contradiction has to beat both.
  const double slack =
      tol::kPrimalFeasibility * std::max({1.0, proof.weight, std::fabs(proof.required)}) +
      1e-12 * magnitude;
  return margin > slack;
}

std::vector<ConflictLiteral> minimize_conflict(
    std::vector<ConflictLiteral> literals, bool keep_last, int budget,
    const std::function<bool(const std::vector<ConflictLiteral>&)>& verify, int* checks_used) {
  int checks = 0;
  const std::size_t last = literals.empty() ? 0 : literals.size() - 1;
  std::vector<bool> dropped(literals.size(), false);
  std::vector<ConflictLiteral> trial;
  for (std::size_t k = 0; k < literals.size() && checks < budget; ++k) {
    if (keep_last && k == last) continue;
    trial.clear();
    for (std::size_t q = 0; q < literals.size(); ++q) {
      if (q != k && !dropped[q]) trial.push_back(literals[q]);
    }
    ++checks;
    if (verify(trial)) dropped[k] = true;
  }
  std::vector<ConflictLiteral> kept;
  for (std::size_t q = 0; q < literals.size(); ++q) {
    if (!dropped[q]) kept.push_back(literals[q]);
  }
  if (checks_used != nullptr) *checks_used = checks;
  return kept;
}

bool ConflictStore::add(std::vector<ConflictLiteral> literals, ConflictSource source,
                        std::int64_t node) {
  if (capacity_ == 0) return false;
  if (known_.count(literals) > 0) {
    ++duplicates_;
    return false;
  }
  if (entries_.size() >= capacity_) evict();
  known_.insert(literals);
  Entry entry;
  entry.literals = std::move(literals);
  entry.source = source;
  entry.id = next_id_++;
  entry.created = node;
  entry.last_used = node;
  entries_.push_back(std::move(entry));
  return true;
}

void ConflictStore::evict() {
  // The least useful tenth, at least one. Sorted on a total order - uses, then how long
  // since last used, then age - so a rerun forgets exactly the same conflicts.
  const std::size_t count = std::max<std::size_t>(1, entries_.size() / 10);
  std::vector<std::size_t> order(entries_.size());
  for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const Entry& x = entries_[a];
    const Entry& y = entries_[b];
    if (x.uses != y.uses) return x.uses < y.uses;
    if (x.last_used != y.last_used) return x.last_used < y.last_used;
    return x.id < y.id;
  });
  std::vector<bool> drop(entries_.size(), false);
  for (std::size_t k = 0; k < count; ++k) drop[order[k]] = true;
  std::vector<Entry> kept;
  kept.reserve(entries_.size() - count);
  for (std::size_t k = 0; k < entries_.size(); ++k) {
    if (drop[k]) {
      known_.erase(entries_[k].literals);
    } else {
      kept.push_back(std::move(entries_[k]));
    }
  }
  entries_ = std::move(kept);
  evicted_ += static_cast<std::int64_t>(count);
}

std::string conflicts_to_json(const ConflictStore& store, const ConflictStats& stats) {
  std::string out = "{\n";
  out += fmt::format(
      "  \"detected\": {},\n  \"learned\": {},\n  \"rejected\": {},\n  \"too_long\": {},\n"
      "  \"minimized\": {},\n  \"duplicates\": {},\n  \"evicted\": {},\n  \"checks\": {},\n"
      "  \"nodes_pruned\": {},\n  \"tightenings\": {},\n  \"seconds\": {:.6f},\n",
      stats.detected, stats.learned, stats.rejected, stats.too_long, stats.minimized,
      store.duplicates(), store.evicted(), stats.checks, stats.nodes_pruned, stats.tightenings,
      stats.seconds);
  out += "  \"conflicts\": [";
  bool first = true;
  for (const ConflictStore::Entry& entry : store.entries()) {
    out += first ? "\n" : ",\n";
    first = false;
    out += fmt::format("    {{\"id\": {}, \"source\": \"{}\", \"uses\": {}, \"literals\": [",
                       entry.id, to_string(entry.source), entry.uses);
    for (std::size_t k = 0; k < entry.literals.size(); ++k) {
      const ConflictLiteral& literal = entry.literals[k];
      out += fmt::format("{}[{}, \"{}\", {:.17g}]", k == 0 ? "" : ", ", literal.column,
                         literal.is_upper ? "<=" : ">=", literal.value);
    }
    out += "]}";
  }
  out += first ? "]\n}\n" : "\n  ]\n}\n";
  return out;
}

}  // namespace sankhya::mip
