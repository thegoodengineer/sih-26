// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solution pool (#225). See solution_pool.hpp for the references and the
// reasoning; this file is the bookkeeping.

#include "solution_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sankhya::mip {

SolutionPool::SolutionPool(std::vector<Index> integer_columns, std::size_t capacity,
                           bool diversity)
    : integer_columns_(std::move(integer_columns)),
      capacity_(capacity),
      diversity_(diversity) {}

std::vector<std::int64_t> SolutionPool::key_of(const std::vector<double>& x) const {
  std::vector<std::int64_t> key;
  key.reserve(integer_columns_.size());
  for (const Index j : integer_columns_) {
    key.push_back(std::llround(x[static_cast<std::size_t>(j)]));
  }
  return key;
}

std::size_t SolutionPool::hamming(const std::vector<std::int64_t>& a,
                                  const std::vector<std::int64_t>& b) {
  std::size_t distance = 0;
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t k = 0; k < n; ++k) {
    if (a[k] != b[k]) ++distance;
  }
  return distance + (std::max(a.size(), b.size()) - n);
}

void SolutionPool::offer(double objective, const std::vector<double>& x) {
  if (capacity_ == 0) return;
  std::vector<std::int64_t> key = key_of(x);

  // One plan per integer assignment. A second point with the same assignment differs only in
  // its continuous values, which the LP with the integers fixed would settle anyway, so the
  // better of the two stands for both.
  for (std::size_t k = 0; k < entries_.size(); ++k) {
    if (entries_[k].key != key) continue;
    if (objective >= entries_[k].objective) return;
    entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(k));
    break;
  }

  Entry entry;
  entry.objective = objective;
  entry.x = x;
  entry.key = std::move(key);
  // upper_bound, so an equal objective lands after the members already holding it: the
  // earlier arrival keeps its place, which is also the order the incumbent was chosen in.
  const auto at =
      std::upper_bound(entries_.begin(), entries_.end(), objective,
                       [](double value, const Entry& held) { return value < held.objective; });
  entries_.insert(at, std::move(entry));

  while (entries_.size() > capacity_) evict_one();
}

void SolutionPool::evict_one() {
  if (!diversity_ || entries_.size() < 3) {
    entries_.pop_back();
    return;
  }
  // DIVERSITY: remove the member that adds least variety - the one whose nearest neighbour
  // in the pool is nearest - never the best, and on a tie the worse objective. What survives
  // is spread out rather than ten neighbours of the optimum.
  std::size_t victim = entries_.size() - 1;
  std::size_t victim_nearest = std::numeric_limits<std::size_t>::max();
  for (std::size_t k = 1; k < entries_.size(); ++k) {
    std::size_t nearest = std::numeric_limits<std::size_t>::max();
    for (std::size_t other = 0; other < entries_.size(); ++other) {
      if (other == k) continue;
      nearest = std::min(nearest, hamming(entries_[k].key, entries_[other].key));
    }
    // `<=` walks toward the worse objective on a tie, because entries_ is ascending.
    if (nearest <= victim_nearest) {
      victim_nearest = nearest;
      victim = k;
    }
  }
  entries_.erase(entries_.begin() + static_cast<std::ptrdiff_t>(victim));
}

double SolutionPool::cutoff() const {
  if (diversity_ || !full()) return std::numeric_limits<double>::infinity();
  return entries_.back().objective;
}

std::vector<SolutionPool::Entry> SolutionPool::finish(double incumbent_objective,
                                                      const std::vector<double>& incumbent_x,
                                                      double relative_gap) const {
  std::vector<Entry> out;
  if (capacity_ == 0) return out;

  // The incumbent leads, exactly as reported. It is normally entries_[0] already, but not
  // always: a point within the incumbent test's 1e-12 of it can sort ahead, and a pool of
  // one can have replaced it. Placing it explicitly makes pool[0] == the solution by
  // construction rather than by the usual order of events.
  Entry first;
  first.objective = incumbent_objective;
  first.x = incumbent_x;
  first.key = key_of(incumbent_x);
  out.push_back(std::move(first));

  const double limit =
      incumbent_objective + relative_gap * std::max(1.0, std::fabs(incumbent_objective));
  for (const Entry& entry : entries_) {
    if (out.size() >= capacity_) break;
    if (entry.key == out.front().key) continue;
    if (std::isfinite(limit) && entry.objective > limit) continue;
    out.push_back(entry);
  }
  return out;
}

}  // namespace sankhya::mip
