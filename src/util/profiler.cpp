// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the solve profiler: a region tree, counters, and its two reports (#285).

#include "util/profiler.hpp"

#include <algorithm>
#include <functional>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace sankhya {

bool parse_profile_mode(std::string_view text, ProfileMode* out) noexcept {
  if (text == "off") {
    *out = ProfileMode::kOff;
  } else if (text == "basic") {
    *out = ProfileMode::kBasic;
  } else if (text == "detailed") {
    *out = ProfileMode::kDetailed;
  } else {
    return false;
  }
  return true;
}

const char* to_string(ProfileMode mode) noexcept {
  switch (mode) {
    case ProfileMode::kOff: return "off";
    case ProfileMode::kBasic: return "basic";
    case ProfileMode::kDetailed: return "detailed";
  }
  return "off";
}

int Profiler::child_named(int parent, std::string_view name) {
  const std::vector<int>& siblings =
      parent < 0 ? roots_ : regions_[static_cast<std::size_t>(parent)].children;
  for (const int index : siblings) {
    if (regions_[static_cast<std::size_t>(index)].name == name) return index;
  }
  Region region;
  region.name = std::string(name);
  region.parent = parent;
  regions_.push_back(std::move(region));
  const auto index = static_cast<int>(regions_.size() - 1);
  if (parent < 0) {
    roots_.push_back(index);
  } else {
    regions_[static_cast<std::size_t>(parent)].children.push_back(index);
  }
  return index;
}

int Profiler::enter(std::string_view name) {
  const int parent = open_.empty() ? -1 : open_.back();
  const int index = child_named(parent, name);
  open_.push_back(index);
  return index;
}

void Profiler::leave(int region, double seconds) {
  // Scopes are RAII, so they close in the reverse of the order they opened; a mismatch would
  // be a scope that outlived its parent, and closing whatever is on top keeps the stack sane.
  if (!open_.empty() && open_.back() == region) open_.pop_back();
  if (region < 0 || static_cast<std::size_t>(region) >= regions_.size()) return;
  Region& r = regions_[static_cast<std::size_t>(region)];
  r.inclusive_seconds += seconds;
  ++r.calls;
}

void Profiler::record(std::string_view name, double seconds, std::int64_t calls) {
  if (mode_ == ProfileMode::kOff) return;
  const int parent = open_.empty() ? -1 : open_.back();
  Region& r = regions_[static_cast<std::size_t>(child_named(parent, name))];
  r.inclusive_seconds += seconds;
  r.calls += calls;
}

void Profiler::count(std::string_view name, std::int64_t amount) {
  if (mode_ == ProfileMode::kOff) return;
  for (Counter& c : counters_) {
    if (c.name == name) {
      c.value += amount;
      return;
    }
  }
  counters_.push_back(Counter{std::string(name), amount});
}

double Profiler::exclusive_seconds(int region) const {
  const Region& r = regions_[static_cast<std::size_t>(region)];
  double children = 0.0;
  for (const int c : r.children)
    children += regions_[static_cast<std::size_t>(c)].inclusive_seconds;
  return std::max(0.0, r.inclusive_seconds - children);
}

std::string Profiler::path(int region) const {
  std::string text;
  for (int at = region; at >= 0; at = regions_[static_cast<std::size_t>(at)].parent) {
    text = text.empty() ? regions_[static_cast<std::size_t>(at)].name
                        : regions_[static_cast<std::size_t>(at)].name + "/" + text;
  }
  return text;
}

void Profiler::merge(const Profiler& other) {
  if (&other == this) return;  // folding a tree into itself while walking it
  // Walk the other tree top-down, finding or creating the same path here. Recursion depth is
  // the nesting depth of scopes, a handful.
  const std::function<void(int, int)> fold = [&](int theirs, int parent_here) {
    const Region& r = other.regions_[static_cast<std::size_t>(theirs)];
    const int here = child_named(parent_here, r.name);
    regions_[static_cast<std::size_t>(here)].inclusive_seconds += r.inclusive_seconds;
    regions_[static_cast<std::size_t>(here)].calls += r.calls;
    for (const int child : r.children) fold(child, here);
  };
  for (const int root : other.roots_) fold(root, -1);
  for (const Counter& c : other.counters_) count(c.name, c.value);
}

std::string Profiler::format_text() const {
  std::string text = fmt::format("Performance profile ({})\n", to_string(mode_));
  text += fmt::format("  {:<44} {:>11} {:>11} {:>9}\n", "region", "inclusive", "exclusive",
                      "calls");
  const std::function<void(int, int)> line = [&](int region, int depth) {
    const Region& r = regions_[static_cast<std::size_t>(region)];
    const std::string label = std::string(static_cast<std::size_t>(2 * depth), ' ') + r.name;
    text += fmt::format("  {:<44} {:>10.4f}s {:>10.4f}s {:>9}\n", label, r.inclusive_seconds,
                        exclusive_seconds(region), r.calls);
    for (const int child : r.children) line(child, depth + 1);
  };
  for (const int root : roots_) line(root, 0);
  if (!counters_.empty()) {
    text += "  counters\n";
    for (const Counter& c : counters_)
      text += fmt::format("    {:<42} {:>12}\n", c.name, c.value);
  }
  return text;
}

std::string Profiler::format_json() const {
  nlohmann::json blob;
  blob["mode"] = to_string(mode_);
  blob["regions"] = nlohmann::json::array();
  for (std::size_t k = 0; k < regions_.size(); ++k) {
    const auto index = static_cast<int>(k);
    const Region& r = regions_[k];
    blob["regions"].push_back({{"path", path(index)},
                               {"parent_path", r.parent < 0 ? std::string() : path(r.parent)},
                               {"inclusive_seconds", r.inclusive_seconds},
                               {"exclusive_seconds", exclusive_seconds(index)},
                               {"calls", r.calls}});
  }
  blob["counters"] = nlohmann::json::object();
  for (const Counter& c : counters_) blob["counters"][c.name] = c.value;
  return blob.dump(2);
}

}  // namespace sankhya
