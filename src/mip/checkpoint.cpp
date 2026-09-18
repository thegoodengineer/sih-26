// SPDX-License-Identifier: Apache-2.0
// SANKHYA - branch-and-bound checkpoint (de)serialization (#287). The format and its
// guarantees are on the declarations.

#include "mip/checkpoint.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "sankhya/version.hpp"

namespace sankhya::mip {
namespace {

using nlohmann::json;

constexpr const char* kFormat = "sankhya-tree-checkpoint";

/// FNV-1a over the bytes of `text`, the same function Model::fingerprint uses.
std::string checksum(const std::string& text) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const char c : text) {
    hash ^= static_cast<std::uint64_t>(static_cast<unsigned char>(c));
    hash *= 1099511628211ULL;
  }
  return fmt::format("{:016x}", hash);
}

/// JSON has no infinity; a string stands for one.
json number(double v) {
  if (std::isfinite(v)) return v;
  if (std::isnan(v)) return "nan";
  return v > 0 ? "inf" : "-inf";
}

bool read_number(const json& j, double* out) {
  if (j.is_number()) {
    *out = j.get<double>();
    return true;
  }
  if (const auto* text = j.get_ptr<const json::string_t*>(); text != nullptr) {
    const std::string& s = *text;
    if (s == "inf") {
      *out = std::numeric_limits<double>::infinity();
      return true;
    }
    if (s == "-inf") {
      *out = -std::numeric_limits<double>::infinity();
      return true;
    }
  }
  return false;  // "nan" and anything else: a checkpoint never legitimately holds one
}

json state_of(const TreeCheckpoint& c) {
  json s;
  s["model_fingerprint"] = c.model_fingerprint;
  s["problem_class"] = c.problem_class;
  s["num_rows"] = c.num_rows;
  s["num_cols"] = c.num_cols;
  s["integrality_tolerance"] = c.integrality_tolerance;
  s["have_incumbent"] = c.have_incumbent;
  s["incumbent_x"] = json::array();
  for (const double v : c.incumbent_x) s["incumbent_x"].push_back(number(v));
  s["nodes_explored"] = c.nodes_explored;
  s["nodes_pruned"] = c.nodes_pruned;
  s["pseudo_down_sum"] = c.pseudo_down_sum;
  s["pseudo_up_sum"] = c.pseudo_up_sum;
  s["pseudo_down_count"] = c.pseudo_down_count;
  s["pseudo_up_count"] = c.pseudo_up_count;
  s["open"] = json::array();
  for (const CheckpointNode& n : c.open) {
    json node;
    node["bound"] = number(n.bound);
    node["depth"] = n.depth;
    node["estimate"] = number(n.estimate);
    node["domain"] = json::array();
    for (const CheckpointChange& d : n.domain) {
      node["domain"].push_back({d.column, d.is_upper, number(d.value)});
    }
    s["open"].push_back(std::move(node));
  }
  return s;
}

/// A string field, or empty when it is missing or not a string. Read through get_ptr rather
/// than get<std::string>(), whose inlined path trips GCC's -Wnull-dereference inside nlohmann.
std::string text_of(const json& object, const char* key) {
  const auto it = object.find(key);
  if (it == object.end()) return {};
  const auto* text = it->get_ptr<const json::string_t*>();
  return text == nullptr ? std::string() : *text;
}

bool fail(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
  return false;
}

}  // namespace

bool write_checkpoint(const std::string& path, const TreeCheckpoint& checkpoint,
                      std::string* error) {
  json root;
  root["format"] = kFormat;
  root["version"] = kCheckpointVersion;
  root["sankhya"] = {{"version", version_string()}, {"commit", git_commit()}};
  const json state = state_of(checkpoint);
  root["checksum"] = checksum(state.dump());
  root["state"] = state;
  const std::string text = root.dump(1);

  const std::string temporary = path + ".tmp";
  std::FILE* out = std::fopen(temporary.c_str(), "wb");
  if (out == nullptr) return fail(error, fmt::format("{}: cannot open for writing", temporary));
  const bool written = std::fwrite(text.data(), 1, text.size(), out) == text.size();
  const bool closed = std::fclose(out) == 0;
  if (!written || !closed) {
    std::remove(temporary.c_str());
    return fail(error, fmt::format("{}: write failed; the previous checkpoint is untouched",
                                   temporary));
  }
  // The move is the commit point. std::filesystem::rename replaces an existing target on
  // every platform this builds on (MoveFileEx with REPLACE_EXISTING on Windows).
  std::error_code code;
  std::filesystem::rename(temporary, path, code);
  if (code) {
    std::remove(temporary.c_str());
    return fail(error, fmt::format("{}: could not replace the checkpoint ({}); the previous "
                                   "one is untouched",
                                   path, code.message()));
  }
  return true;
}

bool read_checkpoint(const std::string& path, TreeCheckpoint* out, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail(error, fmt::format("{}: cannot open", path));
  std::stringstream buffer;
  buffer << in.rdbuf();
  const json root = json::parse(buffer.str(), nullptr, /*allow_exceptions=*/false);
  if (root.is_discarded() || !root.is_object()) {
    return fail(error, fmt::format("{}: not a checkpoint (the file does not parse)", path));
  }
  if (text_of(root, "format") != kFormat) {
    return fail(error, fmt::format("{}: not a SANKHYA tree checkpoint", path));
  }
  if (!root.contains("version") || !root["version"].is_number_integer() ||
      root["version"].get<int>() != kCheckpointVersion) {
    return fail(error,
                fmt::format("{}: checkpoint format version {} - this build reads "
                            "version {} only, and does not guess at another",
                            path, root.contains("version") ? root["version"].dump() : "missing",
                            kCheckpointVersion));
  }
  if (!root.contains("state") || !root.contains("checksum") ||
      checksum(root["state"].dump()) != text_of(root, "checksum")) {
    return fail(error, fmt::format("{}: the checksum does not match - the file is corrupt or "
                                   "was edited, and is refused",
                                   path));
  }
  const json& s = root["state"];
  TreeCheckpoint c;
  try {
    c.model_fingerprint = text_of(s, "model_fingerprint");
    c.problem_class = text_of(s, "problem_class");
    c.num_rows = s.at("num_rows").get<Index>();
    c.num_cols = s.at("num_cols").get<Index>();
    c.integrality_tolerance = s.at("integrality_tolerance").get<double>();
    c.have_incumbent = s.at("have_incumbent").get<bool>();
    for (const json& v : s.at("incumbent_x")) {
      double x = 0.0;
      if (!read_number(v, &x)) return fail(error, "a non-number in the incumbent");
      c.incumbent_x.push_back(x);
    }
    c.nodes_explored = s.at("nodes_explored").get<Count>();
    c.nodes_pruned = s.at("nodes_pruned").get<Count>();
    c.pseudo_down_sum = s.at("pseudo_down_sum").get<std::vector<double>>();
    c.pseudo_up_sum = s.at("pseudo_up_sum").get<std::vector<double>>();
    c.pseudo_down_count = s.at("pseudo_down_count").get<std::vector<Count>>();
    c.pseudo_up_count = s.at("pseudo_up_count").get<std::vector<Count>>();
    for (const json& node : s.at("open")) {
      CheckpointNode n;
      if (!read_number(node.at("bound"), &n.bound) ||
          !read_number(node.at("estimate"), &n.estimate)) {
        return fail(error, "a non-number in an open node");
      }
      n.depth = node.at("depth").get<Index>();
      for (const json& d : node.at("domain")) {
        CheckpointChange change;
        change.column = d.at(0).get<Index>();
        change.is_upper = d.at(1).get<bool>();
        if (!read_number(d.at(2), &change.value))
          return fail(error, "a non-number in a domain");
        n.domain.push_back(change);
      }
      c.open.push_back(std::move(n));
    }
  } catch (const json::exception& e) {
    // nlohmann reports a missing or mistyped field by exception; the checksum matched, so
    // this is a file written by something other than this build.
    return fail(error,
                fmt::format("{}: the checkpoint's state is malformed ({})", path, e.what()));
  }
  *out = std::move(c);
  return true;
}

}  // namespace sankhya::mip
