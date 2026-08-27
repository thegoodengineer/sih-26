// SPDX-License-Identifier: Apache-2.0
// SANKHYA - logger implementation.

#include "sankhya/logging.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <string>

#include <nlohmann/json.hpp>

namespace sankhya {
namespace {

/// A blank cell for a quantity the engine does not track, so that columns stay aligned.
std::string metric_cell(double value, int width) {
  if (value < 0.0) return fmt::format("{:>{}}", "-", width);
  return fmt::format("{:>{}.2e}", value, width);
}

}  // namespace

const char* to_string(LogLevel level) noexcept {
  switch (level) {
    case LogLevel::kOff: return "off";
    case LogLevel::kError: return "error";
    case LogLevel::kWarning: return "warning";
    case LogLevel::kInfo: return "info";
    case LogLevel::kVerbose: return "verbose";
    case LogLevel::kDebug: return "debug";
  }
  return "unknown";
}

bool parse_log_level(std::string_view text, LogLevel* out) noexcept {
  std::string lowered(text);
  std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (lowered == "off") {
    *out = LogLevel::kOff;
    return true;
  }
  if (lowered == "error") {
    *out = LogLevel::kError;
    return true;
  }
  if (lowered == "warning") {
    *out = LogLevel::kWarning;
    return true;
  }
  if (lowered == "info") {
    *out = LogLevel::kInfo;
    return true;
  }
  if (lowered == "verbose") {
    *out = LogLevel::kVerbose;
    return true;
  }
  if (lowered == "debug") {
    *out = LogLevel::kDebug;
    return true;
  }
  return false;
}

void Logger::write(LogLevel level, std::string_view line) {
  if (!enabled(level)) return;
  const char* prefix = "";
  if (level == LogLevel::kError)
    prefix = "error: ";
  else if (level == LogLevel::kWarning)
    prefix = "warning: ";
  std::fputs(prefix, stream_);
  std::fwrite(line.data(), 1, line.size(), stream_);
  std::fputc('\n', stream_);
}

void Logger::begin_iteration_table() {
  if (!enabled(LogLevel::kInfo)) return;
  in_node_table_ = false;
  rows_since_header_ = 0;
  write(LogLevel::kInfo, "");
  write(LogLevel::kInfo, fmt::format("{:>10}  {:>16}  {:>11}  {:>11}  {:>9}", "Iteration",
                                     "Objective", "Primal Inf", "Dual Inf", "Time"));
  write(LogLevel::kInfo, std::string(std::size_t{10 + 2 + 16 + 2 + 11 + 2 + 11 + 2 + 9}, '-'));
}

void Logger::iteration(Count iteration_number, double objective, double primal_infeasibility,
                       double dual_infeasibility, double seconds) {
  write_progress_iteration(iteration_number, objective, seconds);
  if (!enabled(LogLevel::kInfo)) return;
  if (header_interval_ > 0 && rows_since_header_ >= header_interval_) {
    begin_iteration_table();
  }
  write(LogLevel::kInfo, fmt::format("{:>10}  {:>16.8e}  {}  {}  {:>8.2f}s", iteration_number,
                                     objective, metric_cell(primal_infeasibility, 11),
                                     metric_cell(dual_infeasibility, 11), seconds));
  ++rows_since_header_;
}

void Logger::begin_node_table() {
  if (!enabled(LogLevel::kInfo)) return;
  in_node_table_ = true;
  rows_since_header_ = 0;
  write(LogLevel::kInfo, "");
  write(LogLevel::kInfo, fmt::format("{:>10}  {:>8}  {:>16}  {:>16}  {:>8}  {:>9}", "Nodes",
                                     "Open", "Incumbent", "Best Bound", "Gap", "Time"));
  write(LogLevel::kInfo,
        std::string(std::size_t{10 + 2 + 8 + 2 + 16 + 2 + 16 + 2 + 8 + 2 + 9}, '-'));
}

void Logger::node(Count nodes, Count open_nodes, double incumbent, double dual_bound,
                  double relative_gap, double seconds) {
  write_progress_node(nodes, incumbent, dual_bound, relative_gap, seconds);
  if (!enabled(LogLevel::kInfo)) return;
  if (header_interval_ > 0 && rows_since_header_ >= header_interval_) {
    begin_node_table();
  }
  // An absent incumbent is passed as an infinity; print a dash rather than "inf".
  const std::string incumbent_cell =
      std::isinf(incumbent) ? fmt::format("{:>16}", "-") : fmt::format("{:>16.8e}", incumbent);
  const std::string gap_cell = (relative_gap < 0.0 || std::isinf(relative_gap))
                                   ? fmt::format("{:>8}", "-")
                                   : fmt::format("{:>7.2f}%", relative_gap * 100.0);
  write(LogLevel::kInfo,
        fmt::format("{:>10}  {:>8}  {}  {:>16.8e}  {}  {:>8.2f}s", nodes, open_nodes,
                    incumbent_cell, dual_bound, gap_cell, seconds));
  ++rows_since_header_;
}

Logger& default_logger() {
  static Logger logger(stdout, LogLevel::kInfo);
  return logger;
}

void Logger::enable_progress_output(const std::string& path) {
  if (progress_stream_ != nullptr) std::fclose(progress_stream_);
  progress_stream_ = std::fopen(path.c_str(), "w");
  if (progress_stream_ == nullptr) {
    warning("could not open progress output file '{}'; continuing without it", path);
  }
}

void Logger::write_progress_iteration(Count iteration_number, double objective,
                                      double seconds) {
  if (progress_stream_ == nullptr) return;
  const nlohmann::json line = {
      {"elapsed_s", seconds},   {"iterations", iteration_number}, {"nodes", nullptr},
      {"best_bound", objective}, {"best_integer", nullptr},        {"gap_pct", nullptr},
  };
  const std::string text = line.dump();
  std::fwrite(text.data(), 1, text.size(), progress_stream_);
  std::fputc('\n', progress_stream_);
  std::fflush(progress_stream_);
}

void Logger::write_progress_node(Count nodes, double incumbent, double dual_bound,
                                 double relative_gap, double seconds) {
  if (progress_stream_ == nullptr) return;
  // An absent incumbent arrives as an infinity (see node()); an absent gap as either a
  // negative value or an infinity, matching the console table's own "not tracked" cases.
  const bool have_incumbent = !std::isinf(incumbent);
  const bool have_gap = !std::isinf(relative_gap) && relative_gap >= 0.0;
  const nlohmann::json line = {
      {"elapsed_s", seconds},
      {"iterations", nullptr},
      {"nodes", nodes},
      {"best_bound", dual_bound},
      {"best_integer", have_incumbent ? nlohmann::json(incumbent) : nlohmann::json(nullptr)},
      {"gap_pct",
       have_gap ? nlohmann::json(relative_gap * 100.0) : nlohmann::json(nullptr)},
  };
  const std::string text = line.dump();
  std::fwrite(text.data(), 1, text.size(), progress_stream_);
  std::fputc('\n', progress_stream_);
  std::fflush(progress_stream_);
}

Logger::~Logger() {
  if (progress_stream_ != nullptr) std::fclose(progress_stream_);
}

}  // namespace sankhya
