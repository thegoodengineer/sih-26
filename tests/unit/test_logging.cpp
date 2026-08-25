// SPDX-License-Identifier: Apache-2.0
// SANKHYA - logger tests.
//
// The iteration table is a deliverable, not decoration: the demo shows it to judges and the
// benchmark harness greps it. These tests pin the level filtering and the column layout so
// that a formatting change is a deliberate act rather than an accident.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/logging.hpp"

namespace sankhya {
namespace {

/// Runs `body` with a Logger pointed at a scratch file and returns the lines it wrote.
template <typename Body>
std::vector<std::string> capture(LogLevel level, Body&& body) {
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("sankhya_log_test_" +
       std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()) +
       ".txt");
  std::FILE* stream = std::fopen(path.string().c_str(), "w");
  if (stream == nullptr) {
    ADD_FAILURE() << "could not open scratch log file " << path.string();
    return {};
  }
  {
    Logger logger(stream, level);
    body(logger);
  }
  std::fclose(stream);

  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    lines.push_back(line);
  }
  in.close();
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  return lines;
}

TEST(Logging, LevelNamesRoundTrip) {
  for (const LogLevel level : {LogLevel::kOff, LogLevel::kError, LogLevel::kWarning,
                               LogLevel::kInfo, LogLevel::kVerbose, LogLevel::kDebug}) {
    LogLevel parsed = LogLevel::kOff;
    ASSERT_TRUE(parse_log_level(to_string(level), &parsed)) << to_string(level);
    EXPECT_EQ(parsed, level);
  }
  LogLevel unused = LogLevel::kInfo;
  EXPECT_FALSE(parse_log_level("chatty", &unused));
}

TEST(Logging, ParseIsCaseInsensitive) {
  LogLevel parsed = LogLevel::kOff;
  ASSERT_TRUE(parse_log_level("VERBOSE", &parsed));
  EXPECT_EQ(parsed, LogLevel::kVerbose);
}

TEST(Logging, LevelFiltersMessages) {
  const std::vector<std::string> lines = capture(LogLevel::kWarning, [](Logger& log) {
    log.error("boom");
    log.warning("careful");
    log.info("chatter");
    log.verbose("more chatter");
    log.debug("most chatter");
  });
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(lines[0], "error: boom");
  EXPECT_EQ(lines[1], "warning: careful");
}

TEST(Logging, OffSuppressesEverything) {
  const std::vector<std::string> lines = capture(LogLevel::kOff, [](Logger& log) {
    log.error("boom");
    log.info("chatter");
  });
  EXPECT_TRUE(lines.empty());
}

TEST(Logging, NullStreamIsSafe) {
  Logger logger(nullptr, LogLevel::kDebug);
  EXPECT_FALSE(logger.enabled(LogLevel::kError));
  logger.error("this must not crash");
  logger.begin_iteration_table();
  logger.iteration(1, 1.0, 1.0, 1.0, 1.0);
}

TEST(Logging, FormatArgumentsAreSubstituted) {
  const std::vector<std::string> lines = capture(LogLevel::kInfo, [](Logger& log) {
    log.info("{} rows, {} columns, {:.3f} density", 12, 34, 0.5);
  });
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0], "12 rows, 34 columns, 0.500 density");
}

TEST(Logging, IterationTableHasAlignedColumns) {
  const std::vector<std::string> lines = capture(LogLevel::kInfo, [](Logger& log) {
    log.set_iteration_header_interval(0);
    log.begin_iteration_table();
    log.iteration(0, 0.0, 3.2e1, 0.0, 0.0);
    log.iteration(17, -4.6475314e2, 0.0, 1.5e-9, 0.03);
  });
  // blank line, header, rule, two rows
  ASSERT_EQ(lines.size(), 5u);
  EXPECT_EQ(lines[0], "");
  EXPECT_NE(lines[1].find("Iteration"), std::string::npos);
  EXPECT_NE(lines[1].find("Objective"), std::string::npos);
  EXPECT_NE(lines[1].find("Primal Inf"), std::string::npos);
  EXPECT_NE(lines[1].find("Dual Inf"), std::string::npos);
  EXPECT_EQ(lines[2], std::string(lines[2].size(), '-'));
  EXPECT_EQ(lines[1].size(), lines[2].size());
  // Every data row is the same width as the header, which is what keeps a long log
  // readable and greppable by column.
  EXPECT_EQ(lines[3].size(), lines[1].size());
  EXPECT_EQ(lines[4].size(), lines[1].size());
  EXPECT_NE(lines[4].find("17"), std::string::npos);
}

TEST(Logging, IterationTableBlanksUntrackedMetrics) {
  const std::vector<std::string> lines = capture(LogLevel::kInfo, [](Logger& log) {
    log.set_iteration_header_interval(0);
    log.begin_iteration_table();
    log.iteration(3, 1.0, -1.0, -1.0, 0.5);  // negative == "not tracked"
  });
  ASSERT_EQ(lines.size(), 4u);
  // A dash, never a negative infeasibility, which would read as a real measurement.
  EXPECT_NE(lines[3].find('-'), std::string::npos);
  EXPECT_EQ(lines[3].find("-1.0"), std::string::npos);
}

TEST(Logging, IterationHeaderRepeatsOnInterval) {
  const std::vector<std::string> lines = capture(LogLevel::kInfo, [](Logger& log) {
    log.set_iteration_header_interval(2);
    log.begin_iteration_table();
    for (int i = 0; i < 5; ++i) log.iteration(i, 0.0, 0.0, 0.0, 0.0);
  });
  int headers = 0;
  for (const std::string& line : lines) {
    if (line.find("Iteration") != std::string::npos) ++headers;
  }
  EXPECT_EQ(headers, 3);  // initial + after row 2 + after row 4
}

TEST(Logging, NodeTableBlanksAnAbsentIncumbent) {
  const std::vector<std::string> lines = capture(LogLevel::kInfo, [](Logger& log) {
    log.set_iteration_header_interval(0);
    log.begin_node_table();
    log.node(1, 1, kInfinity, -3.5, kInfinity, 0.01);
    log.node(40, 12, 8.0, 7.6, 0.05, 0.42);
  });
  ASSERT_EQ(lines.size(), 5u);
  EXPECT_EQ(lines[3].find("inf"), std::string::npos);
  EXPECT_NE(lines[4].find("5.00%"), std::string::npos);
  EXPECT_EQ(lines[3].size(), lines[1].size());
  EXPECT_EQ(lines[4].size(), lines[1].size());
}

}  // namespace
}  // namespace sankhya
