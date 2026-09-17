// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CPU/CUDA PDHG numerical agreement tests (issue #283).
//
// Solves the nine committed Netlib instances (data/netlib/reference.json) with both the
// CPU PDHG engine (algorithm=pdhg, gpu=false) and the CUDA PDHG engine (algorithm=pdhg,
// gpu=true) and asserts that the resulting objectives satisfy:
//
//   |obj_cpu - obj_cuda| <= kAgreementTol * max(1, |obj_cpu|, |obj_cuda|)
//
// where kAgreementTol = 1e-9.  Tolerance rationale: GPU floating-point execution may
// reorder operations relative to CPU, producing differences at the 1e-13..1e-15 level
// (well within the 1e-9 budget); a genuine numerical regression produces much larger
// divergence and is caught here.  The threshold is documented here, not scattered across
// individual assertions, so it can be reviewed and tightened in one place.
//
// When the CUDA backend is not compiled in (gpu=true falls back to pdhg-cpu), the test
// calls GTEST_SKIP so that "skipped" is distinguishable from "passed" in CI output.
// Depends on #16 and #17.

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sankhya/io.hpp"
#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya {
namespace {

std::string repository_path(const char* relative) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path().parent_path() / relative)
      .string();
}

// Scale-aware objective comparison.  Documented in the file header.
constexpr double kAgreementTol = 1e-9;

Options pdhg_regression_options(bool gpu) {
  Options o;
  o.set_bool("log_to_console", false);
  o.set_string("algorithm", "pdhg");
  o.set_bool("gpu", gpu);
  // The polish (#229) hands the PDHG point to the interior-point engine and tests a
  // different code path.  Disabled here so the comparison is purely between the two
  // first-order engines.
  o.set_bool("pdhg_polish", false);
  // 1e-8: the tighter of the two project-standard tolerances (ENGINEERING_RULES.md). Note
  // that kAgreementTol = 1e-9 is TIGHTER than this stopping tolerance, not looser: the two
  // runs are expected to take the same iterates (same restarts, same step sizes) and differ
  // only by floating-point reordering inside each mat-vec, so their final objectives should
  // agree far below the stopping tolerance. If the CUDA path ever takes a different
  // trajectory (a restart decided differently by a 1-ulp change), this test says so, and
  // the right response is to understand why, not to loosen kAgreementTol.
  o.set_double("pdhg_tolerance", 1e-8);
  // Netlib small instances need up to ~200k iterations at 1e-4 (adlittle, test_pdhg.cpp).
  // At 1e-8 the count is higher; 1e6 is the budget here.
  o.set_int("iteration_limit", 1000000);
  return o;
}

// Returns true when the CUDA backend actually ran rather than falling back to pdhg-cpu.
bool cuda_was_used(const Solution& s) {
  return s.algorithm.find("cuda") != std::string::npos ||
         s.algorithm.find("gpu") != std::string::npos;
}

// The nine instances committed in data/netlib/reference.json, ordered small to large.
const std::vector<const char*> kNetlibInstances = {
    "afiro", "sc50a", "sc50b", "adlittle", "blend", "share2b", "sc105", "stocfor1", "israel",
};

// =========================================================================================

TEST(PdhgCudaRegression, NineNetlibInstancesAgreeToOnePart1e9) {
  // --- CUDA availability probe ---
  // Solve the smallest committed instance (afiro) with gpu=true.  If the engine falls back
  // to the CPU path the algorithm field is "pdhg-cpu" and the test skips rather than fails,
  // so "skipped" and "passed" are distinguishable in CI.
  const std::string probe_path = repository_path("data/netlib/afiro.mps");
  Model probe_model;
  const io::ReadResult probe_read = io::read_model(probe_path, &probe_model);
  ASSERT_TRUE(probe_read.ok) << probe_path << ": " << probe_read.error
                             << " (instance is committed; a failure here means the file moved)";
  // One iteration is enough to learn which engine ran; a full solve at 1e-8 here would
  // cost every CPU-only CI run a solve it then throws away.
  Options probe_options = pdhg_regression_options(/*gpu=*/true);
  probe_options.set_int("iteration_limit", 1);
  const Solution probe = solve(probe_model, probe_options);
  if (!cuda_was_used(probe)) {
    GTEST_SKIP() << "CUDA backend not in this build (gpu=true ran as \"" << probe.algorithm
                 << "\"): skipped, not passed.  Depends on #16 and #17.";
  }

  // --- Nine Netlib instances ---
  int agreed = 0;
  int failed = 0;

  for (const char* name : kNetlibInstances) {
    const std::string path =
        repository_path((std::string("data/netlib/") + name + ".mps").c_str());
    Model model;
    const io::ReadResult r = io::read_model(path, &model);
    ASSERT_TRUE(r.ok) << path << ": " << r.error;

    const Solution cpu = solve(model, pdhg_regression_options(/*gpu=*/false));
    const Solution gpu = solve(model, pdhg_regression_options(/*gpu=*/true));

    EXPECT_EQ(cpu.status, SolveStatus::kOptimal) << name << " CPU: " << cpu.message;
    EXPECT_EQ(gpu.status, SolveStatus::kOptimal) << name << " CUDA: " << gpu.message;
    if (cpu.status != SolveStatus::kOptimal || gpu.status != SolveStatus::kOptimal) {
      ++failed;
      continue;
    }

    const double scale = std::max({1.0, std::fabs(cpu.objective), std::fabs(gpu.objective)});
    const double diff = std::fabs(cpu.objective - gpu.objective);
    const bool ok = diff <= kAgreementTol * scale;

    std::cout << name << ": cpu=" << cpu.objective << "  cuda=" << gpu.objective
              << "  rel=" << diff / scale << (ok ? "  PASS" : "  FAIL") << "\n";

    EXPECT_TRUE(ok) << name << ": objective mismatch\n"
                    << "  CPU  objective : " << cpu.objective << "\n"
                    << "  CUDA objective : " << gpu.objective << "\n"
                    << "  relative diff  : " << diff / scale << "\n"
                    << "  allowed        : " << kAgreementTol;
    if (ok) {
      ++agreed;
    } else {
      ++failed;
    }
  }

  std::cout << "CPU/CUDA agreement: " << agreed << "/"
            << static_cast<int>(kNetlibInstances.size())
            << " instances passed (tolerance=" << kAgreementTol << ")\n";
  EXPECT_EQ(failed, 0);
}

}  // namespace
}  // namespace sankhya
