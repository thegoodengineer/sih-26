# SPDX-License-Identifier: Apache-2.0
# SANKHYA - test suite.
#
# One binary, many test files. gtest_discover_tests registers each TEST() with ctest
# individually so that a failure names the case rather than the binary.

include(GoogleTest)

add_executable(sankhya_tests
  unit/test_sparse.cpp
  unit/test_sparse_vector.cpp
  unit/test_arena.cpp
  unit/test_c_api.cpp
  unit/test_certificate.cpp
  unit/test_cuts.cpp
  unit/test_miqp.cpp
  unit/test_options.cpp
  unit/test_model.cpp
  unit/test_logging.cpp
  unit/test_mps_reader.cpp
  unit/test_lp_reader.cpp
  unit/test_presolve.cpp
  unit/test_simplex.cpp
  unit/test_dual_simplex.cpp
  unit/test_dense_lu.cpp
  unit/test_sparse_lu.cpp
  unit/test_ldl.cpp
  unit/test_ipm.cpp
  unit/test_writer.cpp
  unit/test_pdhg.cpp
  unit/test_qp.cpp
  unit/test_branch_and_bound.cpp
  # The verification spine (Phase 3). The oracle is exact-arithmetic reference code
  # that lives only in the test binary; nothing in src/ can reach it.
  oracles/rational_simplex.cpp
  oracles/lp_generator.cpp
  oracles/test_rational_oracle.cpp
  oracles/test_fuzz_against_oracle.cpp
  robustness/test_robustness.cpp)

# nlohmann_json is PRIVATE to sankhya_objects, so the stats-blob test has to ask for it
# directly rather than inheriting it.
target_link_libraries(sankhya_tests
  PRIVATE sankhya_core GTest::gtest GTest::gtest_main fmt::fmt
          nlohmann_json::nlohmann_json sankhya_flags)

# tests/ for the shared helpers, src/ so the unit tests can reach implementation headers
# such as src/simplex/dense_lu.hpp that are deliberately not part of the public interface.
target_include_directories(sankhya_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/src")

gtest_discover_tests(sankhya_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 900)
