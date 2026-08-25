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
  unit/test_options.cpp
  unit/test_model.cpp
  unit/test_logging.cpp
  unit/test_mps_reader.cpp
  unit/test_lp_reader.cpp
  unit/test_simplex.cpp
  unit/test_dense_lu.cpp)

target_link_libraries(sankhya_tests
  PRIVATE sankhya_core GTest::gtest GTest::gtest_main fmt::fmt sankhya_flags)

# tests/ for the shared helpers, src/ so the unit tests can reach implementation headers
# such as src/simplex/dense_lu.hpp that are deliberately not part of the public interface.
target_include_directories(sankhya_tests PRIVATE
  "${PROJECT_SOURCE_DIR}/tests"
  "${PROJECT_SOURCE_DIR}/src")

gtest_discover_tests(sankhya_tests
  DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 300)
