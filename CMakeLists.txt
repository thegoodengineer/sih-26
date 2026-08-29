# SPDX-License-Identifier: Apache-2.0
# SANKHYA - build system.
#
# Layout of targets:
#   sankhya_objects  object library, compiled once, PIC
#   sankhya_core     static library      (the one the CLI and tests link)
#   sankhya_shared   shared library      (what the Python bindings and the C API consumer use)
#   sankhya-cli      command line front end
#   sankhya_tests    GoogleTest binary
#
# Warning flags are applied PER TARGET, never globally, so that -Werror governs our code
# and not the dependencies fetched below.

cmake_minimum_required(VERSION 3.20)

project(sankhya
        VERSION 0.1.0
        DESCRIPTION "Indigenous GPU-accelerated LP/MILP/QP optimization solver"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -----------------------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------------------

option(SANKHYA_ENABLE_CUDA  "Build the CUDA backend (Phase 4)"          OFF)
option(SANKHYA_BUILD_PYTHON "Build the pybind11 bindings (Phase 10)"    OFF)
option(SANKHYA_BUILD_TESTS  "Build the GoogleTest suite"                ON)
option(SANKHYA_SANITIZE     "Build with AddressSanitizer + UBSan"       OFF)
option(SANKHYA_WITH_ZLIB    "Read gzip-compressed model files"          ON)

# -----------------------------------------------------------------------------------------
# Build identification. Every benchmark CSV records the commit, so the binary must know it.
# -----------------------------------------------------------------------------------------

find_package(Git QUIET)
set(SANKHYA_GIT_COMMIT "unknown")
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE SANKHYA_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(SANKHYA_GIT_COMMIT STREQUAL "")
    set(SANKHYA_GIT_COMMIT "unknown")
  endif()
endif()

# -----------------------------------------------------------------------------------------
# Dependencies. Per CLAUDE.md: non-solver libraries only. Each entry is justified in
# docs/PROVENANCE.md, which is the artifact a judge reads.
# -----------------------------------------------------------------------------------------

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG        10.2.1
  GIT_SHALLOW    TRUE)

FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.4.2
  GIT_SHALLOW    TRUE)

set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
  GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(fmt cli11 nlohmann_json)

# Dependency headers are pulled into our translation units, so our -Werror set is applied to
# their code as well as ours - fmt 10.2.1's ranges.h trips -Wtautological-compare on GCC 16,
# for instance. Re-export each dependency's include directories as SYSTEM so the compiler
# suppresses diagnostics originating inside them while still policing every line we write.
function(sankhya_mark_include_dirs_system)
  foreach(dep IN LISTS ARGN)
    if(NOT TARGET ${dep})
      continue()
    endif()
    get_target_property(dep_includes ${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
      set_target_properties(${dep} PROPERTIES
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${dep_includes}")
    endif()
  endforeach()
endfunction()

sankhya_mark_include_dirs_system(fmt CLI11 nlohmann_json)

if(SANKHYA_WITH_ZLIB)
  # Prefer a system zlib; only fetch and build one when the platform has none.
  find_package(ZLIB QUIET)
  if(ZLIB_FOUND)
    message(STATUS "sankhya: using system zlib ${ZLIB_VERSION_STRING}")
    set(SANKHYA_ZLIB_TARGET ZLIB::ZLIB)
  else()
    FetchContent_Declare(zlib
      GIT_REPOSITORY https://github.com/madler/zlib.git
      GIT_TAG        v1.3.1
      GIT_SHALLOW    TRUE)
    set(ZLIB_BUILD_EXAMPLES OFF CACHE INTERNAL "")
    FetchContent_MakeAvailable(zlib)
    # madler/zlib names its static target zlibstatic and does not export a namespaced alias.
    set(SANKHYA_ZLIB_TARGET zlibstatic)
    target_include_directories(zlibstatic PUBLIC
      "${zlib_SOURCE_DIR}" "${zlib_BINARY_DIR}")
    message(STATUS "sankhya: building bundled zlib 1.3.1")
  endif()
endif()

if(SANKHYA_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
endif()

# -----------------------------------------------------------------------------------------
# Warning and sanitizer configuration, applied per target.
# -----------------------------------------------------------------------------------------

add_library(sankhya_flags INTERFACE)

if(MSVC)
  target_compile_options(sankhya_flags INTERFACE /W4 /WX /permissive- /Zc:__cplusplus)
else()
  target_compile_options(sankhya_flags INTERFACE
    -Wall -Wextra -Werror
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wcast-qual
    -Wdouble-promotion
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference)
endif()

if(MINGW)
  # Many Windows boxes have an older MinGW earlier on PATH than the toolchain that built
  # the binary; its libstdc++-6.dll is then loaded first and the process dies at start-up
  # with STATUS_ENTRYPOINT_NOT_FOUND (0xC0000139) before main() runs. Linking the runtime
  # statically makes the produced .exe self-contained and removes the whole failure mode.
  target_link_options(sankhya_flags INTERFACE -static-libgcc -static-libstdc++ -static)
endif()

if(SANKHYA_SANITIZE)
  if(MSVC)
    message(FATAL_ERROR "SANKHYA_SANITIZE is not supported with MSVC")
  endif()
  target_compile_options(sankhya_flags INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(sankhya_flags INTERFACE -fsanitize=address,undefined)
endif()

# -----------------------------------------------------------------------------------------
# The solver core
# -----------------------------------------------------------------------------------------

set(SANKHYA_CORE_SOURCES
  src/api/c_api.cpp
  src/core/model.cpp
  src/core/solve.cpp
  src/io/line_reader.cpp
  src/io/lp_reader.cpp
  src/io/mps_reader.cpp
  src/io/writer.cpp
  src/qp/convexity.cpp
  src/qp/qp_condat_vu.cpp
  src/la/lu.cpp
  src/la/scaling.cpp
  src/la/sparse.cpp
  src/simplex/dense_lu.cpp
  src/mip/branch_and_bound.cpp
  src/presolve/presolve.cpp
  src/pdhg/pdhg.cpp
  src/simplex/primal_simplex.cpp
  src/util/logging.cpp
  src/util/options.cpp
  src/util/version.cpp)

add_library(sankhya_objects OBJECT ${SANKHYA_CORE_SOURCES})

target_include_directories(sankhya_objects PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

# src/ is PRIVATE: it lets one module include another module's implementation header
# (src/mip needs src/simplex, and later phases need more of these) without a chain of
# ../ paths. It is deliberately not PUBLIC - nothing outside the library may depend on
# a header that is not in include/sankhya.
target_include_directories(sankhya_objects PRIVATE "${PROJECT_SOURCE_DIR}/src")

target_link_libraries(sankhya_objects
  PUBLIC  fmt::fmt
  # nlohmann_json is used only by src/io/writer.cpp to emit the benchmark stats blob. It is
  # header-only, so PRIVATE keeps it out of every consumer's include path.
  PRIVATE nlohmann_json::nlohmann_json sankhya_flags)

target_compile_definitions(sankhya_objects PRIVATE
  SANKHYA_VERSION="${PROJECT_VERSION}"
  SANKHYA_GIT_COMMIT="${SANKHYA_GIT_COMMIT}"
  SANKHYA_BUILD_TYPE="$<CONFIG>"
  SANKHYA_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

if(SANKHYA_ENABLE_CUDA)
  enable_language(CUDA)
  target_compile_definitions(sankhya_objects PUBLIC SANKHYA_ENABLE_CUDA)
endif()

if(SANKHYA_WITH_ZLIB AND SANKHYA_ZLIB_TARGET)
  target_link_libraries(sankhya_objects PUBLIC ${SANKHYA_ZLIB_TARGET})
  target_compile_definitions(sankhya_objects PUBLIC SANKHYA_WITH_ZLIB)
endif()

add_library(sankhya_core STATIC $<TARGET_OBJECTS:sankhya_objects>)
target_link_libraries(sankhya_core PUBLIC sankhya_objects)

add_library(sankhya_shared SHARED $<TARGET_OBJECTS:sankhya_objects>)
target_link_libraries(sankhya_shared PUBLIC sankhya_objects)
set_target_properties(sankhya_shared PROPERTIES
  OUTPUT_NAME sankhya
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

# -----------------------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------------------

add_executable(sankhya-cli apps/sankhya-cli/main.cpp)
target_link_libraries(sankhya-cli
  PRIVATE sankhya_core CLI11::CLI11 nlohmann_json::nlohmann_json sankhya_flags)
set_target_properties(sankhya-cli PROPERTIES OUTPUT_NAME sankhya)

# -----------------------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------------------

if(SANKHYA_BUILD_TESTS)
  add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------------------
# Configuration summary - printed so a build log records exactly what was built.
# -----------------------------------------------------------------------------------------

message(STATUS "")
message(STATUS "SANKHYA ${PROJECT_VERSION} (${SANKHYA_GIT_COMMIT})")
message(STATUS "  build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA       : ${SANKHYA_ENABLE_CUDA}")
message(STATUS "  sanitizers : ${SANKHYA_SANITIZE}")
message(STATUS "  tests      : ${SANKHYA_BUILD_TESTS}")
message(STATUS "  zlib       : ${SANKHYA_WITH_ZLIB}")
message(STATUS "")
