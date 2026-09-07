// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the one place that talks to the OpenMP runtime (#57).
#pragma once

#include <cstdint>

#include "sankhya/logging.hpp"
#include "sankhya/options.hpp"

#ifdef SANKHYA_HAVE_OPENMP
#include <omp.h>
#endif

namespace sankhya {

/// Apply the `threads` option to the OpenMP runtime, once per solve. 0 means one thread per
/// hardware core (the runtime's default). Without OpenMP the option is acknowledged in the
/// log and nothing else happens: there is no thread pool to size.
inline void apply_thread_option(const Options& options, Logger& logger) {
  const std::int64_t requested = options.get_int("threads");
#ifdef SANKHYA_HAVE_OPENMP
  if (requested > 0) omp_set_num_threads(static_cast<int>(requested));
  logger.debug("Threads: {} worker(s) for the column loops (OpenMP)",
               requested > 0 ? static_cast<int>(requested) : omp_get_max_threads());
#else
  if (requested != 1) {
    logger.info("Threads: {} requested, but this build has no OpenMP; running single-threaded",
                requested);
  }
#endif
}

}  // namespace sankhya
