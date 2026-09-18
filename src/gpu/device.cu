// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU device probe.
//
// Intentionally thin: enumerate devices, pick device 0, report name and memory.
// No allocation, no kernel launch, no transfer. Compiled only when
// SANKHYA_ENABLE_CUDA is ON; the guard lives in CMakeLists.txt.

#include "device.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>

namespace sankhya::gpu {

bool device_available(std::string* description) {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    if (description)
      *description = std::string("CUDA driver error: ") + cudaGetErrorString(err);
    return false;
  }
  if (count == 0) {
    if (description) *description = "no CUDA device found";
    return false;
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    if (description) *description = "cudaGetDeviceProperties failed";
    return false;
  }
  if (description) {
    char buf[512];  // prop.name is char[256]; suffix adds ~40 chars
    std::snprintf(buf, sizeof(buf), "%s (compute %d.%d, %.0f MiB VRAM)", prop.name, prop.major,
                  prop.minor, static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0));
    *description = buf;
  }
  return true;
}

bool device_free_memory(std::size_t* free_bytes, std::size_t* total_bytes) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
  std::size_t free_val = 0, total_val = 0;
  if (cudaMemGetInfo(&free_val, &total_val) != cudaSuccess) return false;
  if (free_bytes) *free_bytes = free_val;
  if (total_bytes) *total_bytes = total_val;
  return true;
}

}  // namespace sankhya::gpu
