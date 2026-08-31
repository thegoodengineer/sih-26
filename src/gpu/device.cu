// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA device availability probe.

#include "gpu/device.hpp"

#include <cuda_runtime.h>

#include <string>

namespace sankhya::gpu {

bool device_available(std::string* description) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);

  if (count_status != cudaSuccess) {
    if (description != nullptr) {
      *description = cudaGetErrorString(count_status);
    }
    return false;
  }

  if (device_count == 0) {
    if (description != nullptr) {
      *description = "no CUDA device is visible";
    }
    return false;
  }

  cudaDeviceProp properties{};
  const cudaError_t property_status = cudaGetDeviceProperties(&properties, 0);

  if (property_status != cudaSuccess) {
    if (description != nullptr) {
      *description = cudaGetErrorString(property_status);
    }
    return false;
  }

  if (description != nullptr) {
    *description =
        std::string(properties.name) + " (" +
        std::to_string(properties.totalGlobalMem / (1024ULL * 1024ULL)) + " MiB)";
  }

  return true;
}

}  // namespace sankhya::gpu