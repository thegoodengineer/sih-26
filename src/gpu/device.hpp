// SPDX-License-Identifier: Apache-2.0
// SANKHYA - GPU device probe.
//
// Compiled only when SANKHYA_ENABLE_CUDA is set. Included behind that guard from any
// translation unit that needs to know whether a device is present before touching it.
#pragma once

#include <cstddef>
#include <string>

namespace sankhya::gpu {

/// Return true when at least one CUDA-capable device is present and ready.
///
/// On success *description is a human-readable line:
///   "GeForce RTX 4050 Laptop GPU (compute 8.9, 6144 MiB VRAM)"
/// On failure *description holds the reason (no device, driver error, …).
/// description may be null.
[[nodiscard]] bool device_available(std::string* description);

/// Query free and total memory on the selected CUDA device (device 0).
/// Returns false when no device is available or the query fails.
/// Either pointer may be null if the caller does not need that value.
[[nodiscard]] bool device_free_memory(std::size_t* free_bytes, std::size_t* total_bytes);

}  // namespace sankhya::gpu
