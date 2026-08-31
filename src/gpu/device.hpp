// SPDX-License-Identifier: Apache-2.0
// SANKHYA - CUDA device availability interface.

#pragma once

#include <string>

namespace sankhya::gpu {

/// Check whether a CUDA device is available and optionally describe it.
[[nodiscard]] bool device_available(std::string* description);

}  // namespace sankhya::gpu