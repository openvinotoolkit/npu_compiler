//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace vpux {
namespace arch37xx {

// Base of frequency values used in tables (in MHz).
constexpr uint32_t FREQ_BASE = 700;
// Step of frequency for each entry in tables (in MHz).
constexpr uint32_t FREQ_STEP = 150;
// Default perf_clk value after dividing by the default frequency divider
constexpr double PERF_CLK_DEFAULT_VALUE_MHZ = 38.4;

}  // namespace arch37xx
}  // namespace vpux
