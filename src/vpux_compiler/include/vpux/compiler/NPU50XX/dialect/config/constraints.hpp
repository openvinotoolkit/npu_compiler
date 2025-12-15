//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace vpux {
namespace arch50xx {

// Details see E#180168
// Base of frequency values used in tables (in MHz).
constexpr uint32_t FREQ_BASE = 1000;
// Step of frequency for each entry in tables (in MHz).
constexpr uint32_t FREQ_STEP = 250;

}  // namespace arch50xx
}  // namespace vpux
