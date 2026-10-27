//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux::VPU {
/* @brief
 * Static class for generating MPEEngine attributes.
 */
class MPEEngineConfig {
public:
    static bool isNewWeightTableFormatSupportedWithDwOps([[maybe_unused]] config::ArchKind arch) {
        return false;
    }
};

}  // namespace vpux::VPU
