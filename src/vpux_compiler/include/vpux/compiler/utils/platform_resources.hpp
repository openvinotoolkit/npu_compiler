//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <string>

namespace vpux {

constexpr Byte VPUX37XX_CMX_WORKSPACE_SIZE = Byte(1936_KB);
constexpr Byte VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE = Byte(
        static_cast<int64_t>(static_cast<double>(VPUX37XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO));

constexpr Byte VPUX40XX_CMX_WORKSPACE_SIZE = Byte(1439_KB);
constexpr Byte VPUX40XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE = Byte(
        static_cast<int64_t>(static_cast<double>(VPUX40XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO));
constexpr Byte VPUX50XX_CMX_WORKSPACE_SIZE = Byte(1439_KB);
constexpr uint32_t VPUX50XX_CMX_WORKSPACE_OFFSET = 0x18000;
constexpr Byte VPUX50XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE = Byte(
        static_cast<int64_t>(static_cast<double>(VPUX50XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO));

constexpr Byte VPUX5010_CMX_WORKSPACE_SIZE = Byte(1439_KB);
constexpr Byte VPUX5010_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE = Byte(
        static_cast<int64_t>(static_cast<double>(VPUX5010_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO));

constexpr int VPUX37XX_MAX_DPU_GROUPS = 2;
constexpr int VPUX40XX_MAX_DPU_GROUPS = 6;
constexpr int VPUX5010_MAX_DPU_GROUPS = 3;
constexpr int VPUX50XX_MAX_DPU_GROUPS = VPUX5010_MAX_DPU_GROUPS;

constexpr int VPUX37XX_MAX_SHAVES_PER_TILE = 2;
constexpr int VPUX40XX_MAX_SHAVES_PER_TILE = 2;
constexpr int VPUX50XX_MAX_SHAVES_PER_TILE = 2;

constexpr int VPUX37XX_MAX_DMA_PORTS = 2;
constexpr int VPUX40XX_MAX_DMA_PORTS = 2;
constexpr int VPUX50XX_MAX_DMA_PORTS = 2;

}  // namespace vpux
