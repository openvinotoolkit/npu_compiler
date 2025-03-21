//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <string>

namespace vpux {

constexpr Byte VPUX37XX_CMX_WORKSPACE_SIZE = Byte(1936_KB);
constexpr Byte VPUX37XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE =
        Byte(static_cast<double>(VPUX37XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO);

constexpr Byte VPUX40XX_CMX_WORKSPACE_SIZE =
        Byte(1440_KB);  // Error from feasibleAllication if 1449_KB; See E62792 and E60873
constexpr Byte VPUX40XX_CMX_WORKSPACE_FRAGMENTATION_AWARE_SIZE =
        Byte(static_cast<double>(VPUX40XX_CMX_WORKSPACE_SIZE.count()) * FRAGMENTATION_AVOID_RATIO);
constexpr int VPUX37XX_MAX_DPU_GROUPS = 2;
constexpr int VPUX40XX_MAX_DPU_GROUPS = 6;

constexpr int VPUX37XX_MAX_DMA_PORTS = 2;
constexpr int VPUX40XX_MAX_DMA_PORTS = 2;

}  // namespace vpux
