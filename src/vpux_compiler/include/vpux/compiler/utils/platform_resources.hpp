//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <string>

namespace vpux {

constexpr uint32_t CMX_BASE_ADDR = 0x40000000;
constexpr uint32_t CMX_WORKSPACE_OFFSET = 0x18000;
constexpr Byte CMX_SHAVE_STACK_SIZE = Byte(7_KB);

// CMX workspace sizes grouped by hardware capacity
constexpr Byte CMX_WORKSPACE_SIZE_1936KB = Byte(1936_KB);
constexpr Byte CMX_WORKSPACE_SIZE_1439KB = Byte(1439_KB);
constexpr Byte CMX_WORKSPACE_SIZE_1951KB = Byte(1951_KB);

// Maximum DPU cluster counts
constexpr int DPU_GROUPS_1 = 1;
constexpr int DPU_GROUPS_2 = 2;
constexpr int DPU_GROUPS_3 = 3;
constexpr int DPU_GROUPS_4 = 4;
constexpr int DPU_GROUPS_6 = 6;

// Maximum SHAVE ACT executors per tile
constexpr int MAX_SHAVES_PER_TILE_2 = 2;
constexpr int MAX_SHAVES_PER_TILE_4 = 4;

// Maximum DMA engine ports
constexpr int MAX_DMA_PORTS_2 = 2;
constexpr int MAX_DMA_PORTS_4 = 4;

// Maximum HW barriers per tile
constexpr int MAX_BARRIERS_PER_TILE_16 = 16;
constexpr int MAX_BARRIERS_PER_TILE_32 = 32;

}  // namespace vpux
