//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Types.h>
#include <array>

namespace vpux::VPU::arch60xx {

constexpr std::array<int64_t, 9> supportedChannelsForNPU60XX = {256, 224, 192, 160, 128, 96, 64, 32, 16};

vpux::SmallVector<int64_t> getChannelsSupportedByKernelOptimization(vpux::ArrayRef<int64_t> workloadsChannels,
                                                                    int64_t maxSlotsSum);

}  // namespace vpux::VPU::arch60xx
