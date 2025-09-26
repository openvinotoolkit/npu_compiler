//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"
#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

namespace vpux::VPU::arch40xx {

/**
 * @brief These functions check if the given channels are supported for "Kernel Optimisation for DW" (L1aOpt) on NPU40XX
 * architecture.
 *
 * @details After correct-NCE-workloads pass is executed, channels are split on [16, 32, 64].
 * If "Kernel Optimization for DW" (L1aOpt) matches requirements, we can apply this optimization.
 *
 * Requirements:
 * - kernel_x must be 3
 * - stride_x must be 1
 * - Valid workload_size_z are 16 or 32 for both 8-bit and 16-bit operations
 * - Exception: supported workload_size_z=64 for 8-bit, not added because it causes performance regression.
 *
 * Performance considerations:
 * - This is a performance optimization for DW operations with KX = 3 and SX = 1. Hardware has specific support for
 *   these workloads.
 * - The optimization is disabled if the total workload number is greater than the max barrier slot number,
 *   as this would cause the workload to be executed in linearization mode.
 */
bool hasAnyChannelSupportedByKernelOptimization(ArrayRef<int64_t> supportedChannels, const int64_t KX,
                                                const int64_t SX) {
    return (std::find(supportedChannels.begin(), supportedChannels.end(),
                      VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16) != supportedChannels.end() ||
            std::find(supportedChannels.begin(), supportedChannels.end(),
                      VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32) != supportedChannels.end()) &&
           KX == 3 && SX == 1;
}

SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels,
                                                              const int64_t maxSlotsSum) {
    SmallVector<int64_t> supportedChannels;

    auto workloadChannelsMeetRequirement16 = llvm::all_of(workloadsChannels, [&](const auto& channel) {
        return channel % VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 == 0;
    });
    auto workloadChannelsMeetRequirement32 = llvm::all_of(workloadsChannels, [&](const auto& channel) {
        return channel % VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32 == 0;
    });

    int64_t workloadNumInTotal32 = 0;
    for (auto channel : workloadsChannels) {
        workloadNumInTotal32 += (channel / VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32);
    }
    const auto workloadNumInTotal16 = workloadNumInTotal32 + 1;

    if (workloadNumInTotal32 < maxSlotsSum && workloadChannelsMeetRequirement32) {
        supportedChannels = {VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32};
    } else if (workloadNumInTotal16 < maxSlotsSum && workloadChannelsMeetRequirement16) {
        supportedChannels = {VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32,
                             VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16};
    }

    return supportedChannels;
}

bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface) {
    return false;
}

}  // namespace vpux::VPU::arch40xx
