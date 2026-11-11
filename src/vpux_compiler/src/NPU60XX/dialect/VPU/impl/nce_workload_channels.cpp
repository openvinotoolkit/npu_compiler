//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

namespace vpux::VPU::arch60xx {

/**
 * @brief These functions check if the given channels are supported for "Kernel Optimisation for DW" (L1aOpt) on NPU60XX
 * architecture.
 *
 * @details After correct-NCE-workloads pass is executed, channels are split on [16, 32, 64, 96, 128, 160, 192,
 * 224, 256]. If "Kernel Optimization for DW" (L1aOpt) matches requirements, we can apply this optimization.
 *
 * Requirements:
 * - kernel_x must be greater than 1
 * - stride_x must be 1 or 2
 * - Valid workload_size_z are 16, 32, 64, 96, 128 up to 256 in multiples of 32, for both 8-bit and 16-bit operations
 *
 * Performance considerations:
 * - This is a performance optimization for DW operations with KX > 1 and SX ∈ {1, 2}. Hardware has specific support for
 *   these workload configurations
 * - The optimization is disabled if the total workload number is greater than the max barrier slot number,
 *   as this would cause the workload to be executed in linearization mode.
 */
SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels,
                                                              const int64_t maxSlotsSum) {
    SmallVector<int64_t> supportedWorkloadChannels;

    llvm::DenseSet<int64_t> supportedWorkloadChannelsSet;
    int64_t totalWorkloadChannels = 0;

    auto appendWorkloadChannel = [&](int64_t wlChannel, ArrayRef<int64_t> supportedChannels) -> void {
        for (auto supportedChannel : supportedChannels) {
            while (wlChannel >= supportedChannel) {
                supportedWorkloadChannelsSet.insert(supportedChannel);
                wlChannel -= supportedChannel;

                totalWorkloadChannels++;
            }
        }
    };

    for (const auto& workloadsChannel : workloadsChannels) {
        appendWorkloadChannel(workloadsChannel, supportedChannelsForNPU60XX);
    }

    // Convert set to a vector and sort it in descending order
    for (const auto& channel : supportedWorkloadChannelsSet) {
        supportedWorkloadChannels.push_back(channel);
    }
    std::sort(supportedWorkloadChannels.begin(), supportedWorkloadChannels.end(), std::greater<int64_t>());

    return totalWorkloadChannels < maxSlotsSum ? std::move(supportedWorkloadChannels) : SmallVector<int64_t>{};
}

}  // namespace vpux::VPU::arch60xx
