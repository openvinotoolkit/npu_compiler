//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

SmallVector<int64_t> VPU::getSupportedChannelsDW(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return {64, 32, 16};
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}

bool VPU::hasAnyChannelSupportedByKernelOptimization(mlir::Operation* op, ArrayRef<int64_t> supportedChannels,
                                                     const int64_t KX, const int64_t SX) {
    const auto arch = config::getArch(op);

    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::hasAnyChannelSupportedByKernelOptimization();
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return VPU::arch40xx::hasAnyChannelSupportedByKernelOptimization(supportedChannels, KX, SX);
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}

SmallVector<int64_t> VPU::getChannelsSupportedByKernelOptimization(mlir::Operation* op,
                                                                   ArrayRef<int64_t> workloadsChannels,
                                                                   const int64_t maxSlotsSum) {
    const auto arch = config::getArch(op);

    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::getChannelsSupportedByKernelOptimization();
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return VPU::arch40xx::getChannelsSupportedByKernelOptimization(workloadsChannels, maxSlotsSum);
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}

bool VPU::isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp) {
    const auto arch = config::getArch(nceOp);

    switch (arch) {
    case config::ArchKind::NPU37XX:
        return VPU::arch37xx::isNCEPermuteOffsetsCorrectionNeeded(nceOp);
    case config::ArchKind::NPU40XX:
    case config::ArchKind::NPU50XX:
        return VPU::arch40xx::isNCEPermuteOffsetsCorrectionNeeded(nceOp);
    case config::ArchKind::UNKNOWN:
    default:
        VPUX_THROW("Unexpected architecture {0}", arch);
    }
}
