//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace vpux::VPU::arch40xx {

bool isSmallKernelOptimizationSupported(mlir::Operation* op, const int64_t KX, const int64_t SX,
                                        ArrayRef<VPU::DPUWorkloadOp> workloads) {
    const auto isFp16Input = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType().isF16();
    auto is16Workload = false;
    auto is64Workload = false;
    const auto workloadChannelsMeetRequirement = llvm::all_of(workloads, [&](auto workload) {
        const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
        is16Workload |= wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16;
        is64Workload |= wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT64;
        return isFp16Input ? wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 ||
                                     wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32
                           : (wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 ||
                              wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32 ||
                              wlSizes[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT64) &&
                                     !(is16Workload && is64Workload);
    });

    return KX == 3 && SX == 1 && workloadChannelsMeetRequirement;
}

bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t SX, ArrayRef<int64_t> workloadOutSz,
                                       bool isFp16Input) {
    // L1Opt can be enabled when kernelX = 3 and strideX = 1
    if (KX != 3 || SX != 1) {
        return false;
    }

    // Float16 input align to 16 generates performance regression.
    return isFp16Input ? workloadOutSz[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32
                       : workloadOutSz[Dims4D::Act::C.ind()] % VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16 == 0;
}

}  // namespace vpux::VPU::arch40xx
