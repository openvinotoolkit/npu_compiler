//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"
#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

bool hasAnyChannelSupportedByKernelOptimization() {
    return false;
}

SmallVector<int64_t> getChannelsSupportedByKernelOptimization() {
    return {};
}

bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp) {
    auto outputType = nceOp->getResult(0).getType();

    return mlir::isa<vpux::VPU::DistributedTensorType>(outputType);
}

}  // namespace vpux::VPU::arch37xx
