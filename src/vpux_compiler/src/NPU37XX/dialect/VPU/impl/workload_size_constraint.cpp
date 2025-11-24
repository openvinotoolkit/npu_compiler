//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/workload_size_constraint.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;

// No further workload split is needed for depthwise operation
bool VPU::arch37xx::WorkloadSizeConstraint::doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const {
    VPUX_UNUSED(op);
    return false;
}

// Passthrough the already supported workload channels
SmallVector<int64_t> VPU::arch37xx::WorkloadSizeConstraint::getChannelsSupportedBySmallSpatialComputeDwOp(
        ArrayRef<int64_t> workloadsChannels) const {
    return SmallVector<int64_t>(workloadsChannels);
}
