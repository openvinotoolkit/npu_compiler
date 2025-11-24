//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"

using namespace vpux;

bool VPU::WorkloadSizeConstraint::doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const {
    return self->doesDWOperationNeedWorkloadSplit(op);
}

SmallVector<int64_t> VPU::WorkloadSizeConstraint::getChannelsSupportedBySmallSpatialComputeDwOp(
        ArrayRef<int64_t> workloadsChannels) const {
    return self->getChannelsSupportedBySmallSpatialComputeDwOp(workloadsChannels);
}
