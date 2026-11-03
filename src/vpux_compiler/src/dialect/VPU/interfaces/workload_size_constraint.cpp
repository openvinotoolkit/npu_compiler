//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"

using namespace vpux;

bool VPU::WorkloadSizeConstraint::doesDWOperationNeedWorkloadSplit(mlir::Operation* op,
                                                                   ShapeRef inputShapeOverride) const {
    return self->doesDWOperationNeedWorkloadSplit(op, inputShapeOverride);
}

bool VPU::WorkloadSizeConstraint::checkDWOperationWorkloadLimit(mlir::Operation* op, const OutputTiling& tiles) const {
    return self->checkDWOperationWorkloadLimit(op, tiles);
}

SmallVector<int64_t> VPU::WorkloadSizeConstraint::getChannelsSupportedBySmallSpatialComputeDwOp(
        ArrayRef<int64_t> workloadsChannels) const {
    return self->getChannelsSupportedBySmallSpatialComputeDwOp(workloadsChannels);
}
