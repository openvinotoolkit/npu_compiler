//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"

namespace vpux::VPU::arch60xx {

struct WorkloadSizeConstraint final {
    bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op, ShapeRef inputShapeOverride) const;
    bool checkDWOperationWorkloadLimit(mlir::Operation* op, const OutputTiling& tiles) const;
    SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(ArrayRef<int64_t> workloadsChannels) const;
};

}  // namespace vpux::VPU::arch60xx
