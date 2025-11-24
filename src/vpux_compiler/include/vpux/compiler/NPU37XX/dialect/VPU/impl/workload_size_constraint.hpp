//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"

namespace vpux::VPU::arch37xx {

struct WorkloadSizeConstraint final {
    bool doesDWOperationNeedWorkloadSplit(mlir::Operation* op) const;
    SmallVector<int64_t> getChannelsSupportedBySmallSpatialComputeDwOp(ArrayRef<int64_t> workloadsChannels) const;
};

}  // namespace vpux::VPU::arch37xx
