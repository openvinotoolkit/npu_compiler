//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"

using namespace vpux;

namespace vpux::VPU::arch40xx {

bool hasAnyChannelSupportedByKernelOptimization(ArrayRef<int64_t> supportedChannels, int64_t KX, int64_t SX);
SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels, int64_t maxSlotsSum);
bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp);

}  // namespace vpux::VPU::arch40xx
