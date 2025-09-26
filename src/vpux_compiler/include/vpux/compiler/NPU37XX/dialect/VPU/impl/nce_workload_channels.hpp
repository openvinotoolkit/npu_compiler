//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Types.h>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/nce_workload_channels.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

bool hasAnyChannelSupportedByKernelOptimization();
SmallVector<int64_t> getChannelsSupportedByKernelOptimization();
bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp);

}  // namespace vpux::VPU::arch37xx
