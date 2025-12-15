//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"

#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

bool isSmallKernelOptimizationSupported() {
    // Small kernel optimization is not supported on NPU37XX.
    return false;
}

bool doesWorkloadSupportSmallKernelOpt() {
    // Small kernel optimization is not supported on NPU37XX.
    return false;
}

}  // namespace vpux::VPU::arch37xx
