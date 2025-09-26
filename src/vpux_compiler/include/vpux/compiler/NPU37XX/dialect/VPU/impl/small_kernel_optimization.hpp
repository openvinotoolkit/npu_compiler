//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

using namespace vpux;

namespace vpux::VPU::arch37xx {

bool isSmallKernelOptimizationSupported();
bool doesWorkloadSupportSmallKernelOpt();

}  // namespace vpux::VPU::arch37xx
