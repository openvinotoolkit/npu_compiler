//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/small_kernel_optimization.hpp"

using namespace vpux;

namespace vpux::VPU::arch50xx {

bool isSmallKernelOptimizationSupported(mlir::Operation* op, int64_t KX, int64_t KY, int64_t SX,
                                        ArrayRef<VPU::DPUWorkloadOp> workloads);
bool doesWorkloadSupportSmallKernelOpt(int64_t KX, int64_t SX, ArrayRef<int64_t> workloadOutSz, bool isFp16Input,
                                       [[maybe_unused]] int64_t KY, [[maybe_unused]] int64_t padLeft);

}  // namespace vpux::VPU::arch50xx
