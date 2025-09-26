//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

namespace vpux {
namespace VPU {

bool isSmallKernelOptimizationSupported(mlir::Operation* op, config::ArchKind arch, int64_t KX, int64_t KY, int64_t SX,
                                        ArrayRef<VPU::DPUWorkloadOp> workloads);
bool doesWorkloadSupportSmallKernelOpt([[maybe_unused]] config::ArchKind arch, int64_t KX, int64_t SX,
                                       ArrayRef<int64_t> workloadOutSz, bool isFp16Input, [[maybe_unused]] int64_t KY,
                                       [[maybe_unused]] int64_t padLeft);

}  // namespace VPU
}  // namespace vpux
