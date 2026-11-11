//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"

namespace vpux::VPU::arch60xx {

bool isSmallKernelOptimizationSupported(ArrayRef<VPU::DPUWorkloadOp> workloads);

}  // namespace vpux::VPU::arch60xx
