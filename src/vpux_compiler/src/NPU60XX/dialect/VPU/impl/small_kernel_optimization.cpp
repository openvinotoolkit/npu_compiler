//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

using namespace vpux;

namespace vpux::VPU::arch60xx {

bool isSmallKernelOptimizationSupported(ArrayRef<VPU::DPUWorkloadOp> workloads) {
    return llvm::all_of(workloads, [&](auto workload) {
        const auto wlSizes = workload.getConstOutputSizes();
        return llvm::is_contained(supportedChannelsForNPU60XX, wlSizes[Dims4D::Act::C.ind()]);
    });
}

}  // namespace vpux::VPU::arch60xx
