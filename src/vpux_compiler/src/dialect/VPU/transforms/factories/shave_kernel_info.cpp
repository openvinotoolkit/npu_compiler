//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_kernel_info.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/shave_kernel_info.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/shave_kernel_info.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

std::unique_ptr<VPU::ShaveKernelInfo> VPU::getShaveKernelInfo(mlir::Operation* op) {
    const auto arch = config::getArch(op);
    switch (arch) {
    case config::ArchKind::NPU37XX: {
        return std::make_unique<VPU::arch37xx::ShaveKernelInfo>(op);
    }
    default: {
        return std::make_unique<VPU::arch40xx::ShaveKernelInfo>(op);
    }
    }
}
