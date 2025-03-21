//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/transforms/factories/shave_kernel_info.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/shave_kernel_info.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/shave_kernel_info.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

std::unique_ptr<VPU::ShaveKernelInfo> VPU::getShaveKernelInfo(mlir::Operation* op) {
    const auto arch = VPU::getArch(op);
    switch (arch) {
    case VPU::ArchKind::NPU37XX: {
        return std::make_unique<VPU::arch37xx::ShaveKernelInfo>(op);
    }
    default: {
        return std::make_unique<VPU::arch40xx::ShaveKernelInfo>(op);
    }
    }
}
