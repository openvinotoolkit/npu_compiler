//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/impl/shave_kernel_info.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"

using namespace vpux;

Bit VPU::arch40xx::ShaveKernelInfo::getShaveVectorSize() const {
    // MVN kernel on LNL currently uses 256 bit vectorization
    // see: E#101157, E#28163
    if (mlir::isa<IE::MVNOp, VPU::MVNOp>(_swOp)) {
        return Bit(256);
    }
    // Default to use 37XX kernels
    return VPU::arch37xx::ShaveKernelInfo::getShaveVectorSize();
}
