//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"

#include <algorithm>
#include <cstddef>

using namespace vpux;

bool VPU::hasMaxKernelSize(mlir::Operation* op) {
    auto module = getModuleOp(op);
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp != nullptr) {
        auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(VPU::MAX_KERNEL_SIZE);
        if (attrValue != nullptr) {
            return true;
        }
    }
    return false;
}

int64_t VPU::getMaxKernelSize(mlir::Operation* op) {
    return VPU::getConstraint<int64_t>(op, VPU::MAX_KERNEL_SIZE);
}
