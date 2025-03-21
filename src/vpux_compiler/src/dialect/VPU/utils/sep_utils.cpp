//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/sep_utils.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool hasSEOption(mlir::ModuleOp module, StringRef seOption) {
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch SEP option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(seOption);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find IE.OptionOp to fetch SEP option");
        return false;
    }
    return static_cast<bool>(attrValue.getOptionValue());
}

bool VPU::hasEnableSEPtrsOperations(mlir::ModuleOp module) {
    return hasSEOption(module, ENABLE_SE_PTRS_OPERATIONS);
}

bool VPU::hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module) {
    return hasSEOption(module, ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS);
}
