//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;

bool hasAdaptiveStrippingOption(mlir::ModuleOp module, StringRef option) {
    auto pipelineOptionOp = module.lookupSymbol<IE::PipelineOptionsOp>(VPU::PIPELINE_OPTIONS);
    if (pipelineOptionOp == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find PipelineOptions to fetch adaptive stripping option");
        return false;
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<IE::OptionOp>(option);
    if (attrValue == nullptr) {
        auto logger = vpux::Logger::global();
        logger.trace("Failed to find IE.OptionOp to fetch adaptive stripping option");
        return false;
    }
    return static_cast<bool>(attrValue.getOptionValue());
}

bool VPU::hasEnableAdaptiveStripping(mlir::ModuleOp module) {
    return hasAdaptiveStrippingOption(module, ENABLE_ADAPTIVE_STRIPPING);
}
