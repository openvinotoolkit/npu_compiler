//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/config/utils/setup_pipeline_options_utils.hpp"
#include "vpux/compiler/dialect/config/IR/ops.hpp"

using namespace vpux;

config::PipelineOptionsOp config::getPipelineOptionsOp(mlir::MLIRContext& ctx, mlir::ModuleOp moduleOp) {
    auto pipelineOptionsOp = moduleOp.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    const auto hasPipelineOptions = pipelineOptionsOp != nullptr;
    auto optionsBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());

    if (!hasPipelineOptions) {
        pipelineOptionsOp =
                optionsBuilder.create<config::PipelineOptionsOp>(mlir::UnknownLoc::get(&ctx), config::PIPELINE_OPTIONS);
        pipelineOptionsOp.getOptions().emplaceBlock();
    }

    return pipelineOptionsOp;
}

std::optional<bool> config::tryGetBoolPassOption(mlir::ModuleOp module, StringRef attrName) {
    auto pipelineOptionOp = module.lookupSymbol<config::PipelineOptionsOp>(config::PIPELINE_OPTIONS);
    auto logger = vpux::Logger::global();

    if (pipelineOptionOp == nullptr) {
        logger.trace("Failed to find PipelineOptions to fetch '{0}' option", attrName);
        return {};
    }

    auto attrValue = pipelineOptionOp.lookupSymbol<config::OptionOp>(attrName);
    if (attrValue == nullptr) {
        logger.trace("Failed to find config.OptionOp to fetch '{0}' option", attrName);
        return {};
    }

    auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(attrValue.getOptionValue());
    if (boolAttr == nullptr) {
        logger.trace("Failed to cast config.OptionOp to BoolAttr for '{0}'", attrName);
        return {};
    }
    return boolAttr.getValue();
}
