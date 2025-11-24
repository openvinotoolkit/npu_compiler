//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/adjust_for_vpu_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void IE::AdjustForVPUPipelineStrategy::registerRewriters(RewriterRegistry& registry, Logger& log) const {
    IE::registerMergeTileWithSliceRewriters(registry, log);
    IE::registerConvertLargeConvToMultiConvWithAddRewriters(registry, log);
    IE::registerMergeWeightsSharedConvRewriters(registry, log);
    IE::registerPerAxisFQConcatRewriters(registry, log);
    IE::registerConvertShuffleChannelsRewriters(registry, log);
    IE::registerFusePadOpsRewriters(registry, log);
    IE::registerFuseActivationOpsRewriters(registry, _enableFuseClamp, log);
}

std::unique_ptr<IDynamicRewriterStrategy> IE::createAdjustForVPUPipelineStrategy(mlir::func::FuncOp,
                                                                                 bool enableFuseClamp) {
    return std::make_unique<IE::AdjustForVPUPipelineStrategy>(enableFuseClamp);
}
