//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/factories/batch_op_processing_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

void IE::BatchOpProcessingPipelineStrategy::registerRewriters(RewriterRegistry& registry, Logger& log) const {
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(5);
    IE::registerMatMulInputsTo2dRewriters(registry, log, benefitLevels, 0, _enableGroupedMatMul);
    IE::registerEliminateReshapeRoundtripInSDPARewriters(registry, log, benefitLevels, 1);
    IE::registerPropagateOpThroughBatchConcatRewriters(registry, log, benefitLevels, 4);
}

std::unique_ptr<IDynamicRewriterStrategy> IE::createBatchOpProcessingPipelineStrategy(mlir::func::FuncOp,
                                                                                      bool enableGroupedMatMul) {
    return std::make_unique<IE::BatchOpProcessingPipelineStrategy>(enableGroupedMatMul);
}
