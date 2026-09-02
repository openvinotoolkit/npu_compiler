//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/initial_low_precision_transformations_pipeline_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

namespace vpux::IE::arch37xx {

void InitialLowPrecisionTransformationsPipelineStrategy::registerRewriters(RewriterRegistry& registry,
                                                                           Logger& log) const {
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(3);
    IE::registerDecomposeMultiZPQuantizationRewriters(registry, benefitLevels, 0, log);
    IE::registerWeightsDequantizeToDynamicDequantizeRewriters(registry, benefitLevels, 1, log);
    IE::registerConsolidateWeightsDequantizationRewriters(registry, benefitLevels, 2, log);
}
}  // namespace vpux::IE::arch37xx
