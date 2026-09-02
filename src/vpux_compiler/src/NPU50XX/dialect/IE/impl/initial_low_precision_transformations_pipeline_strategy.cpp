//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/initial_low_precision_transformations_pipeline_strategy.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters.hpp"

namespace vpux::IE::arch50xx {

void InitialLowPrecisionTransformationsPipelineStrategy::registerRewriters(RewriterRegistry& registry,
                                                                           Logger& log) const {
    SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(4);
    IE::registerDecomposeMultiZPQuantizationRewriters(registry, benefitLevels, 0, log);
    IE::registerWeightsDequantizeToDynamicDequantizeRewriters(registry, benefitLevels, 1, log);
    IE::registerConsolidateWeightsDequantizationRewriters(registry, benefitLevels, 2, log);
    IE::registerConsolidateActivationFP8QuantizationRewriters(registry, benefitLevels, 3, log);
}
}  // namespace vpux::IE::arch50xx
