//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/convert_to_mixed_precision_strategy.hpp"

namespace vpux::IE::arch37xx {
class ConvertToMixedPrecisionStrategy final : public vpux::IE::IConvertToMixedPrecisionStrategy {
public:
    ConvertToMixedPrecisionStrategy(const bool enableFloatInQuantWeightsMixedMode)
            : IConvertToMixedPrecisionStrategy(enableFloatInQuantWeightsMixedMode) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
};

}  // namespace vpux::IE::arch37xx
