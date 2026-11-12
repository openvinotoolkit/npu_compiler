//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU50XX/dialect/IE/impl/convert_to_mixed_precision_strategy.hpp"

namespace vpux::IE::arch60xx {
class ConvertToMixedPrecisionStrategy final : public vpux::IE::arch50xx::ConvertToMixedPrecisionStrategy {
public:
    ConvertToMixedPrecisionStrategy(const bool enableFloatInQuantWeightsMixedMode)
            : arch50xx::ConvertToMixedPrecisionStrategy(enableFloatInQuantWeightsMixedMode) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
};

}  // namespace vpux::IE::arch60xx
