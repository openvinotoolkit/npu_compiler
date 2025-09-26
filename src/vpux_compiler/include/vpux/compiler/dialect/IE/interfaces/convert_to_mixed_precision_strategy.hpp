//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {

class IConvertToMixedPrecisionStrategy {
public:
    IConvertToMixedPrecisionStrategy(const bool enableFloatInQuantWeightsMixedMode)
            : _enableFloatInQuantWeightsMixedMode(enableFloatInQuantWeightsMixedMode) {
    }

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;

    virtual ~IConvertToMixedPrecisionStrategy() = default;

protected:
    bool _enableFloatInQuantWeightsMixedMode = true;
};

}  // namespace vpux::IE
