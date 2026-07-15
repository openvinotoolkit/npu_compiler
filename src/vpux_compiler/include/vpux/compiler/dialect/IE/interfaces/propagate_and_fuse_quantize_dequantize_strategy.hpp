//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>

namespace vpux::IE {

class IPropagateAndFuseQuantizeDequantizeStrategy {
public:
    IPropagateAndFuseQuantizeDequantizeStrategy(bool seOpsEnabled): _seOpsEnabled(seOpsEnabled) {
    }

    virtual void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const = 0;

    virtual ~IPropagateAndFuseQuantizeDequantizeStrategy() = default;

protected:
    bool _seOpsEnabled = false;
};

}  // namespace vpux::IE
