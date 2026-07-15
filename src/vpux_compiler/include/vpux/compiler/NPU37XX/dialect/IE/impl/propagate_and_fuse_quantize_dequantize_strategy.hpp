//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/interfaces/propagate_and_fuse_quantize_dequantize_strategy.hpp"

namespace vpux::IE::arch37xx {

class PropagateAndFuseQuantizeDequantizeStrategy : public vpux::IE::IPropagateAndFuseQuantizeDequantizeStrategy {
public:
    PropagateAndFuseQuantizeDequantizeStrategy(bool seOpsEnabled)
            : IPropagateAndFuseQuantizeDequantizeStrategy(seOpsEnabled) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
};

}  // namespace vpux::IE::arch37xx
