//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/NPU37XX/dialect/IE/impl/propagate_and_fuse_quantize_dequantize_strategy.hpp"

namespace vpux::IE::arch60xx {

class PropagateAndFuseQuantizeDequantizeStrategy final :
        public vpux::IE::arch37xx::PropagateAndFuseQuantizeDequantizeStrategy {
public:
    PropagateAndFuseQuantizeDequantizeStrategy(bool seOpsEnabled)
            : arch37xx::PropagateAndFuseQuantizeDequantizeStrategy(seOpsEnabled) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
};

}  // namespace vpux::IE::arch60xx
