//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/propagate_and_fuse_quantize_dequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/propagate_and_fuse_quantize_dequantize.hpp"

namespace vpux::IE::arch37xx {

void PropagateAndFuseQuantizeDequantizeStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::FuseDequantizeWithMultiplier>(ctx, log);
    patterns.add<vpux::IE::PropagateDequantize<IE::DequantizeOp>>(ctx, log.nest(), _seOpsEnabled);
}

}  // namespace vpux::IE::arch37xx
