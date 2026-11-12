//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/propagate_and_fuse_quantize_dequantize_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/propagate_and_fuse_quantize_dequantize.hpp"

namespace vpux::IE::arch60xx {

void PropagateAndFuseQuantizeDequantizeStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    arch37xx::PropagateAndFuseQuantizeDequantizeStrategy::addPatterns(patterns, log);

    auto ctx = patterns.getContext();
    patterns.add<vpux::IE::PropagateDynamicDequantize>(ctx, log.nest(), _seOpsEnabled);
}

}  // namespace vpux::IE::arch60xx
