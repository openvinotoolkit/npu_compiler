//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_quantized_ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE::arch50xx {

//
// FuseQuantizedOpsStrategy
//

void FuseQuantizedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<FuseWithConv>(ctx, log);
    patterns.add<FuseWithGroupConv>(ctx, log);
    patterns.add<FuseWithEltwiseConverter<IE::AddOp>>(ctx, log);
    patterns.add<FuseWithEltwiseConverter<IE::SubtractOp>>(ctx, log);
    // #E157147: Disable fuse quantized for multiply. It will be enabled once it is optimal.
    patterns.add<FuseWithSlice>(ctx, log);
    patterns.add<FuseWithMaxPool>(ctx, log);
    patterns.add<FuseWithTile>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceMeanOp>>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceSumOp>>(ctx, log);
    patterns.add<FuseWithAveragePool>(ctx, log);
    patterns.add<FuseWithConcat>(ctx, log);
    patterns.add<FuseWithMatMul>(ctx, log);
    patterns.add<FuseWithPostOp>(ctx, log);
    // TODO: optimize for SEP Pad & Roll
    if (_seOpsEnabled) {
        patterns.add<FuseWithInterpolate>(ctx, log);
        patterns.add<FuseWithTransposedConv>(ctx, log);
    }
}

}  // namespace vpux::IE::arch50xx
