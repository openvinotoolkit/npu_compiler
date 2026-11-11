//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"

namespace vpux::IE::arch60xx {

//
// FuseOutstandingQuantStrategy
//

void FuseOutstandingQuantStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithAvgPool>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithNCEOp<IE::ConvolutionOp>>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithNCEOp<IE::MatMulOp>>(ctx, log);
    // E-134083 - Full support of IE::MultiplyOp and IE::SubtractOp is pending on
    // updating mixed precision support for PTL.
    // Once that is resolved, add patterns for IE::MultiplyOp and IE::SubtractOp as well.
}

}  // namespace vpux::IE::arch60xx
