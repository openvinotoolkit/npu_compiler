//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"

namespace vpux::IE::arch50xx {

//
// FuseOutstandingQuantStrategy
//

void FuseOutstandingQuantStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithAvgPool>(ctx, log);

    // E-134083 - Full support of IE::MultiplyOp and IE::SubtractOp is pending on
    // updating mixed precision support for PTL.
    // When E-134083 is resolved, add patterns for MultiplyOp and SubtractOp.
}

}  // namespace vpux::IE::arch50xx
