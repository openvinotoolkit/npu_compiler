//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_outstanding_quant_strategy.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"

namespace vpux::IE::arch37xx {

//
// FuseOutstandingQuantStrategy
//

void FuseOutstandingQuantStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    patterns.add<vpux::IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>>(ctx, log);
    patterns.add<vpux::IE::QuantizeWithAvgPool>(ctx, log);
}

}  // namespace vpux::IE::arch37xx
