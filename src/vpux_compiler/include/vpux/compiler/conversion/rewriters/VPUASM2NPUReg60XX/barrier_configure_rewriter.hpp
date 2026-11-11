//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg60xx {

class BarrierRewriter final : public mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp> {
public:
    BarrierRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPUASM::ConfigureBarrierOp>(ctx), _log(log) {
        setDebugName("ConfigureBarrier_VPUASM2NPUReg60XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::ConfigureBarrierOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};
}  // namespace vpuasm2npureg60xx
}  // namespace vpux
