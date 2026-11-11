//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUASM/ops.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux {
namespace vpuasm2npureg60xx {

class WorkItemRewriter final : public mlir::OpRewritePattern<VPUASM::WorkItemOp> {
public:
    WorkItemRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPUASM::WorkItemOp>(ctx), _log(log) {
        setDebugName("WorkItem_VPUASM2NPUReg40XXRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPUASM::WorkItemOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace vpuasm2npureg60xx
}  // namespace vpux
