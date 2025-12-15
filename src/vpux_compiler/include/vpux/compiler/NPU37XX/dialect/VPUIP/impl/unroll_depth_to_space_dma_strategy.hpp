//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

#include <mlir/IR/MLIRContext.h>

namespace vpux::VPUIP::arch37xx {
class SingleClusterDepthToSpaceDMARewriter final : public mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp> {
public:
    SingleClusterDepthToSpaceDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

class UnrollDepthToSpaceDMAStrategy : public IIterativeWalkPassStrategy {
public:
    UnrollDepthToSpaceDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount);

    void addPatterns(SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const final;

private:
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
};

}  // namespace vpux::VPUIP::arch37xx
