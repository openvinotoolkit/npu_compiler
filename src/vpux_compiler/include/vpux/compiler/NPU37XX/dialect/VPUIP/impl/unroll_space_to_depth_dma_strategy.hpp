//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/task.hpp"

namespace vpux::VPUIP::arch37xx {

class SingleClusterSpaceToDepthDMARewriter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp> {
public:
    SingleClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, mlir::PatternRewriter& rewriter) const;
    void unrollBlocksFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollBlocksFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollBlocksFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                    mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNCHW2NCHW(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNHWC2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;
    void unrollDepthFirstNCHW2NHWC(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask,
                                   mlir::PatternRewriter& rewriter) const;

    void createSpaceToDepthDMASubOp(VPUIP::SpaceToDepthDMAOp origOp, vpux::VPURT::TaskOp vpurtTask, ShapeRef subShape,
                                    int64_t srcOffset, int64_t dstOffset, VPUIP::DMADescriptorAttr dmaDescriptor,
                                    int64_t port, mlir::PatternRewriter& rewriter) const;

private:
    int64_t _dmaPortCount;
    Logger _log;
};

class UnrollSpaceToDepthDMAStrategy : public IIterativeWalkPassStrategy {
public:
    UnrollSpaceToDepthDMAStrategy(mlir::MLIRContext* ctx, int64_t dmaPortCount);
    void addPatterns(SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const final;

private:
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
};

}  // namespace vpux::VPUIP::arch37xx
