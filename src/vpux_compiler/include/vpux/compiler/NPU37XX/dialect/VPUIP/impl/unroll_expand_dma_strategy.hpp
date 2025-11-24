//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

namespace vpux::VPUIP::arch37xx {

class SingleClusterExpandDMARewriter final : public mlir::OpRewritePattern<VPUIP::ExpandDMAOp> {
public:
    SingleClusterExpandDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("SingleClusterExpandDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void createTilesForLargeSize(VPUIP::ExpandDMAOp origOp, VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                 mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollSingleTile(VPUIP::ExpandDMAOp origOp,
                                         VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                         mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

class MultiClusterExpandDMARewriter final : public mlir::OpRewritePattern<VPUIP::ExpandDMAOp> {
public:
    MultiClusterExpandDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("MultiClusterExpandDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::ExpandDMAOp origOp, VPURT::TaskOp vpurtTask,
                                     VPUIP::DistributedBufferType distributedType,
                                     VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                                     mlir::PatternRewriter& rewriter) const;
    void unrollDuplicated(mlir::Location loc, VPUIP::ExpandDMAOp origOp, VPURT::TaskOp vpurtTask,
                          VPUIP::DistributedBufferType distributedType,
                          VPUIP::ExpandDmaDescriptorGenerator dmaDescriptorGenerator,
                          mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

class UnrollExpandDMAStrategy : public IGreedilyPassStrategy {
public:
    UnrollExpandDMAStrategy(int64_t dmaPortCount);

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const final;

private:
    int64_t _dmaPortCount;
};

}  // namespace vpux::VPUIP::arch37xx
