//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/dma_descriptor_generator.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::VPUIP {

class SingleClusterExpandDMARewriter final : public mlir::OpRewritePattern<VPUIP::ExpandDMAOp> {
public:
    SingleClusterExpandDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log)
            : mlir::OpRewritePattern<VPUIP::ExpandDMAOp>(ctx), _log(log), _ctx(ctx), _dmaPortCount(dmaPortCount) {
        setDebugName("SingleClusterExpandDMARewriter");

        _cmxNameAttr = mlir::FlatSymbolRefAttr::get(ctx, stringifyEnum(VPU::MemoryKind::CMX_NN));
    }

    mlir::LogicalResult matchAndRewrite(VPUIP::ExpandDMAOp expandDmaOp, mlir::PatternRewriter& rewriter) const final;

private:
    void createTilesForLargeSize(VPUIP::ExpandDMAOp origOp, VPUIP::ExpandDmaDescriptorGenerator& dmaDescriptorGenerator,
                                 mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollSingleTile(VPUIP::ExpandDMAOp expandDmaOp,
                                         VPUIP::ExpandDmaDescriptorGenerator& dmaDescriptorGenerator,
                                         mlir::PatternRewriter& rewriter) const;

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
    void unrollSegmentedOrOverlapped(mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp, VPURT::TaskOp vpurtTask,
                                     VPUIP::DistributedBufferType distributedType,
                                     VPUIP::ExpandDmaDescriptorGenerator& dmaDescriptorGenerator,
                                     mlir::PatternRewriter& rewriter) const;
    void unrollDuplicated(mlir::Location loc, VPUIP::ExpandDMAOp expandDmaOp, VPURT::TaskOp vpurtTask,
                          VPUIP::DistributedBufferType distributedType,
                          VPUIP::ExpandDmaDescriptorGenerator& dmaDescriptorGenerator,
                          mlir::PatternRewriter& rewriter) const;

    Logger _log;
    mlir::MLIRContext* _ctx;
    int64_t _dmaPortCount;
    mlir::FlatSymbolRefAttr _cmxNameAttr;
};

}  // namespace vpux::VPUIP
