//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP {

vpux::NDTypeInterface changeShape(vpux::NDTypeInterface originType, ShapeRef shape, ShapeRef offset, int64_t padAxis);
bool isMultiClusterPerAxisTileDMA(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp);

class SingleClusterPerAxisTileDMARewriter final : public mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp> {
public:
    SingleClusterPerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

class MultiClusterPerAxisTileDMARewriter final : public mlir::OpRewritePattern<VPUIP::PerAxisTileDMAOp> {
public:
    MultiClusterPerAxisTileDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, bool useDMADescriptorAttr,
                                       Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlapped(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                                    VPUIP::DistributedBufferType distributedType,
                                                    mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicated(VPUIP::PerAxisTileDMAOp perAxisTileDMAOp,
                                         VPUIP::DistributedBufferType distributedType,
                                         mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
    bool _useDMADescriptorAttr;
};

}  // namespace vpux::VPUIP
