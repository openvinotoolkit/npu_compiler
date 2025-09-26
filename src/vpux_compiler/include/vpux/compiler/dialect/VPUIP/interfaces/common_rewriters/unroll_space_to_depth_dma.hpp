//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP {

bool isMultiClusterSpaceToDepthDMAOp(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp);

class SingleClusterSpaceToDepthDMARewriter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp> {
public:
    SingleClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

class MultiClusterSpaceToDepthDMARewriter final : public mlir::OpRewritePattern<VPUIP::SpaceToDepthDMAOp> {
public:
    MultiClusterSpaceToDepthDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::SpaceToDepthDMAOp spaceToDepthDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

}  // namespace vpux::VPUIP
