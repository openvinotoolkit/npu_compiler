//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"

namespace vpux::VPUIP {

bool isMultiClusterDepthToSpaceDMAOp(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp);

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

class MultiClusterDepthToSpaceDMARewriter final : public mlir::OpRewritePattern<VPUIP::DepthToSpaceDMAOp> {
public:
    MultiClusterDepthToSpaceDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::DepthToSpaceDMAOp depthToSpaceDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

}  // namespace vpux::VPUIP
