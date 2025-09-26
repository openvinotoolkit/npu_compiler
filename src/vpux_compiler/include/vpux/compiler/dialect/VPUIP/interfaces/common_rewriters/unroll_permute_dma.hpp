//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"

namespace vpux::VPUIP {

bool isMultiClusterPermuteDMA(VPUIP::PermuteDMAOp permuteDMAOp);

class SingleClusterPermuteDMARewriter final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    SingleClusterPermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unroll(VPUIP::PermuteDMAOp permuteDMAOp, mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

class MultiClusterPermuteDMARewriter final : public mlir::OpRewritePattern<VPUIP::PermuteDMAOp> {
public:
    MultiClusterPermuteDMARewriter(mlir::MLIRContext* ctx, int64_t dmaPortCount, Logger log);

    mlir::LogicalResult matchAndRewrite(VPUIP::PermuteDMAOp permuteDMAOp, mlir::PatternRewriter& rewriter) const final;

private:
    mlir::LogicalResult unrollSegmentedOrOverlappedOutput(VPUIP::PermuteDMAOp permuteDMAOp,
                                                          VPUIP::DistributedBufferType distributedType,
                                                          mlir::AffineMap memPerm,
                                                          mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicatedOutput(VPUIP::PermuteDMAOp permuteDMAOp,
                                               VPUIP::DistributedBufferType distributedType, mlir::AffineMap memPerm,
                                               mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicatedInputAndOutput(VPUIP::PermuteDMAOp permuteDMAOp, mlir::AffineMap memPerm,
                                                       mlir::PatternRewriter& rewriter) const;

    mlir::LogicalResult unrollDuplicatedInput(VPUIP::PermuteDMAOp permuteDMAOp, mlir::AffineMap memPerm,
                                              mlir::PatternRewriter& rewriter) const;

    int64_t _dmaPortCount;
    Logger _log;
};

}  // namespace vpux::VPUIP
