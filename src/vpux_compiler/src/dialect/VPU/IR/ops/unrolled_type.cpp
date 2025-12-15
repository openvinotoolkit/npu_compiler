//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/PatternMatch.h>
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"

using namespace vpux;

//
// fold
//

mlir::OpFoldResult vpux::VPU::UnrolledTypeOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// FuseCopies
//

namespace {

class FuseUnrolledTypes final : public mlir::OpRewritePattern<VPU::UnrolledTypeOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(VPU::UnrolledTypeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseUnrolledTypes::matchAndRewrite(VPU::UnrolledTypeOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    auto producerUnrolledOp = origOp.getInput().getDefiningOp<VPU::UnrolledTypeOp>();
    if (producerUnrolledOp == nullptr) {
        return mlir::failure();
    }
    // The I/O types of this CopyOp chain should not contain Distributed types
    auto isDistributedType = [](mlir::Value val) {
        auto distributedIf = mlir::dyn_cast_or_null<VPU::DistributedTypeInterface>(val.getType());
        return distributedIf != nullptr && distributedIf.containsDistributedTypes();
    };
    if (isDistributedType(producerUnrolledOp.getInput()) || isDistributedType(producerUnrolledOp.getOutput()) ||
        isDistributedType(origOp.getInput()) || isDistributedType(origOp.getOutput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<VPU::UnrolledTypeOp>(origOp, origOp.getType(), producerUnrolledOp.getInput());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::VPU::UnrolledTypeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseUnrolledTypes>(ctx);
}
