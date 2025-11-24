//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/wrap_vf_base_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPU::VF {

mlir::LogicalResult WrapVFRewriterBase::matchAndRewrite(VPU::VerticalFusionOpInterface origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    if (!opNeedsTobeWrapped(origOp)) {
        _log.trace("Operation '{0}' at '{1}' does not need to be wrapped", origOp->getName(), origOp->getLoc());
        return mlir::failure();
    }
    wrapIntoVFRegion(origOp, rewriter);
    return mlir::success();
}

void WrapVFRewriterBase::wrapIntoVFRegion(VPU::VerticalFusionOpInterface op, mlir::PatternRewriter& rewriter) const {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    const SmallVector<int64_t> one(inputType.getRank(), 1);

    auto tilingStrategyArray = op->hasAttr(tilingStrategy) ? mlir::cast<mlir::ArrayAttr>(op->getAttr(tilingStrategy))
                                                           : getIntArrayAttr(op->getContext(), one);
    const auto bodyBuilder = [op](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(op->getOperands(), newOperands);
        auto* newOp = builder.clone(*op, mapper);
        newOp->removeAttr(tilingStrategy);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };
    rewriter.replaceOpWithNewOp<VPU::VerticalFusionOp>(op, op->getResultTypes(), op->getOperands(), bodyBuilder,
                                                       tilingStrategyArray);
}
}  // namespace vpux::VPU::VF
