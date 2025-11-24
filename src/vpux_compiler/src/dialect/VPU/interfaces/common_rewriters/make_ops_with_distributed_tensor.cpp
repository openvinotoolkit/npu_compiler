//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/make_ops_with_distributed_tensor.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;
using namespace VPU;

mlir::LogicalResult VPU::ClusteredOpRewriter::matchAndRewrite(VPU::ClusteredOpInterface origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!_isOpLegalToRewrite(origOp)) {
        return matchFailed(_log, rewriter, origOp, "unsupported op");
    }

    if (!origOp->hasAttr(multiClusterStrategy)) {
        return matchFailed(_log, rewriter, origOp, "can't rewrite clustered op without multicluster strategy");
    }

    SmallVector<mlir::Value> distributedCopyOps{};
    const auto& operandLookup = _inputTypeLookup.at(origOp.getOperation());
    for (auto& operand : origOp->getOpOperands()) {
        auto copyOp = createDistributedUnrolledTypeIn(rewriter, origOp, operand.get(),
                                                      operandLookup.at(operand.getOperandNumber()));
        distributedCopyOps.push_back(copyOp.getResult());
    }

    SmallVector<mlir::Type> distributedOutputTypes{};
    for (const auto& origOutput : origOp->getResults()) {
        _log.trace("[{0}] Got tag: {1}\n", getDebugName(), origOutput);
        distributedOutputTypes.push_back(_typeLookup.at(origOutput));
    }

    auto* newOp = rewriter.clone(*origOp);
    for (auto operand : origOp->getOperands() | indexed) {
        newOp->setOperand(operand.index(), distributedCopyOps[operand.index()]);
    }

    SmallVector<mlir::Value> newCopyOutputs;
    rewriter.setInsertionPointAfter(newOp);
    for (auto result : newOp->getResults() | indexed) {
        result.value().setType(distributedOutputTypes[result.index()]);
        auto outputCopyOp = rewriter.create<VPU::UnrolledTypeOp>(
                newOp->getLoc(), origOp->getResult(result.index()).getType(), result.value());
        newCopyOutputs.push_back(outputCopyOp->getResult(0));
    }

    newOp->removeAttr(multiClusterStrategy);

    rewriter.replaceOp(origOp, newCopyOutputs);
    return mlir::success();
}

//
// NCEEltwiseRewriter
//

mlir::LogicalResult VPU::NCEEltwiseRewriter::matchAndRewrite(VPU::NCEEltwiseOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    auto distributedOutputTensorType = _typeLookup.at(origOp->getResult(0));

    SmallVector<mlir::Value> distributedCopyOps{};
    const auto& operandLookup = _inputTypeLookup.at(origOp.getOperation());

    if (origOp.getInput1() == origOp.getInput2()) {
        auto& operand = origOp->getOpOperand(0);
        const auto distributedActivationCopyOp = createDistributedUnrolledTypeIn(
                rewriter, clusteredOp, origOp.getInput1(), operandLookup.at(operand.getOperandNumber()));
        distributedCopyOps.push_back(distributedActivationCopyOp->getResult(0));
    } else {
        for (auto& operand : origOp->getOpOperands()) {
            auto copyOp = createDistributedUnrolledTypeIn(rewriter, clusteredOp, operand.get(),
                                                          operandLookup.at(operand.getOperandNumber()));
            distributedCopyOps.push_back(copyOp.getResult());
        }
    }

    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(multiClusterStrategy)) {
        newOp->removeAttr(multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::UnrolledTypeOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0));

    rewriter.replaceOp(origOp, outputCopyOp);

    return mlir::success();
}
