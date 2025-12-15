//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/make_ops_with_distributed_tensor.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

using namespace vpux;

//
// NCEPermuteRewriter
//

namespace {

class NCEPermuteRewriter final : public mlir::OpRewritePattern<VPU::NCEPermuteOp> {
public:
    NCEPermuteRewriter(mlir::MLIRContext* ctx, bool enableExplicitDistributionInfoAttr, Logger log)
            : mlir::OpRewritePattern<VPU::NCEPermuteOp>(ctx),
              _enableExplicitDistributionInfoAttr(enableExplicitDistributionInfoAttr),
              _log(log) {
        setDebugName("NCEPermuteRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEPermuteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    vpux::VPU::UnrolledTypeOp buildInputUnrolledType(mlir::PatternRewriter& rewriter,
                                                     VPU::ClusteredOpInterface clusteredOp, mlir::Value input,
                                                     mlir::Type distType) const;
    bool _enableExplicitDistributionInfoAttr = false;
    Logger _log;
};

mlir::LogicalResult NCEPermuteRewriter::matchAndRewrite(VPU::NCEPermuteOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    if (!origOp.getMultiClusterStrategy().has_value()) {
        return matchFailed(_log, rewriter, origOp, "The operation does not have multi-cluster strategy.");
    }

    auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(origOp.getOperation());
    VPUX_THROW_UNLESS(clusteredOp != nullptr, "Operation '{0}' cannot be converted to VPU::ClusteredOpInterface",
                      origOp);

    const auto nextConv = getNextCompressConv(origOp);
    const auto strategy = nextConv == nullptr ? VPU::MultiClusterStrategy::SplitOverHeight
                                              : VPU::MultiClusterStrategy::SplitOverHeightOverlapped;

    auto outputTensorType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto numClusters = VPU::getOptimalNumClusters(clusteredOp, outputTensorType.getShape(), strategy);
    const auto activationTensorDistributionMode = getActivationTensorDistributionMode(clusteredOp, strategy);
    const auto activationTensorNumTiles = getActivationTensorNumTiles(clusteredOp, numClusters, strategy);

    const auto activationAlignment = getActivationTensorAlignment(clusteredOp, numClusters, strategy);

    const auto neutralPads = VPU::Padding(0, 0, 0, 0);
    auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(origOp.getOperation());
    const auto overlapParams =
            VPU::OverlapDistributionParams(nceOp.getKernelSizeVal(), neutralPads, nceOp.getStridesVal());

    auto inputTensorType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());

    const auto uniformDistributedSegments = VPU::getUniformDistributedSegments(
            clusteredOp, inputTensorType.getShape().raw(), activationTensorDistributionMode, activationTensorNumTiles,
            activationAlignment.value_or(SmallVector<int64_t>{}));

    const auto inputDistType = VPU::createDistributedTensorType(
            clusteredOp, inputTensorType, activationTensorDistributionMode, activationTensorNumTiles, numClusters,
            activationAlignment.has_value() ? activationAlignment.value() : ArrayRef<int64_t>{},
            uniformDistributedSegments, _enableExplicitDistributionInfoAttr, overlapParams);
    const auto fusedDistType =
            fuseOverlapParams(clusteredOp, inputDistType, nextConv, _enableExplicitDistributionInfoAttr);

    const auto inputCopyOp = buildInputUnrolledType(rewriter, clusteredOp, origOp.getInput(), fusedDistType);

    const auto distributedOutputTensorType =
            getDistributedOutputTypeFromOp(origOp, origOp->getResult(0).getType(), numClusters, strategy);

    SmallVector<mlir::Value> distributedCopyOps{inputCopyOp->getResult(0)};
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), mlir::ValueRange{distributedCopyOps});
    auto* newOp = rewriter.clone(*origOp, mapper);
    newOp->getResult(0).setType(distributedOutputTensorType);
    if (newOp->hasAttr(vpux::VPU::multiClusterStrategy)) {
        newOp->removeAttr(vpux::VPU::multiClusterStrategy);
    }
    auto outputCopyOp =
            rewriter.create<VPU::UnrolledTypeOp>(newOp->getLoc(), origOp->getResult(0).getType(), newOp->getResult(0));

    rewriter.replaceOp(origOp, outputCopyOp);

    return mlir::success();
}

vpux::VPU::UnrolledTypeOp NCEPermuteRewriter::buildInputUnrolledType(mlir::PatternRewriter& rewriter,
                                                                     VPU::ClusteredOpInterface clusteredOp,
                                                                     mlir::Value input, mlir::Type distType) const {
    rewriter.setInsertionPoint(clusteredOp);
    auto distributedInputCopyOp = rewriter.create<VPU::UnrolledTypeOp>(clusteredOp.getLoc(), distType, input);

    return distributedInputCopyOp;
}

}  // namespace

//
// MakeOpsWithDistributedTensorStrategy
//

void VPU::arch37xx::MakeOpsWithDistributedTensorStrategy::addPatterns(mlir::RewritePatternSet& patterns,
                                                                      Logger& log) const {
    auto ctx = patterns.getContext();
    patterns.add<VPU::ClusteredOpRewriter>(
            ctx, _typeLookup, _inputTypeLookup,
            [](VPU::ClusteredOpInterface op) {
                return !(mlir::isa<VPU::NCEEltwiseOp>(op) || mlir::isa<VPU::NCEPermuteOp>(op));
            },
            log);
    patterns.add<VPU::NCEEltwiseRewriter>(ctx, _typeLookup, _inputTypeLookup, log);
    patterns.add<NCEPermuteRewriter>(ctx, _enableExplicitDistributionInfoAttr, log);
}
