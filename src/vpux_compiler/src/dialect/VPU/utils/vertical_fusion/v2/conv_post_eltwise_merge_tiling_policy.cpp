//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/conv_post_eltwise_merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <algorithm>
#include <queue>

namespace vpux::VPU::VF::v2 {
namespace {

SmallVector<mlir::OpOperand*> getLinkedComputeUses(VFConfig& currentConfig, VFConfig& prevConfig) {
    auto currentOp = currentConfig.getSubgraph();
    auto prevOp = prevConfig.getSubgraph();
    if (currentOp != nullptr && prevOp != nullptr) {
        const auto linkedArgs = getLinkedArgumentsBetweenVFOps(currentOp, prevOp);
        if (linkedArgs.size() != 1) {
            return {};
        }
        return details::getComputeUses(linkedArgs.front());
    }

    SmallVector<mlir::OpOperand*> uses;
    const auto& prevOps = prevConfig.getVFOperations();
    const auto& currentOps = currentConfig.getVFOperations();
    for (auto* op : currentOps) {
        if (mlir::isa<VPU::TilingViewLikeOpInterface>(op)) {
            continue;
        }
        for (auto& operand : op->getOpOperands()) {
            auto* parentOp = operand.get().getDefiningOp();
            while (parentOp != nullptr && isPureViewOp(parentOp)) {
                parentOp = parentOp->getOperand(0).getDefiningOp();
            }
            if (prevOps.contains(parentOp)) {
                uses.push_back(&operand);
            }
        }
    }
    return uses;
}

mlir::Operation* getMappedOpInMergedVF(mlir::Operation* op, VPU::VerticalFusionOp prevOp,
                                       VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp mergedOp) {
    return details::getMappedOpInMergedVF(op, prevOp, currentOp, mergedOp);
}

}  // namespace

ConvPostEltwiseMergeTilingPolicy::ConvPostEltwiseMergeTilingPolicy(bool enableVerticalFusionPipelining, Logger log,
                                                                   VFCacheAnalysis& cache)
        : MergeTilingPolicy(MergeTilingPolicyType::ConvPostEltwise, cache, enableVerticalFusionPipelining, log) {
}

StringRef ConvPostEltwiseMergeTilingPolicy::getPatternName() const {
    return "ConvPostEltwiseMergeTilingPolicy";
}

bool ConvPostEltwiseMergeTilingPolicy::match(VFConfig& prevConfig, VFConfig& currentConfig) const {
    return matchPattern(prevConfig, currentConfig);
}

SmallVector<VFSplit> ConvPostEltwiseMergeTilingPolicy::getSplits(DimArrRef, DimArrRef, VFConfig& vfConfig,
                                                                 const VFSplit& prevVFSplit, const VFSplit&) const {
    VPUX_THROW_WHEN(prevVFSplit.empty(), "Previous VF split is empty");
    auto getTilingDim = [&]() {
        if (prevVFSplit.size() == 1) {
            return prevVFSplit.begin()->first;
        }
        auto hasChannelTiling = llvm::any_of(prevVFSplit, [](const auto& item) {
            return item.first == Dims4D::Act::C;
        });
        if (hasChannelTiling) {
            auto convOp = vfConfig.getOutputs().front();
            auto nTilesOnDim = Shape(restoreTilingBySplit(getShape(convOp->getResult(0)).size(), prevVFSplit));
            auto unrollSpatialFirst = isSpatialFirstNestedTiling(convOp, nTilesOnDim);
            if (!unrollSpatialFirst) {
                return Dims4D::Act::C;
            }
        }
        auto iter = llvm::find_if(prevVFSplit, [](const auto& item) {
            return isSpatialDim(item.first);
        });
        VPUX_THROW_WHEN(iter == prevVFSplit.end(), "No spatial dimension found in previous VF split");
        return iter->first;
    };
    Dim outerUnrollDim = getTilingDim();
    auto newSplit = prevVFSplit;
    newSplit[outerUnrollDim] = std::nullopt;
    return {std::move(newSplit)};
}

bool ConvPostEltwiseMergeTilingPolicy::adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentVF,
                                                                  VPU::VerticalFusionOp prevVF,
                                                                  VPU::VerticalFusionOp mergedOp) const {
    VFConfig currentConfig(currentVF, _cache, _enableVerticalFusionPipelining);
    VFConfig prevConfig(prevVF, _cache, _enableVerticalFusionPipelining);

    auto convOp = prevConfig.getOperationsForTiling().front();
    auto eltwiseOp = currentConfig.getOperationsForTiling().front();

    auto linkedArgUses = getLinkedComputeUses(currentConfig, prevConfig);
    auto linkedUseIt = llvm::find_if(linkedArgUses, [&](auto* use) {
        return use->getOwner() == eltwiseOp;
    });
    if (linkedUseIt == linkedArgUses.end()) {
        return false;
    }

    const auto operandIdx = (*linkedUseIt)->getOperandNumber();
    auto mappedConvOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(
            getMappedOpInMergedVF(convOp, prevVF, currentVF, mergedOp));
    auto mappedEltwiseOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(
            getMappedOpInMergedVF(eltwiseOp, prevVF, currentVF, mergedOp));
    if (mappedConvOp == nullptr || mappedEltwiseOp == nullptr) {
        return false;
    }

    if (!mappedConvOp.getMultiClusterStrategy().has_value()) {
        return true;
    }

    SmallVector<mlir::Operation*> viewOps;
    auto viewOp = (*linkedUseIt)->get().getDefiningOp();
    while (viewOp != nullptr && isPureViewOp(viewOp)) {
        viewOps.emplace_back(viewOp);
        viewOp = viewOp->getOperand(0).getDefiningOp();
    }
    std::reverse(viewOps.begin(), viewOps.end());

    SmallVector<mlir::Operation*> mappedViewOps;
    mappedViewOps.reserve(viewOps.size());
    for (auto* op : viewOps) {
        auto* mappedViewOp = getMappedOpInMergedVF(op, prevVF, currentVF, mergedOp);
        if (mappedViewOp == nullptr) {
            return false;
        }
        mappedViewOps.push_back(mappedViewOp);
    }

    OpWithViewInputs parentInfo{checked_cast<int64_t>(operandIdx), mappedConvOp, std::move(mappedViewOps)};
    DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy> rollbackStrategy;
    auto alignedStrategy = alignMCStrategy(parentInfo, mappedEltwiseOp, rollbackStrategy);
    if (mlir::failed(alignedStrategy)) {
        return false;
    }
    mappedEltwiseOp.setMultiClusterStrategy(alignedStrategy.value());
    return true;
}

mlir::FailureOr<VPU::MultiClusterStrategy> ConvPostEltwiseMergeTilingPolicy::alignMCStrategy(
        const OpWithViewInputs& parentOpInfo, VPU::ClusteredOpInterface userOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const {
    auto parentOp = parentOpInfo.clusteredOp;
    if (!parentOp.getMultiClusterStrategy().has_value()) {
        return mlir::failure();
    }

    auto parentStrategy = details::getMultiClusterStrategy(parentOp, rollbackStrategy);
    auto parentOutType = getDistributedOutputType(parentOp, parentOp->getResult(0), parentStrategy);
    if (parentOutType == nullptr) {
        return mlir::failure();
    }

    auto parentOutDataType = mlir::cast<VPU::DistributedTensorType>(parentOutType.getDistributedTypes().front());
    auto inferredInputType = inferDistributedTypeThroughViewOps(parentOutDataType, parentOpInfo.viewOps);
    if (inferredInputType == nullptr) {
        return mlir::failure();
    }

    auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(userOp->getResult(0).getType());
    if (outType == nullptr) {
        return mlir::failure();
    }

    SmallVector<VPU::MultiClusterStrategy> potentialStrategies;
    potentialStrategies.push_back(VPU::MultiClusterStrategy::SplitOverHeight);
    potentialStrategies.push_back(VPU::MultiClusterStrategy::SplitOverKernel);

    for (const auto& strategy : potentialStrategies) {
        auto numClusters = getOptimalNumClusters(userOp, outType.getShape(), strategy);
        if (!userOp.checkStrategyCompatibility(strategy, checked_cast<size_t>(numClusters))) {
            continue;
        }

        auto userInType = getDistributedInputType(userOp, userOp->getOperand(parentOpInfo.operandIdx), strategy);
        if (userInType == nullptr) {
            continue;
        }
        auto userInDataType = mlir::cast<VPU::DistributedTensorType>(userInType.getDistributedTypes().front());
        if (areDistributionAttrsCompatible(inferredInputType, userInDataType, true).failed()) {
            continue;
        }
        return strategy;
    }

    return mlir::failure();
}

int64_t ConvPostEltwiseMergeTilingPolicy::getMinimumNumber(Dim, const VFSplit&, ArrayRef<int64_t>, ArrayRef<int64_t>,
                                                           VFConfig&, VFConfig&, size_t) const {
    return MINIMUM_LENGTH_TILING;
}

int64_t ConvPostEltwiseMergeTilingPolicy::getMaximalNumber(Dim dim, const VFSplit&, VFConfig& vfConfig) const {
    return getTilingLimit(dim, vfConfig);
}

bool ConvPostEltwiseMergeTilingPolicy::hasFallbackSplits() const {
    return false;
}

bool ConvPostEltwiseMergeTilingPolicy::matchPattern(VFConfig& prevConfig, VFConfig& currentConfig) const {
    const auto prevOps = prevConfig.getOperationsForTiling();
    const auto currentOps = currentConfig.getOperationsForTiling();
    if (prevOps.size() != 1 || currentOps.size() != 1) {
        return false;
    }

    auto hasSliceOp = llvm::any_of(currentConfig.getVFOperations(), [](mlir::Operation* op) {
        return mlir::isa<VPU::SliceOp>(op);
    });
    if (hasSliceOp) {
        // Skip the pattern if there is a slice op, since the slice op may introduce additional tiling restriction that
        // are not present in the previous conv.
        return false;
    }

    auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(prevOps.front());
    if (convOp == nullptr) {
        return false;
    }

    if (!convOp->hasAttr(VPU::pinnedStrategy)) {
        return false;
    }

    auto tilingDims = parseIntArrayAttr<int64_t>(prevConfig.getSubgraph().getTilingStrategyAttr());
    auto vfSplit = getVFTilingSplit(tilingDims);
    if (vfSplit.empty()) {
        return false;
    }

    auto* eltwiseOp = currentOps.front();
    return mlir::isa<VPU::SWOpInterface>(eltwiseOp) && eltwiseOp->hasTrait<VPU::EltwiseOp>();
}

}  // namespace vpux::VPU::VF::v2
