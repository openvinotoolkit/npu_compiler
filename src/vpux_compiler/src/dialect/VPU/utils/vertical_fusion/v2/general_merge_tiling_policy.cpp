//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/general_merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <queue>

namespace vpux::VPU::VF::v2 {
GeneralMergeTilingPolicy::GeneralMergeTilingPolicy(bool enableVerticalFusionPipelining, Logger log,
                                                   VFCacheAnalysis& cache)
        : MergeTilingPolicy(MergeTilingPolicyType::General, cache, enableVerticalFusionPipelining, log) {
}

StringRef GeneralMergeTilingPolicy::getPatternName() const {
    return "GeneralMergeTilingPolicy";
}

bool GeneralMergeTilingPolicy::match(VFConfig&, VFConfig&) const {
    return true;
}

mlir::FailureOr<VPU::MultiClusterStrategy> GeneralMergeTilingPolicy::alignMCStrategy(
        const OpWithViewInputs& parentOpInfo, VPU::ClusteredOpInterface userOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const {
    auto parentOp = parentOpInfo.clusteredOp;
    if (!VF::v2::supportMultiClusterStrategyAdjustmentInVF(parentOp.getOperation())) {
        return mlir::failure();
    }
    auto parentStrategy = parentOp.getMultiClusterStrategy().value();
    auto userStrategy = details::getMultiClusterStrategy(userOp, rollbackStrategy);
    auto userInType = getDistributedInputType(userOp, userOp->getOperand(parentOpInfo.operandIdx), userStrategy);
    if (userInType == nullptr) {
        return mlir::failure();
    }
    auto userInDataType = mlir::cast<VPU::DistributedTensorType>(userInType.getDistributedTypes().front());

    SmallVector<VPU::MultiClusterStrategy> potentialStrategies = {VPU::MultiClusterStrategy::HKSwitch,
                                                                  VPU::MultiClusterStrategy::SplitOverHeight,
                                                                  VPU::MultiClusterStrategy::SplitOverKernel};
    auto outType = mlir::cast<vpux::NDTypeInterface>(parentOp->getResult(0).getType());
    for (const auto& strategy : potentialStrategies) {
        if (parentStrategy == strategy) {
            continue;
        }

        const auto numClusters = getOptimalNumClusters(parentOp, outType.getShape(), strategy);
        if (!parentOp.checkStrategyCompatibility(strategy, checked_cast<size_t>(numClusters))) {
            continue;
        }

        if ((strategy == VPU::MultiClusterStrategy::SplitOverHeight ||
             strategy == VPU::MultiClusterStrategy::HKSwitch) &&
            !parentOp.isOperationSplitOverHeightCompatible(TileInfo(ShapeRef()))) {
            continue;
        } else if (strategy == VPU::MultiClusterStrategy::SplitOverKernel &&
                   !parentOp.isOperationSplitOverKernelCompatible(ShapeRef(), ShapeRef(), ShapeRef())) {
            continue;
        }

        auto newOutType = getDistributedOutputTypeFromOp(parentOp, outType, numClusters, strategy);
        if (newOutType == nullptr) {
            continue;
        }
        auto newOutDataType = mlir::cast<VPU::DistributedTensorType>(newOutType.getDistributedTypes().front());
        auto inferredOutType = inferDistributedTypeThroughViewOps(newOutDataType, parentOpInfo.viewOps);
        if (inferredOutType == nullptr) {
            continue;
        }
        if (areDistributionAttrsCompatible(inferredOutType, userInDataType, true).failed()) {
            continue;
        }
        return strategy;
    }
    return mlir::failure();
}

SmallVector<VFSplit> GeneralMergeTilingPolicy::getSplits(DimArrRef dimsToCheck, DimArrRef allowedDims,
                                                         VFConfig& vfConfig, const VFSplit&,
                                                         const VFSplit& curVFSplit) const {
    return getSplitFromDimArr(dimsToCheck, allowedDims, vfConfig, isMultiDimTilingPerformant(vfConfig, curVFSplit));
}

int64_t GeneralMergeTilingPolicy::getMinimumNumber(Dim dim, const VFSplit& split, ArrayRef<int64_t> currentTiling,
                                                   ArrayRef<int64_t> prevTiling, VFConfig& currentConfig, VFConfig&,
                                                   size_t linkNumber) const {
    auto isSpatialDimTiling = llvm::all_of(split, [](const auto& item) {
        return isSpatialDim(item.first);
    });
    auto hasChannelTiling = llvm::any_of(split, [](const auto& item) {
        return item.first == Dims4D::Act::C;
    });

    if (split.size() > 1 && (isSpatialDimTiling || hasChannelTiling)) {
        return MINIMUM_LENGTH_TILING;
    }
    std::unordered_map<mlir::Operation*, vpux::Dim> curOpDimMap;
    auto curInputAxes = backInferVFTilingDim(currentConfig, dim, curOpDimMap);
    return std::max(currentTiling[dim.ind()], prevTiling[curInputAxes.value()[linkNumber].ind()]);
}

bool GeneralMergeTilingPolicy::adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp,
                                                          VPU::VerticalFusionOp mergedOp) const {
    const auto hasMultipleParentVFOps = llvm::count_if(currentOp->getOperands(), [](mlir::Value operand) {
                                            return operand.getDefiningOp<VPU::VerticalFusionOp>() != nullptr;
                                        }) > 1;
    if (hasMultipleParentVFOps) {
        // In case the current VF can fuse with other parent VFs without MC strategy adjustment, which may has better
        // performance than adjusting MC strategy on prevOp, we choose to not adjust MC strategy for current VF.
        return false;
    }

    const auto currInputArgs = getLinkedArgumentsBetweenVFOps(currentOp, prevOp);
    if (currInputArgs.empty() || currInputArgs.front() == nullptr) {
        // No link found between current op and previous op
        return false;
    }

    auto firstOperand = currentOp.getOperand(currInputArgs.front().getArgNumber());
    for (const auto& currInputArg : currInputArgs) {
        auto operand = currentOp.getOperand(currInputArg.getArgNumber());
        if (operand != firstOperand) {
            // There are multiple links between current op and previous op and
            // they come from different source operands
            return false;
        }
    }

    auto currInputArg = currInputArgs.front();
    auto argUses = details::getComputeUses(currInputArg);
    if (argUses.empty()) {
        // No user is found for the input argument, no need to adjust MC strategy
        return false;
    }

    auto firstUse = argUses.front();
    auto firstClusteredUser = mlir::dyn_cast_if_present<VPU::ClusteredOpInterface>(firstUse->getOwner());
    if (firstClusteredUser == nullptr) {
        return false;
    }
    auto firstUserInType = mlir::cast<VPU::DistributedTensorType>(
            getDistributedInputType(firstClusteredUser, firstClusteredUser->getOperand(firstUse->getOperandNumber()))
                    .getDistributedTypes()
                    .front());

    auto allUsersHaveCompatibleInputType = llvm::all_of(argUses, [&firstUserInType](auto* use) {
        auto clusteredUserOp = mlir::dyn_cast_if_present<VPU::ClusteredOpInterface>(use->getOwner());
        if (clusteredUserOp == nullptr) {
            return false;
        }
        auto userInType = mlir::cast<VPU::DistributedTensorType>(
                getDistributedInputType(clusteredUserOp, clusteredUserOp->getOperand(use->getOperandNumber()))
                        .getDistributedTypes()
                        .front());
        return areDistributionAttrsCompatible(firstUserInType, userInType, true).succeeded();
    });
    if (!allUsersHaveCompatibleInputType) {
        return false;
    }

    auto userOpInMergedVF = mlir::dyn_cast_if_present<VPU::ClusteredOpInterface>(
            details::getMappedOpInMergedVF(firstUse->getOwner(), prevOp, currentOp, mergedOp));

    std::queue<VPU::ClusteredOpInterface> adjustedOpList;
    DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy> rollBackStrategy;

    adjustedOpList.push(userOpInMergedVF);
    while (!adjustedOpList.empty()) {
        auto op = adjustedOpList.front();
        adjustedOpList.pop();
        auto parentOps = details::getParentOpWithViewInputs(op);

        if (parentOps.empty()) {
            continue;
        }

        auto checkSingleOutputParents = llvm::all_of(parentOps, [](auto& opToAdjust) {
            return opToAdjust.clusteredOp->getNumResults() == 1;
        });
        if (!checkSingleOutputParents) {
            // The current implementation of MC strategy adjustment only supports parent ops with single output.
            return false;
        }

        auto strategy = details::getMultiClusterStrategy(op, rollBackStrategy);
        auto allParentsAligned = llvm::all_of(parentOps, [&](auto& opToAdjust) {
            auto& parentOp = opToAdjust.clusteredOp;
            auto& viewOps = opToAdjust.viewOps;

            auto parentOutType = getDistributedOutputType(parentOp, parentOp->getResult(0),
                                                          parentOp.getMultiClusterStrategy().value());
            if (parentOutType == nullptr) {
                return false;
            }
            auto parentOutDataType =
                    mlir::cast<VPU::DistributedTensorType>(parentOutType.getDistributedTypes().front());
            auto castedParentOutType = inferDistributedTypeThroughViewOps(parentOutDataType, viewOps);
            if (castedParentOutType == nullptr) {
                return false;
            }
            auto userInType = getDistributedInputType(op, op->getOperand(opToAdjust.operandIdx), strategy);
            if (userInType == nullptr) {
                return false;
            }
            // For NCE(SOK) -> NCE.Eltwise(SOH), the DMA cost is not accurate for the eltwise op, especially
            // when the input has duplicated distribution, and it has bad impact on the scheduling decision. So we
            // disable the alignment for this case.
            auto userInDataType = mlir::cast<VPU::DistributedTensorType>(userInType.getDistributedTypes().front());
            if (mlir::isa<VPU::NCEOpInterface>(op.getOperation()) && op->hasTrait<VPU::EltwiseOp>()) {
                const auto inDistribution = VPU::DistributionInfo::getClassFromAttr(userInDataType.getDistribution());
                const auto outType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
                        getDistributedOutputType(op, op->getResult(0)).getDistributedTypes().front());
                const auto outDistribution = VPU::DistributionInfo::getClassFromAttr(outType.getDistribution());
                const auto isInputSegmentedLike = isSegmentedLikeDistributionMode(userInDataType, inDistribution);
                const auto isOutputSegmentedLike = isSegmentedLikeDistributionMode(outType, outDistribution);
                if (!isInputSegmentedLike && isOutputSegmentedLike) {
                    return false;
                }
            }

            return areDistributionAttrsCompatible(castedParentOutType, userInDataType, true).succeeded();
        });

        if (allParentsAligned) {
            continue;
        }

        for (auto& opToAdjust : parentOps) {
            auto newParentStrategy = alignMCStrategy(opToAdjust, op, rollBackStrategy);
            if (mlir::failed(newParentStrategy)) {
                // fail to adjust parent op's mc strategy
                return false;
            }
            rollBackStrategy[opToAdjust.clusteredOp] = newParentStrategy.value();
            adjustedOpList.push(opToAdjust.clusteredOp);
        }
    }

    for (auto& item : rollBackStrategy) {
        _log.trace("adjust MC strategy: set {0} for op {1} in merged VF ", item.second, item.first->getLoc());
        item.first.setMultiClusterStrategy(item.second);
    }

    VFConfig mergedConfig(mergedOp, _cache, _enableVerticalFusionPipelining);
    auto inputOps = mergedConfig.getInputs();
    auto isInputOpStrategyChanged = llvm::any_of(inputOps, [&](mlir::Operation* inputOp) {
        return rollBackStrategy.find(mlir::cast<VPU::ClusteredOpInterface>(inputOp)) != rollBackStrategy.end();
    });
    if (isInputOpStrategyChanged) {
        for (auto operand : mergedOp->getOperands()) {
            if (auto parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>()) {
                auto storage = std::make_unique<TilingOperationStorage>();
                auto tilingArray = parseIntArrayAttr<int64_t>(parentVFOp.getTilingStrategyAttr());
                auto tilingRegions = calculateTilingRegions(parentVFOp, tilingArray, _log, storage);
                if (mlir::failed(tilingRegions)) {
                    _log.trace("The parent op has its distributed mode changed, which caused a failure in calculating "
                               "tiling regions for op {0}",
                               parentVFOp->getLoc());
                    return false;
                }
            }
        }
    }

    return true;
}

int64_t GeneralMergeTilingPolicy::getMaximalNumber(Dim dim, const VFSplit& split, VFConfig& vfConfig) const {
    auto maxTiles = getTilingLimit(dim, vfConfig);
    if (split.size() > 1) {
        if (isSpatialDim(dim)) {
            auto otherDimSum = getVFTilesLen(split);
            maxTiles = divUp(maxTiles, otherDimSum);
        } else {
            maxTiles = getTilingLimit(dim, vfConfig, true);
        }
    }
    return maxTiles;
}

bool GeneralMergeTilingPolicy::hasFallbackSplits() const {
    return true;
}

}  // namespace vpux::VPU::VF::v2
