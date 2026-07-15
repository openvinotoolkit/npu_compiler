//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_vf_region_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/IRMapping.h>

namespace vpux::VPU::VF::v2 {

namespace {

bool tileOnSameDims(const VFSplit& curVFSplit, const VFSplit& preVFSplit, const int64_t linkNumber,
                    VFConfig& currentConfig, VFConfig& prevConfig,
                    std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
    if (curVFSplit.size() != preVFSplit.size()) {
        return false;
    }

    if (curVFSplit.size() == 1 && !isSpatialDim(curVFSplit.begin()->first)) {
        // if both VFs are only tiled on C dim and any of them is NCE with weights,
        // try to search for 2D tiling to compare
        const auto isNCEWithWeights = [](mlir::Operation* op) {
            auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
            return nceOp != nullptr && nceOp.getWeightsOperand() != nullptr;
        };

        if (llvm::any_of(currentConfig.getOperationsForTiling(), isNCEWithWeights) ||
            llvm::any_of(prevConfig.getOperationsForTiling(), isNCEWithWeights)) {
            return false;
        }
    }

    for (const auto& item : curVFSplit) {
        auto curTilingDim = item.first;
        auto curInputAxesResult = VPU::backInferVFTilingDim(currentConfig, curTilingDim, opDimMap);
        if (mlir::failed(curInputAxesResult)) {
            return false;
        }
        auto curInputAxes = curInputAxesResult.value();
        const auto isTiledOnPreVF = preVFSplit.find(curInputAxes[linkNumber]) != preVFSplit.end();
        if (!isTiledOnPreVF) {
            return false;
        }
    }
    return true;
}

bool isTilingDimChangedViewOp(VPU::TilingViewLikeOpInterface viewOp) {
    auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(viewOp->getOperand(0).getType());
    auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(viewOp->getResult(0).getType());
    if (inType == nullptr || outType == nullptr) {
        return false;
    }

    // Strong signal for permute-like behavior
    if (inType.getRank() == outType.getRank() && inType.getDimsOrder() != outType.getDimsOrder()) {
        return true;
    }

    // Fallback: tiling-dim remap behavior
    for (int64_t i = 0; i < inType.getRank(); ++i) {
        auto inDim = vpux::Dim(i);
        auto outDims = viewOp.inferTilingDim(inDim);
        if (outDims.size() != 1 || outDims.front() != inDim) {
            return true;
        }
    }
    return false;
}

SmallVector<OpWithViewInputs> getParentOpWithViewInputs(mlir::Operation* op) {
    SmallVector<OpWithViewInputs> parentOps;
    for (auto& item : op->getOpOperands()) {
        auto operand = item.get();
        auto idx = item.getOperandNumber();

        auto parent = operand.getDefiningOp();
        SmallVector<mlir::Operation*> viewOps;
        while (parent != nullptr && isPureViewOp(parent)) {
            viewOps.push_back(parent);
            parent = parent->getOperand(0).getDefiningOp();
        }
        // Reverse: the loop above collects view ops from consumer toward
        // producer, but inferDistributedTypeThroughViewOps iterates forward
        // from the producer distribution, so producer-to-consumer order is required.
        std::reverse(viewOps.begin(), viewOps.end());
        if (auto clusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(findParent(operand))) {
            parentOps.push_back({idx, clusteredOp, std::move(viewOps)});
        }
    }
    return parentOps;
}

// Collect all non-viewlike uses of the VF block argument by traversing through chains of
// TilingViewLikeOpInterface ops
SmallVector<mlir::OpOperand*> getComputeUses(mlir::BlockArgument currInputArg) {
    SmallVector<mlir::OpOperand*> uses;
    SmallVector<mlir::OpOperand*> worklist;
    for (auto& use : currInputArg.getUses()) {
        worklist.push_back(&use);
    }
    while (!worklist.empty()) {
        auto* use = worklist.pop_back_val();
        auto userOp = use->getOwner();
        if (mlir::isa<VPU::TilingViewLikeOpInterface>(userOp)) {
            // Skip view-like op, but add its users to the worklist
            for (auto& user : userOp->getUses()) {
                worklist.push_back(&user);
            }
        } else {
            uses.push_back(use);
        }
    }
    return uses;
}

mlir::Operation* getMappedOpInMergedVF(mlir::Operation* op, VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currOp,
                                       VPU::VerticalFusionOp mergedOp) {
    auto vfOp = op->getParentOfType<VPU::VerticalFusionOp>();
    VPUX_THROW_UNLESS(vfOp == prevOp || vfOp == currOp,
                      "Operation {0} does not belong to previous VF {1} or current VF {2}", op->getLoc(),
                      prevOp->getLoc(), currOp->getLoc());
    auto opIdx = std::distance(vfOp.getBody()->getOperations().begin(), mlir::Block::iterator(op));
    if (vfOp == prevOp) {
        return &*std::next(mergedOp.getBody()->getOperations().begin(), opIdx);
    }

    auto prevOutputOp = prevOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    auto prevOutputOpIdx =
            std::distance(prevOp.getBody()->getOperations().begin(), mlir::Block::iterator(prevOutputOp));
    return &*std::next(mergedOp.getBody()->getOperations().begin(), opIdx + prevOutputOpIdx + 1);
}

VPU::MultiClusterStrategy getMultiClusterStrategy(
        VPU::ClusteredOpInterface clusteredOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) {
    auto iter = rollbackStrategy.find(clusteredOp);
    if (iter != rollbackStrategy.end()) {
        return iter->second;
    }
    return clusteredOp.getMultiClusterStrategy().value();
}
}  // namespace

StrategyCost MergeVFRegionRewriter::extractVFCost(VFConfig& vfConfig) const {
    auto vfOp = vfConfig.getSubgraph();
    auto tilingDims = parseIntArrayAttr<int64_t>(vfOp.getTilingStrategyAttr());
    auto vfSplit = getVFTilingSplit(tilingDims);

    auto operations = vfConfig.getOperationsForTiling();
    if (operations.empty()) {
        return 0;
    }

    if (vfSplit.empty() || operations.size() == 1) {
        return extractBaselineCost(operations.back(), ShapeRef(tilingDims), _vpunnCostFunction, _log);
    }

    auto vfCase = VFCase(vfConfig, vfSplit);

    auto scenario = detectScenario(vfConfig);

    vfCase.setScheduling(std::move(scenario));
    return vfCase.getCost(_vpunnCostFunction, _log);
}

bool MergeVFRegionRewriter::isMCStrategyAligned(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const {
    // Get input arg of current vf region corresponding to previous vf op
    auto currInputArg = getLinkedArgumentBetweenVFOps(currentOp, prevOp);
    VPUX_THROW_UNLESS(currInputArg != nullptr,
                      "No corresponding input argument found for current VF region {0} with previous VF region {1}",
                      currentOp, prevOp);

    // Check if previous output op has MC strategy
    const auto prevOutDistType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
            getDistributedOutputType(prevOp, currentOp.getOperand(currInputArg.getArgNumber())));
    const auto isPrevOutOpWithMCStrategy = prevOutDistType != nullptr;
    if (!isPrevOutOpWithMCStrategy) {
        return true;
    }

    // Here we check whether the MC strategies are aligned between the previous VF output and all
    // compute consumers in the current VF region. The check accounts for view-like ops (e.g. PermuteCast,
    // ShapeCast) that may appear between the block argument and the actual compute op, since these ops
    // can remap the multi-cluster axis. For example, in a SOK -> PermuteCast -> SOH chain, the PermuteCast
    // may transpose the split-over-kernel distribution into a split-over-height layout.
    //
    // The algorithm works as follows:
    // 1. For each direct user of the linked block argument, collect the chain of TilingViewLikeOpInterface
    //    ops until a compute (non-view-like) op is reached.
    // 2. Compute the "actual" input distributed type that the compute op expects for its operand
    //    (actualCurrInDistType).
    // 3. Starting from the previous VF output's distributed type (prevOutDistType), propagate it through
    //    the collected view-like op chain via inferDistributedTypeThroughViewOps to obtain the "inferred"
    //    distributed type that would arrive at the compute op after passing through those view-like ops
    //    (inferredCurrInDistType).
    // 4. Compare the inferred type against the actual type using areDistributionAttrsCompatible. If they
    //    are incompatible (e.g. the clustering axis changed in a way the compute op does not expect),
    //    the strategies are considered misaligned and the function returns false.
    auto hasCompatibleMCStrategy = true;
    for (auto currInputOp : currInputArg.getUsers()) {
        SmallVector<mlir::Operation*> currInputViewLikeOps;
        while (mlir::isa<VPU::TilingViewLikeOpInterface>(currInputOp) && currInputOp->hasOneUse()) {
            currInputViewLikeOps.push_back(currInputOp);
            currInputOp = *(currInputOp->getUsers().begin());
        }
        // isLegalFusion has ensured currInputOp is clustered op with MC strategy
        auto currInputOperand = currInputViewLikeOps.empty() ? mlir::cast<mlir::Value>(currInputArg)
                                                             : currInputViewLikeOps.back()->getResult(0);
        auto actualCurrInDistType = mlir::cast<VPU::DistributedTensorType>(
                getDistributedInputType(currInputOp, currInputOperand).getDistributedTypes().front());
        auto inferredCurrInDistType = inferDistributedTypeThroughViewOps(prevOutDistType, currInputViewLikeOps);
        if (inferredCurrInDistType == nullptr ||
            areDistributionAttrsCompatible(inferredCurrInDistType, actualCurrInDistType, true).failed()) {
            hasCompatibleMCStrategy = false;
            break;
        }
        // For NCE(SOK) -> NCE.Eltwise(SOH), the DMA cost is not accurate for the eltwise, especially
        // when the input has duplicated distribution, and it has bad impact on the scheduling decision. So we
        // disable VF for this case.

        if (mlir::isa<VPU::NCEOpInterface>(currInputOp) && currInputOp->hasTrait<VPU::EltwiseOp>()) {
            const auto inDistribution = VPU::DistributionInfo::getClassFromAttr(actualCurrInDistType.getDistribution());
            const auto outType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
                    getDistributedOutputType(currInputOp, currInputOp->getResult(0)).getDistributedTypes().front());
            const auto outDistribution = VPU::DistributionInfo::getClassFromAttr(outType.getDistribution());
            const auto isInputSegmentedLike = isSegmentedLikeDistributionMode(actualCurrInDistType, inDistribution);
            const auto isOutputSegmentedLike = isSegmentedLikeDistributionMode(outType, outDistribution);
            if (!isInputSegmentedLike && isOutputSegmentedLike) {
                hasCompatibleMCStrategy = false;
                break;
            }
        }
    }

    return hasCompatibleMCStrategy;
}

bool MergeVFRegionRewriter::adjustMCStrategyInMergedVF(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp,
                                                       VPU::VerticalFusionOp mergedVF) const {
    auto currInputArg = getLinkedArgumentBetweenVFOps(currentOp, prevOp);
    auto argUses = getComputeUses(currInputArg);
    if (argUses.empty()) {
        // No user is found for the input argument, no need to adjust MC strategy
        return false;
    }
    auto hasMultipleParentVFOps = llvm::count_if(currentOp->getOperands(), [](mlir::Value operand) {
                                      return operand.getDefiningOp<VPU::VerticalFusionOp>() != nullptr;
                                  }) > 1;
    if (hasMultipleParentVFOps) {
        // In case the current VF can fuse with other parent VFs without MC strategy adjustment, which may has better
        // performance than adjusting MC strategy on prevOp, we choose to not adjust MC strategy for current VF.
        return false;
    }

    auto firstUse = argUses.front();
    auto firstClusteredUser = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(firstUse->getOwner());
    if (firstClusteredUser == nullptr) {
        return false;
    }
    auto firstUserInType = mlir::cast<VPU::DistributedTensorType>(
            getDistributedInputType(firstClusteredUser, firstClusteredUser->getOperand(firstUse->getOperandNumber()))
                    .getDistributedTypes()
                    .front());

    auto allUsersHaveCompatibleInputType = llvm::all_of(argUses, [&firstUserInType](auto* use) {
        auto clusteredUserOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(use->getOwner());
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

    auto userOpInMergedVF = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(
            getMappedOpInMergedVF(firstUse->getOwner(), prevOp, currentOp, mergedVF));

    std::queue<VPU::ClusteredOpInterface> adjustedOpList;
    DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy> rollBackStrategy;

    adjustedOpList.push(userOpInMergedVF);
    while (!adjustedOpList.empty()) {
        auto op = adjustedOpList.front();
        adjustedOpList.pop();
        auto parentOps = getParentOpWithViewInputs(op);

        if (parentOps.empty()) {
            continue;
        }

        auto strategy = getMultiClusterStrategy(op, rollBackStrategy);
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

    VFConfig mergedConfig(mergedVF, _enableVerticalFusionPipelining);
    auto inputOps = mergedConfig.getInputs();
    auto isInputOpStrategyChanged = llvm::any_of(inputOps, [&](mlir::Operation* inputOp) {
        return rollBackStrategy.find(mlir::cast<VPU::ClusteredOpInterface>(inputOp)) != rollBackStrategy.end();
    });
    if (isInputOpStrategyChanged) {
        for (auto operand : mergedVF->getOperands()) {
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

mlir::FailureOr<VPU::MultiClusterStrategy> MergeVFRegionRewriter::alignMCStrategy(
        const OpWithViewInputs& parentOpInfo, VPU::ClusteredOpInterface userOp,
        const DenseMap<VPU::ClusteredOpInterface, VPU::MultiClusterStrategy>& rollbackStrategy) const {
    auto parentOp = parentOpInfo.clusteredOp;
    if (!VF::v2::supportMultiClusterStrategyAdjustmentInVF(parentOp.getOperation())) {
        return mlir::failure();
    }
    auto parentStrategy = parentOp.getMultiClusterStrategy().value();
    auto userStrategy = getMultiClusterStrategy(userOp, rollbackStrategy);
    auto userInType = getDistributedInputType(userOp, userOp->getOperand(parentOpInfo.operandIdx), userStrategy);
    if (userInType == nullptr) {
        return mlir::failure();
    }
    auto userInDataType = mlir::cast<VPU::DistributedTensorType>(userInType.getDistributedTypes().front());

    SmallVector<VPU::MultiClusterStrategy> potentialStrategies = {VPU::MultiClusterStrategy::HKSwitch,
                                                                  VPU::MultiClusterStrategy::SplitOverHeight,
                                                                  VPU::MultiClusterStrategy::SplitOverKernel};
    auto isNCEOp = mlir::isa<VPU::NCEOpInterface>(parentOp.getOperation());
    for (const auto& strategy : potentialStrategies) {
        if (parentStrategy == strategy) {
            continue;
        }
        if ((!isNCEOp || parentOp->hasAttr(VPU::isInPlace)) && strategy == VPU::MultiClusterStrategy::HKSwitch) {
            // HKswitch is only supported for non-inplace NCE ops
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

        auto outType = mlir::cast<vpux::NDTypeInterface>(parentOp->getResult(0).getType());
        auto numClusters = getOptimalNumClusters(parentOp, outType.getShape(), strategy);
        if (!parentOp.checkStrategyCompatibility(strategy, checked_cast<size_t>(numClusters))) {
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

std::optional<VFCase> MergeVFRegionRewriter::findVFCase(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                        VPU::VerticalFusionOp mergedVFOp) const {
    if (!isLegalFusion(currentOp, prevOp)) {
        return std::nullopt;
    }
    if (isMCStrategyAligned(currentOp, prevOp) || adjustMCStrategyInMergedVF(currentOp, prevOp, mergedVFOp)) {
        return findVFTiling(mergedVFOp, prevOp, currentOp);
    }
    return std::nullopt;
}

bool MergeVFRegionRewriter::canMergeVFOpsWithoutCostCheck(VFCase&) const {
    return false;
}

bool MergeVFRegionRewriter::canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const {
    auto ops = vfConfig.getOperationsForTiling();

    // TODO: E#215747 - fix suboptimal vf tiling happening when the parent on the weights tensor has inaccurate cost for
    // small sizes
    auto hasParentOnWeights = llvm::any_of(ops, [](mlir::Operation* op) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        return nceOp != nullptr && nceOp.getWeightsOperand() != nullptr &&
               findParent(nceOp.getWeightsOperand()) != nullptr &&
               llvm::is_contained(mlir::cast<VPU::VerticalFusionOpInterface>(op).restrictedFusionAxes(),
                                  Dims4D::Act::C);
    });
    return !opsNeedTiling && (!vfConfig.isPipelined() || hasParentOnWeights);
}

MergeVFRegionRewriter::IVFSchedulingPtr MergeVFRegionRewriter::detectScenario(VFConfig& vfConfig) const {
    VFSchedulingFactory costFactory(_enablePrefetchTiling);
    auto scenarioKind = vfConfig.getSubgraph().getScenario().has_value() ? vfConfig.getSubgraph().getScenario().value()
                        : _enablePrefetchTiling                          ? VFScenario::WEIGHTS_PREFETCHING
                                                                         : VFScenario::MINIMAL;
    return costFactory.createVFScenario(scenarioKind, _log);
}

std::optional<VFCase> MergeVFRegionRewriter::findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                                          VPU::VerticalFusionOp currentOp) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VFConfig currentConfig(currentOp, _enableVerticalFusionPipelining);
    VFConfig prevConfig(prevOp, _enableVerticalFusionPipelining);

    auto curVFSplit = getVFTilingSplit(currentTiling);
    auto preVFSplit = getVFTilingSplit(prevTiling);

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them
    VFConfig vfConfig(mergedOp, _enableVerticalFusionPipelining, prevHasTiling, curHasTiling);

    bool opsNeedTiling = prevHasTiling || curHasTiling;
    if (canSkipMergeVF(vfConfig, opsNeedTiling)) {
        return std::nullopt;
    }

    // Record the operation and its corresponding tiling dim when back-infer subgraph
    std::unordered_map<mlir::Operation*, vpux::Dim> opDimMap;

    bool checkIfNextMergeBetter = false;
    if (!vfConfig.isPipelined() && currentOp->hasOneUse() &&
        mlir::isa<VPU::VerticalFusionOp>(*currentOp->getUsers().begin())) {
        checkIfNextMergeBetter = true;
    }

    // If mergedOp can not pipeline, but currentOp + userOp can pipeline, and mergedOp's tile dim is userOp's
    // restricted axes, then we will block the mergeOp. For example Conv + Add + Softmax, the three operations
    // can not be vertically fused due to restricted axes on Softmax. Conv + Add can not pipeline while
    // Add + Softmax can. We prefer Add vertical fusion with Softmax.
    const auto isNextMergeCanBePipelined = [&](const std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
        auto nextVFOp = mlir::cast<VPU::VerticalFusionOp>(*mergedOp->getUsers().begin());

        llvm::SetVector<mlir::Operation*> operations;
        const auto currentOps = currentConfig.getOperationsForTiling();
        operations.insert(currentOps.begin(), currentOps.end());

        VFConfig nextConfig(nextVFOp, _enableVerticalFusionPipelining);
        const auto nextOps = nextConfig.getOperationsForTiling();
        operations.insert(nextOps.begin(), nextOps.end());

        VFConfig mergeCurrWithNextConfig(operations);
        if (!mergeCurrWithNextConfig.isPipelined()) {
            return false;
        }

        auto linkedArg = getLinkedArgumentBetweenVFOps(nextVFOp, mergedOp);
        if (linkedArg == nullptr) {
            return false;
        }

        auto outputOpDimIt = opDimMap.find(currentConfig.getOutputs().back());
        if (outputOpDimIt == opDimMap.end()) {
            return false;
        }
        auto linkedInputDim = outputOpDimIt->second;

        auto nextViewOps = nextConfig.getVFOperations() | filtered([](mlir::Operation* op) {
                               return mlir::isa<VPU::TilingViewLikeOpInterface>(op);
                           });
        auto isInferedDimChanged = llvm::any_of(nextViewOps, [&](mlir::Operation* viewOp) {
            return isTilingDimChangedViewOp(mlir::cast<VPU::TilingViewLikeOpInterface>(viewOp));
        });

        if (isInferedDimChanged) {
            // If the inferred tiling dim is changed, skip it for simplicity
            return false;
        }

        for (auto* operation : nextConfig.getOperationsForTiling()) {
            auto vfOperation = mlir::cast<VPU::VerticalFusionOpInterface>(operation);
            auto restrictedAxes = vfOperation.restrictedFusionAxes();
            if (restrictedAxes.empty()) {
                continue;
            }
            if (llvm::is_contained(restrictedAxes, linkedInputDim)) {
                return true;
            }
        }
        return false;
    };

    // Only for current VF Op check to skip restricted dims
    // E.g., VF{conv} -> VF{conv}, the first VF can support CTiling, but the second cannot
    const auto isRegionRestrictedDim = [&](const std::unordered_map<mlir::Operation*, vpux::Dim>& opDimMap) {
        // Skip the case of ConvolutionOp whose weights has split over output channel tiling strategy
        if (isTileOverOutputChannel(currentConfig)) {
            return false;
        }

        for (auto* operation : currentConfig.getOperationsForTiling()) {
            auto vfOperation = mlir::cast<VPU::VerticalFusionOpInterface>(operation);
            auto restrictedAxes = vfOperation.restrictedFusionAxes();
            if (restrictedAxes.empty()) {
                continue;
            }

            if (llvm::find(currentConfig.getInputs(), operation) != currentConfig.getInputs().end()) {
                // skip inputs which has no connection with previous operation
                if (llvm::none_of(operation->getOperands(), [&](mlir::Value value) {
                        if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
                            return currentOp.getOperand(argument.getArgNumber()).getDefiningOp() == prevOp;
                        }
                        return false;
                    })) {
                    continue;
                }
            }
            VPUX_THROW_WHEN(opDimMap.find(operation) == opDimMap.end(), "Operation {0} is not in the map",
                            operation->getLoc());
            auto dim = opDimMap.at(operation);
            if (llvm::find(restrictedAxes, dim) != restrictedAxes.end()) {
                return true;
            }
        }
        return false;
    };

    const auto linkNumber = getLinkNumber(currentOp, prevOp);

    const auto getMinimumNumber = [&](auto dim, const VFSplit& split) -> int64_t {
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
    };

    const auto getMaximalNumber = [&](auto dim, const VFSplit& split) -> int64_t {
        auto maxTiles = getTilingLimit(dim, vfConfig);
        if (split.size() > 1) {
            // 2D tiling
            if (isSpatialDim(dim)) {
                auto otherDimSum = getVFTilesLen(split);
                maxTiles = divUp(maxTiles, otherDimSum);
            } else {
                // This will happen when try to re-adjust the channel tile size to avoid over-tiling on channel dim
                maxTiles = getTilingLimit(dim, vfConfig, true);
            }
        }
        return maxTiles;
    };

    VPU::VFSubgraphUserSetter setter(currentOp, mergedOp);

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations().getArrayRef(), _log);
    if (allowedDims.empty()) {
        return std::nullopt;
    }
    DimArr dimsToCheck;
    if (tileOnSameDims(curVFSplit, preVFSplit, linkNumber, currentConfig, prevConfig, opDimMap)) {
        // If the current and previous VF splits are on the same dimensions, we can try to check the common dimensions
        // first
        for (auto& item : curVFSplit) {
            dimsToCheck.push_back(item.first);
        }
    } else {
        // Otherwise, we check all allowed dimensions
        dimsToCheck = allowedDims;
    }

    const SplitFilterFn splitFilter = [&](Dim dim, const VFSplit&) -> bool {
        std::unordered_map<mlir::Operation*, vpux::Dim> curOpDimMap;

        if (mlir::failed(backInferVFTilingDim(currentConfig, dim, curOpDimMap))) {
            return true;
        }

        // Skip current merge if a better (pipelined) merge with the next VF block is possible.
        if (checkIfNextMergeBetter && isNextMergeCanBePipelined(curOpDimMap)) {
            return true;
        }

        return isRegionRestrictedDim(curOpDimMap);
    };

    auto enableMultiDimTiling = isMultiDimTilingPerformant(vfConfig, curVFSplit);
    auto splits = getSplitFromDimArr(dimsToCheck, allowedDims, vfConfig, enableMultiDimTiling);

    auto mergedCase = findBestVFCaseFromSplits(vfConfig, splits, getMinimumNumber, getMaximalNumber, _vpunnCostFunction,
                                               _log, mergedOp.getContext(), splitFilter);
    if (mergedCase.has_value()) {
        return std::move(mergedCase.value());
    }

    // If no valid case found, try to check dims that are not in dimsToCheck. For example, if the current VF and
    // previous VF has tiled on same dimensions W. Then the allowedDims will only contains dimW instead. If merge on
    // dimW is not optimal, the compiler can still have the change to merge on other supported dimensions like dimH,
    // dimH&dimW, etc.
    DimArr restAllowedDims;
    llvm::copy_if(allowedDims, std::back_inserter(restAllowedDims), [&](const Dim& dim) {
        return llvm::find(dimsToCheck, dim) == dimsToCheck.end();
    });
    auto splitsWithLowPriority = getSplitFromDimArr(restAllowedDims, allowedDims, vfConfig, enableMultiDimTiling);
    mergedCase = findBestVFCaseFromSplits(vfConfig, splitsWithLowPriority, getMinimumNumber, getMaximalNumber,
                                          _vpunnCostFunction, _log, mergedOp.getContext(), splitFilter);
    return mergedCase;
}

bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                VFCase& mergedCase) const {
    VPUX_THROW_WHEN(!mergedCase.isInitialized(), "Incorrect tiling strategy for VF");
    if (canMergeVFOpsWithoutCostCheck(mergedCase)) {
        return true;
    }

    // Compare the cost between merged VF subgraph and 2 subgraphs with the spill
    VFConfig prevOpConfig(prevOp, _enableVerticalFusionPipelining);
    VFConfig currentOpConfig(currentOp, _enableVerticalFusionPipelining);

    const auto prevCost = extractVFCost(prevOpConfig);
    const auto currentCost = extractVFCost(currentOpConfig);

    {
        // Change the IR so that merged VF substitutes current operation and previous op to
        // calculate correct cost.
        // The IR will change back when the setter is destroyed.
        VPU::VFSubgraphUserSetter setter(currentOp, mergedCase.getConfig().getSubgraph());
        if (!isVFMergeProfitable(mergedCase, prevCost, currentCost, _vpunnCostFunction, _log)) {
            return false;
        }
    }

    return true;
}

mlir::LogicalResult MergeVFRegionRewriter::matchAndRewrite(VPU::VerticalFusionOp vfOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("Starting vertical fusion for region with VerticalFusionOp {0} at location {1}", vfOp, vfOp->getLoc());
    if (vfOp.getIsManualConfigured()) {
        _log.trace("Skipping vertical fusion for manually configured VerticalFusionOp at location {0}", vfOp->getLoc());
        return mlir::failure();
    }

    VPU::VerticalFusionOp vfBlock = nullptr;
    VPU::VerticalFusionOp parentVFOp = nullptr;
    for (auto operand : vfOp->getOperands()) {
        parentVFOp = operand.getDefiningOp<VPU::VerticalFusionOp>();
        vfBlock = nullptr;

        if (parentVFOp == nullptr || parentVFOp.getIsManualConfigured()) {
            continue;
        }

        _log.trace("Analyzing vertical fusion region with parent VerticalFusionOp {0} at location {1}", parentVFOp,
                   parentVFOp->getLoc());

        const bool allInOldBlock = llvm::all_of(parentVFOp->getUsers(), [&](auto user) {
            return user == vfOp;
        });
        // if not all user of current parent VF go to the same block
        // try further with next operand
        // For situations
        // Operation1
        //   |      |          |
        //   |     Operation2
        //     Eltwise         Operation3
        // in case Operation1's user goes to Operation3, which cannot be fused with Eltwise
        // switch to Operation2, try to merge it with Eltwise
        if (!allInOldBlock) {
            continue;
        }

        vfBlock = fuseOpsInBlock(rewriter, vfOp, parentVFOp.getOperation());
        auto vfCase = findVFCase(parentVFOp, vfOp, vfBlock);
        if (!vfCase.has_value() || !checkVFCostFunction(parentVFOp, vfOp, vfCase.value())) {
            // Drop all references to vfBlock to avoid it being added back to the rewriter
            // worklist.
            vfBlock->dropAllReferences();
            rewriter.eraseOp(vfBlock);
            vfBlock = nullptr;
            // Add support for NCE task, if merging activation failed, continue to merge weights.
            // E-141686: A general solution to merge more subgraph for more VF ops.
            if (checkOtherVFInput(vfOp, parentVFOp)) {
                continue;
            }
            return mlir::failure();
        }

        break;
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merged subgraph {0}", vfBlock);
    fuseBlocks(rewriter, vfOp, vfBlock);

    return mlir::success();
}
}  // namespace vpux::VPU::VF::v2
