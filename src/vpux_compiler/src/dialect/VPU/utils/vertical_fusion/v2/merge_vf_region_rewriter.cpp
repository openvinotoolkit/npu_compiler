//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_vf_region_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy_utils.hpp"
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

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Value.h>

namespace vpux::VPU::VF::v2 {

namespace {

using MCStrategyAttrSnapshot = SmallVector<std::pair<mlir::Operation*, mlir::Attribute>>;

MCStrategyAttrSnapshot snapshotMCStrategyAttrs(VPU::VerticalFusionOp vfOp) {
    MCStrategyAttrSnapshot snapshot;
    for (auto& operation : vfOp.getBody()->without_terminator()) {
        if (!mlir::isa<VPU::ClusteredOpInterface>(operation)) {
            continue;
        }
        snapshot.emplace_back(&operation, operation.getAttr(VPU::multiClusterStrategy));
    }
    return snapshot;
}

void restoreMCStrategyAttrs(const MCStrategyAttrSnapshot& snapshot) {
    for (const auto& [operation, strategyAttr] : snapshot) {
        if (strategyAttr != nullptr) {
            operation->setAttr(VPU::multiClusterStrategy, strategyAttr);
        } else {
            operation->removeAttr(VPU::multiClusterStrategy);
        }
    }
}

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
        const auto& curInputAxes = curInputAxesResult.value();
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

SmallVector<MergeTilingPolicyType> getSupportedMergeTilingPolicyTypes() {
    SmallVector<MergeTilingPolicyType> policyTypes{MergeTilingPolicyType::ConvPostEltwise,
                                                   MergeTilingPolicyType::General};
    return policyTypes;
}

}  // namespace

MergeVFRegionRewriter::MergeVFRegionRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining,
                                             bool enablePrefetchTiling,
                                             const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log,
                                             VFCacheAnalysis& cache)
        : MergeVFRegionBaseRewriter<VFCase>(ctx, enableVerticalFusionPipelining, enablePrefetchTiling, costFunction,
                                            log),
          _cache(cache) {
    const auto policyTypes = getSupportedMergeTilingPolicyTypes();
    _tilingPolicies.reserve(policyTypes.size());
    llvm::copy(llvm::map_range(policyTypes,
                               [&](auto policyType) {
                                   return createMergeTilingPolicy(policyType, _enableVerticalFusionPipelining,
                                                                  _log.nest(), _cache);
                               }),
               std::back_inserter(_tilingPolicies));
}

VFConfig MergeVFRegionRewriter::createVFConfig(VPU::VerticalFusionOp vfOp) const {
    return VFConfig(vfOp, _cache, _enableVerticalFusionPipelining);
}

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

StrategyCost MergeVFRegionRewriter::getOriginalVFCost(VPU::VerticalFusionOp vfOp) const {
    VFConfig vfConfig(vfOp, _cache, _enableVerticalFusionPipelining, true, true);
    return extractVFCost(vfConfig);
}

bool MergeVFRegionRewriter::isMCStrategyAligned(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const {
    // Get input arg of current vf region corresponding to previous vf op
    const auto currInputArgs = getLinkedArgumentsBetweenVFOps(currentOp, prevOp);
    VPUX_THROW_UNLESS(!currInputArgs.empty(),
                      "No corresponding input argument found for current VF region {0} with previous VF region {1}",
                      currentOp, prevOp);

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
    // 4. Once the inferred producer type and actual consumer type are resolved, delegate the
    //    current-state compatibility decision (distribution attrs + NCE eltwise segmented-like guard)
    //    to checkCurrentMCStrategyCompatibility, the single source of truth shared with the SCF path.
    //    If they are incompatible (e.g. the clustering axis changed in a way the compute op does not
    //    expect), the strategies are considered misaligned and the function returns false.
    auto hasCompatibleMCStrategy = true;
    for (const auto& currInputArg : currInputArgs) {
        // Check if previous output op has MC strategy
        const auto prevOutDistType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
                getDistributedOutputType(prevOp, currentOp.getOperand(currInputArg.getArgNumber())));
        const auto isPrevOutOpWithMCStrategy = prevOutDistType != nullptr;
        if (!isPrevOutOpWithMCStrategy) {
            // if previous op does not have distributed type for one output, we assume the other outputs
            // are not distributed either and the the op does not have an MC strategy
            return true;
        }

        // Derive the specific producer op from the yield operand corresponding to the linked result,
        // not always the last yield operand. This handles multi-result VF ops correctly.
        auto prevOutputResult = mlir::cast<mlir::OpResult>(currentOp.getOperand(currInputArg.getArgNumber()));
        auto* prevOutputOp =
                prevOp.getBody()->getTerminator()->getOperand(prevOutputResult.getResultNumber()).getDefiningOp();

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
                !checkCurrentMCStrategyCompatibility(inferredCurrInDistType, actualCurrInDistType, prevOutputOp,
                                                     currInputOp, currInputOperand)) {
                hasCompatibleMCStrategy = false;
                break;
            }
        }
    }

    return hasCompatibleMCStrategy;
}

std::optional<VFCase> MergeVFRegionRewriter::findVFCase(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                        VPU::VerticalFusionOp mergedVFOp) const {
    if (!isLegalFusion(currentOp, prevOp)) {
        _log.trace("Illegal fusion between {0} and {1}", currentOp->getLoc(), prevOp->getLoc());
        return std::nullopt;
    }
    const auto isMCStrategyAlreadyAligned = isMCStrategyAligned(currentOp, prevOp);
    const auto supportedPolicyTypes = getSupportedMergeTilingPolicyTypes();

    for (const auto& tilingPolicy : _tilingPolicies) {
        if (!llvm::is_contained(supportedPolicyTypes, tilingPolicy->getPolicyType())) {
            continue;
        }

        VFConfig prevConfig(prevOp, _cache, _enableVerticalFusionPipelining);
        VFConfig currentConfig(currentOp, _cache, _enableVerticalFusionPipelining);
        if (!tilingPolicy->match(prevConfig, currentConfig)) {
            continue;
        }

        auto mcStrategyAttrSnapshot = snapshotMCStrategyAttrs(mergedVFOp);
        if (isMCStrategyAlreadyAligned || tilingPolicy->adjustMCStrategyInMergedVF(currentOp, prevOp, mergedVFOp)) {
            auto mergedCase = findVFTiling(mergedVFOp, prevOp, currentOp, *tilingPolicy);
            if (mergedCase.has_value()) {
                return mergedCase;
            }
        }
        restoreMCStrategyAttrs(mcStrategyAttrSnapshot);
    }

    if (!isMCStrategyAlreadyAligned) {
        _log.trace("MC strategies between {0} and {1} cannot be aligned.", currentOp->getLoc(), prevOp->getLoc());
    } else {
        _log.trace("No valid tiling found for merged VF {0} with {1} and {2}", mergedVFOp->getLoc(), prevOp->getLoc(),
                   currentOp->getLoc());
    }
    return std::nullopt;
}

bool MergeVFRegionRewriter::canMergeVFOpsWithoutCostCheck(VFCase&) const {
    return false;
}

bool MergeVFRegionRewriter::canSkipMergeVF(VFConfig& vfConfig, bool opsNeedTiling) const {
    if (opsNeedTiling) {
        return false;
    }

    const auto& ops = vfConfig.getOperationsForTiling();

    // TODO: E#215747 - fix suboptimal vf tiling happening when the parent on the weights tensor has inaccurate cost
    // for small sizes
    auto hasParentOnWeights = llvm::any_of(ops, [](mlir::Operation* op) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        return nceOp != nullptr && nceOp.getWeightsOperand() != nullptr &&
               findParent(nceOp.getWeightsOperand()) != nullptr &&
               llvm::is_contained(mlir::cast<VPU::VerticalFusionOpInterface>(op).restrictedFusionAxes(),
                                  Dims4D::Act::C);
    });

    return !vfConfig.isPipelined() || hasParentOnWeights;
}

bool MergeVFRegionRewriter::checkOtherVFInput(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp) const {
    auto lastOp = currentOp.getBody()->getTerminator()->getOperands().back().getDefiningOp();
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    if (!hasTiling(currentTiling) && mlir::isa_and_present<VPU::NCEOpInterface>(lastOp) &&
        lastOp->hasTrait<VPU::EltwiseOp>() && !currentOp->hasOneUse()) {
        // E#222642: NCE eltwise is usually DMA-bound. Fusing an untiled multi-user VF can force output spilling and
        // perturb user scheduling.
        return false;
    }
    return true;
}

MergeVFRegionRewriter::IVFSchedulingPtr MergeVFRegionRewriter::detectScenario(VFConfig& vfConfig) const {
    VFSchedulingFactory costFactory(_enablePrefetchTiling);
    auto scenarioKind = vfConfig.getSubgraph().getScenario().has_value() ? vfConfig.getSubgraph().getScenario().value()
                        : _enablePrefetchTiling                          ? VFScenario::WEIGHTS_PREFETCHING
                                                                         : VFScenario::MINIMAL;
    return costFactory.createVFScenario(scenarioKind, _log);
}

std::optional<VFCase> MergeVFRegionRewriter::findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                                          VPU::VerticalFusionOp currentOp,
                                                          const MergeTilingPolicy& tilingPolicy) const {
    const auto currentTiling = parseIntArrayAttr<int64_t>(currentOp.getTilingStrategy());
    const auto prevTiling = parseIntArrayAttr<int64_t>(prevOp.getTilingStrategy());

    VFConfig currentConfig(currentOp, _cache, _enableVerticalFusionPipelining);
    VFConfig prevConfig(prevOp, _cache, _enableVerticalFusionPipelining);

    auto curVFSplit = getVFTilingSplit(currentTiling);
    auto preVFSplit = getVFTilingSplit(prevTiling);

    bool curHasTiling = hasTiling(currentTiling);
    bool prevHasTiling = hasTiling(prevTiling);
    // in case both subgraphs have tiling, check if they match
    // if there is only one subgraph with tiling, check if it's allowed
    // to tile second one with such axis
    // if both doesn't have tiling, check if there is at least one
    // allowed axis for both of them
    VFConfig vfConfig(mergedOp, _cache, _enableVerticalFusionPipelining, prevHasTiling, curHasTiling);

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
        const auto& currentOps = currentConfig.getOperationsForTiling();
        operations.insert(currentOps.begin(), currentOps.end());

        VFConfig nextConfig(nextVFOp, _cache, _enableVerticalFusionPipelining);
        const auto& nextOps = nextConfig.getOperationsForTiling();
        operations.insert(nextOps.begin(), nextOps.end());

        VFConfig mergeCurrWithNextConfig(operations, _cache);
        if (!mergeCurrWithNextConfig.isPipelined()) {
            return false;
        }

        const auto linkedArgs = getLinkedArgumentsBetweenVFOps(nextVFOp, mergedOp);
        if (linkedArgs.empty()) {
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

    VPU::VFSubgraphUserSetter setter(currentOp, mergedOp);

    DimArr allowedDims = getAllowedDims(vfConfig.getVFOperations().getArrayRef(), _log, true);
    if (allowedDims.empty()) {
        return std::nullopt;
    }
    DimArr dimsToCheck;
    if (tileOnSameDims(curVFSplit, preVFSplit, linkNumber, currentConfig, prevConfig, opDimMap)) {
        // If the current and previous VF splits are on the same dimensions, we can try to check the common
        // dimensions first
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

    _log.trace("Using policy {0} for merged VF {1}", tilingPolicy.getPatternName(), mergedOp->getLoc());
    auto getBestVFCase = [&](ArrayRef<VFSplit> splits) {
        return findBestVFCaseFromSplits(
                vfConfig, splits,
                [&](Dim dim, const VFSplit& split) {
                    return tilingPolicy.getMinimumNumber(dim, split, currentTiling, prevTiling, currentConfig,
                                                         prevConfig, linkNumber);
                },
                [&](Dim dim, const VFSplit& split) {
                    return tilingPolicy.getMaximalNumber(dim, split, vfConfig);
                },
                _vpunnCostFunction, _log, mergedOp.getContext(), splitFilter);
    };

    auto splits = tilingPolicy.getSplits(dimsToCheck, allowedDims, vfConfig, preVFSplit, curVFSplit);
    auto mergedCase = getBestVFCase(splits);
    if (mergedCase.has_value()) {
        return std::move(mergedCase.value());
    }

    if (!tilingPolicy.hasFallbackSplits()) {
        return std::nullopt;
    }

    // If no valid case found, try to check dims that are not in dimsToCheck. For example, if the current VF and
    // previous VF has tiled on same dimensions W. Then the allowedDims will only contains dimW instead. If merge on
    // dimW is not optimal, the compiler can still have the change to merge on other supported dimensions like dimH,
    // dimH&dimW, etc.
    DimArr restAllowedDims;
    llvm::copy_if(allowedDims, std::back_inserter(restAllowedDims), [&](const Dim& dim) {
        return llvm::find(dimsToCheck, dim) == dimsToCheck.end();
    });
    auto splitsWithLowPriority = tilingPolicy.getSplits(restAllowedDims, allowedDims, vfConfig, preVFSplit, curVFSplit);
    return getBestVFCase(splitsWithLowPriority);
}

bool MergeVFRegionRewriter::checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                                                VFCase& mergedCase) const {
    VPUX_THROW_WHEN(!mergedCase.isInitialized(), "Incorrect tiling strategy for VF");
    if (canMergeVFOpsWithoutCostCheck(mergedCase)) {
        return true;
    }

    // Compare the cost between merged VF subgraph and 2 subgraphs with the spill
    const auto prevCost = getOriginalVFCost(prevOp);
    const auto currentCost = getOriginalVFCost(currentOp);

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
        _log.trace("Merged subgraph candidate {0}", vfBlock.getLoc());
        // `findVFCase` internally compares operations using IR order which is calculated "on the fly" in e.g.
        // `isBeforeInBlock()`
        //  - order recalculation modifies the state and, due to multi-threaded environment, results in a data race
        //  (concurrent writes without synchronization).
        // Initially, this order was invalidated by `fuseOpsInBlock` because it modified IR (added a new operation).
        // As `findVFCase` is expected to not modify the IR, what can be done is an eager recalculation
        // of the operation orders that would ensure later calls to e.g. `isBeforeInBlock()` become concurrent
        // reads instead of concurrent writes (which is no longer a data race).
        vfBlock->getBlock()->recomputeOpOrder();
        vfBlock.getBody()->recomputeOpOrder();
        auto vfCase = findVFCase(parentVFOp, vfOp, vfBlock);
        if (!vfCase.has_value() || !checkVFCostFunction(parentVFOp, vfOp, vfCase.value())) {
            // Drop all references to vfBlock to avoid it being added back to the rewriter
            // worklist.
            vfBlock->dropAllReferences();
            rewriter.eraseOp(vfBlock);
            vfBlock = nullptr;
            if (checkOtherVFInput(vfOp, parentVFOp)) {
                continue;
            }
            _log.trace("Failed to merge subgraph candidate");
            return mlir::failure();
        }

        break;
    }

    if (vfBlock == nullptr) {
        return mlir::failure();
    }

    _log.trace("Merged subgraph {0}", vfBlock->getLoc());
    fuseBlocks(rewriter, vfOp, vfBlock);

    return mlir::success();
}
}  // namespace vpux::VPU::VF::v2
