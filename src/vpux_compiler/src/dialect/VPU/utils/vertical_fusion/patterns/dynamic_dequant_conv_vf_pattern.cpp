//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/dynamic_dequant_conv_vf_pattern.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <algorithm>

using namespace vpux;
using namespace VPU;

namespace {

bool hasMultiDimTilingIncludeChannel(ArrayRef<int64_t> tilingStrategy) {
    if (tilingStrategy.size() != static_cast<size_t>(Dims4D::Act::numDims)) {
        return false;
    }

    return llvm::count_if(tilingStrategy,
                          [](auto tileSize) {
                              return tileSize > 1;
                          }) > 1 &&
           tilingStrategy[Dims4D::Act::C.ind()] > 1;
}

bool isSplitOverKernel(VPU::ClusteredOpInterface clusteredOp) {
    auto strategy = clusteredOp.getMultiClusterStrategy();
    return strategy.has_value() && strategy.value() == VPU::MultiClusterStrategy::SplitOverKernel;
}

bool isSplitOverHeightLike(std::optional<VPU::MultiClusterStrategy> strategy) {
    if (!strategy.has_value()) {
        return false;
    }

    return strategy.value() == VPU::MultiClusterStrategy::SplitOverHeight ||
           strategy.value() == VPU::MultiClusterStrategy::SplitOverHeightOverlapped;
}

void restoreMultiClusterStrategy(VPU::ClusteredOpInterface clusteredOp,
                                 std::optional<VPU::MultiClusterStrategy> strategy) {
    if (strategy.has_value()) {
        clusteredOp.setMultiClusterStrategy(strategy.value());
        return;
    }

    clusteredOp->removeAttr(VPU::multiClusterStrategy);
}

VPU::NCEConvolutionOp getTrailingConvOp(VPU::VerticalFusionOp vfOp) {
    auto* body = vfOp.getBody();
    if (body == nullptr) {
        return nullptr;
    }

    mlir::Operation* lastOp = nullptr;
    for (auto& op : body->without_terminator()) {
        lastOp = &op;
    }

    return mlir::dyn_cast_or_null<VPU::NCEConvolutionOp>(lastOp);
}

VPU::DynamicDequantizeOp getDynamicDequantizeOp(VPU::VerticalFusionOp vfOp) {
    auto* body = vfOp.getBody();
    if (body == nullptr) {
        return nullptr;
    }

    for (auto& op : body->without_terminator()) {
        if (auto dequantOp = mlir::dyn_cast<VPU::DynamicDequantizeOp>(op)) {
            return dequantOp;
        }
    }

    return nullptr;
}

bool fitsDynamicDequantConvCMX(VPU::VF::v2::VFConfig& vfConfig, VPU::DynamicDequantizeOp dequantOp,
                               VPU::NCEConvolutionOp convOp, const VPU::TilingOperationStorage::UPtr& tilingStorage) {
    if (tilingStorage == nullptr) {
        return false;
    }

    const auto index = 0;
    const auto convTiling = tilingStorage->getRef(convOp.getOperation(), index);
    const auto dequantTiling = tilingStorage->getRef(dequantOp.getOperation(), index);
    if (!convTiling.has_value() || !dequantTiling.has_value()) {
        return false;
    }
    const auto& convTilingValue = convTiling.value().get();
    const auto& dequantTilingValue = dequantTiling.value().get();
    auto requiredCMX = vfConfig.getOperationRequiredCMX(dequantOp.getOperation(), dequantTilingValue.second,
                                                        dequantTilingValue.first.tiles);
    const auto& dequantTypes = vfConfig.getOperationTypes(dequantOp.getOperation(), dequantTilingValue.second,
                                                          dequantTilingValue.first.tiles);
    const auto& convTypes =
            vfConfig.getOperationTypes(convOp.getOperation(), convTilingValue.second, convTilingValue.first.tiles);
    VPUX_THROW_WHEN(dequantTypes.empty() || convTypes.empty(),
                    "Cannot get operation types for dequant or conv operations");

    requiredCMX += vfConfig.getOperationRequiredCMX(convOp.getOperation(), convTilingValue.second,
                                                    convTilingValue.first.tiles);
    requiredCMX -= dequantTypes.back().getTotalAllocSize();  // dequant output is used as conv weights (via view ops),
                                                             // so subtract it to avoid double counting
    requiredCMX += convTypes.front().getTotalAllocSize();    // prefetch next tile's conv input

    const auto thresholdCMXSize = VPU::getTotalCMXFragmentationAwareSize(convOp.getOperation());
    return requiredCMX <= thresholdCMXSize;
}

struct ConvAlignmentState {
    VPU::VerticalFusionOp convVF;
    VPU::NCEConvolutionOp convOp;
    std::optional<VPU::MultiClusterStrategy> multiClusterStrategy;
    mlir::ArrayAttr tilingStrategy;
};

std::optional<ConvAlignmentState> getConvAlignmentState(VPU::VerticalFusionOp convVF) {
    if (convVF == nullptr) {
        return std::nullopt;
    }

    auto convOp = getTrailingConvOp(convVF);
    if (convOp == nullptr) {
        return std::nullopt;
    }

    auto convClusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    if (convClusteredOp == nullptr) {
        return std::nullopt;
    }

    return ConvAlignmentState{convVF, convOp, convClusteredOp.getMultiClusterStrategy(),
                              convVF.getTilingStrategyAttr()};
}

void restoreConvAlignmentState(ConvAlignmentState state) {
    auto convClusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(state.convOp.getOperation());
    if (convClusteredOp != nullptr) {
        restoreMultiClusterStrategy(convClusteredOp, state.multiClusterStrategy);
    }

    if (state.tilingStrategy != nullptr) {
        state.convVF.setTilingStrategyAttr(state.tilingStrategy);
        return;
    }

    state.convVF->removeAttr(state.convVF.getTilingStrategyAttrName());
}

}  // namespace

bool DynamicDequantConvVFPattern::alignConvStrategyWithDequant(VPU::VerticalFusionOp dequantVF,
                                                               VPU::VerticalFusionOp convVF, Logger log) const {
    auto* dequantOp = getSingleInnerOp(dequantVF);
    auto dequantClusteredOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(dequantOp);
    if (dequantClusteredOp == nullptr || !isSplitOverKernel(dequantClusteredOp)) {
        return true;
    }

    auto convOp = getTrailingConvOp(convVF);
    if (convOp == nullptr) {
        return false;
    }

    auto convClusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(convOp.getOperation());
    if (convClusteredOp == nullptr) {
        return false;
    }
    if (isSplitOverKernel(convClusteredOp)) {
        return true;
    }
    if (!isSplitOverHeightLike(convClusteredOp.getMultiClusterStrategy())) {
        log.trace("DynamicDequantConvVFPattern skip SOK alignment, Conv has unsupported strategy at {0}",
                  convOp->getLoc());
        return false;
    }

    if (!convClusteredOp.isOperationSplitOverKernelCompatible(ShapeRef(), ShapeRef(), ShapeRef())) {
        log.trace("DynamicDequantConvVFPattern skip SOK alignment, Conv is not SOK compatible at {0}",
                  convOp->getLoc());
        return false;
    }

    auto originalStrategy = convClusteredOp.getMultiClusterStrategy();
    convClusteredOp.setMultiClusterStrategy(VPU::MultiClusterStrategy::SplitOverKernel);

    auto newTilingStrategy = parseIntArrayAttr<int64_t>(convVF.getTilingStrategy());

    const auto maxNumTiles = vpux::getMaxNumTiles(convOp.getOperation(), /*checkMinimalWidthAndHeight=*/true,
                                                  /*checkWorkloadEfficiency=*/true);
    const auto heightIdx = Dims4D::Act::H.ind();
    const auto channelMinNumTiles = 2;
    newTilingStrategy[Dims4D::Act::C.ind()] = channelMinNumTiles;
    newTilingStrategy[heightIdx] = maxNumTiles[heightIdx];

    if (!hasMultiDimTilingIncludeChannel(newTilingStrategy)) {
        log.trace("DynamicDequantConvVFPattern rejected SOK tiling {0} for Conv at {1}", newTilingStrategy,
                  convOp->getLoc());
        restoreMultiClusterStrategy(convClusteredOp, originalStrategy);
        return false;
    }

    convVF.setTilingStrategyAttr(getIntArrayAttr(convVF->getContext(), newTilingStrategy));
    log.trace("DynamicDequantConvVFPattern aligned Conv strategy to SOK with tiling {0} at {1}", newTilingStrategy,
              convOp->getLoc());
    return true;
}

bool DynamicDequantConvVFPattern::hasExpectedChain(VPU::VerticalFusionOp dequantVF,
                                                   VPU::VerticalFusionOp convVF) const {
    auto* dequantOp = getSingleInnerOp(dequantVF);
    if (!mlir::isa_and_nonnull<VPU::DynamicDequantizeOp>(dequantOp)) {
        return false;
    }

    auto* convBody = convVF.getBody();
    if (convBody == nullptr) {
        return false;
    }
    SmallVector<mlir::Operation*> convOps;
    for (auto& op : convBody->without_terminator()) {
        convOps.push_back(&op);
    }

    if (convOps.empty()) {
        return false;
    }

    auto convOp = mlir::dyn_cast<VPU::NCEConvolutionOp>(convOps.back());
    if (convOp == nullptr) {
        return false;
    }
    auto tilingStrategy = parseIntArrayAttr<int64_t>(convVF.getTilingStrategy());
    if (!hasMultiDimTilingIncludeChannel(tilingStrategy)) {
        return false;
    }

    auto dequantVFArg = llvm::find(convVF->getOperands(), dequantVF.getResult(0));
    if (dequantVFArg == convVF->getOperands().end()) {
        return false;
    }

    const auto argIndex = std::distance(convVF->getOperands().begin(), dequantVFArg);
    auto filterBlockArg = convBody->getArgument(argIndex);
    auto filterValue = convOp.getFilter();

    size_t filterViewDepth = 0;
    while (filterValue != filterBlockArg) {
        if (filterViewDepth >= convOps.size()) {
            return false;
        }

        auto* filterViewOp = filterValue.getDefiningOp();
        if (filterViewOp == nullptr || filterViewOp->getBlock() != convBody || !VPU::isPureViewOp(filterViewOp) ||
            filterViewOp->getNumOperands() == 0) {
            return false;
        }

        filterValue = filterViewOp->getOperand(0);
        ++filterViewDepth;
    }

    return true;
}

bool DynamicDequantConvVFPattern::hasExpectedMemPermuteProducer(VPU::VerticalFusionOp memPermuteVF,
                                                                VPU::VerticalFusionOp dequantVF) const {
    auto* memPermuteOp = getSingleInnerOp(memPermuteVF);
    if (!mlir::isa_and_present<VPU::MemPermuteOp>(memPermuteOp)) {
        return false;
    }

    auto* dequantOp = getSingleInnerOp(dequantVF);
    auto dynamicDequantizeOp = mlir::dyn_cast_if_present<VPU::DynamicDequantizeOp>(dequantOp);
    if (dynamicDequantizeOp == nullptr) {
        return false;
    }

    const auto memPermuteBlockArgs = VPU::getLinkedArgumentsBetweenVFOps(dequantVF, memPermuteVF);
    return memPermuteBlockArgs.size() == 1 && dynamicDequantizeOp.getInput() == memPermuteBlockArgs.front();
}

VPU::VerticalFusionOp DynamicDequantConvVFPattern::getExpectedMemPermuteProducer(
        VPU::VerticalFusionOp dequantVF) const {
    auto* dequantOp = getSingleInnerOp(dequantVF);
    auto dynamicDequantizeOp = mlir::dyn_cast_if_present<VPU::DynamicDequantizeOp>(dequantOp);
    if (dynamicDequantizeOp == nullptr) {
        return nullptr;
    }

    auto inputBlockArg = mlir::dyn_cast<mlir::BlockArgument>(dynamicDequantizeOp.getInput());
    if (inputBlockArg == nullptr) {
        return nullptr;
    }

    auto producerVF = dequantVF.getOperand(inputBlockArg.getArgNumber()).getDefiningOp<VPU::VerticalFusionOp>();
    if (producerVF == nullptr || getSingleVFUser(producerVF) != dequantVF ||
        !hasExpectedMemPermuteProducer(producerVF, dequantVF)) {
        return nullptr;
    }

    return producerVF;
}

std::optional<SmallVector<VPU::VerticalFusionOp>> DynamicDequantConvVFPattern::getMatchedChain(
        VPU::VerticalFusionOp rootOp, Logger log) const {
    auto dequantVF = rootOp;
    SmallVector<VPU::VerticalFusionOp> chain;
    auto memPermuteVF = getExpectedMemPermuteProducer(dequantVF);
    if (memPermuteVF != nullptr) {
        chain.push_back(memPermuteVF);
    }

    auto convVF = getSingleVFUser(dequantVF);
    if (convVF == nullptr) {
        return std::nullopt;
    }

    if (!alignConvStrategyWithDequant(dequantVF, convVF, log)) {
        return std::nullopt;
    }

    if (!hasExpectedChain(dequantVF, convVF)) {
        return std::nullopt;
    }

    chain.push_back(dequantVF);
    chain.push_back(convVF);
    return chain;
}

std::optional<VPU::VF::v2::VFCase> DynamicDequantConvVFPattern::buildVFCase(
        VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp convVFOp,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) const {
    VPU::VF::v2::VFCacheAnalysis cache(mergedOp.getOperation());
    VPU::VF::v2::VFConfig vfConfig(mergedOp, cache, /*enableVFPipelining=*/true, /*firstVFNeedsTiling=*/true,
                                   /*secondVFNeedsTiling=*/true);
    auto tiling = parseIntArrayAttr<int64_t>(convVFOp.getTilingStrategyAttr());

    auto dequantOp = getDynamicDequantizeOp(mergedOp);
    auto convOp = getTrailingConvOp(mergedOp);
    if (dequantOp == nullptr || convOp == nullptr) {
        return std::nullopt;
    }

    const auto minNumCalc = [&](Dim, const VPU::VF::v2::VFSplit&) {
        return tiling[Dims4D::Act::C.ind()];
    };

    const auto maxNumCalc = [&vfConfig](Dim dim, const VPU::VF::v2::VFSplit&) {
        return VPU::VF::v2::getTilingLimit(dim, vfConfig);
    };

    auto split = VPU::VF::v2::getVFTilingSplit(tiling);
    auto tiledOnSpatialDim = llvm::any_of(split, [](auto& item) {
        return item.second.has_value() && VPU::VF::v2::isSpatialDim(item.first);
    });
    if (!tiledOnSpatialDim) {
        return std::nullopt;
    }

    split[Dims4D::Act::C] = std::nullopt;
    auto vfCase = VPU::VF::v2::findBestVFCaseFromSplits(vfConfig, {split}, minNumCalc, maxNumCalc, costFunction, log,
                                                        mergedOp.getContext());
    if (!vfCase.has_value()) {
        return std::nullopt;
    }

    Dim spatialDim = Dims4D::Act::H;
    for (auto& item : split) {
        if (item.second.has_value() && VPU::VF::v2::isSpatialDim(item.first)) {
            spatialDim = item.first;
            break;
        }
    }

    auto baseTiling = parseIntArrayAttr<int64_t>(vfCase->getTiling());
    const auto maxNumTiles = vpux::getMaxNumTiles(vfConfig.getOutputs().front(), /*checkMinimalWidthAndHeight=*/true,
                                                  /*checkWorkloadEfficiency=*/true);

    auto vfSchedulingChecks = VPU::VF::v2::getSchedulingScenarios(vfConfig, log);
    const auto adjustedMinNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit&) {
        return baseTiling[dim.ind()];
    };
    const auto adjustedMaxNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit&) {
        return maxNumTiles[dim.ind()];
    };

    const auto filterDynamicDequantConvCMXOverflow = [&](Dim dim, const VPU::VF::v2::VFSplit& candidateSplit) {
        auto candidate = VPU::VF::v2::getVFCaseWithTiling(vfConfig, dim, candidateSplit, adjustedMinNumCalc,
                                                          adjustedMaxNumCalc, log, vfSchedulingChecks);
        return !candidate.isInitialized() ||
               !fitsDynamicDequantConvCMX(vfConfig, dequantOp, convOp, candidate.getTilingStorage());
    };

    auto adjustedSplit = VPU::VF::v2::getVFTilingSplit(baseTiling);
    adjustedSplit[spatialDim] = std::nullopt;

    auto fullPrefetchVFCase = VPU::VF::v2::findBestVFCaseFromSplits(
            vfConfig, {std::move(adjustedSplit)}, adjustedMinNumCalc, adjustedMaxNumCalc, costFunction, log,
            mergedOp.getContext(), filterDynamicDequantConvCMXOverflow);
    if (fullPrefetchVFCase.has_value()) {
        return fullPrefetchVFCase;
    }
    return vfCase;
}

mlir::FailureOr<VPU::VerticalFusionOp> DynamicDequantConvVFPattern::tryMerge(
        VPU::VerticalFusionOp rootOp, mlir::RewriterBase& rewriter,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) const {
    log.trace("DynamicDequantConvVFPattern::tryMerge start root={0}", rootOp->getLoc());
    auto convAlignmentState = getConvAlignmentState(getSingleVFUser(rootOp));
    const auto restoreConvAlignmentOnFailure = [&]() {
        if (convAlignmentState.has_value()) {
            restoreConvAlignmentState(convAlignmentState.value());
        }
    };

    auto matchedOps = getMatchedChain(rootOp, log);
    if (!matchedOps.has_value()) {
        log.trace("DynamicDequantConvVFPattern::tryMerge no matched chain");
        restoreConvAlignmentOnFailure();
        return mlir::failure();
    }

    const auto originalTailOp = matchedOps->back();
    rewriter.setInsertionPoint(originalTailOp);
    auto mergedOp = VPU::fuseOpsInBlock(rewriter, matchedOps.value());
    if (mergedOp == nullptr) {
        log.trace("DynamicDequantConvVFPattern::tryMerge fuseOpsInBlock failed for chain root={0}", rootOp->getLoc());
        restoreConvAlignmentOnFailure();
        return mlir::failure();
    }

    auto vfCase = buildVFCase(mergedOp, matchedOps->back(), costFunction, log);
    if (!vfCase.has_value()) {
        log.trace("DynamicDequantConvVFPattern::tryMerge buildVFCase failed, skip fusion at {0}", mergedOp->getLoc());
        rewriter.eraseOp(mergedOp);
        restoreConvAlignmentOnFailure();
        return mlir::failure();
    }

    // Reject SOH-to-SOK alignment when the merged VF can only satisfy the MINIMAL scenario.
    // MINIMAL only proves that tiling is valid; it does not provide the prefetching or pipelining benefit
    // expected to offset the SOK alignment in this pattern.
    if (convAlignmentState.has_value() && isSplitOverHeightLike(convAlignmentState->multiClusterStrategy) &&
        isSplitOverKernel(mlir::cast<VPU::ClusteredOpInterface>(convAlignmentState->convOp.getOperation()))) {
        auto vfScenario = vfCase->getScenario();
        if (vfScenario.has_value() && vfScenario.value() == VPU::VFScenario::MINIMAL) {
            log.trace("DynamicDequantConvVFPattern::tryMerge reject fusion at {0}: VF scenario is MINIMAL",
                      mergedOp->getLoc());
            rewriter.eraseOp(mergedOp);
            restoreConvAlignmentOnFailure();
            return mlir::failure();
        }
    }

    vfCase->approveScheduling();
    mergedOp = vfCase->getConfig().getSubgraph();

    rewriter.replaceOp(originalTailOp, mergedOp.getResult(0));
    for (auto it = matchedOps->rbegin() + 1; it != matchedOps->rend(); ++it) {
        if ((*it) != nullptr && (*it)->use_empty()) {
            log.trace("DynamicDequantConvVFPattern::tryMerge cleanup erase op={0}", (*it)->getLoc());
            rewriter.eraseOp(*it);
        }
    }

    return mergedOp;
}
