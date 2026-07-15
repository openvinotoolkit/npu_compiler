//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/dynamic_dequant_conv_vf_pattern.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/minimal_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;
using namespace VPU;

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
    auto hasMultiDimTilingIncludeChannel = llvm::count_if(tilingStrategy,
                                                          [](auto tileSize) {
                                                              return tileSize > 1;
                                                          }) > 1 &&
                                           tilingStrategy[Dims4D::Act::C.ind()] > 1;
    if (!hasMultiDimTilingIncludeChannel) {
        return false;
    }

    auto dequantVFArg = llvm::find(convVF->getOperands(), dequantVF.getResult(0));
    if (dequantVFArg == convVF->getOperands().end()) {
        return false;
    }

    const auto argIndex = std::distance(convVF->getOperands().begin(), dequantVFArg);
    auto blockArg = convBody->getArgument(argIndex);
    mlir::Value filterInput = blockArg;

    for (auto* viewOp : llvm::ArrayRef(convOps).drop_back()) {
        if (!VPU::isPureViewOp(viewOp) || viewOp->getOperand(0) != filterInput) {
            return false;
        }

        filterInput = viewOp->getResult(0);
    }

    return convOp.getFilter() == filterInput;
}

bool DynamicDequantConvVFPattern::hasExpectedMemPermuteProducer(VPU::VerticalFusionOp memPermuteVF,
                                                                VPU::VerticalFusionOp dequantVF) const {
    auto* memPermuteOp = getSingleInnerOp(memPermuteVF);
    if (!mlir::isa_and_nonnull<VPU::MemPermuteOp>(memPermuteOp)) {
        return false;
    }

    auto* dequantOp = getSingleInnerOp(dequantVF);
    auto dynamicDequantizeOp = mlir::dyn_cast_or_null<VPU::DynamicDequantizeOp>(dequantOp);
    if (dynamicDequantizeOp == nullptr) {
        return false;
    }

    auto memPermuteBlockArg = VPU::getLinkedArgumentBetweenVFOps(dequantVF, memPermuteVF);
    return memPermuteBlockArg != nullptr && dynamicDequantizeOp.getInput() == memPermuteBlockArg;
}

VPU::VerticalFusionOp DynamicDequantConvVFPattern::getExpectedMemPermuteProducer(
        VPU::VerticalFusionOp dequantVF) const {
    auto* dequantOp = getSingleInnerOp(dequantVF);
    auto dynamicDequantizeOp = mlir::dyn_cast_or_null<VPU::DynamicDequantizeOp>(dequantOp);
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
        VPU::VerticalFusionOp rootOp) const {
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
    VPU::VF::v2::VFConfig vfConfig(mergedOp, true, true, true);
    auto tiling = parseIntArrayAttr<int64_t>(convVFOp.getTilingStrategyAttr());

    auto schedulingChecks = VPU::VF::v2::getSchedulingScenarios(vfConfig, log);
    auto minNumCalc = [&](Dim, const VPU::VF::v2::VFSplit&) {
        return tiling[Dims4D::Act::C.ind()];
    };
    const auto maxNumCalc = [&vfConfig](Dim dim, const VPU::VF::v2::VFSplit&) {
        return VPU::VF::v2::getTilingLimit(dim, vfConfig);
    };

    auto split = VPU::VF::v2::getVFTilingSplit(tiling);
    split[Dims4D::Act::C] = std::nullopt;
    auto tiledOnSpatialDim = llvm::any_of(split, [](auto& item) {
        return item.second.has_value() && VPU::VF::v2::isSpatialDim(item.first);
    });
    if (!tiledOnSpatialDim) {
        return std::nullopt;
    }
    return VPU::VF::v2::findBestVFCaseFromSplits(vfConfig, {std::move(split)}, minNumCalc, maxNumCalc, costFunction,
                                                 log, mergedOp.getContext());
}

mlir::FailureOr<VPU::VerticalFusionOp> DynamicDequantConvVFPattern::tryMerge(
        VPU::VerticalFusionOp rootOp, mlir::RewriterBase& rewriter,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log) const {
    log.trace("DynamicDequantConvVFPattern::tryMerge start root={0}", rootOp->getLoc());
    auto matchedOps = getMatchedChain(rootOp);
    if (!matchedOps.has_value()) {
        log.trace("DynamicDequantConvVFPattern::tryMerge no matched chain");
        return mlir::failure();
    }

    const auto originalTailOp = matchedOps->back();
    rewriter.setInsertionPoint(originalTailOp);
    auto mergedOp = VPU::fuseOpsInBlock(rewriter, matchedOps.value());
    if (mergedOp == nullptr) {
        log.trace("DynamicDequantConvVFPattern::tryMerge fuseOpsInBlock failed for chain root={0}", rootOp->getLoc());
        return mlir::failure();
    }

    auto vfCase = buildVFCase(mergedOp, matchedOps->back(), costFunction, log);
    if (!vfCase.has_value()) {
        log.trace("DynamicDequantConvVFPattern::tryMerge buildVFCase failed, skip fusion at {0}", mergedOp->getLoc());
        rewriter.eraseOp(mergedOp);
        return mlir::failure();
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
