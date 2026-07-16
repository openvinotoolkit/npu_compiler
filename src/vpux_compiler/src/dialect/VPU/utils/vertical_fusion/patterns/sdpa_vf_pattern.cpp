//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/sdpa_vf_pattern.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/utils/core/numeric.hpp"

using namespace vpux;
using namespace VPU;

bool SDPAVFPattern::hasExpectedChain(ArrayRef<VPU::VerticalFusionOp> vfOps) const {
    if (vfOps.size() != 4) {
        return false;
    }

    auto* op0 = getSingleInnerOp(vfOps[0]);
    auto* op1 = getSingleInnerOp(vfOps[1]);
    auto* op2 = getSingleInnerOp(vfOps[2]);
    auto* op3 = getSingleInnerOp(vfOps[3]);
    if (op0 == nullptr || op1 == nullptr || op2 == nullptr || op3 == nullptr) {
        return false;
    }

    const auto isNonEltwiseNCEOp = [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op) && !op->hasTrait<VPU::EltwiseOp>();
    };
    const auto isEltwiseLike = [](mlir::Operation* op) {
        return mlir::isa<VPU::NCEOpInterface>(op) && op->hasTrait<VPU::EltwiseOp>();
    };

    return isNonEltwiseNCEOp(op0) && isEltwiseLike(op1) && mlir::isa<VPU::SoftMaxOp>(op2) && isNonEltwiseNCEOp(op3);
}

std::optional<SmallVector<VPU::VerticalFusionOp>> SDPAVFPattern::getMatchedChain(VPU::VerticalFusionOp rootOp) const {
    SmallVector<VPU::VerticalFusionOp> vfOps;
    vfOps.push_back(rootOp);

    auto second = getSingleVFUser(rootOp);
    if (second == nullptr) {
        return std::nullopt;
    }
    vfOps.push_back(second);

    auto third = getSingleVFUser(second);
    if (third == nullptr) {
        return std::nullopt;
    }
    vfOps.push_back(third);

    auto fourth = getSingleVFUser(third);
    if (fourth == nullptr) {
        return std::nullopt;
    }
    vfOps.push_back(fourth);

    if (!hasExpectedChain(vfOps)) {
        return std::nullopt;
    }

    return vfOps;
}

std::optional<VPU::VF::v2::VFCase> SDPAVFPattern::buildVFCase(VPU::VerticalFusionOp mergedOp,
                                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                              Logger log) const {
    VPU::VF::v2::VFConfig vfConfig(mergedOp, true, true, true);
    if (!alignMultiClusterStrategy(vfConfig, log)) {
        return std::nullopt;
    }

    auto allowedDims = getAllowedDims(vfConfig.getVFOperations().getArrayRef(), log);
    if (allowedDims.empty()) {
        return std::nullopt;
    }

    auto schedulingChecks = VPU::VF::v2::getSchedulingScenarios(vfConfig, log);
    auto minNumCalc = [](Dim, const VPU::VF::v2::VFSplit&) {
        return VPU::MINIMUM_LENGTH_TILING;
    };
    const auto getMaxNumCalc = [&](bool multiDimTiling) {
        return [&, multiDimTiling](Dim tilingDim, const VPU::VF::v2::VFSplit& split) {
            auto maxTiles = VPU::VF::v2::getTilingLimit(tilingDim, vfConfig, multiDimTiling);
            if (split.size() > 1) {
                if (VPU::VF::v2::isSpatialDim(tilingDim)) {
                    auto otherDimTiles = VPU::VF::v2::getVFTilesLen(split);
                    maxTiles = divUp(maxTiles, otherDimTiles);
                } else {
                    maxTiles = VPU::VF::v2::getTilingLimit(tilingDim, vfConfig, true);
                }
            }
            return maxTiles;
        };
    };

    auto splits = VPU::VF::v2::getSplitFromDimArr(allowedDims, allowedDims, vfConfig,
                                                  VPU::VF::v2::isMultiDimTilingPerformant(vfConfig, {}));
    const auto maxNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit& split) {
        return getMaxNumCalc(split.size() > 1)(dim, split);
    };

    return VPU::VF::v2::findBestVFCaseFromSplits(vfConfig, splits, minNumCalc, maxNumCalc, costFunction, log,
                                                 mergedOp.getContext());
}

bool SDPAVFPattern::alignMultiClusterStrategy(VPU::VF::v2::VFConfig& vfConfig, Logger log) const {
    // Softmax has some limitations for multi-cluster strategy, so if the pattern is matched, align strategy for other
    // ops;
    auto ops = vfConfig.getOperationsForTiling();
    auto iter = llvm::find_if(ops, [](mlir::Operation* op) {
        return mlir::isa<VPU::SoftMaxOp>(op);
    });
    VPUX_THROW_WHEN(iter == ops.end(), "Softmax op is expected in the matched pattern");
    auto softmax = *iter;
    auto mcStrategy = mlir::cast<VPU::ClusteredOpInterface>(softmax).getMultiClusterStrategy();

    auto moduleOp = softmax->getParentOfType<mlir::ModuleOp>();
    auto tileOp = config::getTileExecutor(moduleOp);
    const auto numTiles = tileOp.getCount();

    auto misAlignedOpSize = 0;
    if (mcStrategy.has_value()) {
        for (auto* op : ops) {
            auto clusteredOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(op);
            if (op == softmax || clusteredOp == nullptr || clusteredOp.getMultiClusterStrategy() == mcStrategy) {
                continue;
            }

            if (!clusteredOp.checkStrategyCompatibility(mcStrategy.value(), numTiles)) {
                return false;
            }
            clusteredOp.setMultiClusterStrategy(mcStrategy.value());
            misAlignedOpSize++;
            if (misAlignedOpSize > 1) {
                log.trace("SDPAVFPattern::alignMultiClusterStrategy failed for too many ops with misaligned multi "
                          "cluster strategy at {0}",
                          vfConfig.getSubgraph()->getLoc());
                return false;
            }
        }
    }
    return true;
}

mlir::FailureOr<VPU::VerticalFusionOp> SDPAVFPattern::tryMerge(VPU::VerticalFusionOp rootOp,
                                                               mlir::RewriterBase& rewriter,
                                                               const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                               Logger log) const {
    log.trace("SDPAVFPattern::tryMerge start root={0}", rootOp->getLoc());
    auto matchedOps = getMatchedChain(rootOp);
    if (!matchedOps.has_value()) {
        log.trace("SDPAVFPattern::tryMerge no matched chain");
        return mlir::failure();
    }
    log.trace("SDPAVFPattern::tryMerge matched chain size={0}", matchedOps->size());

    const auto originalTailOp = matchedOps->back();

    rewriter.setInsertionPoint(originalTailOp);
    auto mergedOp = VPU::fuseOpsInBlock(rewriter, matchedOps.value());
    if (mergedOp == nullptr) {
        log.trace("SDPAVFPattern::tryMerge fuseOpsInBlock failed for chain root={0}", rootOp->getLoc());
        return mlir::failure();
    }

    log.trace("SDPAVFPattern::tryMerge buildVFCase begin {0}", rootOp->getLoc());
    auto vfCase = buildVFCase(mergedOp, costFunction, log);
    if (!vfCase.has_value()) {
        log.trace("SDPAVFPattern::tryMerge buildVFCase failed, skip fusion and return failure at {0}",
                  mergedOp->getLoc());
        rewriter.eraseOp(mergedOp);
        return mlir::failure();
    }

    log.trace("SDPAVFPattern::tryMerge approveScheduling {0}", rootOp->getLoc());
    vfCase->approveScheduling();
    mergedOp = vfCase->getConfig().getSubgraph();

    rewriter.replaceOp(originalTailOp, mergedOp.getResult(0));

    // At this point mergedOp replaces the fully-absorbed chain.
    // The predecessor VF ops (%0, %1, %2) were absorbed into the new blocks during
    // fuseOpsInBlock but their outer VF shells may still exist; erase them now.
    for (auto it = matchedOps->rbegin() + 1; it != matchedOps->rend(); ++it) {
        if ((*it) != nullptr && (*it)->use_empty()) {
            log.trace("SDPAVFPattern::tryMerge cleanup erase op={0}", (*it)->getLoc());
            rewriter.eraseOp(*it);
        }
    }

    return mergedOp;
}
