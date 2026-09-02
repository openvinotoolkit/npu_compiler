//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/sdpa_vf_pattern.hpp"

#include "vpux/compiler/core/tiling.hpp"
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

namespace {

bool isNCEConv(mlir::Operation* op) {
    return mlir::isa<VPU::NCEConvolutionOp>(op);
}

bool isNCEReduce(mlir::Operation* op) {
    return mlir::isa<VPU::NCEReduceOp>(op);
}

bool isEltwiseLike(mlir::Operation* op) {
    return mlir::isa<VPU::NCEOpInterface>(op) && op->hasTrait<VPU::EltwiseOp>();
}

bool isNonEltwiseNCEOp(mlir::Operation* op) {
    return mlir::isa<VPU::NCEOpInterface>(op) && !op->hasTrait<VPU::EltwiseOp>();
}

int64_t getMinTiles(ArrayRef<int64_t> maxNumTilesWithDPUEfficiency, Dim dim, const VPU::VF::v2::VFConfig& vfConfig) {
    const auto isSoftmaxDecomposedSDPA = [&] {
        auto hasSoftmax = false;
        auto hasNCEReduce = false;
        for (auto* op : vfConfig.getOperationsForTiling()) {
            hasSoftmax |= mlir::isa<VPU::SoftMaxOp>(op);
            hasNCEReduce |= mlir::isa<VPU::NCEReduceOp>(op);
        }

        return hasSoftmax && hasNCEReduce;
    }();

    auto minNumTiles = VPU::MINIMUM_LENGTH_TILING;
    // Use `VPU::MIN_REQUIRED_TILES` as the lower bound for SoftMaxDecomposedSDPA.
    // The default `VPU::MINIMUM_LENGTH_TILING` requires the pattern to be split into at least 4 tiles, while VF
    // pipelining can work with 2 or 3 tiles for this pattern.
    // Keep this lower-bound relaxation scoped to some special pattern until the generic policy is revisited.
    // Track: E#229055
    if (dim == Dims4D::Act::H || dim == Dims4D::Act::W) {
        if ((dim.ind() < static_cast<int64_t>(maxNumTilesWithDPUEfficiency.size()) &&
             maxNumTilesWithDPUEfficiency[dim.ind()] < minNumTiles) ||
            isSoftmaxDecomposedSDPA) {
            minNumTiles = VPU::MIN_REQUIRED_TILES;
        }
    }

    return minNumTiles;
}

}  // namespace

std::optional<SmallVector<mlir::Operation*>> SDPAVFPattern::getInnerOps(ArrayRef<VPU::VerticalFusionOp> vfOps,
                                                                        Logger log, StringRef validatorName) const {
    SmallVector<mlir::Operation*> ops;
    ops.reserve(vfOps.size());
    for (auto vfOp : vfOps) {
        auto* innerOp = getSingleInnerOp(vfOp);
        if (innerOp == nullptr) {
            log.trace("SDPAVFPattern::{0} failed: VF op at {1} has no single inner op", validatorName, vfOp->getLoc());
            return std::nullopt;
        }
        ops.push_back(innerOp);
    }
    return ops;
}

bool SDPAVFPattern::hasExpectedSoftmaxDecomposedChain(ArrayRef<VPU::VerticalFusionOp> vfOps, Logger log) const {
    if (vfOps.size() != 6) {
        return false;
    }

    auto ops = getInnerOps(vfOps, log, "hasExpectedSoftmaxDecomposedChain");
    if (!ops.has_value()) {
        return false;
    }

    return isNCEConv((*ops)[0]) && mlir::isa<VPU::SoftMaxOp>((*ops)[1]) && isNCEConv((*ops)[2]) &&
           isNCEReduce((*ops)[3]) && isNCEConv((*ops)[4]) && isEltwiseLike((*ops)[5]);
}

bool SDPAVFPattern::hasExpectedSDPAChain(ArrayRef<VPU::VerticalFusionOp> vfOps, Logger log) const {
    if (vfOps.size() != 4) {
        return false;
    }

    auto ops = getInnerOps(vfOps, log, "hasExpectedSDPAChain");
    if (!ops.has_value()) {
        return false;
    }
    auto softmax = mlir::dyn_cast<VPU::SoftMaxOp>((*ops)[2]);
    if (softmax == nullptr || softmax.getPadSizeAttr() != nullptr) {
        // Skip when softmax has pad_size attribute, which may have performance regressions using pattern based fusion.
        return false;
    }

    return isNonEltwiseNCEOp((*ops)[0]) && isEltwiseLike((*ops)[1]) && isNonEltwiseNCEOp((*ops)[3]);
}

// Attempt to match a SoftmaxDecomposedSDPA pattern: Conv-SoftMax-{Conv, Reduce->Conv}-Eltwise
std::optional<SmallVector<VPU::VerticalFusionOp>> SDPAVFPattern::tryMatchSoftmaxDecomposedSDPA(
        VPU::VerticalFusionOp rootOp, VPU::VerticalFusionOp second, Logger log) const {
    const auto getSingleMatchingVFUser = [&](VPU::VerticalFusionOp producer, auto predicate) -> VPU::VerticalFusionOp {
        VPU::VerticalFusionOp matchedUser = nullptr;
        for (auto* user : producer->getUsers()) {
            auto vfUser = mlir::dyn_cast<VPU::VerticalFusionOp>(user);
            if (vfUser == nullptr) {
                continue;
            }
            auto* innerOp = getSingleInnerOp(vfUser);
            if (innerOp == nullptr || !predicate(innerOp)) {
                continue;
            }
            if (matchedUser != nullptr) {
                return nullptr;
            }
            matchedUser = vfUser;
        }
        return matchedUser;
    };

    auto outputConvVF = getSingleMatchingVFUser(second, isNCEConv);
    auto reduceVF = getSingleMatchingVFUser(second, isNCEReduce);
    if (outputConvVF == nullptr || reduceVF == nullptr) {
        return std::nullopt;
    }

    // Ensure the SoftMax VF output is consumed exactly by the two expected branches.
    if (std::distance(second.getResult(0).use_begin(), second.getResult(0).use_end()) != 2) {
        return std::nullopt;
    }
    for (auto* user : second->getUsers()) {
        if (user != outputConvVF.getOperation() && user != reduceVF.getOperation()) {
            return std::nullopt;
        }
    }

    auto reduceConvVF = getSingleVFUser(reduceVF);
    auto eltwiseVF = getSingleVFUser(outputConvVF);
    if (reduceConvVF == nullptr || eltwiseVF == nullptr) {
        return std::nullopt;
    }

    if (getSingleVFUser(reduceConvVF) != eltwiseVF) {
        return std::nullopt;
    }

    SmallVector<VPU::VerticalFusionOp> vfOps{rootOp, second, outputConvVF, reduceVF, reduceConvVF, eltwiseVF};
    if (!hasExpectedSoftmaxDecomposedChain(vfOps, log)) {
        return std::nullopt;
    }

    log.trace("SDPAVFPattern::tryMatchSoftmaxDecomposedSDPA matched SoftmaxDecomposedSDPA pattern");
    return vfOps;
}

// Attempt to match a SDPA pattern: NCEOp-Eltwise-SoftMax-NCEOp
std::optional<SmallVector<VPU::VerticalFusionOp>> SDPAVFPattern::tryMatchSDPA(VPU::VerticalFusionOp rootOp,
                                                                              VPU::VerticalFusionOp second,
                                                                              Logger log) const {
    auto third = getSingleVFUser(second);
    if (third == nullptr) {
        return std::nullopt;
    }

    auto fourth = getSingleVFUser(third);
    if (fourth == nullptr) {
        return std::nullopt;
    }

    SmallVector<VPU::VerticalFusionOp> vfOps{rootOp, second, third, fourth};
    if (!hasExpectedSDPAChain(vfOps, log)) {
        return std::nullopt;
    }

    return vfOps;
}

std::optional<SmallVector<VPU::VerticalFusionOp>> SDPAVFPattern::getMatchedChain(VPU::VerticalFusionOp rootOp,
                                                                                 Logger log) const {
    auto second = getSingleVFUser(rootOp);
    if (second == nullptr) {
        return std::nullopt;
    }

    auto* rootInnerOp = getSingleInnerOp(rootOp);
    auto* secondInnerOp = getSingleInnerOp(second);
    if (rootInnerOp == nullptr || secondInnerOp == nullptr) {
        return std::nullopt;
    }

    // Check for SoftmaxDecomposedSDPA pattern (Conv-SoftMax-{Conv, Reduce->Conv}-Eltwise)
    if (mlir::isa<VPU::NCEConvolutionOp>(rootInnerOp) && mlir::isa<VPU::SoftMaxOp>(secondInnerOp)) {
        auto result = tryMatchSoftmaxDecomposedSDPA(rootOp, second, log);
        if (result.has_value()) {
            return result;
        }
    }

    // Check for SDPA pattern (NCEOp-Eltwise-SoftMax-NCEOp)
    auto result = tryMatchSDPA(rootOp, second, log);
    if (result.has_value()) {
        return result;
    }

    log.trace("SDPAVFPattern::getMatchedChain no matching SDPA chain found at {0}", rootOp->getLoc());
    return std::nullopt;
}

std::optional<VPU::VF::v2::VFCase> SDPAVFPattern::buildVFCase(VPU::VerticalFusionOp mergedOp,
                                                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                                                              Logger log) const {
    VPU::VF::v2::VFCacheAnalysis cache(mergedOp.getOperation());
    VPU::VF::v2::VFConfig vfConfig(mergedOp, cache, /*enableVFPipelining=*/true, /*firstVFNeedsTiling=*/true,
                                   /*secondVFNeedsTiling=*/true);
    if (!alignMultiClusterStrategy(vfConfig, log)) {
        return std::nullopt;
    }

    auto allowedDims = getAllowedDims(vfConfig.getVFOperations().getArrayRef(), log, true);
    if (allowedDims.empty()) {
        return std::nullopt;
    }

    const auto maxNumTiles = vpux::getMaxNumTiles(vfConfig.getOutputs().front(), /*checkMinimalWidthAndHeight=*/true,
                                                  /*checkWorkloadEfficiency=*/true);
    auto minNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit&) {
        return getMinTiles(maxNumTiles, dim, vfConfig);
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

    const auto useMultiDimTiling = VPU::VF::v2::isMultiDimTilingPerformant(vfConfig, {});
    auto splits = VPU::VF::v2::getSplitFromDimArr(allowedDims, allowedDims, vfConfig, useMultiDimTiling);
    log.trace("SDPAVFPattern::buildVFCase generated splits count={0}, useMultiDimTiling={1} at {2}", splits.size(),
              useMultiDimTiling, mergedOp->getLoc());
    const auto maxNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit& split) {
        return getMaxNumCalc(split.size() > 1)(dim, split);
    };

    auto bestCase = VPU::VF::v2::findBestVFCaseFromSplits(vfConfig, splits, minNumCalc, maxNumCalc, costFunction, log,
                                                          mergedOp.getContext(), nullptr);
    if (!bestCase.has_value()) {
        log.trace("SDPAVFPattern::buildVFCase failed: findBestVFCaseFromSplits returned nullopt at {0}",
                  mergedOp->getLoc());
    }

    return bestCase;
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
    auto matchedOps = getMatchedChain(rootOp, log);
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
