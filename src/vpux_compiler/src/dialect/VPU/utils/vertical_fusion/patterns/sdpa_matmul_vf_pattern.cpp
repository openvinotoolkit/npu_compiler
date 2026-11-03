//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/patterns/sdpa_matmul_vf_pattern.hpp"

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/merge_tiling_policy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <queue>

using namespace vpux;
using namespace VPU;

namespace {

bool isEltwiseLike(mlir::Operation* op) {
    return mlir::isa<VPU::NCEOpInterface>(op) && op->hasTrait<VPU::EltwiseOp>();
}

}  // namespace

std::optional<SmallVector<VPU::VerticalFusionOp>> SDPAMatMulVFPattern::getMatchedChain(VPU::VerticalFusionOp rootOp,
                                                                                       Logger log) const {
    auto second = getUniqUser(rootOp);
    if (second == nullptr) {
        return std::nullopt;
    }

    auto third = getUniqUser(second);
    if (third == nullptr) {
        return std::nullopt;
    }

    auto fourth = getUniqUser(third);
    if (fourth == nullptr) {
        return std::nullopt;
    }

    SmallVector<VPU::VerticalFusionOp> vfOps{rootOp, second, third, fourth};
    auto ops = getInnerOps(vfOps, log, "getMatchedChain");
    if (!ops.has_value()) {
        return std::nullopt;
    }

    auto softmax = mlir::dyn_cast<VPU::SoftMaxOp>((*ops)[2]);
    if (!mlir::isa<VPU::NCEMatMulOp>((*ops)[0]) || !isEltwiseLike((*ops)[1]) || softmax == nullptr ||
        !mlir::isa<VPU::NCEMatMulOp>((*ops)[3])) {
        return std::nullopt;
    }

    log.trace("SDPAMatMulVFPattern matched MatMul-Eltwise-SoftMax-MatMul chain at {0}", rootOp->getLoc());
    return vfOps;
}

bool SDPAMatMulVFPattern::alignMultiClusterStrategy(VPU::VF::v2::VFConfig& vfConfig, Logger log) const {
    const auto& ops = vfConfig.getOperationsForTiling();
    auto lastOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(ops.back());
    if (lastOp == nullptr || !lastOp.getMultiClusterStrategy().has_value()) {
        return false;
    }

    llvm::SmallPtrSet<mlir::Operation*, 16> vfOps(ops.begin(), ops.end());
    VPU::VF::v2::MultiClusterStrategyMap adjustedStrategies;

    std::queue<VPU::ClusteredOpInterface> worklist;
    llvm::SmallPtrSet<mlir::Operation*, 16> visited;
    worklist.push(lastOp);
    while (!worklist.empty()) {
        auto consumerOp = worklist.front();
        worklist.pop();
        if (!visited.insert(consumerOp.getOperation()).second) {
            continue;
        }

        for (const auto& parentInfo : VPU::VF::v2::details::getParentOpWithViewInputs(consumerOp)) {
            auto producerOp = parentInfo.clusteredOp;
            if (!vfOps.contains(producerOp.getOperation())) {
                continue;
            }
            SmallVector<VPU::MultiClusterStrategy> candidates;
            if (auto iter = adjustedStrategies.find(producerOp); iter != adjustedStrategies.end()) {
                candidates.push_back(iter->second);
            } else if (auto strategy = producerOp.getMultiClusterStrategy()) {
                candidates.push_back(strategy.value());
            }
            for (auto strategy : {VPU::MultiClusterStrategy::SplitOverGroup, VPU::MultiClusterStrategy::SplitOverHeight,
                                  VPU::MultiClusterStrategy::SplitOverKernel}) {
                if (llvm::find(candidates, strategy) == candidates.end()) {
                    candidates.push_back(strategy);
                }
            }
            if (!VPU::VF::v2::hasCompatibleDistributedTypes(parentInfo, consumerOp, adjustedStrategies) &&
                !VPU::VF::v2::tryAdjustStrategy(producerOp, parentInfo, consumerOp, adjustedStrategies, candidates)) {
                log.trace("SDPAMatMulVFPattern::alignMultiClusterStrategy failed to align VF at {0}",
                          vfConfig.getSubgraph()->getLoc());
                return false;
            }
            worklist.push(producerOp);
        }
    }

    for (auto& item : adjustedStrategies) {
        item.first.setMultiClusterStrategy(item.second);
    }
    return true;
}

std::optional<VPU::VF::v2::VFCase> SDPAMatMulVFPattern::buildVFCase(
        VPU::VerticalFusionOp mergedOp, ArrayRef<VPU::VerticalFusionOp> matchedOps,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, VFMergeCostMode costMode, Logger log) const {
    VPU::VF::v2::VFCacheAnalysis cache(mergedOp.getOperation());
    VPU::VF::v2::VFConfig vfConfig(mergedOp, cache, /*enableVFPipelining=*/true, /*firstVFNeedsTiling=*/true,
                                   /*secondVFNeedsTiling=*/true);
    if (!alignMultiClusterStrategy(vfConfig, log)) {
        return std::nullopt;
    }

    auto lastMatmul = matchedOps.back();
    auto tilingStrategyAttr = lastMatmul.getTilingStrategyAttr();
    if (tilingStrategyAttr == nullptr) {
        return std::nullopt;
    }

    auto tilingStrategy = parseIntArrayAttr<int64_t>(tilingStrategyAttr);
    auto allEqualToOne = llvm::all_of(tilingStrategy, [](int64_t val) {
        return val == 1;
    });
    if (allEqualToOne) {
        tilingStrategy[DimsGroups5D::Act::G.ind()] = VPU::MIN_REQUIRED_TILES;
    }
    const auto minNumCalc = [&tilingStrategy](Dim dim, const VPU::VF::v2::VFSplit&) {
        return tilingStrategy[dim.ind()];
    };

    const auto maxNumCalc = [&](Dim dim, const VPU::VF::v2::VFSplit&) {
        return tilingStrategy[dim.ind()];
    };

    // set Matmul's strategy for the fused op
    auto split = VPU::VF::v2::getVFTilingSplit(tilingStrategy);
    split.rbegin()->second = std::nullopt;

    SmallVector<VPU::VF::v2::VFSplit> splits = {std::move(split)};
    auto bestCase = VPU::VF::v2::findBestVFCaseFromSplits(vfConfig, splits, minNumCalc, maxNumCalc, costFunction, false,
                                                          log, mergedOp.getContext(), nullptr, costMode);
    if (!bestCase.has_value()) {
        return std::nullopt;
    }
    return bestCase;
}
