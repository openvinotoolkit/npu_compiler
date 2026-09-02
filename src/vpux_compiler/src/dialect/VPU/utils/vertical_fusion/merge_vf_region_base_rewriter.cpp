//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/merge_vf_region_base_rewriter.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_case.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/convert_to_dma_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SmallSet.h>
#include <mlir/IR/IRMapping.h>
#include <cstddef>

namespace vpux {
namespace VPU {

namespace {
// Check if the op doesn't have multi cluster strategy but can have distributed output. For some ops like SpaceToDepth,
// they don't implement the ClusteredOpInteface, which means that it only runs on single tile instead, however, it will
// converted into DMA op with Distributed output type if the distributed output type is compatible with its user. So the
// VF rewriter need to check if the op is supposed to do like this, otherwise the fusion of VF ops might break this
// pattern and casue it can not be fit into CMX. For example, without VF fusion

//   SingleTileOp       SingleTileOp
//      |                    |
//    NCEOp0       =>   DistributedOutput(SEGMENTED)
//      |                    |
//    NCEOp1              NCEOp0(TilingStrategy=[1,1,1,1])
//                           |
//                        NCEOp1(TilingStrategy=[1,1,2,1])

// With the VF fusion, NCEOp0 and NCEOp1 are fused with tiling strategy changed. and SingleTileOp is not able to use
// distributed output type, and it might exceed the CMX size.

//   SingleTileOp       SingleTileOp
//      |                    |
//    NCEOp0       =>   NonDistributedOutput
//      |                    |
//    NCEOp1             VF(TilingStrategy=[1,1,2,1])
//                        [NCEOp0, NCEOp1]

bool isSingleTileOpCompatibleWithDistributedOutput(VPU::TilingBuilderOpInterface op) {
    if (op->hasAttr(multiClusterStrategy)) {
        return false;
    }
    if (isOpTiled(op)) {
        return false;
    }

    SmallVector<Byte> operationNDTypes;
    for (auto type : op->getOperandTypes()) {
        operationNDTypes.push_back(mlir::cast<NDTypeInterface>(type).getTotalAllocSize());
    }
    for (auto type : op->getResultTypes()) {
        operationNDTypes.push_back(mlir::cast<NDTypeInterface>(type).getTotalAllocSize());
    }
    const auto totalAvailableCMXSize = getTotalCMXSize(op.getOperation());
    return vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(op.getOperation()), operationNDTypes) >
           totalAvailableCMXSize;
}

// This function tries to find mergeable input for the currentOp
// Currently only support NCE task with weights
// E-141686: A general solution to merge more subgraph for VFOp.
template <typename VFConfigType>
mlir::FailureOr<VPU::VerticalFusionOp> findMergeableVFInput(VFConfigType& vfConfig) {
    auto currentOp = vfConfig.getSubgraph();
    for (auto* op : vfConfig.getOperationsForTiling()) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr || nceOp.getWeightsOperand() == nullptr) {
            continue;
        }
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(nceOp.getWeightsOperand())) {
            auto parentOp =
                    currentOp.getOperand(blockArg.getArgNumber()).template getDefiningOp<VPU::VerticalFusionOp>();
            if (parentOp != nullptr) {
                return parentOp;
            }
        }
    }
    return mlir::failure();
}
}  // namespace

// This function checks if other inputs of currentOp can be merged
// If prevOp was tried to merge with currentOp, return false
template <typename VFCaseType>
bool MergeVFRegionBaseRewriter<VFCaseType>::checkOtherVFInput(VPU::VerticalFusionOp currentOp,
                                                              VPU::VerticalFusionOp prevOp) const {
    // Check if currentOp has mergeable input
    auto vfConfig = createVFConfig(currentOp);
    auto mergeableOp = findMergeableVFInput(vfConfig);
    if (mlir::failed(mergeableOp)) {
        return false;
    }
    // prevOp was tried to merge with currentOp
    return mergeableOp.value() != prevOp;
}

// This function checks the weights tilingStrategy is split over output channel

template <typename VFCaseType>
bool MergeVFRegionBaseRewriter<VFCaseType>::isTileOverOutputChannel(VFConfigType& vfConfig) const {
    // Check if nceTaskOp has mergeable input, weights of NCE task
    auto nceTaskOp = vfConfig.getSubgraph();
    auto weightsOp = findMergeableVFInput(vfConfig);
    if (mlir::failed(weightsOp)) {
        return false;
    }

    const auto moreThanOne = [](auto value) {
        return value > 1;
    };

    // weights, tiles on OutputChannel dim > 1
    // NCE task, tiles on activation Channel dim > 1
    const auto weightsTilingStrategy = parseIntArrayAttr<int64_t>(weightsOp.value().getTilingStrategy());
    const auto nceTaskTilingStrategy = parseIntArrayAttr<int64_t>(nceTaskOp.getTilingStrategy());
    return weightsTilingStrategy[Dims4D::Filter::OC.ind()] > 1 || llvm::any_of(nceTaskTilingStrategy, moreThanOne);
}

// Get operandNumber for prevOp output in currentOp inputs

template <typename VFCaseType>
bool MergeVFRegionBaseRewriter<VFCaseType>::hasTiling(ArrayRef<int64_t> tilingInfo) const {
    return llvm::any_of(tilingInfo, [](auto i) {
        return i != 1;
    });
}

// Get operandNumber for prevOp output in currentOp inputs
template <typename VFCaseType>
size_t MergeVFRegionBaseRewriter<VFCaseType>::getLinkNumber(VPU::VerticalFusionOp currentOp,
                                                            VPU::VerticalFusionOp prevOp) const {
    auto operands = currentOp->getOperands();
    auto operandIt = llvm::find_if(operands, [&](auto operand) {
        return operand.getDefiningOp() == prevOp;
    });
    VPUX_THROW_WHEN(operandIt == operands.end(),
                    "Cannot find the operand number for the operation {0} in the current block {1}", prevOp, currentOp);
    return std::distance(operands.begin(), operandIt);
}

template <typename VFCaseType>
bool MergeVFRegionBaseRewriter<VFCaseType>::isLegalFusion(VPU::VerticalFusionOp currentOp,
                                                          VPU::VerticalFusionOp prevOp) const {
    for (auto operand : prevOp->getOperands()) {
        auto parent = findParent(operand);
        if (auto tilingOp = mlir::dyn_cast_or_null<VPU::TilingBuilderOpInterface>(parent)) {
            if (isSingleTileOpCompatibleWithDistributedOutput(tilingOp)) {
                // The fusion of the VF ops may break the related copy optimization of the parent op, so we need to
                // skip it in this case
                return false;
            }
        }
    }

    const auto prevBlock = prevOp.getBody();
    const auto parentVFOp = currentOp.getBody();

    auto newOps = prevBlock->getOps<VPU::VerticalFusionOpInterface>();
    auto oldOps = parentVFOp->getOps<VPU::VerticalFusionOpInterface>();

    if (newOps.empty() || oldOps.empty()) {
        return false;
    }

    // Get input args of current vf region corresponding to previous vf op
    const auto currInputArgs = getLinkedArgumentsBetweenVFOps(currentOp, prevOp);
    VPUX_THROW_WHEN(currInputArgs.empty(),
                    "No corresponding input argument found for current VF region {0} with previous VF region {1}",
                    currentOp, prevOp);

    const auto isClusteredOpWithMCStrategy = [](mlir::Operation* op) {
        auto clusterOp = mlir::dyn_cast_or_null<VPU::ClusteredOpInterface>(op);
        return clusterOp != nullptr && clusterOp.getMultiClusterStrategy().has_value();
    };

    for (const auto& currInputArg : currInputArgs) {
        VPUX_THROW_WHEN(
                currInputArg == nullptr,
                "Corresponding input argument found for current VF region {0} with previous VF region {1} is nullptr",
                currentOp, prevOp);
        // Get output op of previous vf region that produces the linked result
        auto prevOutputResult = mlir::cast<mlir::OpResult>(currentOp.getOperand(currInputArg.getArgNumber()));
        auto* prevOutputOp =
                prevOp.getBody()->getTerminator()->getOperand(prevOutputResult.getResultNumber()).getDefiningOp();
        // Check if previous output op has MC strategy
        const auto prevOutDistType = mlir::dyn_cast_if_present<VPU::DistributedTensorType>(
                getDistributedOutputType(prevOp, currentOp.getOperand(currInputArg.getArgNumber())));
        const auto isPrevOutOpWithMCStrategy = prevOutDistType != nullptr;

        // Hard legality gate: only non-adjustable constraints belong here.
        // Distribution compatibility (areDistributionAttrsCompatible) and eltwise segmented-like checks
        // are intentionally NOT checked here — they are current-state checks that may be resolved by
        // isMCStrategyAligned or adjustMCStrategyInMergedVF. Hard-rejecting them here would prevent the
        // adjustment path from running.
        for (auto currInputOp : currInputArg.getUsers()) {
            SmallVector<mlir::Operation*> currInputViewLikeOps;
            while (mlir::isa<VPU::TilingViewLikeOpInterface>(currInputOp) && currInputOp->hasOneUse()) {
                currInputViewLikeOps.push_back(currInputOp);
                currInputOp = *(currInputOp->getUsers().begin());
            }
            const auto isCurrInOpWithMCStrategy = isClusteredOpWithMCStrategy(currInputOp);
            if (isPrevOutOpWithMCStrategy != isCurrInOpWithMCStrategy) {
                return false;
            }
            if (isPrevOutOpWithMCStrategy && isCurrInOpWithMCStrategy) {
                auto currInputOperand = currInputViewLikeOps.empty() ? mlir::cast<mlir::Value>(currInputArg)
                                                                     : currInputViewLikeOps.back()->getResult(0);
                auto actualCurrInDistType = mlir::cast<VPU::DistributedTensorType>(
                        getDistributedInputType(currInputOp, currInputOperand).getDistributedTypes().front());

                if (!checkMCFusionHardLegality(prevOutDistType, actualCurrInDistType, prevOutputOp, currInputOp,
                                               currInputOperand)) {
                    return false;
                }
            }
        }
    }

    return true;
}

/*
 As soon as we don't have logic right now for excluding operations or break subgraph
 check in advance that all users or previous block will be merged to current one
*/
template <typename VFCaseType>
bool MergeVFRegionBaseRewriter<VFCaseType>::waitOtherUsers(VPU::VerticalFusionOp prevOp,
                                                           VPU::VerticalFusionOp currentOp) const {
    if (prevOp->hasOneUse()) {
        return true;
    }

    for (auto user : prevOp->getUsers()) {
        if (!mlir::isa<VPU::VerticalFusionOp>(user)) {
            return false;
        }
        if (user == currentOp) {
            continue;
        }

        const auto userGoToRegion = llvm::any_of(user->getUsers(), [&](auto current) {
            return current != currentOp;
        });

        if (userGoToRegion) {
            return false;
        }
    }

    return true;
}

template <typename VFCaseType>
void MergeVFRegionBaseRewriter<VFCaseType>::fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp,
                                                       VPU::VerticalFusionOp mergedOp) const {
    // The pinnedStrategy is overwritten by VF strategy, remove the attribute
    for (auto& operation : mergedOp.getBody()->without_terminator()) {
        operation.removeAttr(VPU::pinnedStrategy);
    }
    rewriter.replaceOp(currentOp, mergedOp.getResults());
}

template class MergeVFRegionBaseRewriter<VPU::VF::v1::VFCase>;
template class MergeVFRegionBaseRewriter<VPU::VF::v2::VFCase>;

}  // namespace VPU
}  // namespace vpux
