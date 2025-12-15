//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {

FullPrefetchingVFScheduling::FullPrefetchingVFScheduling(Logger log, bool prefetching)
        : WeightsPrefetchingVFScheduling(log, prefetching) {
}

bool FullPrefetchingVFScheduling::validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                           const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto* largest = config.getLargestOp();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize = getInputsSize(config, tilingInfo);
    auto outputSize = getOutputsSize(config, tilingInfo);

    auto opTiling = tilingInfo->get(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    const auto largestOpSize = VPU::getRequiredCMX(
            largest, config.getOperationTypes(largest, opTiling.value().second, opTiling.value().first.tiles));

    auto inputs = config.getInputs();
    llvm::SetVector<mlir::Operation*> ignoredOps(inputs.begin(), inputs.end());
    ignoredOps.insert(largest);
    ignoredOps.insert(config.getOutputs().begin(), config.getOutputs().end());
    auto ops = llvm::set_difference(config.getVFOperations(), ignoredOps);
    auto sharedSize = getSharedSizeByAllTiles(ops.getArrayRef(), config, tilingInfo);
    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(largest);
    return (inputSize + outputSize + largestOpSize + sharedSize + reservedMemory) < thresholdCMXSize;
}

VFScenario FullPrefetchingVFScheduling::getType() const {
    return VFScenario::FULL_PREFETCHING;
}

void FullPrefetchingVFScheduling::correctOutputSpillCost(
        StrategyCost& spillCost, VFConfig& config, const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
        const int64_t index, const int64_t tilesNumber) const {
    _prefetchedCost[index + 1] = spillCost;
    const auto& inputs = config.getInputs();
    StrategyCost nextTileOpCost = 0;
    if (index != tilesNumber - 1) {
        for (auto* input : inputs) {
            auto foundCost = isolatedOperCost.find(input);
            VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *input);
            nextTileOpCost += foundCost->second;
        }
    }

    spillCost = nextTileOpCost < spillCost ? spillCost - nextTileOpCost : 0U;
}

void FullPrefetchingVFScheduling::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const size_t index) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();

    StrategyCost parentCost = 0;
    if (isInput) {
        if (index == 0) {
            return;
        }
        parentCost = std::accumulate(isolatedOperCost.begin(), isolatedOperCost.end(), 0,
                                     [](const StrategyCost previous, const auto& item) {
                                         return previous + item.second;
                                     });
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index - 1);

    } else {
        auto getPrevOp = [&](mlir::Operation* curOp) -> mlir::Operation* {
            auto isInputForCurOp = llvm::find(config.getInputs(), curOp) != config.getInputs().end();
            if (isInputForCurOp) {
                return nullptr;
            }
            auto* parent = findParent(curOp->getOperand(0));
            if (parent == nullptr && (curOp->getNumOperands() > 1 && operation->hasTrait<VPU::EltwiseOp>())) {
                parent = findParent(curOp->getOperand(1));
            }
            return parent;
        };
        auto curOp = operation;
        while (curOp != nullptr) {
            parentCost += getParentCost(curOp, isolatedOperCost);
            curOp = getPrevOp(curOp);
        }
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index);
    }

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}
}  // namespace vpux::VPU::VF::v2
