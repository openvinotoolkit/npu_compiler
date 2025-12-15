//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/prefetch_lastop_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {
PrefetchingLastOpVFScheduling::PrefetchingLastOpVFScheduling(Logger log, bool prefetching)
        : v2::VFScheduling(log, prefetching) {
}

bool PrefetchingLastOpVFScheduling::validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                             const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto outputOperations = config.getOutputs();
    VPUX_THROW_WHEN(outputOperations.empty(), "No output operations found for {0}", config.getSubgraph());
    auto* lastOp = outputOperations.back();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize =
            getInputsSize(config, tilingInfo) - getSharedSizeByAllTiles(config.getInputs(), config, tilingInfo);

    auto opTiling = tilingInfo->get(lastOp, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *lastOp);

    auto largestOpSize = VPU::getRequiredCMX(
            lastOp, config.getOperationTypes(lastOp, opTiling.value().second, opTiling.value().first.tiles));
    largestOpSize -= getSharedSizeByAllTiles({lastOp}, config, tilingInfo);

    auto sharedSize = getSharedSizeByAllTiles(config.getVFOperations().getArrayRef(), config, tilingInfo);

    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(lastOp);
    return inputSize + sharedSize + largestOpSize + reservedMemory < thresholdCMXSize;
}

VFScenario PrefetchingLastOpVFScheduling::getType() const {
    return VFScenario::LASTOP_PREFETCHING;
}

void PrefetchingLastOpVFScheduling::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const size_t index) const {
    StrategyCost parentCost = 0;
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();
    VPUX_THROW_WHEN(config.getOutputs().empty(), "Cannot find outputs for VF {0}", config.getSubgraph());
    auto* lastOp = config.getOutputs().back();
    if (isInput) {
        if (index == 0) {
            return;
        }

        auto foundCost = isolatedOperCost.find(lastOp);
        VPUX_THROW_WHEN(foundCost == isolatedOperCost.end(), "Cannot find the cost for {0}", *lastOp);
        parentCost = foundCost->second;
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index - 1);
    } else {
        parentCost = getParentCost(operation, isolatedOperCost);
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index);
    }

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}
}  // namespace vpux::VPU::VF::v2
