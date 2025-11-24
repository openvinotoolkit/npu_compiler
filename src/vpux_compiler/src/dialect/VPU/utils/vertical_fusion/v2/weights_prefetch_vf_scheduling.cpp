//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/weights_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {
WeightsPrefetchingVFScheduling::WeightsPrefetchingVFScheduling(Logger log, bool prefetching)
        : VFScheduling(log, prefetching) {
}

bool WeightsPrefetchingVFScheduling::validate(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                              const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto* largest = config.getLargestOp();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize =
            getInputsSize(config, tilingInfo) - getSharedSizeByAllTiles(config.getInputs(), config, tilingInfo);
    auto opTiling = tilingInfo->get(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    auto sharedSize = getSharedSizeByAllTiles(config.getVFOperations().getArrayRef(), config, tilingInfo);
    auto largestOpSize = VPU::getRequiredCMX(
            largest, config.getOperationTypes(largest, opTiling.value().second, opTiling.value().first.tiles));
    largestOpSize -= getSharedSizeByAllTiles({largest}, config, tilingInfo);

    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(largest);
    return inputSize + largestOpSize + sharedSize + reservedMemory < thresholdCMXSize;
}

VFScenario WeightsPrefetchingVFScheduling::getType() const {
    return VFScenario::WEIGHTS_PREFETCHING;
}

void WeightsPrefetchingVFScheduling::correctInputPrefetchingCost(
        StrategyCost& prefetchCost, mlir::Operation* operation, VFConfig& config,
        const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost, const size_t index) const {
    const auto isInput = llvm::find(config.getInputs(), operation) != config.getInputs().end();

    StrategyCost parentCost = 0;
    if (isInput) {
        if (index != 0) {
            parentCost = std::accumulate(isolatedOperCost.begin(), isolatedOperCost.end(), 0,
                                         [](const StrategyCost previous, const auto& item) {
                                             return previous + item.second;
                                         });
            reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index - 1);
        } else {
            return;
        }
    } else {
        parentCost = getParentCost(operation, isolatedOperCost);
        reduceCostWithPrefetchedDMA(parentCost, prefetchCost, index);
    }

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}
}  // namespace vpux::VPU::VF::v2
