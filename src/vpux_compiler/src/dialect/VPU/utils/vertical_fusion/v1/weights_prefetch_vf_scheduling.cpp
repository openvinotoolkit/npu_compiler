//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/weights_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

namespace vpux::VPU::VF::v1 {
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
    auto inputSize = getInputsSize(config, tilingInfo);

    auto opTiling = tilingInfo->get(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(largest);
    return inputSize +
                   VPU::getRequiredCMX(largest, config.getOperationTypes(largest, opTiling.value().second,
                                                                         opTiling.value().first.tiles)) +
                   reservedMemory <
           thresholdCMXSize;
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
        } else {
            return;
        }
    } else {
        parentCost = getParentCost(operation, isolatedOperCost);
    }

    prefetchCost = parentCost <= prefetchCost ? prefetchCost - parentCost : 0;
}

VFScenario WeightsPrefetchingVFScheduling::getType() const {
    return VFScenario::WEIGHTS_PREFETCHING;
}

}  // namespace vpux::VPU::VF::v1
