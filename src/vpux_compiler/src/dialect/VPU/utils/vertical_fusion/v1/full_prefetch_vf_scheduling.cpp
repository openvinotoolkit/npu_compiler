//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v1/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

namespace vpux::VPU::VF::v1 {

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
    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(largest);
    return (inputSize + outputSize +
            VPU::getRequiredCMX(largest, config.getOperationTypes(largest, opTiling.value().second,
                                                                  opTiling.value().first.tiles))) +
                   reservedMemory <
           thresholdCMXSize;
}

void FullPrefetchingVFScheduling::correctOutputSpillCost(
        StrategyCost& spillCost, VFConfig& config, const DenseMap<mlir::Operation*, StrategyCost>& isolatedOperCost,
        const int64_t index, const int64_t tilesNumber) const {
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

VFScenario FullPrefetchingVFScheduling::getType() const {
    return VFScenario::FULL_PREFETCHING;
}
}  // namespace vpux::VPU::VF::v1
