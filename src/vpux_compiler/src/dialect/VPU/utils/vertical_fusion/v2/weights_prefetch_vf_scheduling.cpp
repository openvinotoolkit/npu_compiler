//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/weights_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {
WeightsPrefetchingVFScheduling::WeightsPrefetchingVFScheduling(Logger log, bool prefetching,
                                                               std::shared_ptr<const VFCostStrategy> costStrategy)
        : VFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool WeightsPrefetchingVFScheduling::validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                                      const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto* largest = config.getLargestOp();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize =
            getInputsSize(config, tilingInfo) - getSharedSizeByAllTiles(config.getInputs(), config, tilingInfo);
    auto opTiling = tilingInfo->getRef(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    const auto& opTilingValue = opTiling.value().get();
    auto sharedSize = getSharedSizeByAllTiles(config.getVFOperations().getArrayRef(), config, tilingInfo);
    auto largestOpSize = config.getOperationRequiredCMX(largest, opTilingValue.second, opTilingValue.first.tiles);
    largestOpSize -= getSharedSizeByAllTiles({largest}, config, tilingInfo);

    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(largest);
    return inputSize + largestOpSize + sharedSize + reservedMemory < thresholdCMXSize;
}

VFScenario WeightsPrefetchingVFScheduling::getType() const {
    return VFScenario::WEIGHTS_PREFETCHING;
}

}  // namespace vpux::VPU::VF::v2
