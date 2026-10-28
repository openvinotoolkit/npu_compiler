//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/prefetch_lastop_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {
PrefetchingLastOpVFScheduling::PrefetchingLastOpVFScheduling(Logger log, bool prefetching,
                                                             std::shared_ptr<const VFCostStrategy> costStrategy)
        : v2::VFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool PrefetchingLastOpVFScheduling::validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
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

    auto opTiling = tilingInfo->getRef(lastOp, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *lastOp);

    const auto& opTilingValue = opTiling.value().get();
    auto lastOpSize = config.getOperationRequiredCMX(lastOp, opTilingValue.second, opTilingValue.first.tiles);
    lastOpSize -= getSharedSizeByAllTiles({lastOp}, config, tilingInfo);

    auto sharedSize = getSharedSizeByAllTiles(config.getVFOperations().getArrayRef(), config, tilingInfo);

    const auto thresholdCMXSize = getTotalCMXFragmentationAwareSize(lastOp);
    return inputSize + sharedSize + lastOpSize + reservedMemory < thresholdCMXSize;
}

VFScenario PrefetchingLastOpVFScheduling::getType() const {
    return VFScenario::LASTOP_PREFETCHING;
}

}  // namespace vpux::VPU::VF::v2
