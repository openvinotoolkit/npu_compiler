//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/full_prefetch_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"

#include <llvm/ADT/SetOperations.h>

namespace vpux::VPU::VF::v2 {

FullPrefetchingVFScheduling::FullPrefetchingVFScheduling(Logger log, bool prefetching,
                                                         std::shared_ptr<const VFCostStrategy> costStrategy)
        : WeightsPrefetchingVFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool FullPrefetchingVFScheduling::validateScenario(VFConfig& config, const TilingOperationStorage::UPtr& tilingInfo,
                                                   const Byte reservedMemory) const {
    if (tilingInfo == nullptr) {
        return false;
    }

    auto* largest = config.getLargestOp();
    // assuming almost all tiles are same
    const auto index = 0;
    auto inputSize = getInputsSize(config, tilingInfo);
    auto outputSize = getOutputsSize(config, tilingInfo);

    auto opTiling = tilingInfo->getRef(largest, index);
    VPUX_THROW_WHEN(!opTiling.has_value(), "There is no information about tile {0} of operation {1}", index, *largest);
    const auto& opTilingValue = opTiling.value().get();
    const auto largestOpSize = config.getOperationRequiredCMX(largest, opTilingValue.second, opTilingValue.first.tiles);

    const auto& inputs = config.getInputs();
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

}  // namespace vpux::VPU::VF::v2
