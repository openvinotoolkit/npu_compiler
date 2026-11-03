//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/pipelining_double_buffering_vf_scheduling.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_scenario_cost_calculator.hpp"

namespace vpux::VPU::VF::v2 {

PipeliningDoubleBufferingVFScheduling::PipeliningDoubleBufferingVFScheduling(
        Logger log, bool prefetching, std::shared_ptr<const VFCostStrategy> costStrategy)
        : PipeliningVFScheduling(log, prefetching, std::move(costStrategy)) {
}

bool PipeliningDoubleBufferingVFScheduling::isSupportedPattern(const VFConfig& config) const {
    const size_t supportedOpNumber = 2;
    return config.isPipelined() && config.getOperationsForTiling().size() == supportedOpNumber;
}

bool PipeliningDoubleBufferingVFScheduling::validateScenario(VFConfig& config,
                                                             const TilingOperationStorage::UPtr& tilingInfo,
                                                             const Byte reservedMemory) const {
    if (!isSupportedPattern(config)) {
        return false;
    }

    SmallVector<Byte> opNotSharedSizes;
    Byte totalSharedSize = Byte(0);
    if (!calculateBufferSizes(config, tilingInfo, opNotSharedSizes, totalSharedSize)) {
        return false;
    }

    // Buffers holding the same data for every tile are allocated once, the rest need a ping-pong pair so the next
    // tile can be prefetched while both executors are busy with the current one.
    Byte totalNotSharedSize = Byte(0);
    for (auto size : opNotSharedSizes) {
        totalNotSharedSize += size;
    }

    // calculateBufferSizes reports a per-operation footprint, so a tensor produced and consumed inside the region is
    // charged by both its producer and its consumer. Summing those footprints requires discounting it once.
    const auto& operations = config.getOperationsForTiling();
    const int tileIndex = 1;
    for (auto* operation : operations) {
        auto opTiling = tilingInfo->getRef(operation, tileIndex);
        if (!opTiling.has_value()) {
            return false;
        }
        const auto& opTilingValue = opTiling.value().get();
        const auto& tiledTypes = config.getOperationTypes(operation, opTilingValue.second, opTilingValue.first.tiles);
        const auto inputCount = tiledTypes.size() - operation->getNumResults();
        for (auto operandIdx : irange(inputCount)) {
            auto* producer = operation->getOperand(operandIdx).getDefiningOp();
            if (producer != nullptr && llvm::is_contained(operations, producer)) {
                totalNotSharedSize -= tiledTypes[operandIdx].getTotalAllocSize();
            }
        }
    }

    const auto thresholdCMXSize = getTotalCMXSize(config.getOperationsForTiling().front());

    return 2 * totalNotSharedSize + totalSharedSize + reservedMemory <= thresholdCMXSize;
}

VFScenario PipeliningDoubleBufferingVFScheduling::getType() const {
    return VFScenario::VF_DPU_SHV_DOUBLE_BUFFERING;
}

VFPipelineContainer PipeliningDoubleBufferingVFScheduling::getPipelining(
        VFConfig& config, int64_t tilesNumber, const TilingOperationStorage::UPtr& tilingInfo,
        const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction) const {
    VFScenarioCostCalculationContext context(*this, config, tilesNumber, tilingInfo, costFunction, _log, _prefetching);
    PipeliningVFScenarioCostCalculator calculator(true);
    return calculator.getPipelining(context);
}

}  // namespace vpux::VPU::VF::v2
