//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/prefetch_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"

#include <algorithm>

namespace vpux::VPU {

PrefetchTiling::PrefetchTiling(): _log(Logger::global()) {
    _log.setName("PrefetchTiling");
}

llvm::StringRef PrefetchTiling::getName() const {
    return "PREFETCH";
}

bool PrefetchTiling::satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim,
                                             VPU::LayerCostModel& costModel, Logger /*log*/) const {
    return satisfyMemoryConstraintBase(op, nTilesOnDim, costModel, /*minTileCount=*/2);
}

CostInfo PrefetchTiling::calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                                       VPU::LayerCostModel& costModel) const {
    return calculateCostBase(op, tiling, costModel);
}

uint64_t PrefetchTiling::computeTileOverallCost(uint64_t stallCost, uint64_t dmaInCost, uint64_t computeCost,
                                                uint64_t dmaOutCost, bool isFirstTile, bool /*isLastTile*/) const {
    // The first tile has no previous compute to hide DMA-in behind.
    if (isFirstTile) {
        // stallCost + dmaInCost, computeCost + dmaOutCost
        return vpux::addSaturating(vpux::addSaturating(vpux::addSaturating(stallCost, dmaInCost), computeCost),
                                   dmaOutCost);
    }
    // stallCost + max(dmaInCost, computeCost) + dmaOutCost
    return vpux::addSaturating(vpux::addSaturating(stallCost, std::max(dmaInCost, computeCost)), dmaOutCost);
}

Byte PrefetchTiling::calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                                         VPU::LayerCostModel& costModel) const {
    return calculatePeakMemoryBase(op, tiling, costModel, /*minTileCount=*/2);
}

void PrefetchTiling::classifyOperandSize(PeakMemorySizeBuckets& buckets, Byte alignedSize, size_t operandIndex,
                                         size_t numOperands, bool isActivationShared, bool isWeightShared) const {
    if (isActivationShared && operandIndex == 0) {
        buckets.shared += alignedSize;
        return;
    }
    if (isWeightShared && operandIndex == 1) {
        buckets.shared += alignedSize;
        return;
    }
    if (operandIndex < numOperands - 1) {
        // prefetch non shared input operands
        buckets.prefetch += alignedSize;
    }
    buckets.individual += alignedSize;
}

Byte PrefetchTiling::computeTilePeakMemory(const PeakMemorySizeBuckets& buckets, Byte weightTableSize,
                                           Byte sparsityMapSize, bool isWeightShared, bool isWeightTableShared) const {
    auto shared = buckets.shared;
    auto individual = buckets.individual;
    if (isWeightTableShared) {
        shared += weightTableSize;
    } else {
        individual += weightTableSize;
    }
    if (isWeightShared) {
        shared += sparsityMapSize;
    } else {
        individual += sparsityMapSize;
    }
    return individual + shared + buckets.prefetch;
}

LoopScheduleResult PrefetchTiling::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                       vpux::AddressType memorySize) const {
    // The implementation of this function is going to be completed as part of Scheduling refactoring
    // TODO: E#202068
    VPUX_UNUSED(loopRegion);
    VPUX_UNUSED(memorySize);
    VPUX_THROW("PrefetchTiling TODO: getScheduleStrategy()");
}

}  // namespace vpux::VPU
