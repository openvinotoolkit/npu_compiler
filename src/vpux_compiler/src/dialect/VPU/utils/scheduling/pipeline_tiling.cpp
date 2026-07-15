//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/pipeline_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"

#include <algorithm>

namespace vpux::VPU {

PipelineTiling::PipelineTiling(): _log(Logger::global()) {
    _log.setName("PipelineTiling");
}

llvm::StringRef PipelineTiling::getName() const {
    return "PIPELINE";
}

bool PipelineTiling::satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim,
                                             VPU::LayerCostModel& costModel, Logger /*log*/) const {
    return satisfyMemoryConstraintBase(op, nTilesOnDim, costModel, /*minTileCount=*/2);
}

CostInfo PipelineTiling::calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                                       VPU::LayerCostModel& costModel) const {
    return calculateCostBase(op, tiling, costModel);
}

uint64_t PipelineTiling::computeTileOverallCost(uint64_t stallCost, uint64_t dmaInCost, uint64_t computeCost,
                                                uint64_t dmaOutCost, bool isFirstTile, bool isLastTile) const {
    // Note: for PipelineTiling minTileCount=2 is required for satisfyMemoryConstraint,
    // so a tile can never be both first and last at the same time.
    VPUX_THROW_WHEN(isFirstTile && isLastTile,
                    "Invalid tile position flags: a tile cannot be both first and last in PipelineTiling");
    if (isFirstTile) {
        // The first tile has no previous compute to hide DMA-in behind.
        // stallCost + dmaInCost + max(computeCost, dmaOutCost)
        return vpux::addSaturating(vpux::addSaturating(stallCost, dmaInCost), std::max(computeCost, dmaOutCost));
    }
    if (isLastTile) {
        // The last tile has no next compute to hide DMA-out behind.
        // stallCost + max(dmaInCost, computeCost) + dmaOutCost
        return vpux::addSaturating(vpux::addSaturating(stallCost, std::max(dmaInCost, computeCost)), dmaOutCost);
    }
    // stallCost + max(dmaInCost, computeCost, dmaOutCost)
    return vpux::addSaturating(stallCost, std::max(std::max(dmaInCost, computeCost), dmaOutCost));
}

Byte PipelineTiling::calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                                         VPU::LayerCostModel& costModel) const {
    return calculatePeakMemoryBase(op, tiling, costModel, /*minTileCount=*/2);
}

void PipelineTiling::classifyOperandSize(PeakMemorySizeBuckets& buckets, Byte alignedSize, size_t operandIndex,
                                         size_t /*numOperands*/, bool isActivationShared, bool isWeightShared) const {
    if (isActivationShared && operandIndex == 0) {
        buckets.shared += alignedSize;
        return;
    }
    if (isWeightShared && operandIndex == 1) {
        buckets.shared += alignedSize;
        return;
    }
    buckets.individual += alignedSize;
}

Byte PipelineTiling::computeTilePeakMemory(const PeakMemorySizeBuckets& buckets, Byte weightTableSize,
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
    return individual * 2 + shared;
}

LoopScheduleResult PipelineTiling::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                       vpux::AddressType memorySize) const {
    // The implementation of this function is going to be completed as part of Scheduling refactoring
    // TODO: E#202068
    VPUX_UNUSED(loopRegion);
    VPUX_UNUSED(memorySize);
    VPUX_THROW("PipelineTiling TODO: getScheduleStrategy()");
}

}  // namespace vpux::VPU
