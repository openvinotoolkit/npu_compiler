//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/scheduling/isolated_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"

namespace vpux::VPU {

IsolatedTiling::IsolatedTiling(): _log(Logger::global()) {
    _log.setName("IsolatedTiling");
}

llvm::StringRef IsolatedTiling::getName() const {
    return "ISOLATED";
}

bool IsolatedTiling::satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim,
                                             VPU::LayerCostModel& costModel, Logger /*log*/) const {
    return satisfyMemoryConstraintBase(op, nTilesOnDim, costModel, /*minTileCount=*/1);
}

CostInfo IsolatedTiling::calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                                       VPU::LayerCostModel& costModel) const {
    return calculateCostBase(op, tiling, costModel, /*skipDmaForSingleTile=*/true);
}

uint64_t IsolatedTiling::computeTileOverallCost(uint64_t stallCost, uint64_t dmaInCost, uint64_t computeCost,
                                                uint64_t dmaOutCost, bool /*isFirstTile*/, bool /*isLastTile*/) const {
    // stallCost + dmaInCost + computeCost + dmaOutCost
    return vpux::addSaturating(vpux::addSaturating(vpux::addSaturating(stallCost, dmaInCost), computeCost), dmaOutCost);
}

Byte IsolatedTiling::calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                                         VPU::LayerCostModel& costModel) const {
    return calculatePeakMemoryBase(op, tiling, costModel, /*minTileCount=*/1);
}

void IsolatedTiling::classifyOperandSize(PeakMemorySizeBuckets& buckets, Byte alignedSize, size_t operandIndex,
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

Byte IsolatedTiling::computeTilePeakMemory(const PeakMemorySizeBuckets& buckets, Byte weightTableSize,
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
    return individual + shared;
}

LoopScheduleResult IsolatedTiling::getScheduleStrategy(const ComputeRegion& loopRegion,
                                                       vpux::AddressType memorySize) const {
    // The implementation of this function is going to be completed as part of Scheduling refactoring
    // TODO: E#202068
    VPUX_UNUSED(loopRegion);
    VPUX_UNUSED(memorySize);
    VPUX_THROW("IsolatedTiling TODO: getScheduleStrategy()");
}

}  // namespace vpux::VPU
