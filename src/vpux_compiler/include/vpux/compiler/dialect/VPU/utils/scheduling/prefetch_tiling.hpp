//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"

//===----------------------------------------------------------------------===//
//                         PREFETCH TILING SCENARIO
//===----------------------------------------------------------------------===//
//
// OVERVIEW
// ========
//
// Prefetch tiling overlaps the local buffered DMA-in for the next tile with the
// compute of the current tile. While the DPU/SHAVE processes tile N, the DMA engine
// fetches the inputs for tile N+1 into CMX. DMA-out remains serialized after compute.
//
// This scenario requires higher tiling than isolated tiling (shared buffers +
// current tile local buffers + next tile's prefetched input must coexist),
// but significantly reduces idle DMA time.
//
// COST
// ====
//
//   first tile : tileCost = stallCost + dmaInCost + computeCost + dmaOutCost
//   other tiles: tileCost = stallCost + max(dmaInCost, computeCost) + dmaOutCost
//
// DMA-in and compute overlap (the longer one dominates). DMA-out is serial.
//
// MEMORY REQUIREMENTS
// ===================
//
// Peak CMX usage = current tile (inputs + output) + prefetched next-tile inputs
//                + any shared buffers (activations/weights reused across tiles).
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ CMX Memory Layout (during compute of tile N)                                │
// │                                                                             │
// │  ├── input[N] ──┼── output[N] ──┼── prefetch input[N+1] ──┼── shared ──┤    │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// EXECUTION TIMELINE
// ============================
//
// Time ─────────────────────────────────────────────────────────────────────────────────────────────────────────>
//
// SHARED:  |==IN[SHRD]==|----------------------------shared buffer lifetime------------------------------------->
// DMA_IN:               |==IN[0]==|==IN[1]==|                   |==IN[2]==|                   |==IN[3]==|
// Compute:                        |====COMPUTE[0]====|          |====COMPUTE[1]====|          |====COMPUTE[2]====|...
// DMA_OUT:                                           |==OUT[0]==|                  |==OUT[1]==|
//
//                                  IN[1] prefetched while COMPUTE[0] runs
//
// Detailed breakdown:
//   1. DMA_IN[0] loads first tile (no overlap — nothing to overlap with).
//   2. COMPUTE[0] starts; simultaneously DMA_IN[1] prefetches next tile.
//   3. Once COMPUTE[0] finishes, DMA_OUT[0] writes result.
//   4. COMPUTE[1] starts (inputs already in CMX); DMA_IN[2] prefetches.
//   5. Pattern repeats for remaining tiles.
//   6. When a shared buffer is used across a subset of tiles, a DMA operation is needed to bring it into CMX, adding to
//   the overall latency.
//
// Observations:
//   - DMA-in latency is hidden behind compute when computeCost >= dmaInCost.
//   - If dmaInCost > computeCost, compute finishes earlier and waits.
//   - Requires minTileCount >= 2 for memory constraint checking.
//   - Shared buffers (same weights/activations across a subset of tiles) reduce peak memory.
//

namespace vpux::VPU {

class PrefetchTiling : public TemporalTilingScenarioBase {
public:
    PrefetchTiling();
    llvm::StringRef getName() const override;
    bool satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim, VPU::LayerCostModel& costModel,
                                 Logger log) const override;
    CostInfo calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                           VPU::LayerCostModel& costModel) const override;
    Byte calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                             VPU::LayerCostModel& costModel) const override;
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override;

protected:
    uint64_t computeTileOverallCost(uint64_t stallCost, uint64_t dmaInCost, uint64_t computeCost, uint64_t dmaOutCost,
                                    bool isFirstTile, bool isLastTile) const override;
    void classifyOperandSize(PeakMemorySizeBuckets& buckets, Byte alignedSize, size_t operandIndex, size_t numOperands,
                             bool isActivationShared, bool isWeightShared) const override;
    Byte computeTilePeakMemory(const PeakMemorySizeBuckets& buckets, Byte weightTableSize, Byte sparsityMapSize,
                               bool isWeightShared, bool isWeightTableShared) const override;

private:
    Logger _log;
};

}  // namespace vpux::VPU
