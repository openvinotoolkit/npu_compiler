//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"

//===----------------------------------------------------------------------===//
//                         PIPELINE TILING SCENARIO
//===----------------------------------------------------------------------===//
//
// OVERVIEW
// ========
//
// Pipeline tiling (full double-buffering) overlaps DMA-in, compute, and DMA-out
// across different tiles simultaneously. At steady state, three consecutive
// tiles are in-flight: one being loaded, one being computed, one being stored.
//
// This achieves the highest throughput potentially but requires two
// complete sets of non-shared buffers (ping-pong) to reside in CMX at once.
//
// COST
// ====
//
//   first tile: tileCost = stallCost + dmaInCost + max(computeCost, dmaOutCost)
//   mid tiles : tileCost = stallCost + max(dmaInCost, computeCost, dmaOutCost)
//   last tile : tileCost = stallCost + max(dmaInCost, computeCost) + dmaOutCost
//
// - dmaInCost, computeCost, dmaOutCost: overlapped between tiles (the longest dominates)
// - stallCost: DMA cost of reloading a buffer shared for a sub-set of tiles
//
// MEMORY REQUIREMENTS
// ===================
//
// Peak CMX usage = 2 * individual tile buffers + shared buffers.
// The factor of 2 comes from ping-pong: while one slot is used by compute,
// the other is being filled/drained by DMA.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │ CMX Memory Layout (steady state)                                            │
// │                                                                             │
// │  ├───── Slot A (tile N) ─────┼───── Slot B (tile N+1) ─────┼── shared ──┤   │
// │  │ input[N] + output[N]      │ input[N+1] + output[N+1]    │   wt/act   │   │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// EXECUTION TIMELINE
// ============================
//
// Case in which shared buffers are constant across tiles. Pipeline runs uninterrupted.
//
// Time ────────────────────────────────────────────────────────────────────────────────────────────>
//
// SHARED:  |==IN[SHRD]==|----------------------------shared buffer lifetime------------------------------------->
// DMA_IN:               |==IN[0]==|==IN[1]==|  |==IN[2]==|  |==IN[3]==|  |==IN[4]==|
// Compute:                        |===CMP[0]===|===CMP[1]===|===CMP[2]===|===CMP[3]===|===CMP[4]===|
// DMA_OUT:                                     |==OUT[0]==| |==OUT[1]==| |==OUT[2]==| |==OUT[3]==| |==OUT[4]==|
//
//                                 IN[2], CMP[1], OUT[0] all execute concurrently
//
//
// Pipeline phases:
//   - Ramp-up:     Only DMA_IN[0], then DMA_IN[1] + CMP[0] overlap.
//   - Steady-state: DMA_IN[N+2] + CMP[N+1] + DMA_OUT[N] run in parallel.
//   - Ramp-down:   Last compute + last DMA_OUT drain.
//
// Ping-pong memory usage:
//   - Even iterations (0, 2, 4, ...): use Slot A
//   - Odd iterations  (1, 3, 5, ...): use Slot B
//   - Shared buffers (weights, activations, common to a set tiles): one copy
//
// Observations:
//   - Throughput depends on the slowest among DMA-in, compute, or DMA-out.
//   - Requires minTileCount >= 2 for memory constraint checking.
//   - Shared buffers reduce the memory overhead (only non-shared are doubled).
//

namespace vpux::VPU {

class PipelineTiling : public TemporalTilingScenarioBase {
public:
    PipelineTiling();
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
