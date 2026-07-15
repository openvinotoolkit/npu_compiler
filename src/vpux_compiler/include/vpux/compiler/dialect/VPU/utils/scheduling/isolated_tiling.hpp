//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"

//===----------------------------------------------------------------------===//
//                         ISOLATED TILING SCENARIO
//===----------------------------------------------------------------------===//
//
// OVERVIEW
// ========
//
// Isolated tiling is the most conservative tiling/scheduling scenario.
// Each tile is processed in sequence: DMA-in, compute, and DMA-out complete
// for one tile before the next tile begins.
//
// This scenario requires less aggressive tiling compared to prefetch or pipelined
// scenarios, as only one tile's data needs to fit in CMX at a time (it can still
// fill the whole CMX). However, it does not exploit parallelism between DMA and
// compute, usually resulting in higher overall latency.
//
// COST
// ====
//
//   tileCost = stallCost + dmaInCost + computeCost + dmaOutCost
//
// All four components are simply summed.
//
// MEMORY REQUIREMENTS
// ===================
//
// Peak CMX usage = single tile's inputs + outputs + shared buffers (no double-buffering).
// Shared buffers are loaded for a sub-set of tiles; local buffers need to be loaded for each tile
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ CMX Memory Layout (during tile N)                                       │
// │                                                                         │
// │  ├─── input tile N ───┼─── output tile N ───┤── shared ──┤              │
// └─────────────────────────────────────────────────────────────────────────┘
//
// EXECUTION TIMELINE
// ============================
//
// Time ─────────────────────────────────────────────────────────────────────────────────────────────────────────>
//
// SHARED:  |==IN[SHRD]==|----------------------------shared buffer lifetime------------------------------------->
// DMA_IN:               |==IN[0]==|                             |==IN[1]==|
// Compute:                        |====COMPUTE[0]====|                     |====COMPUTE[1]====|
// DMA_OUT:                                           |==OUT[0]==|                             |==OUT[1]==|
//
//          ├───────────────────── tile 0 ───────────────────────┤──────────────── tile 1 ────────────────┤─...
//
// Observations:
//   - Each tile must fully complete before the next one starts.
//   - DMA and compute engines are idle while the other works.
//   - Used as a fallback when prefetch or pipeline strategies
//     cannot fit.
//

namespace vpux::VPU {

class IsolatedTiling : public TemporalTilingScenarioBase {
public:
    IsolatedTiling();
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
