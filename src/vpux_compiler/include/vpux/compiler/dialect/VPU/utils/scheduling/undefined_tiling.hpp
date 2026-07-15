//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_scenario_base.hpp"

//===----------------------------------------------------------------------===//
//                    UNDEFINED TILING - MEMORY ALLOCATION
//===----------------------------------------------------------------------===//
//
// ALGORITHM SUMMARY
// =================
//
// This module implements a memory allocation strategy for temporal tiling of
// operations across multiple loop iterations. The goal is to minimize memory
// reloads while fitting all buffers within available CMX memory.
//
// PHASE 1: BUFFER ANALYSIS & SORTING
// ----------------------------------
//   1. Extract compute operations and their input/output buffers
//   2. Detect inner-loop dependencies (output of iteration N used in N+1)
//   3. Sort operations by buffer frequency (most common buffers first)
//   4. Build frequency table: operand_idx -> buffer -> usage_count
//
// PHASE 2: SHARED BUFFER RESERVATION
// ----------------------------------
//   - Identify globally shared buffers (used in ALL iterations)
//   - Reserve memory for these at fixed locations (top of memory)
//   - Calculate remaining available memory for per-iteration buffers
//
// PHASE 3: MEMORY ALLOCATION
// --------------------------
//   - Allocate buffers in order of minimum reloads (most shared first)
//   - Use ping-pong allocation: alternate slots between iterations
//   - When memory exhausted, reuse previous allocation slots
//
// This allocator implements two strategies for temporal tiling memory management.
// Each strategy trades off memory efficiency vs. reload overhead.
//
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │                         MEMORY LAYOUT (All Strategies)                      │
// │                                                                             │
// │  Address: 0                                              memorySize         │
// │           ├──────── Local Working Area ────────┼──── Global Shared ────┤    │
// │           │  (grows upward ->)                 │  (reserved at top)    │    │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY 1: PIPELINING (Highest Performance)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Requirements: 2× largest iteration size fits in available memory
// Benefit: Zero buffer reloads between iterations
//
// Memory Layout:
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │  Slot A (even iterations)  │  Slot B (odd iterations)  │  Global Buffers    │
// │  iter 0, 2, 4, ...         │  iter 1, 3, 5, ...        │  (shared)          │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Execution Timeline:
//   Iter 0: [==== Slot A ====]
//   Iter 1:                    [==== Slot B ====]
//   Iter 2: [==== Slot A ====]  (reuses Slot A, no reload)
//
// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY 2: PREFETCHING (Fallback - Always Works)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Requirements: None (always applicable)
// Benefit: Handles any buffer pattern
//
// Memory Layout:
// ┌─────────────────────────────────────────────────────────────────────────────┐
// │  Sequential allocation with LRU-style reuse on overflow  │  Global Buffers  │
// └─────────────────────────────────────────────────────────────────────────────┘
//
// Allocation Policy:
//   1. Allocate sequentially from offset 0
//   2. On overflow: reuse oldest allocation for same operand position
//   3. Track live buffers to avoid conflicts
//
// PHASE 4: SCHEDULE GENERATION
// ----------------------------
//   - Generate explicit allocation/deallocation schedule per operation
//   - Detect address conflicts with alive buffers
//   - Order DMAs to prioritize those without deallocations
//

namespace vpux::VPU {

class UndefinedTiling : public TemporalTilingScenarioBase {
public:
    UndefinedTiling();
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
