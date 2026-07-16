//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/schedule_builder_utils.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/temporal_tiling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/scheduling/tiling_scenario_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/sibling_ops_analysis.hpp"
#include "vpux/compiler/dialect/VPU/utils/tile_utils.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/utils/core/small_vector.hpp"

namespace vpux::VPU {
class NCEOpInterface;
class TilingBuilderOpInterface;

// TODO E#216915: Investigate correct value that controls the number of tiles considered in peak memory calculation
inline constexpr size_t PEAK_MEMORY_NUM_TILES_SEARCH_HEURISTIC = 10;

struct PeakMemorySizeBuckets {
    Byte shared = Byte(0);
    Byte individual = Byte(0);
    Byte prefetch = Byte(0);
};

class TemporalTilingScenarioBase : public TilingScenarioInterface {
public:
    TemporalTilingScenarioBase() = default;
    virtual ~TemporalTilingScenarioBase();
    TemporalTilingScenarioBase(const TemporalTilingScenarioBase&) = delete;
    TemporalTilingScenarioBase& operator=(const TemporalTilingScenarioBase&) = delete;
    TemporalTilingScenarioBase(TemporalTilingScenarioBase&&) = delete;
    TemporalTilingScenarioBase& operator=(TemporalTilingScenarioBase&&) = delete;

    // Returns a human-readable identifier for this tiling scenario
    llvm::StringRef getName() const override = 0;
    // Checks whether the given tile count per dimension fits within CMX memory for this scenario
    bool satisfyMemoryConstraint(mlir::Operation* op, const Shape& nTilesOnDim, VPU::LayerCostModel& costModel,
                                 Logger log) const override = 0;
    // Estimates the execution cost (DMA + compute + stall cycles) for the given tiling
    CostInfo calculateCost(mlir::Operation* op, const OutputTiling& tiling,
                           VPU::LayerCostModel& costModel) const override = 0;
    // Computes the peak CMX memory footprint required to execute the given tiling
    Byte calculatePeakMemory(mlir::Operation* op, const OutputTiling& tiling,
                             VPU::LayerCostModel& costModel) const override = 0;
    // Determines the loop schedule for a given compute region assuming current tiling scenario
    LoopScheduleResult getScheduleStrategy(const ComputeRegion& loopRegion,
                                           vpux::AddressType memorySize) const override = 0;

protected:
    // Shared base implementation of satisfyMemoryConstraint with a configurable minimum tile count
    bool satisfyMemoryConstraintBase(mlir::Operation* op, const Shape& nTilesOnDim, VPU::LayerCostModel& costModel,
                                     size_t minTileCount) const;

    // Shared base implementation of calculateCost; subclasses override computeTileOverallCost and computeSharedFlags
    CostInfo calculateCostBase(mlir::Operation* op, const OutputTiling& tiling, VPU::LayerCostModel& costModel,
                               bool skipDmaForSingleTile = false) const;

    // Combines per-tile cost components into a single overall cost value (scenario-specific formula)
    virtual uint64_t computeTileOverallCost(uint64_t stallCost, uint64_t dmaInCost, uint64_t computeCost,
                                            uint64_t dmaOutCost, bool isFirstTile, bool isLastTile) const = 0;

    // Shared base implementation of calculatePeakMemory with a configurable minimum tile count
    // It calls computeSharedFlags and classifyOperandSize to determine how to bucket operand sizes for peak memory
    // estimation, then calls computeTilePeakMemory
    Byte calculatePeakMemoryBase(mlir::Operation* op, const OutputTiling& tiling, VPU::LayerCostModel& costModel,
                                 size_t minTileCount) const;

    // Determines whether activations/weights are shared across tiles by comparing
    // back-inferred input offsets between the first two tiles
    std::pair<bool, bool> computeSharedFlags(mlir::Operation* op, const OutputTiling& tiling) const;

    // Assigns an aligned operand size to the appropriate peak-memory bucket (shared/individual/prefetch)
    virtual void classifyOperandSize(PeakMemorySizeBuckets& buckets, Byte alignedSize, size_t operandIndex,
                                     size_t numOperands, bool isActivationShared, bool isWeightShared) const = 0;

    // Computes the final peak memory for a single tile from classified bucket sizes
    virtual Byte computeTilePeakMemory(const PeakMemorySizeBuckets& buckets, Byte weightTableSize, Byte sparsityMapSize,
                                       bool isWeightShared, bool isWeightTableShared) const = 0;
};

}  // namespace vpux::VPU
