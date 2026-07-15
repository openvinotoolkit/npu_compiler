//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/mem_size.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::VPU {
class TilingBuilderOpInterface;

struct CostInfo {
    uint64_t dmaCost = 0;
    uint64_t computeCost = 0;
    uint64_t overallCost = 0;
    double efficiency = 0;

    bool operator==(const CostInfo& other) const {
        return dmaCost == other.dmaCost && computeCost == other.computeCost && overallCost == other.overallCost &&
               efficiency == other.efficiency;
    }
};

struct TemporalTilingInfo {
    Shape tilingShape;
    CostInfo costInfo;
    vpux::SmallString scenarioName;
};

struct TemporalTilingSearchSpaceConfig;

using LastScenarioSearchStopPredicate = bool (*)(Byte peakMemory, Byte cmxSize,
                                                 const TemporalTilingSearchSpaceConfig& config);

bool defaultLastScenarioSearchStopPredicate(Byte peakMemory, Byte cmxSize,
                                            const TemporalTilingSearchSpaceConfig& config);

struct TemporalTilingSearchSpaceConfig {
    // Candidate minimum H/W tile sizes. Smaller values allow more spatial tiles and expand the search range.
    SmallVector<int64_t> nceTargetSpatialSizes = {4, 8, 16};
    // Candidate minimum C tile sizes. Smaller values allow more channel tiles; each value is combined with each
    // spatial size to form max-tiling limits for 2D/3D search.
    SmallVector<int64_t> nceTargetChannelSizes = {16, 32, 64};
    // Stop last-scenario (Pipeline) search when peakMemory <= cmxSize * memoryUsageBasedTerminator.
    // Terminate tiling number search earlier to avoid inefficient over-tiling.
    double memoryUsageBasedTerminator = 0.5;
    // Predicate that terminates last-scenario candidate collection, preventing inefficient over-tiling.
    LastScenarioSearchStopPredicate lastScenarioSearchStopPredicate = defaultLastScenarioSearchStopPredicate;
};

const TemporalTilingSearchSpaceConfig& getDefaultTemporalTilingSearchSpaceConfig();

DimArr getDimsForTiling(mlir::Operation* op, Logger log);

// Adjusts nTilesOnDim downward until the current tiling is compatible with the selected multi-cluster strategy.
// Always succeeds as it will return a single tile if no compatible tiling is found
void ensureNTilesIsCompatibleWithMultiClusterDown(mlir::Operation* op, Shape& nTilesOnDim, Dim dimToTile,
                                                  ShapeRef outputShape, const Logger& log);

// Adjusts nTilesOnDim upward within maxTiling until a multi-cluster-compatible tiling is found.
// Returns failure if the search range is exhausted without finding a compatible shape.
mlir::LogicalResult ensureNTilesIsCompatibleWithMultiClusterUp(mlir::Operation* op, Shape& nTilesOnDim, Dim dimToTile,
                                                               ShapeRef outputShape, ArrayRef<int64_t> maxTiling,
                                                               const Logger& log);
bool isInPlaceOperation(mlir::Operation* op);
bool isActivationTiledOnLowestDim(VPU::TilingBuilderOpInterface tilingOp, const TileInfo& outTile,
                                  const DimsOrder& inOrder);
SmallVector<SmallVector<int64_t>> getTilingMaxLimits(
        mlir::Operation* op,
        const TemporalTilingSearchSpaceConfig& config = getDefaultTemporalTilingSearchSpaceConfig());

}  // namespace vpux::VPU
