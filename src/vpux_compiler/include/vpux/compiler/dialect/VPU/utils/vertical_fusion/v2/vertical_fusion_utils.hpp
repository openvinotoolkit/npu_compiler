//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"

namespace vpux::VPU::VF::v2 {

using VFSplit = std::map<Dim, std::optional<int64_t>>;

// check if whole operation is in CMX
bool isCmxOperation(mlir::Operation* operation, const bool checkTilingType);

// check if previous operation has some DDR users apart from VF
bool hasBeforeDDRUsers(mlir::Operation* prevOp, mlir::Operation* nextOp);

// Check if the op has multi view op user with shape changed, which will cause the output to be spilled
bool hasOutputSpilledForDifferentDataSizeUses(mlir::Operation* op);

// Check if the op's output is tiled on same axis as the distributed output type's tiling axis
bool outputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op);

// Check if the op's input is tiled on same axis as the distributed input type's tiling axis
bool inputTileAxisIsSameAsMultiClusterStrategy(mlir::Operation* op, mlir::Value operand);

// get the maximal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMaximalValidTilingStrategyFromRange(
        VFConfig& config, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy, Dim tilingAxis,
        TilingOperationStorage::UPtr& opStorage, Logger log);

// get the minimal valid tiling strategy for VF block between the given range of tiling strategy
mlir::FailureOr<SmallVector<int64_t>> getMinimalValidTilingStrategyFromRange(
        VFConfig& config, ArrayRef<int64_t> lowerTilingStrategy, ArrayRef<int64_t> upperTilingStrategy, Dim tilingAxis,
        TilingOperationStorage::UPtr& opStorage, Logger log);

// calculate tiling regions based on particular tiling strategy
mlir::FailureOr<TilingStorage> calculateTilingRegions(VFConfig& config, ArrayRef<int64_t> tilingStrategy, Logger log,
                                                      const TilingOperationStorage::UPtr& opStorage);

// Restore tiling strategy by VF split
SmallVector<int64_t> restoreTilingBySplit(int64_t rank, const VFSplit& split);

// Return Vf tiling split from strategy
VFSplit getVFTilingSplit(ArrayRef<int64_t> tilingStrategy);

// Get number of tiles from split
int64_t getVFTilesLen(const VFSplit& vfSplit);

// calculate limit for number of tiles for set of operations
int64_t getTilingLimit(Dim axis, VFConfig& config, bool tilingOnHW = false);

// if the maxTile is too large, return the cbrt of it if it's a valid max tile candidate
std::optional<int64_t> getCbrtMaxTileCandidate(int64_t minTile, int64_t maxTile);

// Determines if the operand represents shared weights for the operation in Vertical Fusion
bool isOperandSharedWeightsForTiling(mlir::Operation* op, mlir::Value operand, ShapeRef tiledShape);

}  // namespace vpux::VPU::VF::v2
