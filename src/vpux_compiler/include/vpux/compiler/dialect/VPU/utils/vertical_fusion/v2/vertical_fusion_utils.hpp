//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduler_interface.hpp"

#include <functional>
#include <map>
#include <optional>

namespace vpux::VPU {
class LayerVPUNNCost;
}

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

// Check if dim is spatial for 4D activation layouts
bool isSpatialDim(Dim dim);

// Check whether multi-dimensional tiling is performant for the VF subgraph and current split
bool isMultiDimTilingPerformant(VFConfig& config, const VFSplit& currentVFSplit);

// Build candidate single- and multi-dimensional VF splits from the requested dimensions
SmallVector<VFSplit> getSplitFromDimArr(DimArrRef dimsToCheck, DimArrRef allowedDims, VFConfig& config,
                                        bool enableMultiDimTiling);

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

// Get dim for optimization (the one without tiling value)
std::optional<Dim> getNonTiledDimForVFOptimization(const VFSplit& vfSplit);

// Get number of tiles from split
int64_t getVFTilesLen(const VFSplit& vfSplit);

// Build a map from each VF operation to its back-inferred tiling dimension.
mlir::DenseMap<mlir::Operation*, Dim> buildOpDimMap(VFConfig& config, Dim outputDim);

// calculate limit for number of tiles for set of operations
int64_t getTilingLimit(Dim axis, VFConfig& config, bool multiDimTiling = false);

// if the maxTile is too large, return the cbrt of it if it's a valid max tile candidate
std::optional<int64_t> getCbrtMaxTileCandidate(int64_t minTile, int64_t maxTile);

// Determines if the operand represents shared weights for the operation in Vertical Fusion
bool isOperandSharedWeightsForTiling(mlir::Operation* op, mlir::Value operand, const TileInfo& tileInfo);

// Dump VF scheduling trace to JSON file
void printVFSchedulingTrace(mlir::func::FuncOp funcOp, const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction,
                            Logger log);

// Get dim for optimization (the one without tiling value)
std::optional<Dim> getVFOptimizedDim(const VFSplit& vfSplit);

// Get the mapped VF block argument for the operand, the intermediate view like op will be ignored
mlir::BlockArgument getVFBlockArgument(mlir::Value operand);

// Check if the op supports multi cluster strategy adjustment when merging into VF
bool supportMultiClusterStrategyAdjustmentInVF(mlir::Operation* op);

// Find the first non-view-like user of the operation and its operand index
std::optional<std::pair<mlir::Operation*, int64_t>> findFirstNonViewUser(mlir::Operation* operation);

// Check if the eltwise op with sw op user will cause cmx size exceed for the shared input, which will cause dynamic
// spilling
bool cmxSizeExceedForEltwiseOpWithSwOpUser(VFConfig& currentConfig, ArrayRef<mlir::OpResult> parentResults, Logger log);
}  // namespace vpux::VPU::VF::v2
