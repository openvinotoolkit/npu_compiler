//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
class NCEOpInterface;
class TilingBuilderOpInterface;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

constexpr float INPUT_OVERLAP_THRESHOLD = 2.0;

TilingMode getTilingSupportedMode(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling, Logger log);

// Returns a TilingMode which should be applied to operation and whether it can be prefetched. Caller is
// supposed to check if prefetching condition is satisfied as this function aims to be thread safe
// and avoids inspecting parent's strategy.
std::optional<std::pair<TilingMode, bool>> getTilingMode(mlir::Operation* op, bool enablePrefetchTiling, Logger log);

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     TilingMode& mode, Logger log);
mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     Logger log);

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log);
mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log);
mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                      mlir::RewriterBase& rewriter, Logger log);
mlir::Operation* getParentComputeOp(mlir::Operation* op);
bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log);
bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log);
bool hasMultiBranches(mlir::Operation* op);

bool archSupportsSwLayerTiling(config::ArchKind arch);
bool doesNCEOpChannelSatisfyWorkload(mlir::Operation* nceOp, const TileInfo& outputTile);
std::optional<DimArr> getSEPConvTilingOrder(mlir::Operation* op);
std::optional<std::pair<size_t, size_t>> getWorkLoadInformationForNCEWithSparseOutput(
        config::ArchKind arch, ArrayRef<Shape> perClusterShapes, ArrayRef<int64_t> supportedChannels);

/**
 * @brief Get the best hardware layer tiling strategy based on the VPUNN cost model
 * @details
 * This function is used to calculate optimal tiling strategy using VPUNN DMA+DPU time cost :
 * Instead of relying on a predetermined tiling dimension order and the corresponding tiling strategy calculated
 * based on that dimension order, we evaluate all possible tiling orders, and their corresponding strategies for a
 * layer and select the one that minimizes the VPUNN cost.
 * 1. For each operation, first identify the supported tiling mode (Isolated, Pipeline, and Prefetch)
 * 2. If the operation has the necessary interface and shape (4-D), retrieve all possible tiling strategies
 *    for the layer under current multi-cluster strategy and calculate the DMA+DPU costs by using VPUNN cost model.
 * 3. For each tiling strategy candidate, if the calculated cost is valid and less than the current best cost
 *    (initialized to maximum), update it as the best tiling strategy and its cost as the best cost.
 *    If the calculated cost is invalid, skip that tiling strategy.
 * 4. Finally, return the best tiling strategy for the layer which has the least cost.
 * @param op The target operation to tile
 * @param costModel The shared pointer to the LayerCostModel class
 * @param enablePrefetchTiling If this option is enabled, PREFETCH tiling mode is selected, default mode is ISOLATED.
 * @return The best output tiling strategy or a failure
 */

mlir::FailureOr<OutputTiling> getHWLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                       const std::shared_ptr<LayerCostModel>& costModel, Logger log);

mlir::FailureOr<OutputTiling> getBestHWLayerTilingStrategy(mlir::Operation* op, TilingMode tilingMode,
                                                           const std::shared_ptr<LayerCostModel>& costModel,
                                                           bool enablePrefetchTiling, Logger log);

struct StrategyWithCost {
    Shape strategy;
    HwLayerTilingStrategyCosts cost;
};

std::vector<StrategyWithCost> getHwLayerTilingStrategiesWithCost(mlir::Operation* op, TilingMode tilingMode,
                                                                 const std::shared_ptr<LayerCostModel>& costModel,
                                                                 Logger log);

enum class EnableShaveDDRAccessOptimization { TRUE, FALSE, AUTO };

EnableShaveDDRAccessOptimization getShaveDDRAccessOptimizationMode(StringRef enableShaveDDRAccessOptimization);

bool isMultiClusterTilingSupported(mlir::Operation* op);
bool isTilingSupported(mlir::Operation* op);

}  // namespace VPU
}  // namespace vpux
