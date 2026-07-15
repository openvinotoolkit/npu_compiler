//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/cost_model/layer_vpunn_cost.hpp"
#include "vpux/compiler/dialect/VPU/utils/multi_cluster_strategy_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
class NCEOpInterface;
class TilingBuilderOpInterface;
}  // namespace vpux::VPU

namespace vpux {
namespace VPU {

constexpr float INPUT_OVERLAP_THRESHOLD = 2.0;

// Experimental number to compare isolate and pipelining cost to increase tolerance.
// Other limitations are also used in choosing the pipelining tiling mode.
constexpr float PIPELINING_AVAILABLE_RATIO = 0.95f;

// Returns a TilingMode which should be applied to operation and whether it can be prefetched. Caller is
// supposed to check if prefetching condition is satisfied as this function aims to be thread safe
// and avoids inspecting parent's strategy.
std::optional<std::pair<TilingMode, bool>> getTilingMode(mlir::Operation* op, bool enablePrefetchTiling,
                                                         const std::unique_ptr<VPU::LayerVPUNNCost>& layerCost,
                                                         Logger log);

mlir::FailureOr<OutputTiling> getLayerTilingStrategy(VPU::TilingBuilderOpInterface origOp, bool enablePrefetchTiling,
                                                     TilingMode& mode, Logger log);

mlir::LogicalResult checkAndAlignActInputTiling(vpux::VPU::NCEOpInterface nceOp, InputTiling& inputTiling,
                                                vpux::Logger log);
mlir::Value reifyTile(VPU::TilingBuilderOpInterface origOp, const TileInfo& outputTile, mlir::OpBuilder& builder,
                      Logger log);
mlir::LogicalResult applyTileStrategy(VPU::TilingBuilderOpInterface origOp, const OutputTiling& tiles,
                                      mlir::RewriterBase& rewriter, Logger log);
mlir::Operation* getParentComputeOp(mlir::Operation* op);
bool prefetchTilingConditionSatisfied(mlir::Operation* op, Logger log);
bool pipeliningTilingOfSWConditionSatisfied(mlir::Operation* op, const std::unique_ptr<VPU::LayerVPUNNCost>& layerCost,
                                            Logger log);
bool largeConstPipelineConditionSatisfied(mlir::Operation* op, Logger log);
bool hasMultiBranches(mlir::Operation* op);

bool archSupportsSwLayerTiling(config::ArchKind arch);
bool doesNCEOpChannelSatisfyWorkload(mlir::Operation* nceOp, const TileInfo& outputTile);
std::optional<DimArr> getSEPConvTilingOrder(mlir::Operation* op);
std::optional<std::pair<size_t, size_t>> getWorkLoadInformationForNCEWithSparseOutput(
        mlir::Operation* nceOp, ArrayRef<Shape> perClusterShapes, ArrayRef<int64_t> supportedChannels);

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

// Returns a WeightsTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getWeightsTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                             std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origWeightsTable = origOp->getWeightsTable();
    VPUX_THROW_UNLESS(origWeightsTable != nullptr, "The operation {0} doesn't have a WeightsTable", *origOp);

    const auto origWeightsTableShape = getShape(origWeightsTable);
    VPUX_THROW_UNLESS((weightsOutputChannels.has_value() ||
                       origWeightsTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                              origWeightsTableShape[Dim(1)] == 1 && origWeightsTableShape[Dim(2)] == 1 &&
                              origWeightsTableShape[Dim(3)] == VPU::NCEInvariant::WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected WeightsTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origWeightsTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the WeightsTable corresponds to its own output channel
    TileInfo weightsTableTile(origWeightsTableShape);
    weightsTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    weightsTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return weightsTableTile;
}

// Returns a DataPointerTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getDataPointerTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                                 std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origDataPointerTable = origOp->getWeightTableDataPtr();
    VPUX_THROW_UNLESS(origDataPointerTable != nullptr, "The operation {0} doesn't have a DataPointerTable", *origOp);

    const auto origDataPointerTableShape = getShape(origDataPointerTable);
    VPUX_THROW_UNLESS(
            (weightsOutputChannels.has_value() ||
             origDataPointerTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                    origDataPointerTableShape[Dim(1)] == 1 && origDataPointerTableShape[Dim(2)] == 1 &&
                    origDataPointerTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
            "Unexpected DataPointerTable shape notation or order: {0} with output shape of {1}"
            "\nProbably, we need to update this logic",
            origDataPointerTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the WeightsTable corresponds to its own output channel
    TileInfo dataPointerTableTile(origDataPointerTableShape);
    dataPointerTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    dataPointerTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return dataPointerTableTile;
}

// Returns a ScaleTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getScaleTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                           std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origScaleTable = origOp->getWeightTableScale();
    VPUX_THROW_UNLESS(origScaleTable != nullptr, "The operation {0} doesn't have a ScaleTable", *origOp);

    const auto origScaleTableShape = getShape(origScaleTable);
    VPUX_THROW_UNLESS((weightsOutputChannels.has_value() ||
                       origScaleTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                              origScaleTableShape[Dim(1)] == 1 && origScaleTableShape[Dim(2)] == 1 &&
                              origScaleTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected ScaleTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origScaleTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the ScaleTable corresponds to its own output channel
    TileInfo scaleTableTile(origScaleTableShape);
    scaleTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    scaleTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return scaleTableTile;
}

// Returns a BiasTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getBiasTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                          std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origBiasTable = origOp->getWeightTableBias();
    VPUX_THROW_UNLESS(origBiasTable != nullptr, "The operation {0} doesn't have a BiasTable", *origOp);

    const auto origBiasTableShape = getShape(origBiasTable);
    VPUX_THROW_UNLESS((weightsOutputChannels.has_value() ||
                       origBiasTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                              origBiasTableShape[Dim(1)] == 1 && origBiasTableShape[Dim(2)] == 1 &&
                              origBiasTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
                      "Unexpected BiasTable shape notation or order: {0} with output shape of {1}"
                      "\nProbably, we need to update this logic",
                      origBiasTableShape, getShape(origOp->getOutput()));

    // Each N-wise batch of the BiasTable corresponds to its own output channel
    TileInfo biasTableTile(origBiasTableShape);
    biasTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    biasTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return biasTableTile;
}

// Returns a ZeroPointTable tile required to produce the specific output tile
template <typename ConcreteOp>
TileInfo getZeroPointTableTile(ConcreteOp* origOp, const vpux::TileInfo& outputTile,
                               std::optional<int64_t> weightsOutputChannels = std::nullopt) {
    const auto origZeroPointTable = origOp->getWeightZeroPoints();
    VPUX_THROW_UNLESS(origZeroPointTable != nullptr, "The operation {0} doesn't have a ZeroPointTable", *origOp);

    const auto origZeroPointTableShape = getShape(origZeroPointTable);
    VPUX_THROW_UNLESS(
            (weightsOutputChannels.has_value() ||
             origZeroPointTableShape[Dim(0)] == getShape(origOp->getOutput())[Dims4D::Act::C]) &&
                    origZeroPointTableShape[Dim(1)] == 1 && origZeroPointTableShape[Dim(2)] == 1 &&
                    origZeroPointTableShape[Dim(3)] == VPU::NCEInvariant::NEW_WEIGHT_TABLE_NUM_ELEMENTS_PER_OC,
            "Unexpected ZeroPointTable shape notation or order: {0} with output shape of {1}"
            "\nProbably, we need to update this logic",
            origZeroPointTableShape, getShape(origOp->getOutput()));

    TileInfo zeroPointTableTile(origZeroPointTableShape);
    zeroPointTableTile.offsets[Dim(0)] = outputTile.offsets[Dims4D::Act::C];
    zeroPointTableTile.shape[Dim(0)] =
            weightsOutputChannels.has_value() ? weightsOutputChannels.value() : outputTile.shape[Dims4D::Act::C];
    return zeroPointTableTile;
}

// Adjust paddings attributes for tiled input
template <typename ConcreteOp>
void adjustPaddings(ConcreteOp* op, const TilingInfo& inputTiling) {
    VPUX_THROW_UNLESS(inputTiling.pads.has_value(), "Missing tile information for paddings");

    auto newPadAttr = getPaddingAttr(op->getContext(), inputTiling.pads.value());

    op->setPadAttr(newPadAttr);
}

// Adjust rawFilterShape attribute for specific output tile
template <typename ConcreteOp>
void adjustRawFilterShape(ConcreteOp* op, const TileInfo& outputTile) {
    auto mixedRawFilterShape = op->getMixedRawFilterShape();
    mixedRawFilterShape[Dims4D::Filter::OC.ind()] =
            mlir::getAsIndexOpFoldResult(op->getContext(), outputTile.shape[Dims4D::Act::C]);

    SmallVector<int64_t> staticValues;
    SmallVector<mlir::Value> dynamicValues;
    mlir::dispatchIndexOpFoldResults(mixedRawFilterShape, dynamicValues, staticValues);

    op->setStaticRawFilterShape(staticValues);
    op->getRawFilterShapeMutable().assign(dynamicValues);
}

}  // namespace VPU
}  // namespace vpux
