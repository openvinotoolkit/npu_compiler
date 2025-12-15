//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Support/LogicalResult.h>

namespace vpux::VPU {

OutputTiling DetectionOutputSortOpOutputTiling(const vpux::TileInfo& firstOutputTile) {
    // Output 0 top_k_confidence    [ 1, 1, numClasses, numBoxes ]
    // Output 1 indices             [ 1, 1, numClasses, numPriors ]
    // Output 2 sizes               [ 1, 1, 1, numClasses ]
    const auto shapeClasses = firstOutputTile.shape[Dims4D::Act::H];
    const auto offsetClasses = firstOutputTile.offsets[Dims4D::Act::H];
    const auto axisClasses = firstOutputTile.axis[Dims4D::Act::H];

    const auto numPriors = firstOutputTile.shape[Dims4D::Act::W];

    const auto indicesShapeSize = 4;
    auto indicesTile = TileInfo(indicesShapeSize);
    indicesTile.shape = Shape{1, 1, shapeClasses, numPriors};
    indicesTile.offsets = Shape{0, 0, offsetClasses, 0};
    indicesTile.axis = Shape{1, 1, axisClasses, 1};

    const auto sizesShapeSize = 4;
    auto sizesTile = TileInfo(sizesShapeSize);
    sizesTile.shape = Shape{1, 1, shapeClasses, 1};
    sizesTile.offsets = Shape{0, 0, offsetClasses, 0};
    sizesTile.axis = Shape{1, 1, axisClasses, 1};

    return OutputTiling{firstOutputTile, std::move(indicesTile), std::move(sizesTile)};
}

InputTiling DetectionOutputSortOpInputTiling(const vpux::TileInfo& firstOutputTile, int numShaves) {
    const auto outputShape = firstOutputTile.shape;
    VPUX_THROW_UNLESS(outputShape.size() == 4, "Expected 4D output shape to be tiled");

    const auto classesDims = outputShape[Dims4D::Act::H];
    const auto classesOffsets = firstOutputTile.offsets[Dims4D::Act::H];
    const auto classesAxis = firstOutputTile.axis[Dims4D::Act::H];

    const auto numPriors = firstOutputTile.shape[Dims4D::Act::W];

    const auto inputRank = 4;
    auto confidenceTile = TileInfo(inputRank);
    confidenceTile.shape = Shape{1, 1, classesDims, numPriors};
    confidenceTile.offsets = Shape{0, 0, classesOffsets, 0};
    confidenceTile.axis = Shape{1, 1, classesAxis, 1};

    auto indicesBufferTile = confidenceTile;

    const auto sortingBufferRank = 4;
    auto sortingBufferTile = TileInfo(sortingBufferRank);
    // 4 buffers of size 256 elements for counting sort
    sortingBufferTile.shape = Shape{1, 1, 4 * numShaves, 256};
    sortingBufferTile.offsets = Shape{0, 0, 0, 0};

    return InputTiling{{std::move(confidenceTile), std::move(indicesBufferTile), std::move(sortingBufferTile)}};
}

InputTiling DetectionOutputSortOpInputTilingOnShave(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& firstOutputTile,
                                                    int tileId, int tileCount, Logger /*log*/) {
    auto module = swKernelOp.getOperation()->getParentOfType<mlir::ModuleOp>();
    auto numClusters = config::getTileExecutor(module).getCount();
    auto numTotalShaves = config::getTotalNumOfEngines(module, VPU::ExecutorKind::SHAVE_ACT);

    VPUX_THROW_WHEN(numClusters <= 0, "Unsupported number of clusters: {0}", numClusters);

    auto numShavesOnCluster = numTotalShaves / numClusters;

    auto inputsTiling = DetectionOutputSortOpInputTiling(firstOutputTile, numTotalShaves);

    // This is a workaround for a third input that is used as an auxiliary buffer for the sorting algorithm
    // The kernel requires [1, 1, 4, 256] buffer where it will store intermediate values
    // To achieve that the DetectionOutputSort::build operation creates [1, 1, 4 * 4, 256] buffer
    // After isolated tiling we always have enough buffer memory to divide among 4 shaves
    // TileActShaveKernelTask pass will call this function when it tries to tile onto clusters and shaves
    // When tiling onto clusters, we have two halves with shape [1, 1, 8, 256]
    // When tiling onto shaves, the shape has the required for the kernel shape [1, 1, 4, 256]

    if (tileCount == numClusters) {
        inputsTiling.tiles[2].shape = {1, 1, numShavesOnCluster * 4, 256};
        inputsTiling.tiles[2].offsets = {0, 0, tileId * numShavesOnCluster * 4, 0};
    } else {
        inputsTiling.tiles[2].shape = {1, 1, 4, 256};
        inputsTiling.tiles[2].offsets = {0, 0, tileId * 4, 0};
    }

    return inputsTiling;
}

OutputTiling GRUSequenceOutputTiling(const vpux::TileInfo& firstOutputTile) {
    const auto extractNCW = [](const Shape& values) {
        return Shape{values[Dims4D::Act::N], values[Dims4D::Act::C], values[Dims4D::Act::W]};
    };

    auto outStateShape = extractNCW(firstOutputTile.shape);
    auto outStateOffsets = extractNCW(firstOutputTile.offsets);
    auto outStateAxis = extractNCW(firstOutputTile.axis);
    auto stateOutputTile = vpux::TileInfo(outStateShape, outStateOffsets, outStateAxis);

    return {firstOutputTile, std::move(stateOutputTile)};
}

OutputTiling DynamicQuantizeOutputTiling(const vpux::TileInfo& firstOutputTile) {
    const auto shapeSize = firstOutputTile.shape.size();
    const auto oneShape = Shape(shapeSize, 1);
    const auto zeroShape = Shape(shapeSize, 0);
    const auto scaleTile = TileInfo(oneShape, zeroShape, oneShape);
    const auto zpTile = TileInfo(oneShape, zeroShape, oneShape);

    return {firstOutputTile, std::move(scaleTile), std::move(zpTile)};
}

OutputTiling lstmSequenceOutputTiling(const vpux::TileInfo& firstOutputTile) {
    const auto firstOutputTileShape = firstOutputTile.shape;
    const auto batchSize = firstOutputTileShape[Dims4D::Act::N];
    const auto numDirections = firstOutputTileShape[Dims4D::Act::C];
    const auto hiddenSize = firstOutputTileShape[Dims4D::Act::W];
    const auto secondShape = Shape{batchSize, numDirections, 1, hiddenSize};

    // For the LSTMSequence kernel, each output tile should have the same shape and zero offsets. The tiling
    // infrastructure, specifically the 'divideTiles' function, will accumulate the offsets after each tile, which
    // we will reset here.
    TileInfo newFirstOutputTile(firstOutputTile.shape);

    TileInfo secondTile(secondShape);
    TileInfo thirdTile(secondShape);
    return {std::move(newFirstOutputTile), std::move(secondTile), std::move(thirdTile)};
}

OutputTiling lstmDpuOutputTiling(const vpux::TileInfo& firstOutputTile) {
    const auto extractNCW = [](const Shape& values) {
        return Shape{values[Dims4D::Act::N], values[Dims4D::Act::C], 1, values[Dims4D::Act::W]};
    };
    const auto extractNCWOffset = [](const Shape& values) {
        return Shape{values[Dims4D::Act::N], values[Dims4D::Act::C], 0, values[Dims4D::Act::W]};
    };
    auto outStateShape = extractNCW(firstOutputTile.shape);
    auto outStateOffsets = extractNCWOffset(firstOutputTile.offsets);
    auto outStateAxis = extractNCW(firstOutputTile.axis);
    auto secondTile = vpux::TileInfo(outStateShape, outStateOffsets, outStateAxis);
    auto thirdTile = vpux::TileInfo(outStateShape, outStateOffsets, outStateAxis);
    return {firstOutputTile, std::move(secondTile), std::move(thirdTile)};
}

OutputTiling FlashSDPAOpOutputTiling(const vpux::TileInfo& firstOutputTile, int64_t qkEmbedding) {
    auto maxAndSumTile = TileInfo(firstOutputTile);

    maxAndSumTile.shape[Dims4D::Act::C] = firstOutputTile.shape[Dims4D::Act::C];
    maxAndSumTile.offsets[Dims4D::Act::C] = firstOutputTile.offsets[Dims4D::Act::C];
    maxAndSumTile.axis[Dims4D::Act::C] = firstOutputTile.axis[Dims4D::Act::C];

    maxAndSumTile.shape[Dims4D::Act::H] = firstOutputTile.shape[Dims4D::Act::H];
    maxAndSumTile.offsets[Dims4D::Act::H] = firstOutputTile.offsets[Dims4D::Act::H];
    maxAndSumTile.axis[Dims4D::Act::H] = firstOutputTile.axis[Dims4D::Act::H];

    // Max and Sum outputs have reduced shape, width == 1
    maxAndSumTile.shape[Dims4D::Act::W] = 1;
    maxAndSumTile.offsets[Dims4D::Act::W] = 0;
    maxAndSumTile.axis[Dims4D::Act::W] = 0;

    auto query = TileInfo(firstOutputTile);
    query.shape[Dims4D::Act::W] = qkEmbedding;
    query.offsets[Dims4D::Act::W] = 0;
    query.axis[Dims4D::Act::W] = 0;

    return OutputTiling{firstOutputTile, maxAndSumTile, maxAndSumTile, std::move(query)};
}

InputTiling FlashSDPAOpInputTiling(const vpux::TileInfo& firstOutputTile, ShapeRef keyShape,
                                   std::optional<ShapeRef> attentionMaskShape, std::optional<ShapeRef> scaleShape,
                                   ShapeRef dpuDescriptorBufferShape, ShapeRef weightsTable0Shape,
                                   ShapeRef weightsTable1Shape) {
    const auto targetSeqLen = firstOutputTile.shape[Dims4D::Act::H];
    const auto vEmbedding = firstOutputTile.shape[Dims4D::Act::W];
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];
    const auto qkEmbedding = keyShape[Dims4D::Act::W];
    const auto heads = keyShape[Dims4D::Act::C];

    const auto queryShape = Shape{1, heads, targetSeqLen, qkEmbedding};
    const auto valueShape = Shape{1, heads, vEmbedding, sourceSeqLen};
    const auto auxBufferShape = Shape{1, heads, targetSeqLen, sourceSeqLen};
    const auto runningOutShape = Shape{1, heads, targetSeqLen, vEmbedding};
    const auto runningMaxShape = Shape{1, heads, targetSeqLen, 1};
    const auto& runningSumShape = runningMaxShape;

    auto syncTilesDim = [](const auto& tensorFrom, auto dimFrom, auto& tensorTo, auto dimTo) {
        tensorTo.shape[dimTo] = tensorFrom.shape[dimFrom];
        tensorTo.offsets[dimTo] = tensorFrom.offsets[dimFrom];
        tensorTo.axis[dimTo] = tensorFrom.axis[dimFrom];
    };

    auto queryTile = TileInfo(queryShape);
    auto keyTile = TileInfo(keyShape);
    auto valueTile = TileInfo(valueShape);

    auto auxBufferTile = TileInfo(auxBufferShape);

    auto dpuDescriptorBufferTile = TileInfo(dpuDescriptorBufferShape);
    auto weightsTable0Tile = TileInfo(weightsTable0Shape);
    auto weightsTable1Tile = TileInfo(weightsTable1Shape);

    auto runningOutTile = TileInfo(runningOutShape);
    auto runningMaxTile = TileInfo(runningMaxShape);
    auto runningSumTile = TileInfo(runningSumShape);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, queryTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, queryTile, Dims4D::Act::H);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, keyTile, Dims4D::Act::C);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, valueTile, Dims4D::Act::C);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, auxBufferTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, auxBufferTile, Dims4D::Act::H);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, runningOutTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, runningOutTile, Dims4D::Act::H);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, runningMaxTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, runningMaxTile, Dims4D::Act::H);

    syncTilesDim(firstOutputTile, Dims4D::Act::C, runningSumTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, runningSumTile, Dims4D::Act::H);

    auto inputsTiles = SmallVector<TileInfo>{std::move(queryTile),
                                             std::move(keyTile),
                                             std::move(valueTile),
                                             std::move(auxBufferTile),
                                             std::move(dpuDescriptorBufferTile),
                                             std::move(weightsTable0Tile),
                                             std::move(weightsTable1Tile),
                                             std::move(runningOutTile),
                                             std::move(runningMaxTile),
                                             std::move(runningSumTile)};

    if (attentionMaskShape.has_value()) {
        auto attentionMaskTile = TileInfo(attentionMaskShape.value());

        // Avoid updating the batch size in case of a broadcasted attention mask batch dimension
        if (attentionMaskTile.shape[Dims4D::Act::C] > 1) {
            syncTilesDim(firstOutputTile, Dims4D::Act::C, attentionMaskTile, Dims4D::Act::C);
        }
        syncTilesDim(firstOutputTile, Dims4D::Act::H, attentionMaskTile, Dims4D::Act::H);

        inputsTiles.push_back(attentionMaskTile);
    }

    // Scale is just one value and couldn't be tiled.
    if (scaleShape.has_value()) {
        inputsTiles.push_back(TileInfo(scaleShape.value()));
    }

    return InputTiling{inputsTiles};
}

}  // namespace vpux::VPU
