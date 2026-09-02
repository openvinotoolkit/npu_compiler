//
// Copyright (C) 2024-2026 Intel Corporation
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
    auto numTotalShaves = config::getTotalNumOfEngines(module, config::ExecutorKind::SHAVE_ACT);

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

OutputTiling logSoftmaxTopKOutputTiling(const vpux::TileInfo& firstOutputTile, int64_t axis) {
    auto secondShape = firstOutputTile.shape;
    auto secondOffsets = firstOutputTile.offsets;
    auto secondAxis = firstOutputTile.axis;

    // The innermost dimension will always be 1 for the second output (due to fusing TopK with K=1)
    const auto axisDim = Dim(axis);
    secondShape[axisDim] = 1;
    secondOffsets[axisDim] = 0;
    secondAxis[axisDim] = 1;

    auto secondOutputTile = vpux::TileInfo(secondShape, secondOffsets, secondAxis);

    return {firstOutputTile, std::move(secondOutputTile)};
}

OutputTiling DynamicQuantizeOutputTiling(const vpux::TileInfo& firstOutputTile, ShapeRef scaleShape, ShapeRef zpShape) {
    // Scale and zero-point share the data tensor's layout on the non-quantized axes and carry the
    // quantization granularity on the quantized axis: extent 1 for per-token / per-tensor scales, or
    // the group count for per-group scales. Their tiles must follow firstOutputTile only on the axes
    // that are actually tiled and shared with the parameter tensor (extent > 1); every other axis,
    // including the quantized axis, keeps its full extent because it is never tiled. Using the full
    // scale/zp shape on all axes leaves these outputs untiled and breaks the tiled-shape check in
    // applyTileStrategy, while blindly copying firstOutputTile corrupts the quantized axis once its
    // extent exceeds 1 (per-group case).
    const auto deriveParamTile = [&firstOutputTile](ShapeRef paramShape) {
        auto paramTile = TileInfo(paramShape);
        for (size_t ind = 0; ind < paramShape.size(); ++ind) {
            const auto d = Dim(ind);
            if (firstOutputTile.axis[d] > 1 && paramShape[d] > 1) {
                paramTile.shape[d] = firstOutputTile.shape[d];
                paramTile.offsets[d] = firstOutputTile.offsets[d];
                paramTile.axis[d] = firstOutputTile.axis[d];
            }
        }
        return paramTile;
    };

    auto scaleTile = deriveParamTile(scaleShape);
    auto zpTile = deriveParamTile(zpShape);

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

OutputTiling FlashSDPAOpOutputTiling(const vpux::TileInfo& firstOutputTile) {
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
    maxAndSumTile.axis[Dims4D::Act::W] = 1;

    return OutputTiling{firstOutputTile, maxAndSumTile, maxAndSumTile};
}

InputTiling FlashSDPAOpInputTiling(const vpux::TileInfo& firstOutputTile, int64_t qHeads, ShapeRef keyShape,
                                   std::optional<ShapeRef> attentionMaskShape, ShapeRef auxBufferShape,
                                   ShapeRef dpuDescriptorBufferShape, ShapeRef weightsTable0Shape,
                                   ShapeRef weightsTable1Shape) {
    const auto targetSeqLen = firstOutputTile.shape[Dims4D::Act::H];
    const auto vEmbedding = firstOutputTile.shape[Dims4D::Act::W];
    const auto sourceSeqLen = keyShape[Dims4D::Act::H];
    const auto qkEmbedding = keyShape[Dims4D::Act::W];
    const auto kvHeads = keyShape[Dims4D::Act::C];

    VPUX_THROW_UNLESS(qHeads > 0, "FlashSDPAOpInputTiling expects positive Q head count, got {0}", qHeads);
    VPUX_THROW_UNLESS(kvHeads > 0, "FlashSDPAOpInputTiling expects positive KV head count, got {0}", kvHeads);
    VPUX_THROW_UNLESS(qHeads % kvHeads == 0,
                      "FlashSDPAOpInputTiling expects Q head count {0} to be divisible by KV head count {1}", qHeads,
                      kvHeads);

    // GQA: Q may have more heads than K/V. The output tile's C dimension
    // corresponds to Q heads. For K/V, we need to compute the corresponding
    // K/V head tile based on the group ratio.
    const auto gqaGroupSize = qHeads / kvHeads;  // Q heads per K/V head
    const auto outputTileCOffset = firstOutputTile.offsets[Dims4D::Act::C];
    const auto outputTileCSize = firstOutputTile.shape[Dims4D::Act::C];

    // A Q head tile maps onto the K/V heads via the integer division below, so it must not straddle a
    // group boundary. Two shapes satisfy that: a whole number of groups (the tile then covers
    // 'size / gqaGroupSize' K/V heads), or a subset of a single group (the tile then covers that one
    // K/V head). Only a tile that partially overlaps two groups is rejected, since it would silently
    // drop the heads belonging to the second one.
    const auto spansWholeGroups = (outputTileCOffset % gqaGroupSize == 0) && (outputTileCSize % gqaGroupSize == 0);
    const auto isWithinOneGroup =
            (outputTileCOffset / gqaGroupSize) == ((outputTileCOffset + outputTileCSize - 1) / gqaGroupSize);
    VPUX_THROW_UNLESS(spansWholeGroups || isWithinOneGroup,
                      "FlashSDPAOpInputTiling expects the output tile C range [{0}, {1}) to either span whole GQA "
                      "groups or stay within a single group of size {2}",
                      outputTileCOffset, outputTileCOffset + outputTileCSize, gqaGroupSize);

    const auto queryShape = Shape{1, qHeads, targetSeqLen, qkEmbedding};
    const auto valueShape = Shape{1, kvHeads, sourceSeqLen, vEmbedding};
    const auto runningOutShape = Shape{1, qHeads, targetSeqLen, vEmbedding};
    const auto runningMaxShape = Shape{1, qHeads, targetSeqLen, 1};
    const auto& runningSumShape = runningMaxShape;

    auto syncTilesDim = [](const auto& tensorFrom, auto dimFrom, auto& tensorTo, auto dimTo) {
        tensorTo.shape[dimTo] = tensorFrom.shape[dimFrom];
        tensorTo.offsets[dimTo] = tensorFrom.offsets[dimFrom];
        tensorTo.axis[dimTo] = tensorFrom.axis[dimFrom];
    };

    auto queryTile = TileInfo(queryShape);
    syncTilesDim(firstOutputTile, Dims4D::Act::C, queryTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, queryTile, Dims4D::Act::H);

    auto keyTile = TileInfo(keyShape);
    auto valueTile = TileInfo(valueShape);

    // For GQA: map Q head tile to corresponding K/V head tile
    // E.g., if Q has 32 heads and K/V has 8 heads (group_size=4),
    // and the output tile has C shape=4 offset=8, then K/V tile has C shape=1 offset=2
    if (gqaGroupSize > 1) {
        const auto tileQHeads = firstOutputTile.shape[Dims4D::Act::C];
        const auto tileQOffset = firstOutputTile.offsets[Dims4D::Act::C];

        keyTile.shape[Dims4D::Act::C] = std::max(int64_t{1}, tileQHeads / gqaGroupSize);
        keyTile.offsets[Dims4D::Act::C] = tileQOffset / gqaGroupSize;
        keyTile.axis[Dims4D::Act::C] = firstOutputTile.axis[Dims4D::Act::C];

        valueTile.shape[Dims4D::Act::C] = std::max(int64_t{1}, tileQHeads / gqaGroupSize);
        valueTile.offsets[Dims4D::Act::C] = tileQOffset / gqaGroupSize;
        valueTile.axis[Dims4D::Act::C] = firstOutputTile.axis[Dims4D::Act::C];
    } else {
        syncTilesDim(firstOutputTile, Dims4D::Act::C, keyTile, Dims4D::Act::C);
        syncTilesDim(firstOutputTile, Dims4D::Act::C, valueTile, Dims4D::Act::C);
    }

    auto auxBufferTile = TileInfo(auxBufferShape);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, auxBufferTile, Dims4D::Act::H);

    auto dpuDescriptorBufferTile = TileInfo(dpuDescriptorBufferShape);
    auto weightsTable0Tile = TileInfo(weightsTable0Shape);
    auto weightsTable1Tile = TileInfo(weightsTable1Shape);

    auto runningOutTile = TileInfo(runningOutShape);
    syncTilesDim(firstOutputTile, Dims4D::Act::C, runningOutTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, runningOutTile, Dims4D::Act::H);

    auto runningMaxTile = TileInfo(runningMaxShape);
    syncTilesDim(firstOutputTile, Dims4D::Act::C, runningMaxTile, Dims4D::Act::C);
    syncTilesDim(firstOutputTile, Dims4D::Act::H, runningMaxTile, Dims4D::Act::H);

    auto runningSumTile = TileInfo(runningSumShape);
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

    return InputTiling{inputsTiles};
}

}  // namespace vpux::VPU
