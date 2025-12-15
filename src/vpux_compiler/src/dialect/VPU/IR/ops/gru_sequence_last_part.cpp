//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceLastPartOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceLastPartOpAdaptor gru(operands, attrs, prop);
    if (mlir::failed(gru.verify(loc))) {
        return mlir::failure();
    }

    const auto initialStateType = mlir::cast<vpux::NDTypeInterface>(gru.getInitialHiddenState().getType());
    const auto outputStateType = initialStateType;
    const auto outputStateShape = outputStateType.getShape().raw();
    const auto seqLength = gru.getSeqLength();
    SmallVector<int64_t> middleStateShape = {outputStateShape[0], outputStateShape[1], seqLength, outputStateShape[2]};
    const auto middleStateType = initialStateType.changeShape(ShapeRef(middleStateShape));

    inferredReturnShapes.push_back(middleStateType);
    inferredReturnShapes.push_back(outputStateType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GRUSequenceLastPartOp::backInferTileInfo(const vpux::TileInfo& outputTileY, vpux::Logger) {
    const auto origInputShape = getShape(getFirstPartOutput());
    const auto origInitialHiddenStateShape = getShape(getInitialHiddenState());
    const auto origRShape = getShape(getRecurrenceWeights());
    const auto origBShape = getShape(getBiases());

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    const auto batchSizeDim = 0;
    const auto numDirectionDim = 1;
    const auto seqLengthDim = 2;

    inputTile.shape[Dim(batchSizeDim)] = outputTileY.shape[Dim(batchSizeDim)];
    inputTile.offsets[Dim(batchSizeDim)] = outputTileY.offsets[Dim(batchSizeDim)];
    inputTile.shape[Dim(seqLengthDim)] = outputTileY.shape[Dim(seqLengthDim)];
    inputTile.offsets[Dim(seqLengthDim)] = outputTileY.offsets[Dim(seqLengthDim)];

    initialHiddenStateTile.shape[Dim(batchSizeDim)] = outputTileY.shape[Dim(batchSizeDim)];
    initialHiddenStateTile.offsets[Dim(batchSizeDim)] = outputTileY.offsets[Dim(batchSizeDim)];
    initialHiddenStateTile.shape[Dim(numDirectionDim)] = outputTileY.shape[Dim(numDirectionDim)];
    initialHiddenStateTile.offsets[Dim(numDirectionDim)] = outputTileY.offsets[Dim(numDirectionDim)];

    rTile.shape[Dim(batchSizeDim)] = outputTileY.shape[Dim(numDirectionDim)];
    rTile.offsets[Dim(batchSizeDim)] = outputTileY.offsets[Dim(numDirectionDim)];

    bTile.shape[Dim(batchSizeDim)] = outputTileY.shape[Dim(numDirectionDim)];
    bTile.offsets[Dim(batchSizeDim)] = outputTileY.offsets[Dim(numDirectionDim)];

    return InputTiling{{std::move(inputTile), std::move(initialHiddenStateTile), std::move(rTile), std::move(bTile)}};
}

void vpux::VPU::GRUSequenceLastPartOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputYTile) {
    auto* ctx = this->getContext();
    auto inputTileInfo = inputTiling.tiles[0];
    VPUX_THROW_UNLESS(inputTileInfo.shape[Dim(2)] == outputYTile.shape[Dim(2)],
                      "seq_length dimension in input tile is incompatible with output tile, seq_length dimension of "
                      "input tile is {0}, but it's {1} in output tile",
                      inputTileInfo.shape[Dim(1)], outputYTile.shape[Dim(2)]);
    auto origSeqLength = mlir::dyn_cast_or_null<mlir::IntegerAttr>(getSeqLengthAttr()).getValue().getSExtValue();
    auto tiledSeqLength = inputTileInfo.shape[Dim(2)];
    if (origSeqLength != tiledSeqLength) {
        const auto newSeqLength = getIntAttr(ctx, tiledSeqLength);
        this->setSeqLengthAttr(newSeqLength);
    }
    /**/
    // The num_direction dimension of output can be tiled when direction is BIDIRECTONAL.
    // GRUSequence with BIDIRECTIONAL attribute will be split into two GRUSequence, GRUSequence with FORWARD attribute
    // and GRUSequence with REVERSE attribute when num_direction dimension was tiled.
    auto origDirection = getDirectionAttr().getValue();
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 0) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::FORWARD);
        this->setDirectionAttr(newDirectionAttr);
    }
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL && outputYTile.shape[Dim(1)] == 1 &&
        outputYTile.offsets[Dim(1)] == 1) {
        const auto newDirectionAttr = IE::RNNSequenceDirectionAttr::get(ctx, IE::RNNSequenceDirection::REVERSE);
        this->setDirectionAttr(newDirectionAttr);
    }
}

mlir::FailureOr<OutputTiling> vpux::VPU::GRUSequenceLastPartOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GRUSequence currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    // There are two outputs of GRUSequenceLastPart, the names Y and Ho are from OpenVINO doc.
    // The shape of Y is [batch_size, num_directions, seq_len, hidden_size],
    // and the shape of Ho is [batch_size, num_directions, hidden_size].
    // Ho-tiles can be inferred by Y-tiles.
    const auto outputYType = mlir::cast<vpux::NDTypeInterface>(baseOp->getResult(0).getType());
    const auto outputYShape = outputYType.getShape();
    Shape nTilesOnDimForOutputY(outputYShape.size(), 1);

    const auto isSupportedTileSize = [baseOp, &tilingInfo, outputYShape, log](ShapeRef nTilesOnDim,
                                                                              TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(baseOp, nTilesOnDim, outputYShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    int64_t tileDim = 0;
    while (!isSupportedTileSize(nTilesOnDimForOutputY, tilingMode)) {
        // Dimension of hidden_size can't be tiled.
        if (tileDim >= 3) {
            log.trace("can't get feasible tiling strategy for GRUSequenceLastPart.");
            return mlir::failure();
        }
        if (nTilesOnDimForOutputY[Dim(tileDim)] >= outputYShape[Dim(tileDim)]) {
            ++tileDim;
        } else {
            ++nTilesOnDimForOutputY[Dim(tileDim)];
        }
    }
    log.trace("Isolated tiling strategy: {0}", nTilesOnDimForOutputY);

    auto origTilesY = fillDividedTiles(baseOp, nTilesOnDimForOutputY, outputYShape);
    return origTilesY;
}

void reverseTilesOrderReverseMode(OutputTiling& tilesY, int64_t seqLengthTile, IE::RNNSequenceDirection origDirection) {
    const auto reverse = [seqLengthTile, &tilesY](size_t i) {
        for (size_t j = 0; j < size_t(seqLengthTile / 2); ++j) {
            std::swap(tilesY[i + j], tilesY[i + seqLengthTile - 1 - j]);
        }
    };
    if (origDirection == IE::RNNSequenceDirection::BIDIRECTIONAL) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            if (tilesY[i].offsets[Dim(1)] == 1) {
                reverse(i);
            }
        }
    }
    if (origDirection == IE::RNNSequenceDirection::REVERSE) {
        for (size_t i = 0; i < tilesY.size(); i += seqLengthTile) {
            reverse(i);
        }
    }
}

OutputTiling vpux::VPU::GRUSequenceLastPartOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                               vpux::Logger /*log*/) {
    return GRUSequenceOutputTiling(firstOutputTile);
}
