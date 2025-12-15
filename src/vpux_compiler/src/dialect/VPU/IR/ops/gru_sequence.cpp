//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/tiling_info.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/IRMapping.h>

using namespace vpux;

mlir::LogicalResult vpux::VPU::GRUSequenceOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::GRUSequenceOpAdaptor gru(operands, attrs, prop);
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

vpux::InputTiling vpux::VPU::GRUSequenceOp::backInferTileInfo(const vpux::TileInfo& outputTileY, vpux::Logger) {
    const auto origInputShape = getShape(getInputData());
    const auto origInitialHiddenStateShape = getShape(getInitialHiddenState());
    const auto origWShape = getShape(getWeights());
    const auto origRShape = getShape(getRecurrenceWeights());
    const auto origBShape = getShape(getBiases());

    TileInfo inputTile(origInputShape);
    TileInfo initialHiddenStateTile(origInitialHiddenStateShape);
    TileInfo wTile(origWShape);
    TileInfo rTile(origRShape);
    TileInfo bTile(origBShape);

    inputTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    inputTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    inputTile.shape[Dim(1)] = outputTileY.shape[Dim(2)];
    inputTile.offsets[Dim(1)] = outputTileY.offsets[Dim(2)];

    initialHiddenStateTile.shape[Dim(0)] = outputTileY.shape[Dim(0)];
    initialHiddenStateTile.offsets[Dim(0)] = outputTileY.offsets[Dim(0)];
    initialHiddenStateTile.shape[Dim(1)] = outputTileY.shape[Dim(1)];
    initialHiddenStateTile.offsets[Dim(1)] = outputTileY.offsets[Dim(1)];

    wTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    wTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    rTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    rTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    bTile.shape[Dim(0)] = outputTileY.shape[Dim(1)];
    bTile.offsets[Dim(0)] = outputTileY.offsets[Dim(1)];

    return InputTiling{{inputTile, initialHiddenStateTile, wTile, std::move(rTile), bTile}};
}

void vpux::VPU::GRUSequenceOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputYTile) {
    auto* ctx = this->getContext();
    auto inputTileInfo = inputTiling.tiles[0];
    VPUX_THROW_UNLESS(inputTileInfo.shape[Dim(1)] == outputYTile.shape[Dim(2)],
                      "seq_length dimension in input tile is incompatible with output tile, seq_length dimension of "
                      "input tile is {0}, but it's {1} in output tile",
                      inputTileInfo.shape[Dim(1)], outputYTile.shape[Dim(2)]);
    auto origSeqLength = mlir::dyn_cast_or_null<mlir::IntegerAttr>(getSeqLengthAttr()).getValue().getSExtValue();
    auto tiledSeqLength = inputTileInfo.shape[Dim(1)];
    if (origSeqLength != tiledSeqLength) {
        const auto newSeqLength = getIntAttr(ctx, tiledSeqLength);
        this->setSeqLengthAttr(newSeqLength);
    }
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

mlir::FailureOr<OutputTiling> vpux::VPU::GRUSequenceOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto baseOp = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for GRUSequence currently, for op {0} at '{1}'", baseOp->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(baseOp);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    baseOp->getName());

    // There are two outputs of GRUSequence, the namse Y and Ho are from OpenVINO doc.
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
            log.trace("can't get feasible tiling strategy for GRUSequence.");
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

OutputTiling vpux::VPU::GRUSequenceOp::getOutputTiling(const vpux::TileInfo& firstOutputTile, vpux::Logger /*log*/) {
    return GRUSequenceOutputTiling(firstOutputTile);
}
