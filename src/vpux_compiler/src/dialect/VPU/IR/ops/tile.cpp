//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

mlir::SmallVector<int64_t> calcTileOutputShape(mlir::Value input, llvm::SmallVector<int64_t> repeatsVals) {
    // If number of elements in *"repeats"* is more than shape of *"data"*, then *"data"* will be promoted to
    // "*repeats*" by prepending new axes, e.g. let's shape of *"data"* is equal to (2, 3) and *"repeats"* is equal to
    // [2, 2, 2], then shape of *"data"* will be promoted to (1, 2, 3) and result shape will be (2, 4, 6).
    //
    // If number of elements in *"repeats"* is less than shape of *"data"*, then *"repeats"* will be promoted to
    // "*data*" by prepending 1's to it, e.g. let's shape of *"data"* is equal to (4, 2, 3) and *"repeats"* is equal to
    // [2, 2], then *"repeats"* will be promoted to [1, 2, 2] and result shape will be (4, 4, 6)

    const auto inType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    auto outShape = to_small_vector(inType.getShape());

    while (repeatsVals.size() > outShape.size()) {
        outShape.insert(outShape.begin(), 1);
    }

    auto outShapeIter = std::prev(outShape.end());
    auto repeatsIter = std::prev(repeatsVals.end());
    for (; outShapeIter != std::prev(outShape.begin()) && repeatsIter != std::prev(repeatsVals.begin());
         --outShapeIter, --repeatsIter) {
        *outShapeIter *= *repeatsIter;
    }
    return outShape;
}

}  // namespace

mlir::LogicalResult vpux::VPU::TileOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                        mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                        mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::TileOpAdaptor tile(operands, attrs, prop);
    if (mlir::failed(tile.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(tile.getInput().getType());
    auto outShape = calcTileOutputShape(tile.getInput(), parseIntArrayAttr<int64_t>(tile.getRepeatsValues()));

    const auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::TileOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::TileOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto originalInputShape = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape().raw();
    auto tiledOutputShape = outputTile.shape.raw();
    auto tiledOutputOffsets = outputTile.offsets.raw();
    auto tiledOutputAxis = outputTile.axis.raw();
    auto repeatsValues = parseIntArrayAttr<int64_t>(getRepeatsValuesAttr());
    Shape suggestedNewInputShape, suggestedOffS, suggestedAxis;
    const auto isZero = [](int64_t number) {
        return (number == 0 ? 1 : number);
    };

    // adjust and calc input shape depending on tiled output
    for (size_t i = 0; i < originalInputShape.size(); i++) {
        auto repeatsDimension = tiledOutputShape[i] % originalInputShape[i];
        if (repeatsDimension != 0) {
            // need to tile input and repeats too
            int64_t newRepeatsD = isZero(repeatsValues[i] - 1);
            while (tiledOutputShape[i] % newRepeatsD != 0) {
                newRepeatsD--;
            }
            suggestedNewInputShape.insert(suggestedNewInputShape.end(), tiledOutputShape[i] / newRepeatsD);
        } else {
            // no need to tile from inputs
            suggestedNewInputShape.insert(suggestedNewInputShape.end(), originalInputShape[i]);
        }
        // set axis and offsets
        suggestedOffS.insert(suggestedOffS.end(), tiledOutputOffsets[i] % originalInputShape[i]);
        suggestedAxis.insert(suggestedAxis.end(), tiledOutputAxis[i]);
    }

    return TilingInfo{{TileInfo(suggestedNewInputShape, suggestedOffS, suggestedAxis)}};
}

void vpux::VPU::TileOp::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& outputTile) {
    auto inputTilingTiles = inputTiling.tiles[0];
    auto outputShape = outputTile.shape.raw();
    auto suggestedInShape = inputTilingTiles.shape.raw();

    // repeats values is eq with output shape divided by input shape
    SmallVector<int64_t> newRepeats;
    for (size_t i = 0; i < getRepeatsValues().getValue().size(); i++) {
        newRepeats.push_back(outputShape[i] / suggestedInShape[i]);
    }

    auto newRepeatsAttr = getIntArrayAttr(getContext(), newRepeats);
    setRepeatsValuesAttr(newRepeatsAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::TileOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for SW currently, for op {0} at '{1}'", op->getName(),
                    op->getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(op);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingBuilderOpInterface",
                    op->getName());

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType());
    const auto inputShape = inputType.getShape();
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    const auto dimOrder = inputType.getDimsOrder();

    Shape nTilesOnDim(outputShape.size(), 1);

    auto dimToTileIndex = 0;
    auto dimToTile = dimOrder.dimAt(dimToTileIndex);

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    const auto& maxNumTiles = tilingBuilder.getMaxNumTiles();
    const auto isDimLeftToTile = [&](ShapeRef tileShape) -> bool {
        return (tileShape[dimToTile] < maxNumTiles[dimToTile.ind()]);
    };

    // Only repeatsDim require being divisible when tiling
    auto repeatsValues = parseIntArrayAttr<int64_t>(getRepeatsValuesAttr());
    const auto verifyNumTiles = [&](ShapeRef tileShape) -> bool {
        return (outputShape[dimToTile] / tileShape[dimToTile] == inputShape[dimToTile]) ||
               (inputShape[dimToTile] == 1) || (repeatsValues[dimToTile.ind()] == 1);
    };

    while ((!isSupportedTileSize(nTilesOnDim, tilingMode)) || (!verifyNumTiles(nTilesOnDim))) {
        while (((nTilesOnDim[dimToTile] >= outputShape[dimToTile]) && (!isDimLeftToTile(nTilesOnDim)))) {
            ++dimToTileIndex;
            if (dimToTileIndex >= static_cast<int>(outputShape.size())) {
                VPUX_THROW_WHEN(tilingMode == TilingMode::ISOLATED, "Failed to tile {0} at '{1}'", op->getName(),
                                op->getLoc());
            }
            dimToTile = dimOrder.dimAt(dimToTileIndex);
        }
        ++nTilesOnDim[dimToTile];
    }

    log.trace("Isolated tiling strategy: {0}", nTilesOnDim);
    return fillDividedTiles(op, nTilesOnDim, outputShape);
}
