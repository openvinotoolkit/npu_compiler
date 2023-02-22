//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::GridSampleOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              mlir::Optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::RegionRange,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::GridSampleOpAdaptor gridSample(operands, attrs);

    if (mlir::failed(gridSample.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = gridSample.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inType.getShape().raw();

    const auto gridType = gridSample.grid().getType().cast<vpux::NDTypeInterface>();
    const auto gridShape = gridType.getShape().raw();

    SmallVector<int64_t> outShape = {inputShape[0], inputShape[1], gridShape[1], gridShape[2]};

    const auto outType = inType.changeShape(Shape(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// serialize
//

vpux::EMU::BlobWriter::SpecificTask vpux::VPU::GridSampleOp::serialize(EMU::BlobWriter&) {
    VPUX_THROW("GridSampleOp implemented just on 37xx.");
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::GridSampleOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(input());
    const auto origGridShape = getShape(grid());

    TileInfo inputTile(origInputShape);
    TileInfo gridTile(origGridShape);

    inputTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    inputTile.shape[Dims4D::Act::C] = outputTile.shape[Dims4D::Act::C];

    inputTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    inputTile.offsets[Dims4D::Act::C] = outputTile.offsets[Dims4D::Act::C];

    gridTile.shape[Dims4D::Act::N] = outputTile.shape[Dims4D::Act::N];
    gridTile.shape[Dim(1)] = outputTile.shape[Dims4D::Act::H];
    gridTile.shape[Dim(2)] = outputTile.shape[Dims4D::Act::W];
    gridTile.offsets[Dims4D::Act::N] = outputTile.offsets[Dims4D::Act::N];
    gridTile.offsets[Dim(1)] = outputTile.offsets[Dims4D::Act::H];
    gridTile.offsets[Dim(2)] = outputTile.offsets[Dims4D::Act::W];

    return InputTiling{{inputTile, gridTile}};
}

void vpux::VPU::GridSampleOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

OutputTiling vpux::VPU::GridSampleOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);

    const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputShape = outputType.getShape();

    Shape nTilesOnDimforGridSample(outputShape.size(), 1);
    tilingMode = TilingMode::ISOLATED;
    const auto tilingModeToCheck = tilingMode;

    SmallVector<Dim> tileDimOrder = {Dims4D::Act::N, Dims4D::Act::C, Dims4D::Act::H, Dims4D::Act::W};

    auto tileDimIter = tileDimOrder.begin();
    auto dimToTile = *tileDimIter;

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        return tilingInfo.isSupportedTiling(tiles, tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDimforGridSample, tilingModeToCheck)) {
        if (nTilesOnDimforGridSample[dimToTile] >= outputShape[dimToTile]) {
            dimToTile = *(++tileDimIter);
            if (tileDimIter == tileDimOrder.end()) {
                VPUX_THROW("Unsupported dim to tile: {0}", dimToTile);
            }
        } else {
            ++nTilesOnDimforGridSample[dimToTile];
        }
    }

    auto origTiles = fillDividedTiles(op, nTilesOnDimforGridSample, outputShape);
    return origTiles;
}
