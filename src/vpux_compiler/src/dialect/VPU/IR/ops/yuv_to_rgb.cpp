//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::YuvToRgbOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::YuvToRgbOpAdaptor colorConv(operands, attrs, prop);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(colorConv.getInput1().getType());
    const auto shape = inType.getShape().raw();
    if (shape[3] != 1) {
        return errorAt(loc, "Incorrect input shape format: '{0}'", shape);
    }

    SmallVector<int64_t> outShape{shape[0], shape[1], shape[2], 3};

    if (colorConv.getInput2() == nullptr) {
        VPUX_THROW_UNLESS(colorConv.getInput3() == nullptr, "1xPlane config error");
        VPUX_THROW_UNLESS(((outShape[1] * 2) % 3) == 0, "Invalid height");
        outShape[1] = outShape[1] * 2 / 3;
    }

    const auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::YuvToRgbOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    auto H = Dim(1), W = Dim(2), C = Dim(3);  // N = Dim(0)

    VPUX_THROW_UNLESS(outputTile.shape[H] % 2 == 0 && outputTile.shape[W] % 2 == 0,
                      "Invalid YuvToRgbOp outputTile, output C,H channels are not even");
    auto singlePlane = (getInput2() == nullptr);
    if (!singlePlane) {
        if (getInFmt() == IE::ColorFmt::NV12) {
            TileInfo input1Tile = outputTile;
            TileInfo input2Tile = outputTile;

            input1Tile.shape[C] = 1;
            input2Tile.shape[C] = 2;
            input2Tile.shape[H] = outputTile.shape[H] / 2;
            input2Tile.shape[W] = outputTile.shape[W] / 2;
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;

            return TilingInfo{{input1Tile, input2Tile}};
        } else {
            TileInfo input1Tile = outputTile;
            TileInfo input2Tile = outputTile;
            TileInfo input3Tile = outputTile;

            input1Tile.shape[C] = 1;
            input2Tile.shape[C] = 1;
            input2Tile.shape[H] = outputTile.shape[H] / 2;
            input2Tile.shape[W] = outputTile.shape[W] / 2;
            input2Tile.offsets[H] = outputTile.offsets[H] / 2;
            input2Tile.offsets[W] = outputTile.offsets[W] / 2;

            input3Tile.shape[C] = 1;
            input3Tile.shape[H] = outputTile.shape[H] / 2;
            input3Tile.shape[W] = outputTile.shape[W] / 2;
            input3Tile.offsets[H] = outputTile.offsets[H] / 2;
            input3Tile.offsets[W] = outputTile.offsets[W] / 2;

            return TilingInfo{{input1Tile, input2Tile, input3Tile}};
        }
    } else {
        TileInfo input1Tile(getShape(getInput1()));
        input1Tile = outputTile;

        input1Tile.shape[C] = 1;
        input1Tile.shape[H] = outputTile.shape[H] / 2 * 3;

        return TilingInfo{input1Tile};
    }
}

void vpux::VPU::YuvToRgbOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::YuvToRgbOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = this->getOperation();
    VPUX_THROW_WHEN(tilingMode != TilingMode::ISOLATED,
                    "Only supporting isolated tiling for YuvToRgbOp currently, for op {0} at '{1}'", op->getName(),
                    getLoc());

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    Shape nTilesOnDimforYuv2RGB(outputShape.size(), 1);

    const auto isSupportedTileSize = [op, &tilingInfo, outputShape, log](ShapeRef nTilesOnDim,
                                                                         TilingMode tilingMode) -> bool {
        const auto tiles = fillDividedTiles(op, nTilesOnDim, outputShape);
        if (mlir::failed(tiles)) {
            return false;
        }
        return tilingInfo.isSupportedTiling(tiles.value(), tilingMode, log);
    };

    while (!isSupportedTileSize(nTilesOnDimforYuv2RGB, tilingMode)) {
        if (2 * nTilesOnDimforYuv2RGB[Dims4D::Act::C] < outputShape[Dims4D::Act::C]) {
            nTilesOnDimforYuv2RGB[Dims4D::Act::C]++;
            continue;
        }

        if (2 * nTilesOnDimforYuv2RGB[Dims4D::Act::H] < outputShape[Dims4D::Act::H]) {
            nTilesOnDimforYuv2RGB[Dims4D::Act::H]++;
            continue;
        }

        VPUX_THROW("Operation 'Yuv2RGB' cannot be tiled");
    }

    return fillDividedTiles(op, nTilesOnDimforYuv2RGB, outputShape);
}
