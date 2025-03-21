//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

using namespace vpux;

//
// inferReturnTypes
//

mlir::LogicalResult VPU::DetectionOutputNmsCaffeOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::DetectionOutputNmsCaffeOpAdaptor nmsCaffe(operands, attrs, prop);
    if (mlir::failed(nmsCaffe.verify(loc))) {
        return mlir::failure();
    }

    const auto confidenceType = nmsCaffe.getConfidence().getType().cast<NDTypeInterface>();
    const auto confidenceShape = confidenceType.getShape();
    const auto numClasses = confidenceShape[Dims4D::Act::H];
    const auto topK = nmsCaffe.getTopK();

    const auto boxesType = nmsCaffe.getBoxes().getType().cast<NDTypeInterface>();
    const auto boxesShape = boxesType.getShape();
    const auto outBoxesShape =
            SmallVector<int64_t>{boxesShape[Dims4D::Act::N], numClasses, topK, boxesShape[Dims4D::Act::W]};
    const auto outBoxesType = boxesType.changeShape(Shape(outBoxesShape));

    const auto sizesType = nmsCaffe.getSizes().getType().cast<NDTypeInterface>();

    inferredReturnTypes.push_back(confidenceType);
    inferredReturnTypes.push_back(outBoxesType);
    inferredReturnTypes.push_back(sizesType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

using DimType = Shape::ValueType;

TileInfo tileOnHeight(mlir::Value indices, DimType numClasses, DimType classesOffset, DimType classesAxis) {
    const auto indicesShape = getShape(indices);

    auto tile = TileInfo(indicesShape);
    tile.shape[Dims4D::Act::H] = numClasses;
    tile.offsets[Dims4D::Act::H] = classesOffset;
    tile.axis[Dims4D::Act::H] = classesAxis;

    return tile;
}

TileInfo tileDecodedBoxes(mlir::Value decodedBoxes, DimType numClasses, DimType classesOffset, DimType classesAxis) {
    const auto decodedBoxesShape = getShape(decodedBoxes);

    auto tile = TileInfo(decodedBoxesShape);
    const auto numLocClasses = decodedBoxesShape[Dims4D::Act::C];
    if (numLocClasses > 1) {
        tile.shape[Dims4D::Act::C] = numClasses;
        tile.offsets[Dims4D::Act::C] = classesOffset;
        tile.axis[Dims4D::Act::C] = classesAxis;
    }

    return tile;
}

TileInfo tileSizes(mlir::Value sizes, DimType numClasses, DimType classesOffset, DimType classesAxis) {
    const auto shape = getShape(sizes);

    auto tile = TileInfo(shape);
    tile.shape[Dims4D::Act::W] = numClasses;
    tile.offsets[Dims4D::Act::W] = classesOffset;
    tile.axis[Dims4D::Act::W] = classesAxis;

    return tile;
}

InputTiling vpux::VPU::DetectionOutputNmsCaffeOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                    vpux::Logger /*log*/) {
    // outputTile [1, 1, numClasses, TopK]
    const auto numClasses = outputTile.shape[Dims4D::Act::H];
    const auto classesOffset = outputTile.offsets[Dims4D::Act::H];
    const auto classesAxis = outputTile.axis[Dims4D::Act::H];

    auto confidenceTile = tileOnHeight(getConfidence(), numClasses, classesOffset, classesAxis);
    auto decodedBoxesTile = tileDecodedBoxes(getBoxes(), numClasses, classesOffset, classesAxis);
    auto indicesTile = tileOnHeight(getIndices(), numClasses, classesOffset, classesAxis);
    auto sizesTile = tileSizes(getSizes(), numClasses, classesOffset, classesAxis);
    return InputTiling{
            {std::move(confidenceTile), std::move(decodedBoxesTile), std::move(indicesTile), std::move(sizesTile)}};
}

void vpux::VPU::DetectionOutputNmsCaffeOp::adjustAttrs(const TilingInfo&, const TileInfo& outputTile) {
    const auto outputOffsets = outputTile.offsets;
    VPUX_THROW_UNLESS(outputOffsets.size() == 4,
                      "Expected 4D shape for the first output of DetectionOutputNmsCaffe layer, got {0}",
                      outputOffsets.size());

    const auto classOffset = outputOffsets[Dims4D::Act::H];
    const auto shiftedBackgroundId = getBackgroundId() - classOffset;
    const auto newBackgroundIdAttr = getIntAttr(getContext(), shiftedBackgroundId);

    setBackgroundIdAttr(newBackgroundIdAttr);
}

mlir::FailureOr<OutputTiling> vpux::VPU::DetectionOutputNmsCaffeOp::getTilingStrategy(TilingMode tilingMode,
                                                                                      Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}

OutputTiling vpux::VPU::DetectionOutputNmsCaffeOp::getOutputTiling(const vpux::TileInfo& firstOutputTile,
                                                                   vpux::Logger /*log*/) {
    // Tiling by classes dimension
    // Output 0 confidence    [ 1, 1, numClasses, topK ]
    // Output 1 boxes         [ 1, numClasses, topK, 4 ]
    // Output 2 sizes         [ 1, 1, 1, numClasses ]
    const auto shapeClasses = firstOutputTile.shape[Dims4D::Act::H];
    const auto offsetClasses = firstOutputTile.offsets[Dims4D::Act::H];
    const auto axisClasses = firstOutputTile.axis[Dims4D::Act::H];

    const auto boxesType = getOutBoxes().getType().cast<vpux::NDTypeInterface>();
    const auto boxesShape = boxesType.getShape();
    auto boxesTile = TileInfo(boxesShape);
    boxesTile.shape[Dims4D::Act::C] = shapeClasses;
    boxesTile.offsets[Dims4D::Act::C] = offsetClasses;
    boxesTile.axis[Dims4D::Act::C] = axisClasses;

    const auto sizesShapeSize = 4;
    auto sizesTile = TileInfo(sizesShapeSize);
    sizesTile.shape = Shape{1, 1, 1, shapeClasses};
    sizesTile.offsets = Shape{0, 0, 0, offsetClasses};
    sizesTile.axis = Shape{1, 1, 1, axisClasses};

    return OutputTiling{firstOutputTile, std::move(boxesTile), std::move(sizesTile)};
}
