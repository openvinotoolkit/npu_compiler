//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RDFTUncutOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::RDFTUncutOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(op.getAxesAttr());
    auto signalSize = parseIntArrayAttr<int64_t>(op.getSignalSizeAttr());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outShape = to_small_vector(inType.getShape());

    for (size_t i = 0; i < axes.size(); ++i) {
        if (signalSize[i] != -1) {
            outShape[axes[i]] = signalSize[i];
        }
    }
    // insert complex numbers representation, as output for IRDFT are complex numbers.
    outShape.push_back(2);

    auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::RDFTUncutOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    auto curTile = outputTile;
    auto axes = parseIntArrayAttr<int64_t>(getAxesAttr());
    const auto inShape = getShape(getInput());
    for (auto axis : axes) {
        curTile.shape[Dim(axis)] = inShape[Dim(axis)];
    }
    // remove last  complexs dim
    curTile.shape.pop_back();
    curTile.offsets.pop_back();
    curTile.axis.pop_back();

    TileInfo twiddleTile(getShape(getTwiddleFactors()));
    return TilingInfo{{std::move(curTile), std::move(twiddleTile)}};
}

void vpux::VPU::RDFTUncutOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::RDFTUncutOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return getSWLayerTilingStrategy(getOperation(), tilingMode, log);
}

SmallVector<int64_t> vpux::VPU::RDFTUncutOp::getMaxNumTiles() {
    auto op = getOperation();
    // eliminate axes from possible tiling dims
    auto axes = parseIntArrayAttr<int64_t>(getAxesAttr());
    // add last axis to not allowed split as represent the complex number
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
    const auto outputShape = outputType.getShape();
    axes.push_back(outputShape.size() - 1);

    return vpux::getMaxNumTiles(op, false, false, getMaxNumTilesWithAxesExclusion(op, axes));
}
