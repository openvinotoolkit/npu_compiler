//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::IRDFTLastAxisOp::inferReturnTypes(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
        mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    VPU::IRDFTLastAxisOpAdaptor op(operands, attrs, prop);
    if (mlir::failed(op.verify(loc))) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(op.getAxesAttr());
    auto signalSize = parseIntArrayAttr<int64_t>(op.getSignalSizeAttr());

    const auto inType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
    auto outShape = to_small_vector(inType.getShape());
    // delete last size, 2 in this case
    outShape.pop_back();

    const auto last_axis = axes.back();
    outShape[last_axis] = (outShape[last_axis] - 1) * 2;

    for (size_t i = 0; i < axes.size(); ++i) {
        if (signalSize[i] != -1) {
            outShape[axes[i]] = signalSize[i];
        }
    }

    auto outType = inType.changeShape(ShapeRef(outShape));
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::IRDFTLastAxisOp::backInferTileInfo(const vpux::TileInfo& outputTile,
                                                                vpux::Logger /*log*/) {
    auto curTile = outputTile;
    auto axes = parseIntArrayAttr<int64_t>(getAxesAttr());
    const auto inShape = getShape(getInput());
    for (auto axis : axes) {
        curTile.shape[Dim(axis)] = inShape[Dim(axis)];
    }
    // input have 1 extra dimension complex number representation
    curTile.shape.push_back(2);
    curTile.offsets.push_back(0);
    curTile.axis.push_back(1);

    TileInfo twiddleTile(getShape(getTwiddleFactors()));
    return TilingInfo{{std::move(curTile), std::move(twiddleTile)}};
}

void vpux::VPU::IRDFTLastAxisOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::IRDFTLastAxisOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    auto op = getOperation();
    // eliminate axes from possible tiling dims
    SmallVector<int64_t> axes = parseIntArrayAttr<int64_t>(getAxesAttr());

    return getSWLayerTilingStrategy(op, tilingMode, log, getMaxNumTilesWithAxesExclusion(op, axes));
}
