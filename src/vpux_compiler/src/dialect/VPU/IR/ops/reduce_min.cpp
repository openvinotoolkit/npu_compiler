//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"

#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::ReduceMinOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::ReduceMinOpAdaptor reduceMin(operands, attrs, prop);
    if (mlir::failed(reduceMin.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduceMin.getInput();
    const auto keepDims = reduceMin.getKeepDims();

    auto axesValue = parseIntArrayAttr<int64_t>(reduceMin.getAxesValue());

    return VPU::inferReduceReturnTypes(loc, input, keepDims, axesValue, inferredReturnTypes);
}

//
// fold
//

mlir::OpFoldResult vpux::VPU::ReduceMinOp::fold(FoldAdaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    return nullptr;
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::ReduceMinOp::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger /*log*/) {
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(getInput().getType()).getShape();
    const auto axesValue = getAxesValue();
    const auto keepDims = getKeepDims();

    return backInferReduceTile(outputTile, inShape, axesValue, keepDims);
}

void vpux::VPU::ReduceMinOp::adjustAttrs(const TilingInfo& /*inputTiling*/, const TileInfo& /*outputTile*/) {
}

mlir::FailureOr<OutputTiling> vpux::VPU::ReduceMinOp::getTilingStrategy(TilingMode tilingMode, Logger log) {
    const auto op = getOperation();
    const auto keepDims = getKeepDims();
    SmallVector<int64_t> maxNumTiles;

    if (keepDims) {
        const auto axes = parseIntArrayAttr<int64_t>(getAxesValueAttr());
        maxNumTiles = getMaxNumTilesWithAxesExclusion(op, axes);
    } else {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(getOutput().getType());
        const auto outputShape = outputType.getShape();
        maxNumTiles = to_small_vector(outputShape);
    }

    return vpux::getSWLayerTilingStrategy(op, tilingMode, log, maxNumTiles);
}
