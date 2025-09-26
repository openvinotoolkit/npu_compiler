//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::AvgPool16Op::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::AvgPool16OpAdaptor avgPool(operands, attrs, prop);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(avgPool.getDilations());
    const auto roundingType = avgPool.getRoundingType();

    const auto inType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());

    const auto shapeI64 = inferAvgPool16OutputShape(ShapeInfo::fromNDType(inType), windowStrides, windowDilations,
                                                    dataPaddingBelow, dataPaddingAbove, windowShape, roundingType);

    const auto outType = inType.changeShape(ShapeRef(shapeI64.shape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}

//
// TilingBuilderOpInterface
//

vpux::InputTiling vpux::VPU::AvgPool16Op::backInferTileInfo(const vpux::TileInfo& outputTile, vpux::Logger) {
    const auto origInputShape = getShape(getInput());
    const auto pads = PadInfo(getPadsBegin(), getPadsEnd());
    auto inputTiling = vpux::backInferPoolTile(outputTile, origInputShape, getKernelSize(), getStrides(), pads);
    return inputTiling;
}

void vpux::VPU::AvgPool16Op::adjustAttrs(const TilingInfo& inputTiling, const TileInfo& /*outputTile*/) {
    IE::adjustPaddings(this, inputTiling);
}

mlir::FailureOr<OutputTiling> vpux::VPU::AvgPool16Op::getTilingStrategy(TilingMode tilingMode, Logger log) {
    return vpux::getSWLayerTilingStrategy(this->getOperation(), tilingMode, log);
}
