//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <cstdint>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPoolOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPoolOpAdaptor maxPool(operands, attrs, prop);
    if (mlir::failed(maxPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool.getStrides());
    const auto roundingType = maxPool.getRoundingType();

    const auto inType = mlir::cast<NDTypeInterface>(maxPool.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inType);

    const auto outShapeInfo = inferMaxPoolOutputShape(inShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove,
                                                      windowShape, roundingType);
    auto outShape = outShapeInfo.shape;
    if (maxPool.getOutputChannels().has_value()) {
        outShape[Dims4D::Act::C.ind()] = maxPool.getOutputChannels().value();
    }

    mlir::ArrayAttr outBoundsAttr = !outShapeInfo.bounds.empty() ? getIntArrayAttr(ctx, outShapeInfo.bounds) : nullptr;
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), /*memSpace=*/nullptr, outBoundsAttr);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MaxPoolOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(getKernelSizeAttr());
    const auto strides = parseIntArrayAttr<int64_t>(getStridesAttr());
    const auto padBegin = parseIntArrayAttr<int64_t>(getPadsBeginAttr());
    const auto padEnd = parseIntArrayAttr<int64_t>(getPadsEndAttr());

    auto outShape =
            reifyConvPoolTensors(builder, getInput(), getOutput(), kernelSize, strides, padBegin, padEnd, getLoc());

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
