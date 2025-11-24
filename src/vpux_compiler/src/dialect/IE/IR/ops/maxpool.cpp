//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

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
    auto inShapeInfo = ShapeInfo::fromNDType(inType);
    if (mlir::failed(IE::unpadInputShape(inShapeInfo.shape, maxPool.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    const auto outShapeInfo = inferMaxPoolOutputShape(inShapeInfo, windowStrides, dataPaddingBelow, dataPaddingAbove,
                                                      windowShape, roundingType);
    auto outShape = outShapeInfo.shape;
    if (mlir::failed(IE::padOutputShape(outShape, maxPool.getOutputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    auto outBounds = !outShapeInfo.bounds.empty() ? outShapeInfo.bounds : SmallVector<int64_t>{};
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(outBounds));

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MaxPoolOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    const auto kernelSize = parseIntArrayAttr<int64_t>(getKernelSizeAttr());
    const auto strides = parseIntArrayAttr<int64_t>(getStridesAttr());
    const auto padBegin = parseIntArrayAttr<int64_t>(getPadsBeginAttr());
    const auto padEnd = parseIntArrayAttr<int64_t>(getPadsEndAttr());

    auto outShape = reifyConvPoolTensors(builder, getInput(), getOutput(), nullptr, kernelSize, strides, padBegin,
                                         padEnd, appendLoc(getLoc(), "maxpool"));

    if (mlir::failed(outShape)) {
        return outShape;
    }

    reifiedReturnShapes.emplace_back(std::move(outShape.value()));
    return mlir::success();
}
