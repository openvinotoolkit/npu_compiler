//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/Support/LogicalResult.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::AvgPool16Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::AvgPool16OpAdaptor avgPool(operands, attrs, prop);
    if (mlir::failed(avgPool.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(avgPool.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(avgPool.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(avgPool.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(avgPool.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(avgPool.getDilations());
    const auto roundingType = avgPool.getRoundingType();

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    const auto inType = inputType.getElementType();
    auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    if (mlir::failed(IE::unpadInputShape(inShapeInfo.shape, avgPool.getInputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    auto outShape = inferAvgPool16OutputShape(inShapeInfo, windowStrides, windowDilations, dataPaddingBelow,
                                              dataPaddingAbove, windowShape, roundingType);

    if (mlir::failed(IE::padOutputShape(outShape.shape, avgPool.getOutputPaddingAttr(), loc))) {
        return mlir::failure();
    }

    inferredReturnShapes.emplace_back(outShape.shape, inType);

    return mlir::success();
}
