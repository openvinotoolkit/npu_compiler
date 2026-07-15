//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/verifier_utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::DynamicDequantizeOp::verify() {
    const auto input = getInput();
    if (isTypeSignedOrUnsigned(*this, input).failed()) {
        return mlir::failure();
    }

    const auto inputShape = to_small_vector(mlir::cast<mlir::ShapedType>(input.getType()).getShape());
    const auto scaleShape = to_small_vector(mlir::cast<mlir::ShapedType>(getScale().getType()).getShape());
    if (scaleShape.size() > inputShape.size()) {
        return errorAt(*this, "Scale tensor has rank greater than input tensor.");
    }
    for (auto i : irange(scaleShape.size())) {
        if (scaleShape[i] > 1 && scaleShape[i] != inputShape[i]) {
            return errorAt(*this, "Scale dim doesn't equal input shape.");
        }
    }

    if (const auto zp = getZp()) {
        if (isZeroPointsTypeValidForQuantization(*this, zp).failed()) {
            return mlir::failure();
        }

        const auto zpShape = to_small_vector(mlir::cast<NDTypeInterface>(zp.getType()).getShape());
        if (inputShape.size() != zpShape.size()) {
            return errorAt(*this, "ZeroPoint doesn't have same rank as input tensor.");
        }
        for (auto i : irange(zpShape.size())) {
            if (zpShape[i] > 1 && zpShape[i] != inputShape[i]) {
                return errorAt(*this, "ZeroPoint dim doesn't equal input shape.");
            }
        }
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicDequantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicDequantizeOpAdaptor dynamicDequantize(operands, attrs, prop);
    if (mlir::failed(dynamicDequantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(dynamicDequantize.getInput().getType());
    const auto dstElemType = dynamicDequantize.getDstElemType();
    const auto outDesc = vpux::getTensorAttr(inType);

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);

    return mlir::success();
}
