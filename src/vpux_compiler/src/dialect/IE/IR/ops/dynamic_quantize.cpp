//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

mlir::LogicalResult verifyBroadcastCompatible(mlir::Operation* op, mlir::ArrayRef<int64_t> inputShape,
                                              mlir::Value tensor, llvm::StringRef tensorName) {
    if (tensor == nullptr) {
        return mlir::success();
    }

    const auto tensorShape = to_small_vector(mlir::cast<mlir::ShapedType>(tensor.getType()).getShape());
    if (tensorShape.size() > inputShape.size()) {
        return errorAt(op, "{0} tensor has rank greater than input tensor.", tensorName);
    }

    auto inputIt = inputShape.rbegin();
    auto tensorIt = tensorShape.rbegin();
    for (; tensorIt != tensorShape.rend(); ++tensorIt, ++inputIt) {
        if (*tensorIt > 1 && *tensorIt != *inputIt) {
            return errorAt(op, "{0} tensor is not broadcast-compatible with input tensor.", tensorName);
        }
    }

    return mlir::success();
}

bool hasSameShape(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return true;
    }

    return mlir::cast<mlir::ShapedType>(lhs.getType()).getShape() ==
           mlir::cast<mlir::ShapedType>(rhs.getType()).getShape();
}

}  // namespace

mlir::LogicalResult vpux::IE::DynamicQuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::DynamicQuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    auto scaleOrZpShape = SmallVector<int64_t>{1};
    if (quantize.getMin() != nullptr && quantize.getMax() != nullptr) {
        const auto minType = mlir::cast<mlir::ShapedType>(quantize.getMin().getType());
        scaleOrZpShape = to_small_vector(minType.getShape());
    }

    const auto inType = mlir::cast<mlir::ShapedType>(quantize.getInput().getType());
    const auto quantizedElemType = quantize.getDstElemType();
    inferredReturnShapes.emplace_back(inType.getShape(), quantizedElemType);
    inferredReturnShapes.emplace_back(scaleOrZpShape, inType.getElementType());
    inferredReturnShapes.emplace_back(scaleOrZpShape, quantizedElemType);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::DynamicQuantizeOp::verify() {
    const auto inputShape = to_small_vector(mlir::cast<mlir::ShapedType>(getInput().getType()).getShape());

    if (mlir::failed(verifyBroadcastCompatible(*this, inputShape, getMin(), "Min"))) {
        return mlir::failure();
    }

    if (mlir::failed(verifyBroadcastCompatible(*this, inputShape, getMax(), "Max"))) {
        return mlir::failure();
    }

    if (mlir::failed(verifyBroadcastCompatible(*this, inputShape, getScale(), "Scale"))) {
        return mlir::failure();
    }

    if (mlir::failed(verifyBroadcastCompatible(*this, inputShape, getZeroPoint(), "ZeroPoint"))) {
        return mlir::failure();
    }

    if (!hasSameShape(getMin(), getMax())) {
        return errorAt(*this, "Min and max tensors must have the same shape.");
    }

    if (!hasSameShape(getScale(), getZeroPoint())) {
        return errorAt(*this, "Scale and zero-point tensors must have the same shape.");
    }

    if (getMin() != nullptr && getScale() != nullptr && !hasSameShape(getMin(), getScale())) {
        return errorAt(*this, "Scale tensor must have the same shape as min/max tensors.");
    }

    const auto dstElemType = getDstElemType();
    if (!dstElemType.isInteger(8)) {
        return errorAt(*this, "dstElemType must be an 8-bit integer type, got {0}.", dstElemType);
    }

    const auto outputElemType = mlir::cast<mlir::ShapedType>(getOutput().getType()).getElementType();
    if (outputElemType != dstElemType) {
        return errorAt(*this, "Output element type {0} must match dstElemType {1}.", outputElemType, dstElemType);
    }

    const auto zpElemType = mlir::cast<mlir::ShapedType>(getZeroPoint().getType()).getElementType();
    if (zpElemType != dstElemType) {
        return errorAt(*this, "Zero-point element type {0} must match dstElemType {1}.", zpElemType, dstElemType);
    }

    return mlir::success();
}
