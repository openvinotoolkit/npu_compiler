//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace vpux {
namespace ShaveCodeGen {

mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx) {
    assert(mlir::isa<mlir::RankedTensorType>(ty) && "Ranked tensor type required for getLinalgElementType");
    // Get the math/arith compatible element type for the ty tensor type.
    // math/arith dialects don't accept non-signless integers.
    auto elTy = mlir::cast<mlir::RankedTensorType>(ty).getElementType();

    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elTy)) {
        if (!intTy.isSignless()) {
            return mlir::IntegerType::get(ctx, intTy.getWidth());
        }
        return elTy;
    }

    return elTy;
}

mlir::RankedTensorType getUnpaddedTensorType(mlir::RankedTensorType type, mlir::Location loc,
                                             std::optional<mlir::ArrayAttr> padding) {
    if (!padding) {
        return type;
    }

    auto outShape = SmallVector<int64_t>(type.getShape());
    if (mlir::failed(IE::unpadInputShape(outShape, *padding, loc))) {
        return nullptr;
    }
    auto unpaddedOutTy = mlir::cast<NDTypeInterface>(type).changeShape(ShapeRef(outShape));
    return mlir::cast<mlir::RankedTensorType>(unpaddedOutTy);
}

mlir::RankedTensorType normalizeType(mlir::RankedTensorType type) {
    auto rank = type.getRank();
    auto ndTy = mlir::cast<vpux::NDTypeInterface>(type);
    auto signlessElTy = getLinalgElementType(type, type.getContext());

    if (auto quantTy = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(signlessElTy)) {
        auto storageType = quantTy.getStorageType();
        int width = storageType.getIntOrFloatBitWidth();
        // E206657 Investigate non-signless storage type with quant types
        // Convert to signless storage type for compatibility with MLIR's arith/math dialect operations
        if (!storageType.isSignlessInteger()) {
            // Create a signless storage type
            mlir::Type signlessType = mlir::IntegerType::get(quantTy.getContext(), width);
            // Create a new UniformQuantizedType with signless storage type
            auto newQuantTy = mlir::quant::UniformQuantizedType::get(
                    quantTy.getFlags(), signlessType, quantTy.getExpressedType(), quantTy.getScale(),
                    quantTy.getZeroPoint(), quantTy.getStorageTypeMin(), quantTy.getStorageTypeMax());
            signlessElTy = newQuantTy;
        }
    } else if (auto quantTy = mlir::dyn_cast<mlir::quant::UniformQuantizedPerAxisType>(signlessElTy)) {
        auto storageType = quantTy.getStorageType();
        int width = storageType.getIntOrFloatBitWidth();

        // Order handling for per channel/axis Quantize/Dequantize
        // The quantized dimension should be correctly mapped between logical and memory layout
        // e.g: NCHW logical layout -> quantized dimension is at index 1
        //      NHWC memory layout -> quantized dimension is at index 3
        auto quantDim = quantTy.getQuantizedDimension();
        auto dimOrder = ndTy.getDimsOrder();
        auto inputShape = ndTy.getShape();
        auto memoryShape = dimOrder.toMemoryOrder(inputShape);
        auto newQuantDim = dimOrder.toMemDim(Dim(quantDim)).ind();

        // Convert to signless storage type for compatibility with MLIR's arith/math dialect operations
        mlir::Type finalStorageType = storageType;
        if (!storageType.isSignlessInteger()) {
            // Create a signless storage type
            mlir::Type signlessType = mlir::IntegerType::get(quantTy.getContext(), width);
            finalStorageType = signlessType;
        }
        auto newQuantTy = mlir::quant::UniformQuantizedPerAxisType::get(
                quantTy.getFlags(), finalStorageType, quantTy.getExpressedType(), quantTy.getScales(),
                quantTy.getZeroPoints(), newQuantDim, quantTy.getStorageTypeMin(), quantTy.getStorageTypeMax());
        signlessElTy = newQuantTy;

        auto dstShape = Shape(memoryShape.raw());
        auto retTy = mlir::RankedTensorType::get(llvm::ArrayRef<int64_t>(dstShape.raw()), signlessElTy);
        return retTy;
    }

    auto dstShape = Shape(ndTy.getDimsOrder().toMemoryOrder(ndTy.getShape()).raw());
    auto retTy = ndTy.changeDimsOrder(DimsOrder::fromNumDims(rank)).changeShape(dstShape);
    retTy = retTy.changeElemType(signlessElTy);
    return mlir::cast<mlir::RankedTensorType>(retTy);
}

static bool hasNoPadding(llvm::ArrayRef<int64_t> padding) {
    return llvm::all_of(padding, [](int64_t pad) {
        return pad == 0;
    });
}

// Convert a padding specification given in an operand's logical order to memory order.
static Shape toMemoryPadding(mlir::Value operand, llvm::ArrayRef<int64_t> logicalPadding) {
    return Shape(DimsOrder::fromValue(operand).toMemoryOrder(ShapeRef(logicalPadding)).raw());
}

mlir::Value removePadding(mlir::Value operand, mlir::Value typeConvertedOperand, mlir::PatternRewriter& rewriter,
                          llvm::ArrayRef<int64_t> padding) {
    if (hasNoPadding(padding)) {
        return typeConvertedOperand;
    }

    auto loc = operand.getLoc();
    const auto paddedType = mlir::cast<mlir::ShapedType>(typeConvertedOperand.getType());
    const auto rank = paddedType.getRank();
    const auto memoryPadding = toMemoryPadding(operand, padding);

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (int64_t i = 0; i < rank; ++i) {
        sizes[i] = rewriter.getIndexAttr(paddedType.getShape()[i] - memoryPadding.raw()[i]);
    }

    return mlir::tensor::ExtractSliceOp::create(rewriter, loc, typeConvertedOperand, offsets, sizes, strides);
}

mlir::Value addPadding(mlir::Value result, mlir::Value unpaddedResult, mlir::PatternRewriter& rewriter,
                       llvm::ArrayRef<int64_t> padding) {
    if (hasNoPadding(padding)) {
        return unpaddedResult;
    }

    auto loc = result.getLoc();
    const auto elTy = mlir::cast<mlir::RankedTensorType>(unpaddedResult.getType()).getElementType();
    const auto normalizedType = normalizeType(mlir::cast<mlir::RankedTensorType>(result.getType()));
    const auto paddedType = mlir::RankedTensorType::get(normalizedType.getShape(), elTy);
    const auto rank = paddedType.getRank();
    const auto memoryPadding = toMemoryPadding(result, padding);

    SmallVector<mlir::OpFoldResult> low(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> high(rank);
    for (int64_t i = 0; i < rank; ++i) {
        high[i] = rewriter.getIndexAttr(memoryPadding.raw()[i]);
    }

    auto zero = mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(elTy));
    return mlir::tensor::PadOp::create(rewriter, loc, paddedType, unpaddedResult, low, high, zero, /*nofold=*/false)
            ->getResult(0);
}

}  // namespace ShaveCodeGen
}  // namespace vpux
