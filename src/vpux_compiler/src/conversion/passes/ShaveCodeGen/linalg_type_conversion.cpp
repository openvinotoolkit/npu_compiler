//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
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

mlir::Value removePadding(mlir::Value operand, mlir::Value typeConvertedOperand, mlir::PatternRewriter& rewriter,
                          std::optional<mlir::ArrayAttr> padding) {
    auto loc = operand.getLoc();
    const auto opTy = mlir::cast<vpux::NDTypeInterface>(operand.getType());
    auto rank = opTy.getRank();

    if (!padding) {
        return typeConvertedOperand;
    }

    auto shapedTy = mlir::cast<mlir::ShapedType>(operand.getType());
    auto unpaddedShape = SmallVector<int64_t>(shapedTy.getShape());
    if (mlir::failed(IE::unpadInputShape(unpaddedShape, *padding, loc))) {
        return nullptr;
    }
    auto memUnpaddedShape = Shape(DimsOrder::fromValue(operand).toMemoryOrder(ShapeRef(unpaddedShape)).raw());

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (unsigned i = 0; i < rank; ++i) {
        sizes[i] = rewriter.getIndexAttr(memUnpaddedShape.raw()[i]);
    }

    return mlir::tensor::ExtractSliceOp::create(rewriter, loc, typeConvertedOperand, offsets, sizes, strides);
}

std::tuple<mlir::Value, mlir::Value> emitTensorSlice(mlir::Location loc, SmallVectorImpl<int64_t>& sliceShape,
                                                     mlir::RankedTensorType allocType,
                                                     mlir::PatternRewriter& rewriter) {
    auto rank = allocType.getRank();
    auto outElemTy = allocType.getElementType();
    mlir::Value zero = nullptr;

    if (mlir::isa<mlir::FloatType>(outElemTy)) {
        zero = mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(outElemTy, 0.));
    } else {
        zero = mlir::arith::ConstantOp::create(rewriter, loc, rewriter.getIntegerAttr(outElemTy, 0));
    }

    auto outputEmptyTensor = mlir::tensor::EmptyOp::create(rewriter, loc, allocType.getShape(), outElemTy).getResult();
    auto fillOp =
            mlir::linalg::FillOp::create(rewriter, loc, mlir::ValueRange{zero}, mlir::ValueRange{outputEmptyTensor})
                    .result();

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (unsigned i = 0; i < sliceShape.size(); ++i) {
        sizes[i] = rewriter.getIndexAttr(sliceShape[i]);
    }
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractOp = mlir::tensor::ExtractSliceOp::create(rewriter, loc, fillOp, offsets, sizes, strides);

    return {extractOp, fillOp};
}

}  // namespace ShaveCodeGen
}  // namespace vpux
