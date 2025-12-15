//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/type_padding.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace vpux {
namespace ShaveCodeGen {

mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx) {
    assert(mlir::isa<mlir::RankedTensorType>(ty) && "Ranked tensor type required for getLinalgElementType");
    // Get the math/arith compatible element type for the ty tensor type.
    // math/arith dialects don't accept non-signless integers.
    auto elTy = mlir::cast<NDTypeInterface>(ty).getElementType();
    auto signlessElTy = mlir::isa<mlir::IntegerType>(elTy) && !elTy.isSignlessInteger()
                                ? mlir::IntegerType::get(ctx, getElemTypeSize(elTy).count())
                                : elTy;
    return signlessElTy;
}

mlir::Value convertFromLinalgValue(mlir::Value op, mlir::Type outputTy, mlir::PatternRewriter& rewriter) {
    mlir::Value ret = op;
    auto resultElTy = mlir::cast<NDTypeInterface>(outputTy).getElementType();
    auto elTy = mlir::cast<NDTypeInterface>(op.getType()).getElementType();
    const auto outTy = mlir::cast<vpux::NDTypeInterface>(outputTy);
    bool needsPermute = !outTy.getDimsOrder().isIdentity();
    auto rank = outTy.getRank();

    if (elTy != resultElTy) {
        // Do a tensor.bitcast in order to change the element type.
        auto bcTy = mlir::cast<NDTypeInterface>(ret.getType()).changeElemType(resultElTy);
        ret = rewriter.create<mlir::tensor::BitcastOp>(op.getLoc(), bcTy, ret).getResult();
    }

    if (needsPermute) {
        // Perform a permute cast to get the desired type. This should be a no-op after buffers are allocated.
        ret = rewriter.create<IE::PermuteCastOp>(op.getLoc(), outputTy, ret,
                                                 outTy.getDimsOrder().toAffineMap(rewriter.getContext()),
                                                 rewriter.getMultiDimIdentityMap(rank))
                      ->getResult(0);
    }

    return ret;
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

    auto dstShape = Shape(ndTy.getDimsOrder().toMemoryOrder(ndTy.getShape()).raw());
    auto retTy = ndTy.changeDimsOrder(DimsOrder::fromNumDims(rank)).changeShape(dstShape);
    retTy = retTy.changeElemType(signlessElTy);
    return mlir::cast<mlir::RankedTensorType>(retTy);
}

mlir::Value convertToLinalgValue(mlir::Value operand, mlir::PatternRewriter& rewriter,
                                 std::optional<mlir::ArrayAttr> padding) {
    auto loc = operand.getLoc();
    mlir::Value retVal = operand;
    const auto opTy = mlir::cast<vpux::NDTypeInterface>(operand.getType());
    auto rank = opTy.getRank();
    bool needsPermute = !opTy.getDimsOrder().isIdentity();
    if (needsPermute) {
        // Do a IE.PermuteCast to remove the order.
        auto identMap = rewriter.getMultiDimIdentityMap(rank);
        auto dstShape = Shape(DimsOrder::fromValue(retVal).toMemoryOrder(getShape(retVal)).raw());
        auto retTy = opTy.changeDimsOrder(DimsOrder::fromNumDims(rank)).changeShape(dstShape);
        retVal = rewriter.create<IE::PermuteCastOp>(loc, retTy, retVal, identMap, identMap);
    }

    auto elTy = mlir::cast<NDTypeInterface>(operand.getType()).getElementType();
    auto signlessElTy = getLinalgElementType(operand.getType(), rewriter.getContext());
    if (elTy != signlessElTy) {
        // The input type is a signed/unsigned integer so not compatible
        // with the dialects we need for lowering (math, arith, etc). We need to bitcast this
        // to a signless integer type.
        auto outputTy = mlir::cast<NDTypeInterface>(retVal.getType()).changeElemType(signlessElTy);
        retVal = rewriter.create<mlir::tensor::BitcastOp>(loc, outputTy, retVal)->getResult(0);
    }

    if (!padding) {
        return retVal;
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

    return rewriter.create<mlir::tensor::ExtractSliceOp>(loc, retVal, offsets, sizes, strides);
}

std::tuple<mlir::Value, mlir::Value> emitTensorSlice(mlir::Location loc, SmallVectorImpl<int64_t>& sliceShape,
                                                     mlir::RankedTensorType allocType,
                                                     mlir::PatternRewriter& rewriter) {
    auto rank = allocType.getRank();
    auto outElemTy = allocType.getElementType();
    mlir::Value zero = nullptr;

    if (mlir::isa<mlir::FloatType>(outElemTy)) {
        zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(outElemTy, 0.));
    } else {
        zero = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getIntegerAttr(outElemTy, 0));
    }

    auto outputEmptyTensor = rewriter.create<mlir::tensor::EmptyOp>(loc, allocType.getShape(), outElemTy).getResult();
    auto fillOp =
            rewriter.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero}, mlir::ValueRange{outputEmptyTensor})
                    .result();

    SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<mlir::OpFoldResult> sizes(rank);
    for (unsigned i = 0; i < sliceShape.size(); ++i) {
        sizes[i] = rewriter.getIndexAttr(sliceShape[i]);
    }
    SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractOp = rewriter.create<mlir::tensor::ExtractSliceOp>(loc, fillOp, offsets, sizes, strides);

    return {extractOp, fillOp};
}

}  // namespace ShaveCodeGen
}  // namespace vpux
