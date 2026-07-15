//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// isAvgPoolSupportedElementType
//

bool IE::isAvgPoolSupportedElementType(mlir::Type elemType) {
    // Must stay in sync with IE_AvgPoolOp input constraint in pooling.td:
    //   RankedTensorOf<[F16, F32, F64, quant_QuantizedType, SI32, SI64, SI8, UI8]>
    return elemType.isF16() || elemType.isF32() || elemType.isF64() ||
           mlir::isa<mlir::quant::QuantizedType>(elemType) || elemType.isSignedInteger(32) ||
           elemType.isSignedInteger(64) || elemType.isSignedInteger(8) || elemType.isUnsignedInteger(8);
}

//
// createIdentityAvgPool
//

mlir::Operation* IE::createIdentityAvgPool(mlir::Value input, mlir::Type outType, mlir::OpBuilder& builder,
                                           mlir::Location loc) {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = builder.getContext();

    return builder.create<IE::AvgPoolOp>(loc, outType, input, getIntArrayAttr(ctx, poolKernels),
                                         getIntArrayAttr(ctx, poolStrides), getIntArrayAttr(ctx, pads),
                                         getIntArrayAttr(ctx, pads),
                                         vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR),
                                         mlir::UnitAttr::get(ctx), nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// createIdentityMaxPool
//

mlir::Operation* IE::createIdentityMaxPool(mlir::Value input, mlir::Type outType, mlir::PatternRewriter& rewriter) {
    const SmallVector<int64_t> poolStrides = {1, 1};
    const SmallVector<int64_t> poolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    auto ctx = rewriter.getContext();

    return rewriter.create<IE::MaxPoolOp>(
            appendLoc(input.getLoc(), "to_maxpool"), outType, input, getIntArrayAttr(ctx, poolKernels),
            getIntArrayAttr(ctx, poolStrides), getIntArrayAttr(ctx, pads), getIntArrayAttr(ctx, pads),
            IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR), nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// isQuantizedPurposeAvgPool
//
bool IE::isQuantizedPurposeAvgPool(IE::AvgPoolOp avgPool) {
    if (!isIdentityPooling(avgPool)) {
        return false;
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getOutput().getType());
    if (!inputType.getElementType().isF16() ||
        !mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType())) {
        return false;
    }

    return inputType.getDimsOrder() == outputType.getDimsOrder();
}

//
// isQuantizedAvgPoolPermutation
//
bool IE::isQuantizedAvgPoolPermutation(IE::AvgPoolOp avgPool) {
    if (!isIdentityPooling(avgPool)) {
        return false;
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(avgPool.getOutput().getType());

    // do not check order, cause the pool might be used for permutation as well
    return inputType.getElementType().isF16() &&
           mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType());
}

//
// isAddOutputQuantized
//
bool IE::isAddOutputQuantized(IE::AddOp add) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(add.getInput1().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(add.getOutput().getType());

    // do not check order, cause the pool might be used for permutation as well
    return inputType.getElementType().isF16() &&
           mlir::isa<mlir::quant::UniformQuantizedType>(outputType.getElementType());
}
