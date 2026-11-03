//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/conversions.hpp"
#include "vpux/compiler/conversion/passes/ShaveCodeGen/linalg_type_conversion.hpp"
#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

using namespace vpux;

namespace {

class IESoftMaxToLinalg : public mlir::OpConversionPattern<IE::SoftMaxOp> {
public:
    using mlir::OpConversionPattern<IE::SoftMaxOp>::OpConversionPattern;
    using OpAdaptor = mlir::OpConversionPattern<IE::SoftMaxOp>::OpAdaptor;

    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const final;
};

mlir::LogicalResult IESoftMaxToLinalg::matchAndRewrite(IE::SoftMaxOp op, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    auto inputOrder = DimsOrder::fromValue(op->getOperand(0)).toAffineMap(rewriter.getContext());
    auto outputOrder = DimsOrder::fromValue(op->getResult(0)).toAffineMap(rewriter.getContext());
    auto input = adaptor.getOperands()[0];
    // The softmax axis in input memory order.
    auto axis = mlir::cast<mlir::AffineDimExpr>(mlir::inversePermutation(inputOrder).getResult(op.getAxisInd()))
                        .getPosition();
    // The softmax axis in the output memory order.
    auto outAxis = mlir::cast<mlir::AffineDimExpr>(mlir::inversePermutation(outputOrder).getResult(op.getAxisInd()))
                           .getPosition();
    auto resultType = ShaveCodeGen::normalizeType(mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType()));

    auto elTy = ShaveCodeGen::getLinalgElementType(op->getOperand(0).getType(), rewriter.getContext());
    auto outElTy = resultType.getElementType();
    auto rank = mlir::cast<vpux::NDTypeInterface>(input.getType()).getRank();
    auto loc = op->getLoc();
    bool changesOrder = inputOrder != outputOrder;
    bool hasOutputTypeCast = outElTy != elTy;

    auto srcFloatTy = mlir::dyn_cast<mlir::FloatType>(elTy);
    auto dstFloatTy = mlir::dyn_cast<mlir::FloatType>(outElTy);
    if (hasOutputTypeCast &&
        (srcFloatTy == nullptr || dstFloatTy == nullptr || srcFloatTy.getWidth() == dstFloatTy.getWidth())) {
        return rewriter.notifyMatchFailure(op, "Unsupported SoftMax dstElemType");
    }

    auto pad = op.getPadSize();
    bool hasPad = pad && *pad != 0;

    SmallVector<int64_t> logicalPadding(rank, 0);
    if (hasPad) {
        logicalPadding[op.getAxisInd()] = *pad;
    }

    // The result shape in memory order, with the padding removed.
    auto outShapeRef = resultType.getShape();
    SmallVector<int64_t> outShape(outShapeRef.begin(), outShapeRef.end());
    if (hasPad) {
        outShape[outAxis] -= *pad;
    }

    // Remove the padding from the input.
    input = ShaveCodeGen::removePadding(op->getOperand(0), input, rewriter, logicalPadding);

    // Create the output tensor of the linalg.softmax op.
    mlir::Value out = mlir::tensor::EmptyOp::create(rewriter, loc,
                                                    mlir::cast<mlir::ShapedType>(input.getType()).getShape(), elTy);

    auto linalgOp = mlir::linalg::SoftmaxOp::create(rewriter, loc, mlir::TypeRange{input.getType()}, input, out, axis);

    mlir::Value result = linalgOp->getResult(0);

    if (changesOrder) {
        mlir::Value transposeOut = mlir::tensor::EmptyOp::create(rewriter, loc, outShape, elTy);

        // Permute the result of the softmax op to match the output order.
        SmallVector<int64_t> perm(rank);
        auto transposePerm = outputOrder.compose(mlir::inversePermutation(inputOrder));
        for (int i = 0; i < rank; ++i) {
            perm[i] = mlir::cast<mlir::AffineDimExpr>(transposePerm.getResult(i)).getPosition();
        }
        result = mlir::linalg::TransposeOp::create(rewriter, loc, result, transposeOut, perm)->getResult(0);
    }

    if (hasOutputTypeCast) {
        mlir::Value castOut = mlir::tensor::EmptyOp::create(rewriter, loc, outShape, outElTy);

        auto map = mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
        SmallVector<mlir::utils::IteratorType> iterators(rank, mlir::utils::IteratorType::parallel);

        result = mlir::linalg::GenericOp::create(
                         rewriter, loc, mlir::TypeRange{castOut.getType()}, mlir::ValueRange{result},
                         mlir::ValueRange{castOut}, SmallVector<mlir::AffineMap>{map, map}, iterators,
                         [&](mlir::OpBuilder& b, mlir::Location nestedLoc, mlir::ValueRange args) {
                             mlir::Value casted = nullptr;
                             if (srcFloatTy.getWidth() < dstFloatTy.getWidth()) {
                                 auto ext = mlir::arith::ExtFOp::create(
                                         b, nestedLoc, outElTy, args[0],
                                         mlir::arith::FastMathFlagsAttr::get(b.getContext(),
                                                                             mlir::arith::FastMathFlags::contract));
                                 casted = ext;
                             } else {
                                 auto trunc = mlir::arith::TruncFOp::create(
                                         b, nestedLoc, outElTy, args[0], mlir::arith::RoundingModeAttr{},
                                         mlir::arith::FastMathFlagsAttr::get(b.getContext(),
                                                                             mlir::arith::FastMathFlags::contract));
                                 casted = trunc;
                             }
                             mlir::linalg::YieldOp::create(b, nestedLoc, casted);
                         })
                         ->getResult(0);
    }

    rewriter.replaceOp(op, ShaveCodeGen::addPadding(op->getResult(0), result, rewriter, logicalPadding));

    return mlir::success();
}

}  // namespace

void ShaveCodeGen::populateIESoftmaxToLinalgPatterns(mlir::RewritePatternSet& patternSet,
                                                     mlir::TypeConverter& typeConverter) {
    patternSet.add<IESoftMaxToLinalg>(typeConverter, patternSet.getContext());
}
