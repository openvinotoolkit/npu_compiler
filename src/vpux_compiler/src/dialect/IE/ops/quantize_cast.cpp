//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::QuantizeCastOpAdaptor quantizeCast(operands, attrs);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantizeCast.input().getType().cast<mlir::RankedTensorType>();
    const auto dstElemType = quantizeCast.dstElemType();
    const auto outDesc = IE::getTensorAttr(inType);
    unsigned int outputWidth;
    if (auto quantizedOutput = dstElemType.dyn_cast<mlir::quant::QuantizedType>()) {
        outputWidth = quantizedOutput.getStorageTypeIntegralWidth();
    } else if (auto quantizedOutput = dstElemType.dyn_cast<mlir::IntegerType>()) {
        outputWidth = quantizedOutput.getWidth();
    } else {
        return errorAt(loc, "Unsupported output type: {0}", dstElemType);
    }

    if (auto integerInput = inType.getElementType().dyn_cast<mlir::IntegerType>()) {
        const auto inputWidth = integerInput.getWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Integer input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else if (auto quantizedInput = inType.getElementType().dyn_cast<mlir::quant::QuantizedType>()) {
        const auto inputWidth = quantizedInput.getStorageTypeIntegralWidth();
        if (inputWidth != outputWidth) {
            return errorAt(loc, "Quantized input width ({0}) differs from output width ({1})", inputWidth, outputWidth);
        }
    } else {
        return errorAt(loc, "Unsupported combination of input and output element types: {0} -> {1}",
                       inType.getElementType(), dstElemType);
    }

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}

mlir::OpFoldResult vpux::IE::QuantizeCastOp::fold(vpux::ArrayRef<mlir::Attribute>) {
    return input().getType() == output().getType() ? input() : nullptr;
}

//
// FuseQuantizeCasts
//

namespace {

class FuseQuantizeCasts final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCasts::matchAndRewrite(IE::QuantizeCastOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> IE.QuantizeCast type2 -> IE.QuantizeCast type3 -> Output type3
    // into
    // Input type1 -> IE.QuantizeCast type3 -> Output type3
    auto producerOp = origOp.input().getDefiningOp<IE::QuantizeCastOp>();
    if (producerOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, origOp.output().getType(), producerOp.input(),
                                                    origOp.dstElemType());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void IE::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCasts>(ctx);
}
