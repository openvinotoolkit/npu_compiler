//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::QuantizeOpAdaptor quantize(operands, attrs, prop);
    if (mlir::failed(quantize.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = quantize.getInput().getType().cast<mlir::ShapedType>();
    const auto dstElemType = quantize.getDstElemType();

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType);
    return mlir::success();
}

//
// fold
//

namespace {

mlir::quant::QuantizedType extractQuantizedType(mlir::Value operand) {
    const auto elemType = operand.getType().cast<mlir::ShapedType>().getElementType();
    const auto quantType = elemType.dyn_cast<mlir::quant::QuantizedType>();
    VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
    return quantType;
}

}  // namespace

mlir::OpFoldResult vpux::IE::QuantizeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (auto ephemeral = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        const auto cst = static_cast<Const::ContentAttr>(ephemeral);
        const auto quantType = extractQuantizedType(getOutput());

        return cst.transform().quantize(quantType).castElemType(quantType).get();
    }

    if (auto dequantize = getInput().getDefiningOp<IE::DequantizeOp>()) {
        if (dequantize.getInput().getType() == getOutput().getType()) {
            return dequantize.getInput();
        }
    }

    return nullptr;
}

//
// FuseFQsWithSimilarScales
//

class FuseFQsWithSimilarScales final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    using mlir::OpRewritePattern<IE::QuantizeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseFQsWithSimilarScales::matchAndRewrite(IE::QuantizeOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    // Get the input of origOp
    mlir::Value origOpInput = origOp.getInput();

    // Get the defining operation of the first operand
    mlir::Operation* origOpProducer = origOpInput.getDefiningOp();

    // Check if the producer is a Reshape or AffineReshape
    if (!mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp>(origOpProducer)) {
        return mlir::failure();
    }

    mlir::Operation* reshapeOp = nullptr;
    reshapeOp = origOpProducer;

    // Check if reshapeOp has only one use
    if (!reshapeOp->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    // Get the first operand of the reshapeOp
    mlir::Value reshapeOpInput = reshapeOp->getOperand(0);

    // Get the defining operation of the first operand
    mlir::Operation* reshapeOpProducer = reshapeOpInput.getDefiningOp();

    // Check if the producer is a Dequantize
    if (!mlir::isa_and_nonnull<IE::DequantizeOp>(reshapeOpProducer)) {
        return mlir::failure();
    }

    IE::DequantizeOp dequantizeOp = nullptr;
    dequantizeOp = mlir::dyn_cast<IE::DequantizeOp>(reshapeOpProducer);

    // Check if dequantizeOp has only one use
    if (!dequantizeOp->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    // Get the element types of the Quantize and Dequantize operations
    auto outputTypeQuantize = mlir::cast<mlir::ShapedType>(origOp.getType());
    auto outElemType = outputTypeQuantize.getElementType();

    auto inputTypeDequantize = mlir::cast<mlir::ShapedType>(dequantizeOp.getInput().getType());
    auto inElemType = inputTypeDequantize.getElementType();

    // If the elemTypes are exactly the same, fail the pattern match
    if (outElemType == inElemType) {
        return mlir::failure();
    }

    auto outUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType);
    if (!outUniformType) {
        return mlir::failure();
    }

    auto inUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(inElemType);
    if (!inUniformType) {
        return mlir::failure();
    }

    // Get the scales of the quantized types
    const auto quantizeScale = outUniformType.getScale();
    const auto dequantizeScale = inUniformType.getScale();

    // If the scales are similar, but not within a tolerance, fail pattern match
    const auto quotient = quantizeScale / dequantizeScale;
    if (quotient < 0.99 || quotient > 1.01) {
        return mlir::failure();
    }

    // Set the insertion point to just after the original operation
    rewriter.setInsertionPointAfter(dequantizeOp.getInput().getDefiningOp());

    // Clone the reshapeOp
    auto* clonedReshapeOp = rewriter.clone(*reshapeOp);

    // Update the types of the cloned operation
    clonedReshapeOp->setOperand(0, dequantizeOp.getInput());
    inferReturnTypes(clonedReshapeOp, InferShapedTypeMode::ELEM_TYPE);

    // Update the operand of the second Dequantize operation
    origOp.getOutput().replaceAllUsesWith(clonedReshapeOp->getResult(0));

    return mlir::success();
}

//
// FuseQuantizeWithConvert
//

class FuseQuantizeWithConvert final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    using mlir::OpRewritePattern<IE::QuantizeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

// This rewriter searches for pattern:
// integer_tensor -> [Convert] -> fp_tensor -> [Quantize with Scale 1 and ZeroPoint 0] -> quantized_tensor
// and replaces it with
// integer_tensor -> [QuantizeCast] -> quantized_tensor
mlir::LogicalResult FuseQuantizeWithConvert::matchAndRewrite(IE::QuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    // Get the input of origOp
    mlir::Value origOpInput = origOp.getInput();

    // Get the defining operation of the first operand
    mlir::Operation* origOpProducer = origOpInput.getDefiningOp();

    // Check if the producer is a Convert
    if (!mlir::isa_and_nonnull<IE::ConvertOp>(origOpProducer)) {
        return mlir::failure();
    }

    mlir::Operation* convertOp = origOpProducer;

    // Check if convertOp has only one use
    if (!convertOp->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    // Get the input of the convertOp
    mlir::Value convertOpInput = convertOp->getOperand(0);

    // Get the element type of the Quantize output
    auto outputTypeQuantize = mlir::cast<mlir::ShapedType>(origOp.getType());
    auto outElemType = outputTypeQuantize.getElementType();

    // Get the element type of the Convert input
    auto inputTypeConvert = mlir::cast<mlir::ShapedType>(convertOpInput.getType());
    auto inElemType = inputTypeConvert.getElementType();

    auto outUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(outElemType);
    if (!outUniformType) {
        return mlir::failure();
    }

    auto inIntegerType = mlir::dyn_cast<mlir::IntegerType>(inElemType);
    if (!inIntegerType) {
        return mlir::failure();
    }

    if (inIntegerType.getWidth() != outUniformType.getStorageTypeIntegralWidth()) {
        return mlir::failure();
    }

    // Get the scale of the quantized type
    const auto quantizeScale = outUniformType.getScale();
    const auto quantizeZeroPoint = outUniformType.getZeroPoint();

    if (!isDoubleEqual(quantizeScale, 1.0) || quantizeZeroPoint != 0) {
        return mlir::failure();
    }

    // Set the insertion point to just after the original operation
    rewriter.setInsertionPointAfter(origOp.getOperation());
    auto newQuantizeCastOp = rewriter.create<IE::QuantizeCastOp>(origOp->getLoc(), convertOpInput, outElemType);
    origOp.getOutput().replaceAllUsesWith(newQuantizeCastOp->getResult(0));

    return mlir::success();
}

//
// getCanonicalizationPatterns
//

void vpux::IE::QuantizeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseFQsWithSimilarScales>(ctx);
    patterns.add<FuseQuantizeWithConvert>(ctx);
}
