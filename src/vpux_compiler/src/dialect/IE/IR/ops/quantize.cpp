//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/utils/quantization.hpp>
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/qdq_optimization_aggressive_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

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

    const auto inType = mlir::cast<mlir::RankedTensorType>(quantize.getInput().getType());
    const auto dstElemType = quantize.getDstElemType();
    const auto outDesc = getTensorAttr(inType);

    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);
    return mlir::success();
}

//
// fold
//

namespace {

mlir::quant::QuantizedType extractQuantizedType(mlir::Value operand) {
    const auto elemType = mlir::cast<mlir::ShapedType>(operand.getType()).getElementType();
    const auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType);
    VPUX_THROW_UNLESS(quantType != nullptr, "Type must be quantized, but provided {0}", elemType);
    return quantType;
}

}  // namespace

mlir::OpFoldResult vpux::IE::QuantizeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (auto ephemeral = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
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
    // Check if adaptive stripping is enabled
    auto moduleOp = getModuleOp(origOp);
    auto setAdaptiveStrippingEnabled = VPU::hasEnableAdaptiveStripping(moduleOp);

    if (!setAdaptiveStrippingEnabled) {
        return mlir::failure();
    }

    // Get the input of origOp
    mlir::Value origOpInput = origOp.getInput();

    // Get the defining operation of the first operand
    mlir::Operation* origOpProducer = origOpInput.getDefiningOp();

    // Check if the producer is Reshape, AffineReshape, Slice or Tile
    // TODO: extend this check to other operations that attach ElemTypeInfoOpInterface
    if (!mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp, IE::SliceOp, IE::TileOp, IE::TransposeOp>(
                origOpProducer)) {
        return mlir::failure();
    }

    // Check if operation has only one use
    if (!origOpProducer->getResult(0).hasOneUse()) {
        return mlir::failure();
    }

    // Get the first operand of the operation
    mlir::Value operationInput = origOpProducer->getOperand(0);

    // Get the defining operation of the first operand
    mlir::Operation* operationProducer = operationInput.getDefiningOp();

    // Check if the producer is a Dequantize
    if (!mlir::isa_and_nonnull<IE::DequantizeOp>(operationProducer)) {
        return mlir::failure();
    }

    IE::DequantizeOp dequantizeOp = nullptr;
    dequantizeOp = mlir::dyn_cast<IE::DequantizeOp>(operationProducer);

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
    if (!areQuantizationScalesSimilar(quantizeScale, dequantizeScale)) {
        return mlir::failure();
    }

    // Set the insertion point to just after the original operation
    rewriter.setInsertionPointAfter(dequantizeOp.getInput().getDefiningOp());

    // Clone the operation
    auto* clonedReshapeOp = rewriter.clone(*origOpProducer);

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
