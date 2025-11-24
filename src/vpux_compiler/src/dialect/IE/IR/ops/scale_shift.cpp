//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

//
// FuseScaleAndBias
//

class FuseScaleAndBias final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseScaleAndBias::matchAndRewrite(IE::ScaleShiftOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto C = Dim(1);

    if (!biasOp.getInput().hasOneUse()) {
        return mlir::failure();
    }

    if (biasOp.getWeights() != nullptr) {
        return mlir::failure();
    }

    auto scaleOp = mlir::dyn_cast_or_null<IE::ScaleShiftOp>(biasOp.getInput().getDefiningOp());
    if (scaleOp == nullptr || scaleOp.getBiases() != nullptr) {
        return mlir::failure();
    }

    auto mulOutShape = getShape(scaleOp.getOutput());
    auto weightsShape = getShape(scaleOp.getWeights());
    auto biasShape = getShape(biasOp.getBiases());

    if (mulOutShape.size() != 4) {
        return mlir::failure();
    }
    if (biasShape[C] != weightsShape[C]) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), scaleOp.getInput(), scaleOp.getWeights(),
                                                  biasOp.getBiases());

    return mlir::success();
}

//
// FuseScaleShifts
//

class FuseScaleShifts final : public mlir::OpRewritePattern<IE::ScaleShiftOp> {
public:
    using mlir::OpRewritePattern<IE::ScaleShiftOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ScaleShiftOp scaleShiftOp, mlir::PatternRewriter& rewriter) const final;
};

// Fuse two ScaleShift operations with const weights and biases of the same shape
// TODO: E#143426 Extend this conversion for more generic cases
mlir::LogicalResult FuseScaleShifts::matchAndRewrite(IE::ScaleShiftOp scaleShiftOp,
                                                     mlir::PatternRewriter& rewriter) const {
    if (!scaleShiftOp.getInput().hasOneUse()) {
        return mlir::failure();
    }

    auto inScaleShiftOp = mlir::dyn_cast_or_null<IE::ScaleShiftOp>(scaleShiftOp.getInput().getDefiningOp());
    if (inScaleShiftOp == nullptr) {
        return mlir::failure();
    }

    auto origWeights = scaleShiftOp.getWeights();
    auto origBiases = scaleShiftOp.getBiases();
    auto inputWeights = inScaleShiftOp.getWeights();
    auto inputBiases = inScaleShiftOp.getBiases();

    if (origWeights == nullptr || origBiases == nullptr || inputWeights == nullptr || inputBiases == nullptr) {
        return mlir::failure();
    }

    auto weightsShape = getShape(origWeights);
    auto biasesShape = getShape(origBiases);
    auto inWeightsShape = getShape(inputWeights);
    auto inBiasesShape = getShape(inputBiases);

    if (weightsShape != inWeightsShape || biasesShape != inBiasesShape || weightsShape != biasesShape) {
        return mlir::failure();
    }

    auto origWeightsConst = origWeights.getDefiningOp<Const::DeclareOp>();
    auto origBiasesConst = origBiases.getDefiningOp<Const::DeclareOp>();
    auto inputWeightsConst = inputWeights.getDefiningOp<Const::DeclareOp>();
    auto inputBiasesConst = inputBiases.getDefiningOp<Const::DeclareOp>();

    if (origWeightsConst == nullptr || origBiasesConst == nullptr || inputWeightsConst == nullptr ||
        inputBiasesConst == nullptr) {
        return mlir::failure();
    }

    auto origWeightsVals = IE::getConst(origWeightsConst);
    auto origBiasesVals = IE::getConst(origBiasesConst);
    auto inWeightsVals = IE::getConst(inputWeightsConst);
    auto inBiasesVals = IE::getConst(inputBiasesConst);
    SmallVector<float> newWeightsVec(origWeightsVals.size()), newBiasesVec(origBiasesVals.size());

    // ScaleShift_1(x) = x * weights_1 + biases_1
    // ScaleShift_2(x) = ScaleShift_1(x) * weights_2 + biases_2
    //
    // ScaleShift_result(x) = x * weights_1 * weights_2 + biases_1 * weights_2 + biases_2,
    // where: weights_result = weights_1 * weights_2; biases_result = biases_1 * weights_2 + biases_2
    for (size_t idx = 0; idx < newWeightsVec.size(); idx++) {
        newWeightsVec[idx] = origWeightsVals[idx] * inWeightsVals[idx];
        newBiasesVec[idx] = inBiasesVals[idx] * origWeightsVals[idx] + origBiasesVals[idx];
    }

    auto getContentAttr = [&](ArrayRef<float> values, mlir::Type orgType) {
        // Store the content in FP32 element type precision for more accurate representation and cast back to the
        // original constant element type
        auto fp32ElemType = mlir::Float32Type::get(rewriter.getContext());
        auto ndOrgType = mlir::cast<vpux::NDTypeInterface>(orgType);
        auto baseElemType = ndOrgType.getElementType();
        auto baseType = mlir::cast<mlir::RankedTensorType>(ndOrgType.changeElemType(fp32ElemType));
        auto newAttr = vpux::Const::createConstContent(baseType, values);
        auto newContentAttr = Const::ContentAttr::get(newAttr);
        if (baseElemType != fp32ElemType) {
            newContentAttr = newContentAttr.transform().castElemType(baseElemType).get();
        }
        return newContentAttr;
    };

    auto newWeightsContentAttr = getContentAttr(newWeightsVec, origWeightsConst.getType());
    auto newWeightsOp = rewriter.replaceOpWithNewOp<Const::DeclareOp>(origWeightsConst, origWeightsConst.getType(),
                                                                      std::move(newWeightsContentAttr));

    auto newBiasesContentAttr = getContentAttr(newBiasesVec, origBiasesConst.getType());
    auto newBiasesOp = rewriter.replaceOpWithNewOp<Const::DeclareOp>(origBiasesConst, origBiasesConst.getType(),
                                                                     std::move(newBiasesContentAttr));

    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(scaleShiftOp, scaleShiftOp.getType(), inScaleShiftOp.getInput(),
                                                  newWeightsOp, newBiasesOp);

    return mlir::success();
}

}  // namespace

mlir::LogicalResult vpux::IE::ScaleShiftOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ScaleShiftOpAdaptor scaleShift(operands, attrs, prop);
    if (mlir::failed(scaleShift.verify(loc))) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(scaleShift.getInput().getType());
    auto inShapeInfo = ShapeInfo::fromNDType(inputType);
    const auto outDesc =
            vpux::getTensorAttr(ctx, inputType.getDimsOrder(), /*memSpace=*/nullptr, BoundsRef(inShapeInfo.bounds));
    inferredReturnShapes.emplace_back(inShapeInfo.shape, inputType.getElementType(), outDesc);

    return mlir::success();
}

void vpux::IE::ScaleShiftOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                         mlir::MLIRContext* context) {
    patterns.add<FuseScaleAndBias>(context);
    patterns.add<FuseScaleShifts>(context);
}
