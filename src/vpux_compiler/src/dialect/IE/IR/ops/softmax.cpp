//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

void vpux::IE::SoftMaxOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Value input,
                                mlir::IntegerAttr axisInd, mlir::IntegerAttr padSize) {
    build(odsBuilder, odsState, input, axisInd, padSize, nullptr, {});
}

void vpux::IE::SoftMaxOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState, mlir::Type output,
                                mlir::Value input, mlir::IntegerAttr axisInd, mlir::IntegerAttr padSize) {
    build(odsBuilder, odsState, output, input, axisInd, padSize, nullptr, {});
}

mlir::LogicalResult vpux::IE::SoftMaxOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SoftMaxOpAdaptor softMax(operands, attrs, prop);
    if (mlir::failed(softMax.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(softMax.getInput().getType());
    const auto dstElemType = softMax.getDstElemType();

    auto elemType = dstElemType.value_or(inType.getElementType());

    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), elemType, outDesc);

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::SoftMaxOp::fold(FoldAdaptor) {
    const auto inType = mlir::cast<mlir::ShapedType>(getInput().getType());
    const auto inShape = inType.getShape();
    const auto inRank = inType.getRank();

    auto axis = checked_cast<int64_t>(getAxisInd());

    if (axis < 0) {
        axis += inRank;
    }

    VPUX_THROW_UNLESS(axis >= 0 && axis < inRank, "Wrong SoftMax axis {0}", axis);

    // Avoid folding to Constant with dynamic shape
    if (hasDynamicTensors(getOperation())) {
        return nullptr;
    }

    if (inShape[axis] > 1) {
        return nullptr;
    }

    const auto valueType = mlir::RankedTensorType::get(inShape, mlir::Float32Type::get(getContext()));
    auto attr = mlir::DenseElementsAttr::get(valueType, 1.0f);
    return Const::ContentAttr::get(
            attr, Const::ContentSetup(attr, valueType)
                          .castElemType(mlir::cast<mlir::ShapedType>(getOutput().getType()).getElementType()));
}

mlir::LogicalResult vpux::IE::SoftMaxOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                           mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
    return mlir::success();
}

//
// LegalizeAxisInd
//

namespace {

class LegalizeAxisInd final : public mlir::OpRewritePattern<IE::SoftMaxOp> {
public:
    using mlir::OpRewritePattern<IE::SoftMaxOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SoftMaxOp softmaxOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult LegalizeAxisInd::matchAndRewrite(IE::SoftMaxOp softmaxOp, mlir::PatternRewriter& rewriter) const {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(softmaxOp.getInput().getType());
    int64_t axis = softmaxOp.getAxisInd();

    if (axis >= 0) {
        return mlir::failure();
    }

    int64_t legalizeAxis = vpux::getPositiveAxisInd(softmaxOp.getAxisIndAttr(), inputType.getRank());
    const auto legalizeAxisAttr = getIntAttr(rewriter.getContext(), legalizeAxis);

    rewriter.replaceOpWithNewOp<IE::SoftMaxOp>(softmaxOp, softmaxOp.getInput(), legalizeAxisAttr, nullptr,
                                               softmaxOp.getDstElemTypeAttr(), softmaxOp.getMaskAwareAttr());
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SoftMaxOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<LegalizeAxisInd>(context);
}
