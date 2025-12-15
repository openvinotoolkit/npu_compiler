//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::MaxPool8Op::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MaxPool8OpAdaptor maxPool8(operands, attrs, prop);
    if (mlir::failed(maxPool8.verify(loc))) {
        return mlir::failure();
    }

    const auto dataPaddingBelow = parseIntArrayAttr<int64_t>(maxPool8.getPadsEnd());
    const auto dataPaddingAbove = parseIntArrayAttr<int64_t>(maxPool8.getPadsBegin());
    const auto windowShape = parseIntArrayAttr<int64_t>(maxPool8.getKernelSize());
    const auto windowStrides = parseIntArrayAttr<int64_t>(maxPool8.getStrides());
    const auto windowDilations = parseIntArrayAttr<int64_t>(maxPool8.getDilations());
    const auto roundingType = maxPool8.getRoundingType();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(maxPool8.getInput().getType());
    const auto inType = inputType.getElementType();
    const auto inShape = ShapeInfo::fromNDType(inputType);

    auto outputShape = inferMaxPool8OutputShape(inShape, windowStrides, windowDilations, dataPaddingBelow,
                                                dataPaddingAbove, windowShape, roundingType);

    inferredReturnShapes.emplace_back(outputShape.shape, inType);
    inferredReturnShapes.emplace_back(outputShape.shape, maxPool8.getIndexElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::MaxPool8Op::verify() {
    const auto inRank = mlir::cast<mlir::ShapedType>(getInput().getType()).getRank();
    auto axis = getAxis();

    axis = axis < 0 ? axis + inRank : axis;

    if (axis >= 0 && axis < inRank) {
        return mlir::success();
    }

    return mlir::failure();
}

//
// Canonicalizer
//
namespace {
class NormalizeAxisToPositive final : public mlir::OpRewritePattern<IE::MaxPool8Op> {
public:
    using mlir::OpRewritePattern<IE::MaxPool8Op>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult NormalizeAxisToPositive::matchAndRewrite(IE::MaxPool8Op origOp, mlir::PatternRewriter&) const {
    auto axis = origOp.getAxis();
    if (axis < 0) {
        axis += origOp.getInput().getType().getRank();
        origOp.setAxis(axis);
    } else {
        return mlir::failure();
    }

    return mlir::success();
}

}  // namespace

void vpux::IE::MaxPool8Op::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<NormalizeAxisToPositive>(context);
}
