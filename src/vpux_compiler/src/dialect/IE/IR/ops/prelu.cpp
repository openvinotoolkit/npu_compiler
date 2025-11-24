//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::PReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::PReluOpAdaptor prelu(operands, attrs, prop);
    if (mlir::failed(prelu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(prelu.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::LeakyReluOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::LeakyReluOpAdaptor leaky_relu(operands, attrs, prop);
    if (mlir::failed(leaky_relu.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(leaky_relu.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

namespace {

class UseLeakyRelu final : public mlir::OpRewritePattern<IE::PReluOp> {
public:
    using mlir::OpRewritePattern<IE::PReluOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult UseLeakyRelu::matchAndRewrite(IE::PReluOp origOp, mlir::PatternRewriter& rewriter) const {
    auto negativeSlopeOp = origOp.getNegativeSlope().getDefiningOp<Const::DeclareOp>();
    if (negativeSlopeOp == nullptr || !negativeSlopeOp.getContentAttr().isSplat()) {
        return mlir::failure();
    }

    const auto negativeSlopeContent = negativeSlopeOp.getContent();
    rewriter.replaceOpWithNewOp<IE::LeakyReluOp>(origOp, origOp.getType(), origOp.getInput(),
                                                 rewriter.getF64FloatAttr(negativeSlopeContent.getSplatValue<float>()));

    return mlir::success();
}

}  // namespace

void vpux::IE::PReluOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<UseLeakyRelu>(context);
}
