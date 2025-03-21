//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/core/IR/attributes.hpp"

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::UnsqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::UnsqueezeOpAdaptor unsqueeze(operands, attrs, prop);
    if (mlir::failed(unsqueeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = IE::getAxes(unsqueeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = unsqueeze.getInput();
    const auto inType = input.getType().cast<mlir::RankedTensorType>();
    const auto inShape = inType.getShape();
    const auto inOrder = DimsOrder::fromValue(input);

    const auto outShape = propagateShape(loc, inShape, axes.value());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto outBounds = IE::propagateBoundsAttr(ctx, loc, input, axes.value());
    if (mlir::failed(outBounds)) {
        return mlir::failure();
    }

    const auto outDesc = vpux::getTensorAttr(
            ctx, vpux::VPU::inferUnsqueezeOutputLayout(inOrder.toPermutation(), axes.value(), inShape),
            vpux::getMemorySpace(inType), outBounds.value());

    inferredReturnShapes.emplace_back(ArrayRef(outShape.value()), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::UnsqueezeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseWithReshape
//

namespace {

class FuseWithReshape final : public mlir::OpRewritePattern<IE::UnsqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::UnsqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::UnsqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::UnsqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getAxesValue().has_value()) {
        return mlir::failure();
    }

    const auto axes = IE::getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.value());

    rewriter.replaceOpWithNewOp<IE::UnsqueezeOp>(origOp, origOp.getInput(), nullptr, axesAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::UnsqueezeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<FuseWithReshape>(context);
    patterns.add<ConvertConstToAttr>(context);
}
