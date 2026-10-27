//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/unsqueeze.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"

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
    const auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto inOrder = DimsOrder::fromValue(input);
    const auto outOrder = VPU::inferUnsqueezeOutputLayout(inOrder.toPermutation(), axes.value(), inType.getShape());

    auto outShapeOrFail =
            callOnShapeOf(inType, [&](const auto& inShape) -> mlir::FailureOr<std::pair<Shape, TensorAttr>> {
                const auto outAnyShape = unsqueezeShape(loc, inShape, *axes);
                if (mlir::failed(outAnyShape)) {
                    return mlir::failure();
                }

                auto outShape = extractShape(*outAnyShape);
                const auto outDesc =
                        getTensorAttr(inType, outOrder.toAffineMap(ctx), getMemorySpace(inType), *outAnyShape);
                return std::make_pair(std::move(outShape), outDesc);
            });

    if (mlir::failed(outShapeOrFail)) {
        return mlir::failure();
    }
    const auto [outShape, outDesc] = std::move(*outShapeOrFail);

    inferredReturnShapes.emplace_back(outShape.raw(), inType.getElementType(), outDesc);
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

    VPUX_THROW_WHEN(operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(operands[0])) {
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

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), outputShapeAttr);
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
