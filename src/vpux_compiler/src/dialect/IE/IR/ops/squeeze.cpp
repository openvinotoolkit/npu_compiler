//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// getAxes
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getAxes(IE::SqueezeOpAdaptor squeeze, mlir::Location loc) {
    if (squeeze.getAxes() != nullptr && squeeze.getAxesValue().has_value()) {
        return errorAt(loc, "Ambiguous axes representation");
    }
    if (squeeze.getAxes() == nullptr && !squeeze.getAxesValue().has_value()) {
        return SmallVector<int64_t>();
    }

    if (squeeze.getAxesValue().has_value()) {
        return parseIntArrayAttr<int64_t>(squeeze.getAxesValue().value());
    }

    auto axesConst = squeeze.getAxes().getDefiningOp<Const::DeclareOp>();
    if (axesConst == nullptr) {
        return errorAt(loc, "Only constant axes are supported");
    }

    const auto axesContent = axesConst.getContent();
    auto axes = to_small_vector(axesContent.getValues<int64_t>());
    std::sort(axes.begin(), axes.end());

    const auto inType = mlir::cast<mlir::ShapedType>(squeeze.getInput().getType());
    const auto inRank = inType.getRank();

    for (auto& axis : axes) {
        if (axis < 0) {
            axis += inRank;
        }
    }

    return axes;
}
}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::SqueezeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::SqueezeOpAdaptor squeeze(operands, attrs, prop);
    if (mlir::failed(squeeze.verify(loc))) {
        return mlir::failure();
    }

    const auto axes = ::getAxes(squeeze, loc);
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto input = squeeze.getInput();
    const auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto inShape = inType.getShape();
    const auto inOrder = DimsOrder::fromValue(input);

    SmallVector<int64_t> outShape;

    if (axes->empty()) {
        for (auto dim : inShape) {
            if (dim != 1) {
                outShape.push_back(dim);
            }
        }
        if (outShape.empty()) {
            outShape.push_back(1);
        }
    } else {
        size_t axesInd = 0;
        for (auto inInd : irange(inShape.size())) {
            if (axesInd < axes->size()) {
                const auto nextAxisInd = checked_cast<size_t>(axes.value()[axesInd]);

                if (nextAxisInd < inInd) {
                    return errorAt(loc, "Axis '{0}' was occurred twice", nextAxisInd);
                }

                if (nextAxisInd == inInd) {
                    if (inShape[inInd] != 1) {
                        return errorAt(loc, "Can't exclude '{0}' dimension, it is not equal to 1", nextAxisInd);
                    }

                    ++axesInd;

                    continue;
                }
            }

            outShape.push_back(inShape[inInd]);
        }
    }
    VPUX_THROW_WHEN(mlir::isa<Core::BoundedTensorType>(inType), "{0} doesn't support dynamic shapes",
                    IE::SqueezeOp::getOperationName());
    const auto outDesc = vpux::getTensorAttr(
            ctx, vpux::VPU::inferSqueezeOutputLayout(inOrder.toPermutation(), axes.value(), inShape),
            vpux::getMemorySpace(inType));

    inferredReturnShapes.emplace_back(ArrayRef(outShape), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::SqueezeOp::fold(FoldAdaptor adaptor) {
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

class FuseWithReshape final : public mlir::OpRewritePattern<IE::SqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::SqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseWithReshape::matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }
    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);
    auto reshapeOp = IE::ReshapeOp::create(rewriter, origOp->getLoc(), prevOp->getOperand(0), outputShapeAttr);
    auto outLink = reshapeOp.getOutput();
    auto currentOutType = mlir::dyn_cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto newOutType = mlir::dyn_cast<vpux::NDTypeInterface>(reshapeOp.getOutput().getType());
    auto currentOutDimOrder = currentOutType.getDimsOrder();
    auto newOutDimOrder = newOutType.getDimsOrder();
    // Reshape op is not aware of Layout. Add layout if necessary.
    if (currentOutDimOrder != newOutDimOrder) {
        auto permutation = getPermutationFromOrders(newOutDimOrder, currentOutDimOrder, origOp.getContext());
        auto permuteCast = IE::PermuteCastOp::create(
                rewriter, origOp->getLoc(), reshapeOp.getOutput(),
                mlir::AffineMapAttr::get(DimsOrder::fromValue(origOp.getOutput()).toAffineMap(origOp.getContext())),
                mlir::AffineMapAttr::get(permutation));
        outLink = permuteCast.getOutput();
    }
    rewriter.replaceOp(origOp, outLink);

    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SqueezeOp> {
public:
    using mlir::OpRewritePattern<IE::SqueezeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getAxesValue().has_value()) {
        return mlir::failure();
    }

    const auto axes = ::getAxes(origOp, origOp->getLoc());
    if (mlir::failed(axes)) {
        return mlir::failure();
    }

    const auto axesAttr = getIntArrayAttr(getContext(), axes.value());

    rewriter.replaceOpWithNewOp<IE::SqueezeOp>(origOp, origOp.getInput(), nullptr, axesAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::SqueezeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<FuseWithReshape>(context);
    patterns.add<ConvertConstToAttr>(context);
}
