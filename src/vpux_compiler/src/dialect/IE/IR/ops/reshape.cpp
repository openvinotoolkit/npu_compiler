//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/layout_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/affine_reshape.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

#include <numeric>

using namespace vpux;

//
// getOutShape
//

namespace {

mlir::FailureOr<SmallVector<int64_t>> getOutShape(IE::ReshapeOpAdaptor reshape, mlir::Location loc) {
    if (reshape.getShape() != nullptr && reshape.getShapeValue().has_value()) {
        return errorAt(loc, "Ambiguous shape representation");
    }
    if (reshape.getShape() == nullptr && !reshape.getShapeValue().has_value()) {
        return errorAt(loc, "Missed shape representation");
    }

    if (reshape.getShapeValue().has_value()) {
        return parseIntArrayAttr<int64_t>(reshape.getShapeValue().value());
    }

    auto shapeConst = reshape.getShape().getDefiningOp<Const::DeclareOp>();
    if (shapeConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for shape");
    }

    const auto shapeContent = shapeConst.getContent();
    auto shapeVec = to_small_vector(shapeContent.getValues<int64_t>());

    const auto specialZero = reshape.getSpecialZero();

    const auto zeroDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == 0;
    });
    const auto negativeDims = std::count_if(shapeVec.begin(), shapeVec.end(), [](int64_t v) {
        return v == -1;
    });

    if (negativeDims > 1) {
        return errorAt(loc, "Shape can not contain more than 1 negative value");
    }

    if (!(zeroDims != 0 && specialZero) && negativeDims == 0) {
        return shapeVec;
    } else {
        const auto inShape = to_small_vector(reshape.getInput().getType().cast<mlir::ShapedType>().getShape());

        auto dividend = std::accumulate(inShape.begin(), inShape.end(), int64_t(1), std::multiplies<int64_t>());

        for (size_t i = 0; i < shapeVec.size(); ++i) {
            auto& v = shapeVec[i];

            if (v == 0 && specialZero) {
                if (i >= inShape.size()) {
                    return errorAt(loc, "Shape value at '{0}' is out of range '{1}'", i, inShape.size());
                }

                v = inShape[i];
            }

            if (v > 0) {
                if (dividend % v != 0) {
                    return errorAt(loc, "Shape value at '{0}' ('{1}') is invalid", i, v);
                }

                dividend /= v;
            }
        }

        if (negativeDims > 0) {
            const auto negIt = std::find(shapeVec.begin(), shapeVec.end(), -1);
            VPUX_THROW_UNLESS(negIt != shapeVec.end(), "Shape vector broken");

            *negIt = dividend;
        }

        return shapeVec;
    }
}

}  // namespace

//
// inferReturnTypeComponents
//

mlir::LogicalResult vpux::IE::ReshapeOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::ReshapeOpAdaptor reshape(operands, attrs, prop);
    if (mlir::failed(reshape.verify(loc))) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(reshape, loc);
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto inType = reshape.getInput().getType().cast<mlir::RankedTensorType>();

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape->size()), vpux::getMemorySpace(inType));

    inferredReturnShapes.emplace_back(outShape.value(), inType.getElementType(), outDesc);
    return mlir::success();
}

//
// fold
//

mlir::OpFoldResult vpux::IE::ReshapeOp::fold(FoldAdaptor adaptor) {
    auto operands = adaptor.getOperands();
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    }

    VPUX_THROW_UNLESS(!operands.empty(), "Wrong number of operands : {0}", operands.size());

    if (const auto attr = operands[0].dyn_cast_or_null<Const::ContentAttr>()) {
        return static_cast<Const::ContentAttr>(attr).transform().reshape(vpux::getShape(getOutput())).get();
    }

    return nullptr;
}

//
// FuseReshapes
//

namespace {

class FuseReshapes final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseReshapes::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    auto prevOp = origOp.getInput().getDefiningOp();
    if (prevOp == nullptr) {
        return mlir::failure();
    }
    if (!mlir::isa<IE::SqueezeOp, IE::UnsqueezeOp, IE::ReshapeOp, IE::AffineReshapeOp>(prevOp)) {
        return mlir::failure();
    }

    const auto outputShape = origOp.getType().getShape();
    const auto outputShapeAttr = getIntArrayAttr(getContext(), outputShape);

    auto newOp =
            rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, prevOp->getOperand(0), nullptr, false, outputShapeAttr);
    extendOpLoc(newOp, "fused_with_other");
    return mlir::success();
}

}  // namespace

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const {
    if (origOp.getShapeValue().has_value()) {
        return mlir::failure();
    }

    const auto outShape = getOutShape(origOp, origOp->getLoc());
    if (mlir::failed(outShape)) {
        return mlir::failure();
    }

    const auto outShapeAttr = getIntArrayAttr(getContext(), outShape.value());

    rewriter.replaceOpWithNewOp<IE::ReshapeOp>(origOp, origOp.getInput(), nullptr, false, outShapeAttr);
    return mlir::success();
}

}  // namespace

//
// ConvertToAffineReshape
//

namespace {
// AffineReshapes represent a subset of Reshape ops that can have their output layout inferred from input layout.
// The condition that Reshape must satisfy to be an AffineReshape is to have a clear mapping between the dims of the
// input and output shapes. That happens when each dim of the input shape can either be decomposed into several adjacent
// dims of the output shape OR the multiplication of it with adjacent input dims results in an output dim.
// E.g. n x c x h x w -> n x c x (h * w) (AffineReshape)
//      c x h x w -> (c / k) x k x h x w (AffineReshape)
//      n x c x h x w -> n x (c * w) x h (not AffineReshape - c, w not adjacent)
//      c x h x w -> (c / k) x (h * k) x w (not AffineReshape - not an exact decomposition into dims of the
//      other shape)
//      c x h x w -> c x (h x w) x 1 x 1 (AffineReshape, for the last dims equals to 1 exception)
class ConvertToAffineReshape final : public mlir::OpRewritePattern<IE::ReshapeOp> {
public:
    using mlir::OpRewritePattern<IE::ReshapeOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::ReshapeOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToAffineReshape::matchAndRewrite(IE::ReshapeOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto outputShape = origOp.getType().getShape();
    const auto outShapeAttr = getIntArrayAttr(getContext(), outputShape);

    const auto inShape = origOp.getInput().getType().cast<mlir::ShapedType>().getShape();
    const auto reassociationMap = vpux::IE::getReassociationMap(inShape, outputShape);
    if (mlir::failed(reassociationMap)) {
        return mlir::failure();
    }

    // If no valid output layout can be inferred, don't replace with AffineReshape
    auto inOrder = DimsOrder::fromValue(origOp.getInput());
    const auto outputLayout = Const::inferAffineReshapeOutputLayout(
            inOrder.toPermutation(), getIntArrayOfArray(getContext(), reassociationMap.value()));
    if (!outputLayout.has_value()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(
            origOp, origOp.getInput(), getIntArrayOfArray(getContext(), reassociationMap.value()), outShapeAttr);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FuseReshapes>(ctx);
    patterns.add<ConvertConstToAttr>(ctx);
    patterns.add<ConvertToAffineReshape>(ctx);
}
