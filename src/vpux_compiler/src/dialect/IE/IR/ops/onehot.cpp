//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::OneHotOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = mlir::cast<vpux::NDTypeInterface>(tensor.getType()).getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getDepth())) {
        return errorAt(*this, "Depth should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getOnValue())) {
        return errorAt(*this, "on_value should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getOffValue())) {
        return errorAt(*this, "off_value should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::FailureOr<int64_t> extractDepth(mlir::Location loc, const mlir::Value& depth, mlir::IntegerAttr depthAttr) {
    if (depthAttr != nullptr) {
        return depthAttr.getInt();
    } else if (depth != nullptr) {
        auto depthValue = Const::getSplatValue<int64_t>(depth);
        if (mlir::failed(depthValue)) {
            return errorAt(loc, "OneHot depth must be a const scalar");
        }
        return depthValue;
    }

    return errorAt(loc, "depth is not provided");
}

mlir::LogicalResult vpux::IE::OneHotOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));
    IE::OneHotOpAdaptor oneHot(operands, attrs, prop);

    if (mlir::failed(oneHot.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(oneHot.getInput().getType());
    const auto outElemType = oneHot.getOutputType();

    auto outShape = to_small_vector(inType.getShape());
    const auto axis = oneHot.getAxisAttr();
    auto depth = extractDepth(loc, oneHot.getDepth(), oneHot.getDepthAttrAttr());
    if (mlir::failed(depth)) {
        return mlir::failure();
    }
    int64_t depthVal = checked_cast<int64_t>(*depth);

    if (oneHot.getRankExpanded().value_or(false)) {
        // rank_expanded mode: input rank == output rank.
        // The axis dim of the input is a placeholder 1; replace it with depth.
        const auto axisIdx = axis < 0 ? static_cast<int64_t>(outShape.size()) + axis : axis;
        outShape[axisIdx] = depthVal;
    } else {
        if (axis < 0) {
            outShape.insert(outShape.end() + 1 + axis, depthVal);
        } else {
            outShape.insert(outShape.begin() + axis, depthVal);
        }
    }

    inferredReturnShapes.emplace_back(outShape, outElemType);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::OneHotOp> {
public:
    using mlir::OpRewritePattern<IE::OneHotOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const {
    auto depth = oneHotOp.getDepth();
    auto onValue = oneHotOp.getOnValue();
    auto offValue = oneHotOp.getOffValue();

    if ((depth == nullptr) || (onValue == nullptr) || (offValue == nullptr)) {
        return mlir::failure();
    }

    auto depthConst = depth.getDefiningOp<Const::DeclareOp>();
    auto onValueConst = onValue.getDefiningOp<Const::DeclareOp>();
    auto offValueConst = offValue.getDefiningOp<Const::DeclareOp>();

    const auto isSplat = [](Const::DeclareOp op) {
        return (op != nullptr) && op.getContentAttr().isSplat();
    };

    if (!isSplat(depthConst) || !isSplat(onValueConst) || !isSplat(offValueConst)) {
        return mlir::failure();
    }

    const auto depthContent = depthConst.getContent();
    const auto onValueContent = onValueConst.getContent();
    const auto offValueContent = offValueConst.getContent();

    const auto depthAttrValue = depthContent.getSplatValue<int64_t>();
    const auto onValueAttrValue = onValueContent.getSplatValue<float>();
    const auto offValueAttrValue = offValueContent.getSplatValue<float>();

    rewriter.replaceOpWithNewOp<IE::OneHotOp>(
            oneHotOp, oneHotOp.getType(), oneHotOp.getInput(), nullptr, nullptr, nullptr,
            rewriter.getI64IntegerAttr(depthAttrValue), rewriter.getF64FloatAttr(onValueAttrValue),
            rewriter.getF64FloatAttr(offValueAttrValue), oneHotOp.getAxisAttr(), oneHotOp.getModeAttr(),
            oneHotOp.getRankExpandedAttr(), oneHotOp.getOutputType());

    return mlir::success();
}

//
//
// NormalizeAxis
//

class NormalizeAxis final : public mlir::OpRewritePattern<IE::OneHotOp> {
public:
    using mlir::OpRewritePattern<IE::OneHotOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult NormalizeAxis::matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const {
    const auto axis = oneHotOp.getAxisAttr();
    if (axis >= 0) {
        return mlir::failure();
    }

    const auto outRank =
            static_cast<int64_t>(mlir::cast<vpux::NDTypeInterface>(oneHotOp.getOutput().getType()).getRank());
    const auto normalizedAxis = getIntAttr(rewriter.getContext(), outRank + axis);

    rewriter.replaceOpWithNewOp<IE::OneHotOp>(oneHotOp, oneHotOp.getType(), oneHotOp.getInput(), oneHotOp.getDepth(),
                                              oneHotOp.getOnValue(), oneHotOp.getOffValue(),
                                              oneHotOp.getDepthAttrAttr(), oneHotOp.getOnValueAttrAttr(),
                                              oneHotOp.getOffValueAttrAttr(), normalizedAxis, oneHotOp.getModeAttr(),
                                              oneHotOp.getRankExpandedAttr(), oneHotOp.getOutputTypeAttr());

    return mlir::success();
}

//
// ExpandInputRank
//

class ExpandInputRank final : public mlir::OpRewritePattern<IE::OneHotOp> {
public:
    using mlir::OpRewritePattern<IE::OneHotOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ExpandInputRank::matchAndRewrite(IE::OneHotOp oneHotOp, mlir::PatternRewriter& rewriter) const {
    // Already expanded.
    if (oneHotOp.getRankExpanded().value_or(false)) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(oneHotOp.getInput().getType());
    const auto outType = mlir::cast<vpux::NDTypeInterface>(oneHotOp.getOutput().getType());
    if (inType.getRank() == outType.getRank()) {
        return mlir::failure();
    }

    // Insert a size-1 dimension at the axis position so that input rank == output rank.
    auto inShape = to_small_vector(inType.getShape());
    const auto axis = oneHotOp.getAxisAttr();
    inShape.insert(inShape.begin() + axis, int64_t(1));

    const auto newInShapeAttr = getIntArrayAttr(rewriter.getContext(), inShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(oneHotOp->getLoc(), oneHotOp.getInput(), newInShapeAttr);

    rewriter.replaceOpWithNewOp<IE::OneHotOp>(
            oneHotOp, oneHotOp.getType(), inputReshape, oneHotOp.getDepth(), oneHotOp.getOnValue(),
            oneHotOp.getOffValue(), oneHotOp.getDepthAttrAttr(), oneHotOp.getOnValueAttrAttr(),
            oneHotOp.getOffValueAttrAttr(), oneHotOp.getAxisAttr(), oneHotOp.getModeAttr(),
            /*rank_expanded=*/mlir::BoolAttr::get(rewriter.getContext(), true), oneHotOp.getOutputType());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::OneHotOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<NormalizeAxis>(context);
    patterns.add<ExpandInputRank>(context);
}
