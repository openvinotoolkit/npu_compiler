//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::TopKOp::verify() {
    if (!getK()) {
        return mlir::success();
    }

    auto kNumElements = getK().getType().cast<vpux::NDTypeInterface>().getNumElements();
    if (kNumElements != 1) {
        return errorAt(*this, "K should have only 1 element, while it has {0}", kNumElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::TopKOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::TopKOpAdaptor topK(operands, attrs, prop);
    if (mlir::failed(topK.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = topK.getInput().getType().cast<mlir::ShapedType>();
    const auto inputShape = inType.getShape();

    const auto kValue = getConstOrAttrValue(topK.getK(), topK.getKValueAttr());

    if (mlir::failed(kValue)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
        outShape.push_back(inputShape[i]);
    }
    int64_t axis = topK.getAxis();
    const auto inRank = inType.getRank();
    if (axis < 0) {
        axis += inRank;
    }

    outShape[axis] = kValue.value();

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());
    inferredReturnShapes.emplace_back(outShape, topK.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::TopKOp> {
public:
    using mlir::OpRewritePattern<IE::TopKOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::TopKOp topKOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::TopKOp topKOp, mlir::PatternRewriter& rewriter) const {
    if (topKOp.getKValue()) {
        return mlir::failure();
    }

    const auto kValue = getConstOrAttrValue(topKOp.getK(), topKOp.getKValueAttr());

    if (mlir::failed(kValue)) {
        return mlir::failure();
    }

    const auto kValueAttr = getIntAttr(rewriter.getContext(), kValue.value());

    rewriter.replaceOpWithNewOp<IE::TopKOp>(topKOp, topKOp.getInput(), nullptr, kValueAttr, topKOp.getAxis(),
                                            topKOp.getMode(), topKOp.getSort(), topKOp.getElementType());

    return mlir::success();
}

}  // namespace

void vpux::IE::TopKOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
