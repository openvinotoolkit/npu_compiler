//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::TopKOp::verify() {
    if (!getK()) {
        return mlir::success();
    }

    auto kNumElements = mlir::cast<vpux::NDTypeInterface>(getK().getType()).getNumElements();
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

    const auto inType = mlir::cast<vpux::NDTypeInterface>(topK.getInput().getType());
    const auto inShapeInfo = ShapeInfo::fromNDType(inType);

    const auto outInfo = inferTopKOutputShapeInfo(inShapeInfo, topK.getAxis(), topK.getK(), topK.getKValueAttr(), loc);
    if (mlir::failed(outInfo)) {
        return mlir::failure();
    }

    const auto& outShape = outInfo->shape;
    const mlir::Attribute outTensorAttr =
            vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), BoundsRef(outInfo->bounds));

    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outTensorAttr);
    inferredReturnShapes.emplace_back(outShape, topK.getElementType(), outTensorAttr);

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

    const auto kValue = Const::getConstOrAttrValue(topKOp.getK(), topKOp.getKValueAttr());

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
