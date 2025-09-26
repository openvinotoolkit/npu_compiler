//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/cast_utils.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::QuantizeCastOp::verify() {
    const auto dstElemType = getDstElemType();
    const auto inputElemType = mlir::cast<NDTypeInterface>(getInput().getType()).getElementType();

    if (mlir::failed(isQuantizeCastValid(getLoc(), inputElemType, dstElemType))) {
        return errorAt(getLoc(), "Unsupported quantize cast: '{0}'->'{1}'", inputElemType, dstElemType);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::QuantizeCastOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::QuantizeCastOpAdaptor quantizeCast(operands, attrs, prop);
    if (mlir::failed(quantizeCast.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(quantizeCast.getInput().getType());
    const auto inElemType = inType.getElementType();
    const auto dstElemType = quantizeCast.getDstElemType();

    if (mlir::failed(isQuantizeCastValid(loc, inElemType, dstElemType))) {
        return errorAt(loc, "Unsupported quantize cast: '{0}'->'{1}'", inElemType, dstElemType);
    }

    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), dstElemType, outDesc);

    return mlir::success();
}

mlir::OpFoldResult vpux::IE::QuantizeCastOp::fold(FoldAdaptor adaptor) {
    if (getInput().getType() == getOutput().getType()) {
        return getInput();
    } else if (const auto attr = mlir::dyn_cast_or_null<Const::ContentAttr>(adaptor.getInput())) {
        auto elemType = getDstElemTypeAttr().getValue();
        return attr.transform().castElemType(elemType).get();
    }

    return nullptr;
}

//
// FuseQuantizeCasts
//

namespace {

class FuseQuantizeCasts final : public mlir::OpRewritePattern<IE::QuantizeCastOp> {
public:
    using OpRewritePattern::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::QuantizeCastOp op, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseQuantizeCasts::matchAndRewrite(IE::QuantizeCastOp origOp,
                                                       mlir::PatternRewriter& rewriter) const {
    // Transform
    // Input type1 -> IE.QuantizeCast type2 -> IE.QuantizeCast type3 -> Output type3
    // into
    // Input type1 -> IE.QuantizeCast type3 -> Output type3
    auto producerOp = origOp.getInput().getDefiningOp<IE::QuantizeCastOp>();
    if (producerOp == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::QuantizeCastOp>(origOp, origOp.getOutput().getType(), producerOp.getInput(),
                                                    origOp.getDstElemType());

    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void IE::QuantizeCastOp::getCanonicalizationPatterns(mlir::RewritePatternSet& results, mlir::MLIRContext* ctx) {
    results.add<FuseQuantizeCasts>(ctx);
}
