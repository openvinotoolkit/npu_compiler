//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::HardSigmoidOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::HardSigmoidOpAdaptor hardSigmoid(operands, attrs);
    if (mlir::failed(hardSigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = hardSigmoid.input().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::HardSigmoidOp> {
public:
    using mlir::OpRewritePattern<IE::HardSigmoidOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::HardSigmoidOp hardSigmoidOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::HardSigmoidOp hardSigmoidOp,
                                                        mlir::PatternRewriter& rewriter) const {
    auto alpha = hardSigmoidOp.alpha();
    auto beta = hardSigmoidOp.beta();

    if ((alpha == nullptr) || (beta == nullptr)) {
        return mlir::failure();
    }

    auto alphaConst = alpha.getDefiningOp<Const::DeclareOp>();
    auto betaConst = beta.getDefiningOp<Const::DeclareOp>();

    if ((alphaConst == nullptr) || (betaConst == nullptr)) {
        return mlir::failure();
    }

    const auto alphaContent = alphaConst.content();
    const auto betaContent = betaConst.content();

    if ((!alphaContent.isSplat()) || (!betaContent.isSplat())) {
        return mlir::failure();
    }

    const auto alphaValue = alphaContent.getSplatValue<float>();
    const auto betaValue = betaContent.getSplatValue<float>();

    rewriter.replaceOpWithNewOp<IE::HardSigmoidOp>(hardSigmoidOp, hardSigmoidOp.getType(), hardSigmoidOp.input(),
                                                   nullptr, nullptr, rewriter.getF64FloatAttr(alphaValue),
                                                   rewriter.getF64FloatAttr(betaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::HardSigmoidOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
