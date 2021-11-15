//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/IE/ops.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::CumSumOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::CumSumOpAdaptor cumsum(operands, attrs);
    if (mlir::failed(cumsum.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = cumsum.input().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();

    inferredReturnShapes.emplace_back(inShape, inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::CumSumOp> {
public:
    using mlir::OpRewritePattern<IE::CumSumOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::CumSumOp cumsumOp, mlir::PatternRewriter& rewriter) const {
    auto axis = cumsumOp.axis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = cumsumOp.axis().getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.content();
    if (!axisContent.isSplat()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::CumSumOp>(cumsumOp, cumsumOp.getType(), cumsumOp.input(), nullptr,
                                              rewriter.getI64IntegerAttr(axisContent.getSplatValue<int64_t>()),
                                              cumsumOp.exclusiveAttr(), cumsumOp.reverseAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::CumSumOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}
