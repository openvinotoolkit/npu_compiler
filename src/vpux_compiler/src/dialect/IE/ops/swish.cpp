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
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/to_ngraph.hpp"

#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::SwishOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::SwishOpAdaptor swish(operands, attrs);
    if (mlir::failed(swish.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = swish.input().getType().cast<mlir::ShapedType>();

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::SwishOp> {
public:
    using mlir::OpRewritePattern<IE::SwishOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::SwishOp swishOp, mlir::PatternRewriter& rewriter) const {
    if (swishOp.beta_value()) {
        return mlir::failure();
    }

    float betaValue = 1.0;

    if (auto beta = swishOp.beta()) {
        auto betaOp = beta.getDefiningOp<Const::DeclareOp>();
        if (betaOp == nullptr) {
            return mlir::failure();
        }

        const auto betaContent = betaOp.content();
        if (!betaContent.isSplat()) {
            return mlir::failure();
        }

        betaValue = betaContent.getSplatValue<float>();
    }

    rewriter.replaceOpWithNewOp<IE::SwishOp>(swishOp, swishOp.getType(), swishOp.input(), nullptr,
                                             rewriter.getF64FloatAttr(betaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::SwishOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertConstToAttr>(context);
}

std::unique_ptr<ngraph::Node> vpux::IE::SwishOp::toNgraph(ngraph::OutputVector &outputs)
{
    return std::make_unique<opset_latest::Swish>(outputs.at(0));
}
