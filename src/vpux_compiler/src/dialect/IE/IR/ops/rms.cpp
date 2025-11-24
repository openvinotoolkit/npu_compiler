//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::RMSOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::RMSOpAdaptor rms(operands, attrs, prop);
    if (mlir::failed(rms.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(rms.getInput().getType());
    const auto gammaType = mlir::cast<mlir::ShapedType>(rms.getGamma().getType());
    const auto inputRank = inType.getRank();
    const auto gammaRank = gammaType.getRank();

    const auto inputWidth = inType.getDimSize(inputRank - 1);
    const auto gammaWidth = gammaType.getDimSize(gammaRank - 1);

    if (inputWidth != gammaWidth) {
        return errorAt(loc, "Input width should be the same as gamma. Got input width = {0} and gamma width = {1}",
                       inputWidth, gammaWidth);
    }

    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType());
    return mlir::success();
}

namespace {

class FoldMulIntoRMSGamma final : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    using mlir::OpRewritePattern<IE::MultiplyOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final {
        auto input1 = mulOp.getInput1();
        auto input2 = mulOp.getInput2();
        // Identify RMS op side and splat const side
        IE::RMSOp rmsOp = input1.getDefiningOp<IE::RMSOp>();
        mlir::Value other = input2;
        if (!rmsOp) {
            rmsOp = input2.getDefiningOp<IE::RMSOp>();
            other = input1;
        }
        if (!rmsOp) {
            return mlir::failure();
        }

        // Require RMS to have single use (this mul) to avoid duplicating RMS
        if (!rmsOp->hasOneUse()) {
            return mlir::failure();
        }

        // Other input must be splat const
        auto scaleConstOp = other.getDefiningOp<Const::DeclareOp>();
        if (scaleConstOp == nullptr) {
            return mlir::failure();
        }

        auto scaleContent = scaleConstOp.getContent();
        if (!scaleContent.isSplat()) {
            return mlir::failure();
        }

        // Gamma must be const
        auto gammaConstOp = rmsOp.getGamma().getDefiningOp<Const::DeclareOp>();
        if (gammaConstOp == nullptr) {
            return mlir::failure();
        }

        // Extract scale value as double/float
        double scaleVal = static_cast<double>(scaleContent.getSplatValue<float>());

        // Create new gamma constant with rescaled content
        auto newGammaContentAttr = gammaConstOp.transformContentAttr().rescale(scaleVal).get();
        if (newGammaContentAttr == nullptr) {
            return mlir::failure();
        }
        auto newGammaConst =
                rewriter.create<Const::DeclareOp>(gammaConstOp.getLoc(), gammaConstOp.getType(), newGammaContentAttr);

        // Recreate RMS op with new gamma
        rewriter.replaceOpWithNewOp<IE::RMSOp>(mulOp, rmsOp.getInput(), newGammaConst.getOutput(),
                                               rmsOp.getEpsilonAttr());

        return mlir::success();
    }
};

}  // namespace

void vpux::IE::RMSOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<FoldMulIntoRMSGamma>(ctx);
}
