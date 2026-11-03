//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

//
// verify
//

mlir::LogicalResult vpux::IE::HardSigmoidOp::verify() {
    int64_t numElements = 0;
    const auto checkNumElements = [&](mlir::Value tensor) {
        if (tensor == nullptr) {
            return true;
        }

        numElements = mlir::cast<vpux::NDTypeInterface>(tensor.getType()).getNumElements();
        return numElements == 1;
    };

    if (!checkNumElements(getAlpha())) {
        return errorAt(*this, "Alpha should have only 1 element, while it has {0}", numElements);
    }

    if (!checkNumElements(getBeta())) {
        return errorAt(*this, "Beta should have only 1 element, while it has {0}", numElements);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::HardSigmoidOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::HardSigmoidOpAdaptor hardSigmoid(operands, attrs, prop);
    if (mlir::failed(hardSigmoid.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::RankedTensorType>(hardSigmoid.getInput().getType());
    const auto outDesc = vpux::getTensorAttr(inType);
    inferredReturnShapes.emplace_back(inType.getShape(), inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::HardSigmoidOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                               mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    reifiedReturnShapes.emplace_back(reifyTrivialTensor(builder, getInput(), getLoc()));
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
    auto alpha = hardSigmoidOp.getAlpha();
    auto beta = hardSigmoidOp.getBeta();

    if ((alpha == nullptr) || (beta == nullptr)) {
        return mlir::failure();
    }

    auto alphaConst = alpha.getDefiningOp<Const::DeclareOp>();
    auto betaConst = beta.getDefiningOp<Const::DeclareOp>();

    const auto isSplat = [](Const::DeclareOp op) {
        return (op != nullptr) && op.getContentAttr().isSplat();
    };
    if (!isSplat(alphaConst) || !isSplat(betaConst)) {
        return mlir::failure();
    }

    const auto alphaContent = alphaConst.getContent();
    const auto alphaValue = alphaContent.getSplatValue<float>();

    const auto betaContent = betaConst.getContent();
    const auto betaValue = betaContent.getSplatValue<float>();

    rewriter.replaceOpWithNewOp<IE::HardSigmoidOp>(hardSigmoidOp, hardSigmoidOp.getType(), hardSigmoidOp.getInput(),
                                                   nullptr, nullptr, rewriter.getF64FloatAttr(alphaValue),
                                                   rewriter.getF64FloatAttr(betaValue));

    return mlir::success();
}

}  // namespace

void vpux::IE::HardSigmoidOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                          mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
