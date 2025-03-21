//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::FullyConnectedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FullyConnectedOpAdaptor fullyConnected(operands, attrs, prop);
    if (mlir::failed(fullyConnected.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = fullyConnected.getInput().getType().cast<mlir::ShapedType>();
    const auto weightsType = fullyConnected.getWeights().getType().cast<mlir::ShapedType>();
    const auto inShape = inType.getShape();
    const auto weightsShape = weightsType.getShape();
    const auto inRank = inShape.size();
    const auto weightsRank = weightsShape.size();

    if (weightsRank != 2 || inRank != 2) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    outShape.push_back(inShape[0]);
    outShape.push_back(weightsShape[0]);

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FullyConnectedOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    SmallVector<mlir::OpFoldResult> outShape;
    outShape.push_back(IE::reifyDim(builder, getInput(), 0, loc));
    outShape.push_back(IE::reifyDim(builder, getWeights(), 0, loc));

    reifiedReturnShapes.emplace_back(std::move(outShape));
    return mlir::success();
}

//
// FuseFCAndBias
//

namespace {

class FuseFCAndBias final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    using mlir::OpRewritePattern<IE::AddOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult FuseFCAndBias::matchAndRewrite(IE::AddOp biasOp, mlir::PatternRewriter& rewriter) const {
    static const auto N = Dim(0);
    static const auto C = Dim(1);

    if (!biasOp.getInput1().hasOneUse()) {
        return mlir::failure();
    }

    if (mlir::failed(IE::getConstParentOp(biasOp.getInput2()))) {
        return mlir::failure();
    }

    auto fullyConnectedOp = mlir::dyn_cast_or_null<IE::FullyConnectedOp>(biasOp.getInput1().getDefiningOp());
    if (fullyConnectedOp == nullptr) {
        return mlir::failure();
    }

    if (fullyConnectedOp.getBias() != nullptr) {
        return mlir::failure();
    }

    auto fcOutShape = getShape(fullyConnectedOp.getOutput());
    auto biasShape = getShape(biasOp.getInput2());

    if (fcOutShape.size() != 2 || biasShape.size() != 2) {
        return mlir::failure();
    }
    if (biasShape[N] != 1) {
        return mlir::failure();
    }
    if (biasShape[C] != fcOutShape[C]) {
        return mlir::failure();
    }

    auto* newFC = rewriter.clone(*fullyConnectedOp);
    newFC->insertOperands(newFC->getNumOperands(), biasOp.getInput2());

    rewriter.replaceOp(biasOp, newFC->getOpResults());

    return mlir::success();
}

}  // namespace

void vpux::IE::FullyConnectedOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                             mlir::MLIRContext* context) {
    patterns.add<FuseFCAndBias>(context);
}
