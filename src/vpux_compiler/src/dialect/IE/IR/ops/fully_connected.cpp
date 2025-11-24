//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

std::tuple<vpux::Shape, vpux::Bounds, vpux::DynamicDimsMask> calcFullyConnectedOutputShape(mlir::Value input,
                                                                                           mlir::Value weights) {
    const auto inType = mlir::cast<mlir::ShapedType>(input.getType());
    const auto weightsType = mlir::cast<mlir::ShapedType>(weights.getType());
    mlir::ShapedType dynType = inType;
    mlir::ShapedType statType = weightsType;
    size_t dynamicIdx = 0;
    size_t staticIdx = 1;

    if (mlir::dyn_cast<Core::BoundedTensorType>(weightsType)) {
        dynType = weightsType;
        statType = inType;
        dynamicIdx = 1;
        staticIdx = 0;
    }

    return callOnShapeOf(dynType, [&](const auto& dynShape) {
        auto outShape = copyShape(dynShape);
        outShape[Dim(dynamicIdx)] = dynShape[Dim(0)];
        outShape[Dim(staticIdx)] = statType.getShape()[0];
        return splitShapeAndRepresentation(outShape);
    });
}
}  // namespace

mlir::LogicalResult vpux::IE::FullyConnectedOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::FullyConnectedOpAdaptor fullyConnected(operands, attrs, prop);
    if (mlir::failed(fullyConnected.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(fullyConnected.getInput().getType());
    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(fullyConnected.getWeights().getType());
    const auto inShape = inType.getShape();
    const auto weightsShape = weightsType.getShape();
    const auto inRank = inShape.size();
    const auto weightsRank = weightsShape.size();

    if (weightsRank != 2 || inRank != 2) {
        return mlir::failure();
    }

    auto [outStaticShape, outBounds, outDimMask] =
            calcFullyConnectedOutputShape(fullyConnected.getInput(), fullyConnected.getWeights());
    SmallVector<int64_t> outShape(outStaticShape.begin(), outStaticShape.end());
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), outBounds, outDimMask);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FullyConnectedOp::reifyResultShapes(
        mlir::OpBuilder& builder, mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    auto loc = getLoc();

    SmallVector<mlir::OpFoldResult> outShape;
    outShape.push_back(reifyDim(builder, getInput(), 0, loc));
    outShape.push_back(reifyDim(builder, getWeights(), 0, loc));

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
    extendOpLoc(newFC, "as_fc");
    newFC->insertOperands(newFC->getNumOperands(), biasOp.getInput2());

    rewriter.replaceOp(biasOp, newFC->getOpResults());

    return mlir::success();
}

}  // namespace

void vpux::IE::FullyConnectedOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns,
                                                             mlir::MLIRContext* context) {
    patterns.add<FuseFCAndBias>(context);
}
