//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;
mlir::LogicalResult vpux::IE::MVNOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::MVNOpAdaptor mvn(operands, attrs, prop);
    if (mlir::failed(mvn.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(mvn.getInput().getType());
    const auto inShape = inType.getShape();
    if (inShape.size() != 4 && inShape.size() != 5) {
        return errorAt(loc, "First input tensor should have 4 or 5 dimensions");
    }

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inType, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        return splitShapeAndRepresentation(outShape);
    });

    SmallVector<int64_t> outShape(outStaticShape.begin(), outStaticShape.end());
    const auto outDesc = vpux::getTensorAttr(ctx, inType.getDimsOrder(), inType.getMemSpace(), outBounds, outDimMask);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}

//
// build
//

void vpux::IE::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                            ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                            ::mlir::FloatAttr eps) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, {}, {});
}

void vpux::IE::MVNOp::build(::mlir::OpBuilder& builder, ::mlir::OperationState& state, ::mlir::Value input,
                            ::mlir::BoolAttr across_channels, ::mlir::BoolAttr normalize_variance,
                            ::mlir::FloatAttr eps, ::mlir::ArrayAttr internal_reshape) {
    build(builder, state, input.getType(), input, across_channels, normalize_variance, eps, internal_reshape, {});
}

//
// LegalizeEpsAttr
//

namespace {
class LegalizeEpsAttr final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    using mlir::OpRewritePattern<IE::MVNOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult LegalizeEpsAttr::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const {
    auto epsAttr = origOp.getEpsAttr();
    if (epsAttr == nullptr) {
        return mlir::failure();
    }

    auto eps = epsAttr.getValueAsDouble();
    auto floatEps = checked_cast<double>(std::numeric_limits<float>::epsilon());
    if (eps >= floatEps) {
        return mlir::failure();
    }

    // Convert double epsilon or smaller value to float epsilon since MVN kernel regards it as float datatype
    const auto newEpsAttr = getFPAttr(rewriter.getContext(), floatEps);
    rewriter.replaceOpWithNewOp<IE::MVNOp>(origOp, origOp.getInput(), origOp.getAcrossChannelsAttr(),
                                           origOp.getNormalizeVarianceAttr(), newEpsAttr);
    return mlir::success();
}

//
// ReshapeBatched
//

class ReshapeBatched final : public mlir::OpRewritePattern<IE::MVNOp> {
public:
    using mlir::OpRewritePattern<IE::MVNOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ReshapeBatched::matchAndRewrite(IE::MVNOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto acrossChannels = origOp.getAcrossChannelsAttr().getValue();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    auto origShape = inputType.getShape();
    if (acrossChannels == false || inputType.getRank() != 4 || origShape[Dims4D::Act::N] == 1) {
        return mlir::failure();
    }

    // acrossChannel batched MVN with shape [N,C,H,W] can be converted into
    // non-acrossChannel non-batched MVN with shape [1,N,C,H*W]
    SmallVector<int64_t> newShape(inputType.getRank(), 1);
    newShape[Dims4D::Act::C.ind()] = origShape[Dims4D::Act::N];
    newShape[Dims4D::Act::H.ind()] = origShape[Dims4D::Act::C];
    newShape[Dims4D::Act::W.ind()] = origShape[Dims4D::Act::H] * origShape[Dims4D::Act::W];
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto inputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), origOp.getInput(),
                                                             nullptr, false, newShapeAttr);

    auto newMvnOp = rewriter.create<IE::MVNOp>(origOp->getLoc(), inputReshape,
                                               mlir::BoolAttr::get(rewriter.getContext(), false),
                                               origOp.getNormalizeVarianceAttr(), origOp.getEpsAttr());

    const auto origShapeAttr = getIntArrayAttr(origOp->getContext(), origShape);
    auto outputReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), newMvnOp, nullptr,
                                                              false, origShapeAttr);

    rewriter.replaceOp(origOp, outputReshape);
    return mlir::success();
}

}  // namespace

//
// getCanonicalizationPatterns
//

void vpux::IE::MVNOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx) {
    patterns.add<LegalizeEpsAttr>(ctx);
    patterns.add<ReshapeBatched>(ctx);
}
