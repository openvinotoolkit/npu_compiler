//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace {

mlir::FailureOr<int64_t> extractAxis(mlir::Location loc, IE::GatherOpAdaptor gather) {
    if (gather.getAxis() != nullptr) {
        auto reshapeAxis = gather.getAxis().getDefiningOp<IE::ReshapeOp>();
        auto axisConst = (reshapeAxis != nullptr) ? reshapeAxis.getInput().getDefiningOp<Const::DeclareOp>()
                                                  : gather.getAxis().getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto axisContent = axisConst.getContent();
        int64_t axisInd = axisContent.getSplatValue<int64_t>();

        if (axisInd < 0) {
            const auto inType = mlir::cast<mlir::ShapedType>(gather.getInput().getType());
            const auto inRank = inType.getRank();
            axisInd += inRank;
            VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Wrong Gather axis {0}", axisInd);
        }

        return axisInd;
    } else if (gather.getAxisValue().has_value()) {
        return gather.getAxisValue().value();
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::IE::GatherOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::GatherOpAdaptor gather(operands, attrs, prop);
    if (mlir::failed(gather.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<mlir::ShapedType>(gather.getInput().getType());
    const auto inputShape = inType.getShape();
    const auto indicesShape = mlir::cast<mlir::ShapedType>(gather.getIndices().getType()).getShape();

    const auto axis = extractAxis(loc, gather);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    SmallVector<int64_t> outShape;
    SmallVector<int64_t> outShapeBounds;

    auto calculateOutputShape = [&](auto& shape, const auto& indicesShape) {
        int64_t batchDims = gather.getBatchDims();
        int64_t axisVal = checked_cast<int64_t>(*axis);
        int64_t indicesRank = gather.getIndicesRank().value_or(indicesShape.size());
        int64_t outRank = inputShape.size() + indicesRank - 1 - batchDims;
        int64_t i = 0;

        for (; i < batchDims; i++) {
            shape.push_back(inputShape[i] & indicesShape[i]);
        }
        for (; i < axisVal; i++) {
            shape.push_back(inputShape[i]);
        }
        for (; i < axisVal + indicesRank - batchDims; i++) {
            shape.push_back(indicesShape[batchDims - axisVal + i]);
        }
        for (; i < outRank; i++) {
            shape.push_back(inputShape[batchDims + 1 - indicesRank + i]);
        }
        // To avoid shape size 0 error, set the shape 1.
        if (shape.empty()) {
            shape.push_back(1);
        }
    };

    calculateOutputShape(outShape, indicesShape);

    auto indicesType = gather.getIndices().getType();
    Bounds bounds;
    if (auto boundedTensor = mlir::dyn_cast<Core::BoundedTensorType>(indicesType)) {
        auto indicesBounds = boundedTensor.getBounds().raw();
        calculateOutputShape(outShapeBounds, indicesBounds);
        bounds = Bounds(outShapeBounds);
    }

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), /*memSpace=*/nullptr, bounds);
    inferredReturnShapes.emplace_back(outShape, inType.getElementType(), outDesc);

    return mlir::success();
}

//
// ConvertConstToAttr
//

namespace {

class ConvertConstToAttr final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    using mlir::OpRewritePattern<IE::GatherOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertConstToAttr::matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
    auto axis = gatherOp.getAxis();
    if (axis == nullptr) {
        return mlir::failure();
    }

    auto axisConst = gatherOp.getAxis().getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return mlir::failure();
    }

    if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
        return mlir::failure();
    }

    const auto axisContent = axisConst.getContent();
    int64_t axisInd = axisContent.getSplatValue<int64_t>();
    if (axisInd < 0) {
        const auto inType = mlir::cast<mlir::ShapedType>(gatherOp.getInput().getType());
        const auto inRank = inType.getRank();
        axisInd += inRank;
    }

    rewriter.replaceOpWithNewOp<IE::GatherOp>(gatherOp, gatherOp.getType(), gatherOp.getInput(), gatherOp.getIndices(),
                                              nullptr, rewriter.getI64IntegerAttr(axisInd), gatherOp.getBatchDims(),
                                              gatherOp.getIndicesRankAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::GatherOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
}
