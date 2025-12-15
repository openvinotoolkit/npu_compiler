//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
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

//
// ConstantFoldGather
//

namespace {

class ConstantFoldGather final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    using mlir::OpRewritePattern<IE::GatherOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;

private:
    template <typename T>
    mlir::LogicalResult foldGatherImpl(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter,
                                       const Const::Content& inputContent, const Const::Content& indicesContent,
                                       int64_t axis) const;
};

mlir::LogicalResult ConstantFoldGather::matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const {
    // Check if input is constant
    auto inputConst = gatherOp.getInput().getDefiningOp<Const::DeclareOp>();
    if (inputConst == nullptr) {
        return mlir::failure();
    }

    // Check if indices is constant
    auto indicesConst = gatherOp.getIndices().getDefiningOp<Const::DeclareOp>();
    if (indicesConst == nullptr) {
        return mlir::failure();
    }

    const auto axis = extractAxis(gatherOp.getLoc(), IE::GatherOpAdaptor(gatherOp));
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto inputContent = inputConst.getContent();
    const auto indicesContent = indicesConst.getContent();
    const auto inputType = mlir::cast<mlir::ShapedType>(gatherOp.getInput().getType());
    const auto elementType = inputType.getElementType();

    if (elementType.isF16()) {
        return foldGatherImpl<vpux::type::float16>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    } else if (elementType.isF32()) {
        return foldGatherImpl<float>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    } else if (mlir::isa<mlir::Float8E5M2Type>(elementType)) {
        return foldGatherImpl<vpux::type::float8_e5m2>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    } else if (mlir::isa<mlir::Float8E4M3FNType>(elementType)) {
        return foldGatherImpl<vpux::type::float8_e4m3>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    } else if (elementType.isSignedInteger(8)) {
        return foldGatherImpl<int8_t>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    } else if (elementType.isUnsignedInteger(8)) {
        return foldGatherImpl<uint8_t>(gatherOp, rewriter, inputContent, indicesContent, axis.value());
    }

    return mlir::failure();
}

template <typename T>
mlir::LogicalResult ConstantFoldGather::foldGatherImpl(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter,
                                                       const Const::Content& inputContent,
                                                       const Const::Content& indicesContent, int64_t axis) const {
    const auto inputType = mlir::cast<mlir::ShapedType>(gatherOp.getInput().getType());
    const auto indicesType = mlir::cast<mlir::ShapedType>(gatherOp.getIndices().getType());
    const auto outputType = mlir::cast<mlir::ShapedType>(gatherOp.getOutput().getType());
    const auto inputShape = inputType.getShape();
    const auto indicesShape = indicesType.getShape();
    const auto outputShape = outputType.getShape();
    const auto batchDims = gatherOp.getBatchDims();
    auto inputValues = inputContent.getValues<T>();
    auto indicesValues = indicesContent.getValues<int64_t>();
    const auto outputSize =
            std::accumulate(outputShape.begin(), outputShape.end(), int64_t(1), std::multiplies<int64_t>());
    SmallVector<T> outputValues(outputSize);

    // Calculate linear index from multi-dimensional index
    auto calculateLinearIndex = [](ArrayRef<int64_t> shape, ArrayRef<int64_t> indices) -> int64_t {
        int64_t linearIndex = 0;
        int64_t stride = 1;
        for (int64_t i = shape.size() - 1; i >= 0; --i) {
            linearIndex += indices[i] * stride;
            stride *= shape[i];
        }
        return linearIndex;
    };

    // Convert linear index to multi-dimensional index
    auto calculateMultiIndex = [](ArrayRef<int64_t> shape, int64_t linearIndex) -> SmallVector<int64_t> {
        SmallVector<int64_t> indices(shape.size());
        for (int64_t i = shape.size() - 1; i >= 0; --i) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
        }
        return indices;
    };

    // Perform gather operation
    for (int64_t outputIdx = 0; outputIdx < outputSize; ++outputIdx) {
        auto outputMultiIdx = calculateMultiIndex(outputShape, outputIdx);

        // Calculate corresponding input index
        SmallVector<int64_t> inputMultiIdx(inputShape.size());

        // Copy batch dimensions
        for (int64_t i = 0; i < batchDims; ++i) {
            inputMultiIdx[i] = outputMultiIdx[i];
        }

        // Copy dimensions before axis
        for (int64_t i = batchDims; i < axis; ++i) {
            inputMultiIdx[i] = outputMultiIdx[i];
        }

        // Get index from indices tensor
        SmallVector<int64_t> indicesMultiIdx(indicesShape.size());
        for (int64_t i = 0; i < batchDims; ++i) {
            indicesMultiIdx[i] = outputMultiIdx[i];
        }
        for (int64_t i = batchDims; i < static_cast<int64_t>(indicesShape.size()); ++i) {
            indicesMultiIdx[i] = outputMultiIdx[axis - batchDims + i];
        }

        // Get gather indice value
        auto indicesLinearIdx = calculateLinearIndex(indicesShape, indicesMultiIdx);
        auto gatheredIdx = indicesValues[indicesLinearIdx];
        if (gatheredIdx < 0) {
            gatheredIdx += inputShape[axis];
        }
        if (gatheredIdx < 0 || gatheredIdx >= inputShape[axis]) {
            // Invalid index
            return mlir::failure();
        }
        inputMultiIdx[axis] = gatheredIdx;

        // Copy dimensions after axis
        int64_t outputOffset = axis + static_cast<int64_t>(indicesShape.size()) - batchDims;
        for (int64_t i = axis + 1; i < static_cast<int64_t>(inputShape.size()); ++i) {
            inputMultiIdx[i] = outputMultiIdx[outputOffset + (i - axis - 1)];
        }

        // Get input value and set
        auto inputLinearIdx = calculateLinearIndex(inputShape, inputMultiIdx);
        outputValues[outputIdx] = inputValues[inputLinearIdx];
    }

    // Create constant output
    const auto outputTensorType = mlir::RankedTensorType::get(outputShape, inputType.getElementType());
    auto newConstOp = Const::createConst(rewriter, gatherOp.getLoc(), outputTensorType, ArrayRef<T>(outputValues));

    rewriter.replaceOp(gatherOp, newConstOp);
    return mlir::success();
}

}  // namespace

void vpux::IE::GatherOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConvertConstToAttr>(context);
    patterns.add<ConstantFoldGather>(context);
}
