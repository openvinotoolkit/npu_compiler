//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/PatternMatch.h>

using namespace vpux;

namespace vpux::IE::Gather {
llvm::FailureOr<int64_t> parseAxis(mlir::Location loc, mlir::ValueRange opInputs) {
    if (opInputs.size() != 3) {
        return errorAt(loc, "Invalid IE.Gather operation: expected 3 inputs got {0}", opInputs.size());
    }

    // Note: by definition in .td file, IE.Reshape's operand #2 is the axis.
    auto reshapeAxis = opInputs[2].getDefiningOp<IE::ReshapeOp>();
    auto axisConst = (reshapeAxis != nullptr) ? reshapeAxis.getInput().getDefiningOp<Const::DeclareOp>()
                                              : opInputs[2].getDefiningOp<Const::DeclareOp>();
    if (axisConst == nullptr) {
        return errorAt(loc, "Only constant input is supported for axis");
    }

    if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
        return errorAt(loc, "Axis value must be a scalar");
    }

    const auto axisContent = axisConst.getContent();
    int64_t axisInd = axisContent.getSplatValue<int64_t>();

    if (axisInd < 0) {
        const auto inType = mlir::cast<mlir::ShapedType>(opInputs[0].getType());
        const auto inRank = inType.getRank();
        axisInd += inRank;
        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Wrong Gather axis {0}", axisInd);
    }

    return axisInd;
}
}  // namespace vpux::IE::Gather
namespace {

auto calculateOutputShape(const llvm::ArrayRef<int64_t>& inputShape, const llvm::ArrayRef<int64_t>& indicesShape,
                          int64_t batchDims, int64_t axisVal, int64_t indicesRank) {
    SmallVector<int64_t> shape;
    int64_t outRank = inputShape.size() + indicesRank - 1 - batchDims;
    VPUX_THROW_UNLESS(outRank >= 0, "Calculated output rank expected to be non-negative, but got {0}", outRank);
    int64_t i = 0;

    for (; i < batchDims; i++) {
        VPUX_THROW_WHEN(inputShape[i] != indicesShape[i],
                        "The first dimensions in Input and Indices shapes are expected to be equal");
        shape.push_back(inputShape[i]);
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
    return shape;
};

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

    const auto inputType = mlir::cast<mlir::ShapedType>(gather.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto indicesType = mlir::cast<mlir::ShapedType>(gather.getIndices().getType());
    const auto indicesShape = indicesType.getShape();

    const auto axis = gather.getAxisValue();
    auto batch = gather.getBatchDims();
    auto rank = gather.getIndicesRank().value_or(indicesShape.size());

    auto outShape = calculateOutputShape(inputShape, indicesShape, batch, axis, rank);

    Bounds bounds;
    if (vpux::details::isDynamicDimValues(outShape)) {
        auto boundedInputTensor = mlir::dyn_cast<Core::BoundedTensorType>(inputType);
        auto boundedIndicesTensor = mlir::dyn_cast<Core::BoundedTensorType>(indicesType);
        auto actualInputShape = boundedInputTensor ? boundedInputTensor.getBounds().raw() : inputShape;
        auto actualIndicesShape = boundedIndicesTensor ? boundedIndicesTensor.getBounds().raw() : indicesShape;
        bounds = Bounds(calculateOutputShape(actualInputShape, actualIndicesShape, batch, axis, rank));
    }

    const auto outDesc =
            vpux::getTensorAttr(ctx, DimsOrder::fromNumDims(outShape.size()), /*memSpace=*/nullptr, bounds);
    inferredReturnShapes.emplace_back(outShape, inputType.getElementType(), outDesc);

    return mlir::success();
}

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

    const auto axis = gatherOp.getAxisValue();
    const auto inputContent = inputConst.getContent();
    const auto indicesContent = indicesConst.getContent();
    const auto inputType = mlir::cast<mlir::ShapedType>(gatherOp.getInput().getType());
    const auto elementType = inputType.getElementType();

    if (elementType.isF16()) {
        return foldGatherImpl<vpux::type::float16>(gatherOp, rewriter, inputContent, indicesContent, axis);
    } else if (elementType.isF32()) {
        return foldGatherImpl<float>(gatherOp, rewriter, inputContent, indicesContent, axis);
    } else if (mlir::isa<mlir::Float8E5M2Type>(elementType)) {
        return foldGatherImpl<vpux::type::float8_e5m2>(gatherOp, rewriter, inputContent, indicesContent, axis);
    } else if (mlir::isa<mlir::Float8E4M3FNType>(elementType)) {
        return foldGatherImpl<vpux::type::float8_e4m3>(gatherOp, rewriter, inputContent, indicesContent, axis);
    } else if (elementType.isSignedInteger(8)) {
        return foldGatherImpl<int8_t>(gatherOp, rewriter, inputContent, indicesContent, axis);
    } else if (elementType.isUnsignedInteger(8)) {
        return foldGatherImpl<uint8_t>(gatherOp, rewriter, inputContent, indicesContent, axis);
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

//
// BypassSignednessConvertIndices
//

namespace {

// Removes a signedness-only integer Convert (e.g. si64->ui64, si32->ui32) feeding the indices operand.
// Gather selects elements by index value, which is identical for both signedness representations of an
// in-range, non-negative index. On NPU the Gather kernel reads indices as signed int32 regardless of the
// declared signedness, so such a Convert is a no-op for execution. The software Convert kernel additionally
// has no implementation for same-width signed->unsigned integer conversions, leaving the destination buffer
// unwritten (zeroed) and corrupting the gathered result. Bypassing the Convert avoids this and feeds the
// original signed indices directly.
class BypassSignednessConvertIndices final : public mlir::OpRewritePattern<IE::GatherOp> {
public:
    using mlir::OpRewritePattern<IE::GatherOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::GatherOp gatherOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult BypassSignednessConvertIndices::matchAndRewrite(IE::GatherOp gatherOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    auto convertOp = gatherOp.getIndices().getDefiningOp<IE::ConvertOp>();
    if (convertOp == nullptr) {
        return mlir::failure();
    }

    const auto srcElemType = mlir::dyn_cast<mlir::IntegerType>(
            mlir::cast<mlir::ShapedType>(convertOp.getInput().getType()).getElementType());
    const auto dstElemType = mlir::dyn_cast<mlir::IntegerType>(
            mlir::cast<mlir::ShapedType>(convertOp.getOutput().getType()).getElementType());
    if (srcElemType == nullptr || dstElemType == nullptr) {
        return mlir::failure();
    }

    // Match only same-width signed->unsigned converts, the case that is value-preserving for non-negative
    // indices yet unsupported by the software Convert kernel.
    if (srcElemType.getWidth() != dstElemType.getWidth() || !srcElemType.isSigned() || !dstElemType.isUnsigned()) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::GatherOp>(gatherOp, gatherOp.getType(), gatherOp.getInput(), convertOp.getInput(),
                                              gatherOp.getAxisValue(), gatherOp.getBatchDims(),
                                              gatherOp.getIndicesRankAttr());
    return mlir::success();
}

}  // namespace

void vpux::IE::GatherOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.add<ConstantFoldGather>(context);
    patterns.add<BypassSignednessConvertIndices>(context);
}
