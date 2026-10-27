//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

using namespace vpux;

mlir::LogicalResult vpux::IE::YuvToRgbOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::OpaqueProperties prop, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    IE::YuvToRgbOpAdaptor colorConv(operands, attrs, prop);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto input = colorConv.getInput1();
    const auto inType = mlir::cast<mlir::RankedTensorType>(input.getType());
    const auto shape = inType.getShape();
    if (shape[Dims4D::Act::W.ind()] != 1) {
        return errorAt(loc, "Incorrect input shape format. Expected Y input to have Width '1', got '{0}'", shape);
    }

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inType, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        outShape[Dims4D::Act::W] = 3;

        if (colorConv.getInput2() == nullptr) {
            outShape[Dims4D::Act::C] = inShape[Dims4D::Act::C] * 2 / 3;
        }

        return splitShapeAndRepresentation(outShape);
    });

    auto outDesc = vpux::getTensorAttr(ctx, DimsOrder::fromValue(input), /*memSpace=*/nullptr, outBounds, outDimMask);

    inferredReturnShapes.emplace_back(outStaticShape.raw(), inType.getElementType(), outDesc);

    return mlir::success();
}

//
// ReifyRankedShapedTypeOpInterface
//

mlir::LogicalResult vpux::IE::YuvToRgbOp::reifyResultShapes(mlir::OpBuilder& builder,
                                                            mlir::ReifiedRankedShapedTypeDims& reifiedReturnShapes) {
    return reifyYuvToRgbTensors(getOperation(), builder, reifiedReturnShapes);
}

//
// ConvertToMultiInputs
//

namespace {

class ConvertToMultiInputs final : public mlir::OpRewritePattern<IE::YuvToRgbOp> {
public:
    using mlir::OpRewritePattern<IE::YuvToRgbOp>::OpRewritePattern;

public:
    mlir::LogicalResult matchAndRewrite(IE::YuvToRgbOp yuvToRgbOp, mlir::PatternRewriter& rewriter) const final;
};

mlir::LogicalResult ConvertToMultiInputs::matchAndRewrite(IE::YuvToRgbOp yuvToRgbOp,
                                                          mlir::PatternRewriter& rewriter) const {
    if (yuvToRgbOp.getInput2() == nullptr) {
        auto inputShape = mlir::cast<vpux::NDTypeInterface>(yuvToRgbOp.getInput1().getType()).getShape();
        const auto inShapeType = mlir::cast<mlir::ShapedType>(yuvToRgbOp.getInput1().getType()).getShape();
        const auto sliceOpLoc = yuvToRgbOp.getLoc();
        auto* ctx = rewriter.getContext();
        enum { N = 0, H = 1, W = 2, C = 3 };

        if (yuvToRgbOp.getInFmt() == IE::ColorFmt::NV12) {
            auto input1_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input2_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            input2_offsets[H] = inShapeType[H] / 3 * 2;

            SmallVector<int64_t> input1_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input2_sizes(inputShape.begin(), inputShape.end());

            input1_sizes[H] = inShapeType[H] / 3 * 2;
            input2_sizes[H] = inShapeType[H] / 3;

            auto input1_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_Y"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, input1_offsets), getIntArrayAttr(ctx, input1_sizes));
            auto input2_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_UV"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, input2_offsets), getIntArrayAttr(ctx, input2_sizes));

            input2_sizes[W] = input2_sizes[W] / 2;
            input2_sizes[C] = 2;
            auto shapeEndAttr = getIntArrayAttr(ctx, input2_sizes);
            auto input2_slice_reshape = IE::ReshapeOp::create(rewriter, appendLoc(sliceOpLoc, "reshape_UV"),
                                                              input2_slice.getResult(), shapeEndAttr);

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, input1_slice.getResult(), input2_slice_reshape,
                                                        nullptr, yuvToRgbOp.getInFmt(), yuvToRgbOp.getOutFmt(),
                                                        yuvToRgbOp.getScaleAttr());
            return mlir::success();

        } else {
            auto inputY_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputUV_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            inputUV_offsets[H] = inShapeType[H] / 3 * 2;

            SmallVector<int64_t> inputY_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputUV_sizes(inputShape.begin(), inputShape.end());

            inputY_sizes[H] = inShapeType[H] / 3 * 2;
            inputY_sizes[W] = inShapeType[W];
            inputUV_sizes[H] = inShapeType[H] / 3;
            inputUV_sizes[W] = inShapeType[W];

            auto inputY_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_Y"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, inputY_offsets), getIntArrayAttr(ctx, inputY_sizes));
            auto inputUV_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_UV"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, inputUV_offsets), getIntArrayAttr(ctx, inputUV_sizes));

            inputUV_sizes[H] = inShapeType[H] / 3 * 2;
            inputUV_sizes[W] = inShapeType[W] / 2;

            auto shape2EndAttr = getIntArrayAttr(ctx, inputUV_sizes);
            auto inputUV_slice_reshape = IE::ReshapeOp::create(rewriter, appendLoc(sliceOpLoc, "reshape_UV"),
                                                               inputUV_slice.getResult(), shape2EndAttr);

            SmallVector<int64_t> inputU_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputV_sizes(inputShape.begin(), inputShape.end());

            inputU_sizes[H] = inputUV_sizes[H] / 2;
            inputU_sizes[W] = inputUV_sizes[W];

            inputV_sizes[H] = inputUV_sizes[H] / 2;
            inputV_sizes[W] = inputUV_sizes[W];

            auto inputU_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputV_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            inputV_offsets[H] = inputUV_sizes[H] / 2;

            auto inputU_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_U"), inputUV_slice_reshape,
                                        getIntArrayAttr(ctx, inputU_offsets), getIntArrayAttr(ctx, inputU_sizes));
            auto inputV_slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "slice_V"), inputUV_slice_reshape,
                                        getIntArrayAttr(ctx, inputV_offsets), getIntArrayAttr(ctx, inputV_sizes));

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, inputY_slice.getResult(), inputU_slice.getResult(),
                                                        inputV_slice.getResult(), yuvToRgbOp.getInFmt(),
                                                        yuvToRgbOp.getOutFmt(), yuvToRgbOp.getScaleAttr());
            return mlir::success();
        }
    }
    return mlir::success();
}

}  // namespace

void vpux::IE::YuvToRgbOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertToMultiInputs>(context);
}
