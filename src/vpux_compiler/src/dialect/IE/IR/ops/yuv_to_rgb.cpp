//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/import/yuv_to_rgb.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/core/IR/tensor_attr.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <cstdint>
#include <vpux/utils/core/small_vector.hpp>

using namespace vpux;
using namespace Dims4D::Act;

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

    // YuvToRgb follows NV12 docs semantics for 2-plane input:
    //   Y  must be [N, H,   W,   1]
    //   UV must be [N, H/2, W/2, 2]
    // with output [N, H, W, 3].
    //
    // In this compiler, YuvToRgb follows the common internal convention: tensor shapes
    // are interpreted in logical NCHW form. External NV12/I420 docs often describe these
    // tensors in NHWC terms, but inside the compiler semantic channel is always C and
    // semantic height is always H. For a combined single-plane input (Y and UV stacked
    // along semantic height), that stacked height shrinks by 2/3 to recover luma height.

    if (shape[C.ind()] != 1) {
        return errorAt(loc, "Incorrect input shape format. Expected Y input to have channels == '1', got '{0}'", shape);
    }

    auto [outStaticShape, outBounds, outDimMask] = callOnShapeOf(inType, [&](const auto& inShape) {
        auto outShape = copyShape(inShape);
        outShape[C] = 3;

        if (colorConv.getInput2() == nullptr) {
            outShape[H] = inShape[H] * 2 / 3;
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
    // This canonicalization runs on the compiler's canonical YuvToRgb representation:
    // order == NHWC with logical shape indices interpreted as [N, C, H, W].
    // For single-plane input, it splits stacked Y/UV data into multi-input form.
    const auto dimsOrder = DimsOrder::fromValue(yuvToRgbOp.getInput1());
    if (dimsOrder != DimsOrder::NHWC) {
        return mlir::failure();
    }

    if (yuvToRgbOp.getInput2() == nullptr) {
        auto inputShape = mlir::cast<vpux::NDTypeInterface>(yuvToRgbOp.getInput1().getType()).getShape();
        const auto inShapeType = mlir::cast<mlir::ShapedType>(yuvToRgbOp.getInput1().getType()).getShape();
        const auto sliceOpLoc = yuvToRgbOp.getLoc();
        auto* ctx = rewriter.getContext();

        if (yuvToRgbOp.getInFmt() == IE::ColorFmt::NV12) {
            auto input1Offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input2Offsets = SmallVector<int64_t>(inputShape.size(), 0);

            input2Offsets[H.ind()] = inShapeType[H.ind()] / 3 * 2;

            SmallVector<int64_t> input1Sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input2Sizes(inputShape.begin(), inputShape.end());

            input1Sizes[H.ind()] = inShapeType[H.ind()] / 3 * 2;
            input2Sizes[H.ind()] = inShapeType[H.ind()] / 3;

            auto input1Slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceY"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, input1Offsets), getIntArrayAttr(ctx, input1Sizes));
            auto input2Slice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceUV"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, input2Offsets), getIntArrayAttr(ctx, input2Sizes));

            input2Sizes[W.ind()] = input2Sizes[W.ind()] / 2;
            input2Sizes[C.ind()] = 2;
            auto shapeEndAttr = getIntArrayAttr(ctx, input2Sizes);
            auto input2SliceReshape = IE::ReshapeOp::create(rewriter, appendLoc(sliceOpLoc, "reshapeUV"),
                                                            input2Slice.getResult(), shapeEndAttr);
            const auto input2OrderAttr =
                    mlir::AffineMapAttr::get(DimsOrder::fromValue(input2Slice.getResult()).toAffineMap(ctx));
            auto input2SliceLayoutCast = IE::LayoutCastOp::create(rewriter, appendLoc(sliceOpLoc, "layoutCastUV"),
                                                                  input2SliceReshape.getResult(), input2OrderAttr);

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(
                    yuvToRgbOp, input1Slice.getResult(), input2SliceLayoutCast.getResult(), nullptr,
                    yuvToRgbOp.getInFmt(), yuvToRgbOp.getOutFmt(), yuvToRgbOp.getScaleAttr());
            return mlir::success();

        } else {
            auto inputYOffsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputUVOffsets = SmallVector<int64_t>(inputShape.size(), 0);

            inputUVOffsets[H.ind()] = inShapeType[H.ind()] / 3 * 2;

            SmallVector<int64_t> inputYSizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputUVSizes(inputShape.begin(), inputShape.end());

            inputYSizes[H.ind()] = inShapeType[H.ind()] / 3 * 2;
            inputYSizes[W.ind()] = inShapeType[W.ind()];
            inputUVSizes[H.ind()] = inShapeType[H.ind()] / 3;
            inputUVSizes[W.ind()] = inShapeType[W.ind()];

            auto inputYSlice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceY"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, inputYOffsets), getIntArrayAttr(ctx, inputYSizes));
            auto inputUVSlice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceUV"), yuvToRgbOp.getInput1(),
                                        getIntArrayAttr(ctx, inputUVOffsets), getIntArrayAttr(ctx, inputUVSizes));

            inputUVSizes[H.ind()] = inShapeType[H.ind()] / 3 * 2;
            inputUVSizes[W.ind()] = inShapeType[W.ind()] / 2;

            auto shape2EndAttr = getIntArrayAttr(ctx, inputUVSizes);
            auto inputUVSliceReshape = IE::ReshapeOp::create(rewriter, appendLoc(sliceOpLoc, "reshapeUV"),
                                                             inputUVSlice.getResult(), shape2EndAttr);
            const auto inputUVOrderAttr =
                    mlir::AffineMapAttr::get(DimsOrder::fromValue(inputUVSlice.getResult()).toAffineMap(ctx));
            auto inputUVSliceLayoutCast = IE::LayoutCastOp::create(rewriter, appendLoc(sliceOpLoc, "layoutCastUV"),
                                                                   inputUVSliceReshape.getResult(), inputUVOrderAttr);

            SmallVector<int64_t> inputUSizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> inputVSizes(inputShape.begin(), inputShape.end());

            inputUSizes[H.ind()] = inputUVSizes[H.ind()] / 2;
            inputUSizes[W.ind()] = inputUVSizes[W.ind()];

            inputVSizes[H.ind()] = inputUVSizes[H.ind()] / 2;
            inputVSizes[W.ind()] = inputUVSizes[W.ind()];

            auto inputUOffsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto inputVOffsets = SmallVector<int64_t>(inputShape.size(), 0);
            inputVOffsets[H.ind()] = inputUVSizes[H.ind()] / 2;

            auto inputUSlice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceU"), inputUVSliceLayoutCast,
                                        getIntArrayAttr(ctx, inputUOffsets), getIntArrayAttr(ctx, inputUSizes));
            auto inputVSlice =
                    IE::SliceOp::create(rewriter, appendLoc(sliceOpLoc, "sliceV"), inputUVSliceLayoutCast,
                                        getIntArrayAttr(ctx, inputVOffsets), getIntArrayAttr(ctx, inputVSizes));

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, inputYSlice.getResult(), inputUSlice.getResult(),
                                                        inputVSlice.getResult(), yuvToRgbOp.getInFmt(),
                                                        yuvToRgbOp.getOutFmt(), yuvToRgbOp.getScaleAttr());
            return mlir::success();
        }
    }
    return mlir::success();
}

}  // namespace

//
// Canonicalization helpers for IE::YuvToRgb operation
//

mlir::Value IE::YuvToRgb::canonicalizeYuvToRgbInput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                                    llvm::StringRef locSuffix) {
    if (input == nullptr) {
        return nullptr;
    }

    // Normalize every imported YUV input through an explicit transpose to keep
    // YuvToRgb wrapped symmetrically by data-movement ops only.
    const auto ctx = builder.getContext();
    auto transposeLoc = appendLoc(loc, locSuffix);
    auto transpose = IE::TransposeOp::create(builder, transposeLoc, input, nullptr,
                                             mlir::AffineMapAttr::get(DimsOrder::NWCH.toAffineMap(ctx)));
    return transpose.getOutput();
}

mlir::Value IE::YuvToRgb::restoreYuvToRgbOutput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value output) {
    if (output == nullptr) {
        return nullptr;
    }

    // Restore external OpenVINO contract with an explicit data transpose.
    const auto ctx = builder.getContext();
    auto transpose = IE::TransposeOp::create(builder, appendLoc(loc, "restoreOutputOrder"), output, nullptr,
                                             mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx)));
    return transpose.getOutput();
}

void vpux::IE::YuvToRgbOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertToMultiInputs>(context);
}
