//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

mlir::LogicalResult vpux::IE::YuvToRgbOp::inferReturnTypeComponents(
        mlir::MLIRContext* ctx, Optional<mlir::Location> optLoc, mlir::ValueShapeRange operands,
        mlir::DictionaryAttr attrs, mlir::RegionRange,
        SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes) {
    const auto loc = optLoc.getValueOr(mlir::UnknownLoc::get(ctx));

    IE::YuvToRgbOpAdaptor colorConv(operands, attrs);
    if (mlir::failed(colorConv.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = colorConv.input1().getType().cast<mlir::ShapedType>();
    const auto shape = inType.getShape();
    if (shape[3] != 1) {
        return errorAt(loc, "Incorrect input shape format: '{0}'", shape);
    }

    SmallVector<int64_t> outShape{shape[0], shape[1], shape[2], 3};

    if (colorConv.input2() == nullptr) {
        VPUX_THROW_UNLESS(colorConv.input3() == nullptr, "1xPlane config error");
        VPUX_THROW_UNLESS(((outShape[1] * 2) % 3) == 0, "Invalid height");
        outShape[1] = outShape[1] * 2 / 3;
    }

    inferredReturnShapes.emplace_back(outShape, inType.getElementType());

    return mlir::success();
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
    if (yuvToRgbOp.input2() == nullptr) {
        const auto cmxAvailableBytes = vpux::VPU::getTotalCMXSize(yuvToRgbOp).to<Byte>().count();

        const auto outputByteSize =
                yuvToRgbOp.output().getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        const auto outputSizeBytes =
                yuvToRgbOp.output().getType().cast<NDTypeInterface>().getShape().totalSize() * outputByteSize;

        const auto inputByteSize =
                yuvToRgbOp.input1().getType().cast<vpux::NDTypeInterface>().getElemTypeSize().to<Byte>().count();
        const auto inputSizeBytes =
                yuvToRgbOp.input1().getType().cast<NDTypeInterface>().getShape().totalSize() * inputByteSize;
        auto requiredCMX = outputSizeBytes + inputSizeBytes;
        if (requiredCMX < cmxAvailableBytes) {
            return mlir::success();
        }

        auto inputShape = yuvToRgbOp.input1().getType().cast<NDTypeInterface>().getShape();
        const auto inShapeType = yuvToRgbOp.input1().getType().cast<mlir::ShapedType>().getShape();
        const auto sliceOpLoc = yuvToRgbOp.getLoc();
        auto* ctx = rewriter.getContext();
        enum { N = 0, H = 1, W = 2, C = 3 };

        if (yuvToRgbOp.inFmt() == IE::ColorFmt::NV12) {
            auto input1_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input2_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            input2_offsets[H] = inShapeType[H] / 3 * 2;

            SmallVector<int64_t> input1_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input2_sizes(inputShape.begin(), inputShape.end());

            input1_sizes[H] = inShapeType[H] / 3 * 2;
            input2_sizes[H] = inShapeType[H] / 3;

            auto input1_slice =
                    rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.input1(), getIntArrayAttr(ctx, input1_offsets),
                                                 getIntArrayAttr(ctx, input1_sizes));
            auto input2_slice =
                    rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.input1(), getIntArrayAttr(ctx, input2_offsets),
                                                 getIntArrayAttr(ctx, input2_sizes));

            input2_sizes[W] = input2_sizes[W] / 2;
            input2_sizes[C] = 2;
            auto shapeEndAttr = getIntArrayAttr(ctx, input2_sizes);
            auto input2_slice_reshape =
                    rewriter.create<IE::ReshapeOp>(sliceOpLoc, input2_slice.result(), nullptr, false, shapeEndAttr);

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, input1_slice.result(), input2_slice_reshape,
                                                        nullptr, yuvToRgbOp.inFmt(), yuvToRgbOp.outFmt());
            return mlir::success();

        } else {
            auto input1_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input2_offsets = SmallVector<int64_t>(inputShape.size(), 0);
            auto input3_offsets = SmallVector<int64_t>(inputShape.size(), 0);

            input2_offsets[H] = inShapeType[H] / 3 * 2;
            input3_offsets[H] = inShapeType[H] / 3 * 2;
            input3_offsets[W] = inShapeType[W] / 2;

            SmallVector<int64_t> input1_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input2_sizes(inputShape.begin(), inputShape.end());
            SmallVector<int64_t> input3_sizes(inputShape.begin(), inputShape.end());

            input1_sizes[H] = inShapeType[H] / 3 * 2;
            input2_sizes[H] = inShapeType[H] / 3;
            input2_sizes[W] = inShapeType[H] / 2;
            input3_sizes[H] = inShapeType[H] / 3;
            input3_sizes[W] = inShapeType[H] / 2;

            auto input1_slice =
                    rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.input1(), getIntArrayAttr(ctx, input1_offsets),
                                                 getIntArrayAttr(ctx, input1_sizes));
            auto input2_slice =
                    rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.input1(), getIntArrayAttr(ctx, input2_offsets),
                                                 getIntArrayAttr(ctx, input2_sizes));
            auto input3_slice =
                    rewriter.create<IE::SliceOp>(sliceOpLoc, yuvToRgbOp.input1(), getIntArrayAttr(ctx, input3_offsets),
                                                 getIntArrayAttr(ctx, input3_sizes));

            rewriter.replaceOpWithNewOp<IE::YuvToRgbOp>(yuvToRgbOp, input1_slice.result(), input2_slice.result(),
                                                        input3_slice.result(), yuvToRgbOp.inFmt(), yuvToRgbOp.outFmt());
            return mlir::success();
        }
    }

    return mlir::success();
}

}  // namespace

void vpux::IE::YuvToRgbOp::getCanonicalizationPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* context) {
    patterns.insert<ConvertToMultiInputs>(context);
}
