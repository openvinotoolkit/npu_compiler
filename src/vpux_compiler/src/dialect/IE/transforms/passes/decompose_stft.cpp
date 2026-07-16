//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSESTFT
#define GEN_PASS_DEF_DECOMPOSESTFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// STFTOpConverter
//

class STFTOpConverter final : public mlir::OpRewritePattern<IE::STFTOp> {
public:
    STFTOpConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::STFTOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::STFTOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult STFTOpConverter::matchAndRewrite(IE::STFTOp origOp, mlir::PatternRewriter& rewriter) const {
    const auto ctx = origOp.getContext();
    const auto loc = origOp.getLoc();

    _log.trace("Decomposing STFT operation at {0}", loc);

    auto signal = origOp.getSignal();
    auto window = origOp.getWindow();
    auto frameSize = origOp.getFrameSize();
    auto frameStep = origOp.getFrameStep();
    auto transposeFrames = origOp.getTransposeFrames();

    _log.trace("Signal type: {0}, Output type: {1}", signal.getType(), origOp.getResult().getType());

    auto frameSizeConstOp = frameSize.getDefiningOp<Const::DeclareOp>();
    auto frameStepConstOp = frameStep.getDefiningOp<Const::DeclareOp>();

    if (!frameSizeConstOp || !frameStepConstOp) {
        _log.error("STFT frame_size and frame_step must be constants for compile-time unrolling");
        return mlir::failure();
    }

    const auto frameSizeContent = frameSizeConstOp.getContent();
    const auto frameStepContent = frameStepConstOp.getContent();
    const auto frameSizeVal = frameSizeContent.getSplatValue<int64_t>();
    const auto frameStepVal = frameStepContent.getSplatValue<int64_t>();

    const auto signalType = mlir::cast<vpux::NDTypeInterface>(signal.getType());
    const auto signalShape = signalType.getShape();
    const auto elemType = signalType.getElementType();

    int64_t batchSize = 1;
    int64_t signalLength = 0;

    if (signalShape.size() == 1) {
        batchSize = 1;
        signalLength = signalShape.raw()[0];
    } else if (signalShape.size() == 2) {
        batchSize = signalShape.raw()[0];
        signalLength = signalShape.raw()[1];
    } else {
        _log.error("STFT input must be 1D or 2D tensor, got {0}D", signalShape.size());
        return mlir::failure();
    }

    if (frameSizeVal <= 0 || frameStepVal <= 0) {
        _log.error("Frame size {0} and step {1} must be positive", frameSizeVal, frameStepVal);
        return mlir::failure();
    }

    if (frameSizeVal > signalLength) {
        _log.error("Frame size {0} cannot be larger than signal length {1}", frameSizeVal, signalLength);
        return mlir::failure();
    }

    const auto numFrames = (signalLength - frameSizeVal) / frameStepVal + 1;
    if (numFrames <= 0) {
        _log.error("Invalid frame configuration results in {0} frames", numFrames);
        return mlir::failure();
    }

    _log.trace("Using Convolution-based STFT decomposition with {0} frames", numFrames);

    // ============================================================
    // CONVOLUTION-BASED APPROACH (replacing Slice + Reshape + Concat)
    // ============================================================

    // Reshape input signal to [batch, 1, signalLength, 1] for convolution (4D)
    SmallVector<int64_t> inputConvShape = {batchSize, 1, signalLength, 1};
    const auto inputConvType = mlir::RankedTensorType::get(inputConvShape, elemType);
    const auto inputConvShapeAttr = getIntArrayAttr(ctx, inputConvShape);

    auto inputReshaped = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "input_reshape_for_conv"), inputConvType, signal,
                                                        inputConvShapeAttr);
    _log.trace("Reshaped input to {0} for convolution", inputConvShape);

    // Create identity convolution weights [frameSizeVal, 1, frameSizeVal]
    // Each output channel extracts one position from the sliding window
    const auto OC = frameSizeVal;  // Output channels
    const auto IC = 1;             // Input channels
    const auto KY = frameSizeVal;  // Kernel height
    const auto KX = 1;             // Kernel width (1D convolution)

    const Shape weightsShape = {OC, IC, KY, KX};
    SmallVector<float> weightsData(weightsShape.totalSize(), 0.0f);

    for (int64_t i = 0; i < frameSizeVal; i++) {
        // Calculate flat index for weights[i, 0, i, 0]
        auto beginIndex = i * KY + i;
        weightsData[beginIndex] = 1.0f;
    }

    const auto weightsElemType = mlir::Float16Type::get(ctx);
    const auto weightsType = mlir::RankedTensorType::get(weightsShape.raw(), weightsElemType);

    // Create the weights constant using the helper function from the file
    auto weightsConstOp = Const::buildWeightsConst(rewriter, appendLoc(loc, "conv_identity_weights"), weightsType,
                                                   ArrayRef(weightsData));

    _log.trace("Created identity convolution weights with shape {0}", weightsShape);

    // Perform Convolution to extract all frames at once
    const auto stridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{frameStepVal, 1});
    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto padsBeginAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto padsEndAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});

    // Output shape: [batch, frameSizeVal, numFrames]
    SmallVector<int64_t> convOutputShape = {batchSize, frameSizeVal, numFrames, 1};

    auto convOp = rewriter.create<IE::ConvolutionOp>(appendLoc(loc, "frame_extraction_conv"),
                                                     inputReshaped.getOutput(),   // input
                                                     weightsConstOp,              // filter
                                                     /*bias=*/nullptr,            // bias
                                                     /*scale=*/nullptr,           // scale
                                                     /*zero_points=*/nullptr,     // zero_points
                                                     stridesAttr,                 // strides
                                                     padsBeginAttr,               // pads_begin
                                                     padsEndAttr,                 // pads_end
                                                     dilationsAttr,               // dilations
                                                     /*post_op=*/nullptr,         // post_op
                                                     /*clamp=*/nullptr,           // clamp
                                                     /*static_scale=*/nullptr,    // static_scale
                                                     /*output_padding=*/nullptr,  // output_padding
                                                     /*input_padding=*/nullptr    // input_padding
    );

    _log.trace("Performed convolution for frame extraction, output shape: {0}\n", convOutputShape);

    // Reshape from 4D [1, 512, 3, 1] to 3D [1, 512, 3]
    SmallVector<int64_t> convReshaped3DShape = {batchSize, frameSizeVal, numFrames};
    const auto convReshaped3DType = mlir::RankedTensorType::get(convReshaped3DShape, elemType);
    const auto convReshaped3DShapeAttr = getIntArrayAttr(ctx, convReshaped3DShape);

    auto convReshaped = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "conv_output_reshape"), convReshaped3DType,
                                                       convOp.getOutput(), convReshaped3DShapeAttr);
    _log.trace("Reshaped conv output from 4D to 3D: {0}\n", convReshaped3DShape);

    // Transpose to [batch, numFrames, frameSizeVal]
    SmallVector<int64_t> transposeOrder = {0, 2, 1};  // Swap channels and width
    SmallVector<int64_t> framesShape = {batchSize, numFrames, frameSizeVal};
    const auto framesType = mlir::RankedTensorType::get(framesShape, elemType);
    const auto permutationMap = mlir::AffineMap::getPermutationMap(transposeOrder, ctx);
    const auto permutationMapAttr = mlir::AffineMapAttr::get(permutationMap);

    auto framesTransposed = rewriter.create<IE::TransposeOp>(appendLoc(loc, "frames_transpose"), framesType,
                                                             convReshaped.getOutput(), nullptr, permutationMapAttr);
    _log.trace("Transposed frames to shape {0}", framesShape);

    // Continue with windowing and RDFT
    mlir::Value windowedFrames = framesTransposed.getOutput();
    if (window) {
        // Apply windowing with NUMPY broadcasting.
        // The window is typically 1D with shape [frameSizeVal]. In 3D, frames are [batch, numFrames, frameSizeVal].
        // NUMPY broadcasting will broadcast the window correctly in 3D.
        // This way, when converted to 4D, it becomes [1, 1, frameSizeVal, 1] (height dimension)
        // instead of [1, frameSizeVal, 1, 1] (channel dimension), and broadcasts correctly without Slice.

        mlir::Value windowForMultiply = window;
        const auto windowType = mlir::cast<vpux::NDTypeInterface>(window.getType());
        const auto windowShape = windowType.getShape();

        // If window is 1D [frameSizeVal], reshape to 3D [1, 1, frameSizeVal] for proper broadcasting
        if (windowShape.size() == 1 && windowShape.raw()[0] == frameSizeVal) {
            SmallVector<int64_t> newWindowShape = {1, 1, frameSizeVal};
            const auto newWindowType = mlir::RankedTensorType::get(newWindowShape, windowType.getElementType());
            auto newWindowShapeAttr = getIntArrayAttr(ctx, newWindowShape);

            windowForMultiply = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "window_reshape_for_broadcast"),
                                                               newWindowType, window,
                                                               newWindowShapeAttr  // shape_value
                                                               )
                                        .getOutput();

            _log.trace("Reshaped window from {0} to {1} for broadcast-compatible windowing", windowShape,
                       newWindowShape);
        }

        auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

        windowedFrames = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "windowing_multiply"),
                                                         framesTransposed.getOutput(), windowForMultiply,
                                                         autoBroadcastAttr, nullptr, nullptr, nullptr, nullptr)
                                 .getOutput();
        _log.trace("Applied windowing function with reshaped window shape: {0}",
                   mlir::cast<vpux::NDTypeInterface>(windowForMultiply.getType()).getShape());
    }

    // RDFT operates on the last dimension of the input tensor
    const auto windowedFramesType = mlir::cast<vpux::NDTypeInterface>(windowedFrames.getType());
    const auto windowedFramesRank = windowedFramesType.getRank();
    SmallVector<int64_t> rdftAxes = {static_cast<int64_t>(windowedFramesRank - 1)};
    SmallVector<int64_t> rdftSignalSize = {frameSizeVal};

    const auto rdftAxesAttr = getIntArrayAttr(ctx, rdftAxes);
    const auto rdftSignalSizeAttr = getIntArrayAttr(ctx, rdftSignalSize);

    // RDFT output: [batch, numFrames, frameSizeVal/2+1, 2]
    SmallVector<int64_t> rdftOutputShape = {batchSize, numFrames, frameSizeVal / 2 + 1, 2};
    const auto rdftOutputType = mlir::RankedTensorType::get(rdftOutputShape, elemType);

    auto rdftOp = rewriter.create<IE::RDFTOp>(appendLoc(loc, "rdft"), rdftOutputType, windowedFrames, nullptr, nullptr,
                                              rdftAxesAttr, rdftSignalSizeAttr);
    _log.trace("Applied RDFT, output shape: {0}", rdftOutputShape);

    auto rdftOutput = rdftOp.getResult();

    mlir::Value finalOutput = rdftOutput;
    if (transposeFrames) {
        SmallVector<int64_t> transposeOrder = {0, 2, 1, 3};
        SmallVector<int64_t> transposedShape = {batchSize, frameSizeVal / 2 + 1, numFrames, 2};

        const auto transposedType = mlir::RankedTensorType::get(transposedShape, elemType);
        const auto permutationMap = mlir::AffineMap::getPermutationMap(transposeOrder, ctx);
        const auto permutationMapAttr = mlir::AffineMapAttr::get(permutationMap);

        finalOutput = rewriter.create<IE::TransposeOp>(appendLoc(loc, "transpose"), transposedType, rdftOutput, nullptr,
                                                       permutationMapAttr)
                              .getOutput();
        _log.trace("Applied transpose, final shape: {0}", transposedShape);
    }

    if (signalShape.size() == 1) {
        const auto finalOutputType = mlir::cast<vpux::NDTypeInterface>(finalOutput.getType());
        const auto finalOutputShape = finalOutputType.getShape();

        SmallVector<int64_t> squeezedShape;
        for (size_t i = 1; i < finalOutputShape.size(); ++i) {
            squeezedShape.push_back(finalOutputShape.raw()[i]);
        }

        const auto squeezedType = mlir::RankedTensorType::get(squeezedShape, elemType);
        const auto squeezedShapeAttr = getIntArrayAttr(ctx, squeezedShape);

        finalOutput = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "output_squeeze"), squeezedType, finalOutput,
                                                     squeezedShapeAttr)
                              .getOutput();
        _log.trace("Squeezed output to shape: {0}", squeezedShape);
    }

    _log.trace("Final output shape before replacement: {0}",
               mlir::cast<vpux::NDTypeInterface>(finalOutput.getType()).getShape());

    rewriter.replaceOp(origOp, finalOutput);

    _log.trace("Successfully decomposed STFT using convolution-based approach");

    return mlir::success();
}

//
// DecomposeSTFTPass
//

class DecomposeSTFTPass final : public IE::impl::DecomposeSTFTBase<DecomposeSTFTPass> {
public:
    explicit DecomposeSTFTPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void DecomposeSTFTPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addIllegalOp<IE::STFTOp>();
    target.addLegalOp<IE::ConvolutionOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::RDFTOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<STFTOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeSTFTPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeSTFTPass(Logger log) {
    return std::make_unique<DecomposeSTFTPass>(log);
}
