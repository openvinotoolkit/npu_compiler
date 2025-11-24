//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
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

// Helper function to create a slice operation for frame extraction
IE::SliceOp createFrameSlice(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Value input,
                             ArrayRef<int64_t> inputShape, int64_t frameIdx, int64_t frameStep, int64_t frameSize,
                             mlir::Type elemType, mlir::MLIRContext* ctx) {
    SmallVector<int64_t> offsets(inputShape.size(), 0);
    SmallVector<int64_t> sizes = to_small_vector(inputShape);
    offsets[offsets.size() - 1] = frameIdx * frameStep;  // Start at frame offset
    sizes[sizes.size() - 1] = frameSize;                 // Frame size length

    SmallVector<int64_t> frameShape = to_small_vector(inputShape);
    frameShape[frameShape.size() - 1] = frameSize;

    return rewriter.create<IE::SliceOp>(appendLoc(loc, "_frame_{0}_slice", frameIdx),
                                        mlir::RankedTensorType::get(frameShape, elemType), input,
                                        getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sizes));
}

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

    if (window) {
        const auto windowType = mlir::cast<vpux::NDTypeInterface>(window.getType());
        const auto windowShape = windowType.getShape();

        if (windowShape.size() != 1 || windowShape.raw()[0] != frameSizeVal) {
            _log.error("Window shape must be [frame_size={0}], got {1}", frameSizeVal, windowShape);
            return mlir::failure();
        }
    }

    SmallVector<mlir::Value> frames;
    frames.reserve(numFrames);

    SmallVector<int64_t> inputSignalShape = to_small_vector(signalShape.raw());

    for (int64_t frameIdx = 0; frameIdx < numFrames; frameIdx++) {
        auto sliceOp = createFrameSlice(rewriter, loc, signal, inputSignalShape, frameIdx, frameStepVal, frameSizeVal,
                                        elemType, ctx);
        frames.push_back(sliceOp.getResult());
        _log.trace("Created frame {0}: offset={1}, size={2}", frameIdx, frameIdx * frameStepVal, frameSizeVal);
    }

    SmallVector<mlir::Value> reshapedFrames;
    reshapedFrames.reserve(numFrames);

    for (auto frame : frames) {
        SmallVector<int64_t> frameWithAxisShape = {batchSize, 1, frameSizeVal};
        const auto frameWithAxisType = mlir::RankedTensorType::get(frameWithAxisShape, elemType);
        const auto frameWithAxisShapeAttr = getIntArrayAttr(ctx, frameWithAxisShape);

        auto frameReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "_frame_add_axis"), frameWithAxisType,
                                                             frame, nullptr, false, frameWithAxisShapeAttr);
        reshapedFrames.push_back(frameReshapeOp.getOutput());
    }

    const auto concatAxis = 1;
    const auto concatAxisAttr = getIntAttr(ctx, concatAxis);
    auto concatOp = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_frames_concat"), reshapedFrames, concatAxisAttr);

    auto framesResult = concatOp.getResult();
    const auto actualFramesType = mlir::cast<vpux::NDTypeInterface>(framesResult.getType());
    const auto actualFramesShape = actualFramesType.getShape();

    mlir::Value windowedFrames = framesResult;
    if (window) {
        auto autoBroadcastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

        windowedFrames = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_windowing_multiply"), framesResult, window,
                                                         autoBroadcastAttr, nullptr, nullptr, nullptr, nullptr)
                                 .getOutput();
        _log.trace("Applied windowing function");
    }

    SmallVector<int64_t> rdftAxes = {static_cast<int64_t>(actualFramesShape.size() - 1)};
    SmallVector<int64_t> rdftSignalSize = {frameSizeVal};

    const auto rdftAxesAttr = getIntArrayAttr(ctx, rdftAxes);
    const auto rdftSignalSizeAttr = getIntArrayAttr(ctx, rdftSignalSize);

    SmallVector<int64_t> rdftOutputShape = to_small_vector(actualFramesShape.raw());
    rdftOutputShape.back() = frameSizeVal / 2 + 1;
    rdftOutputShape.push_back(2);

    const auto rdftOutputType = mlir::RankedTensorType::get(rdftOutputShape, elemType);

    auto rdftOp = rewriter.create<IE::RDFTOp>(appendLoc(loc, "_rdft"), rdftOutputType, windowedFrames, nullptr, nullptr,
                                              rdftAxesAttr, rdftSignalSizeAttr);

    auto rdftOutput = rdftOp.getResult();

    mlir::Value finalOutput = rdftOutput;
    if (transposeFrames) {
        SmallVector<int64_t> transposeOrder = {0, 2, 1, 3};
        SmallVector<int64_t> transposedShape = to_small_vector(rdftOutputShape);
        std::swap(transposedShape[1], transposedShape[2]);

        const auto transposedType = mlir::RankedTensorType::get(transposedShape, elemType);
        const auto permutationMap = mlir::AffineMap::getPermutationMap(transposeOrder, ctx);
        const auto permutationMapAttr = mlir::AffineMapAttr::get(permutationMap);

        finalOutput = rewriter.create<IE::TransposeOp>(appendLoc(loc, "_transpose"), transposedType, rdftOutput,
                                                       nullptr, permutationMapAttr)
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

        auto squeezeOp = rewriter.create<IE::ReshapeOp>(appendLoc(loc, "_output_squeeze"), squeezedType, finalOutput,
                                                        nullptr, false, squeezedShapeAttr);
        finalOutput = squeezeOp.getOutput();
    }
    rewriter.replaceOp(origOp, finalOutput);

    _log.trace("Successfully decomposed STFT operation with {0} unrolled frames", numFrames);

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
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
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
