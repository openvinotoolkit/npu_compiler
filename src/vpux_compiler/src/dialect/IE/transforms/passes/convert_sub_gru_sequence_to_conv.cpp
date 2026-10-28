//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSUBGRUSEQUENCETOCONV
#define GEN_PASS_DEF_CONVERTSUBGRUSEQUENCETOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool shouldConvertGRUSequenceOp(IE::GRUSequenceOp origOp, Logger log) {
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputData().getType());
    const auto inputShape = inputType.getShape().raw();
    auto inputSize = static_cast<float>(inputShape[2]);
    auto batchSize = static_cast<float>(inputShape[0]);
    auto seqLength = static_cast<float>(origOp.getSeqLength());
    auto hiddenSize = static_cast<float>(origOp.getHiddenSize());
    // IllegalThreshold is used to determine if convert subGRUSequence into convolution. After performance
    // measurement, the pass can improve performance when illegalThreshold less than or equal to 0.1568. The
    // performance improvement is about 0.0% when illegalThreshold equals 0.1568 according to test.
    const auto illegalThreshold = 0.1568;
    auto currentIllegalFactor = inputSize / (batchSize * seqLength * hiddenSize);
    if (currentIllegalFactor > illegalThreshold) {
        return false;
    }
    log.nest(1).trace("The pass is used to convert SubGRUSequence into convoltion for better performance.");
    return true;
}

//
// SubGRUSequenceToConvPass
//

class ConvertSubGRUSequenceToConvPass final :
        public IE::impl::ConvertSubGRUSequenceToConvBase<ConvertSubGRUSequenceToConvPass> {
public:
    explicit ConvertSubGRUSequenceToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

public:
    class GRUSequenceOpConverter;

private:
    void safeRunOnFunc() final;
};

//
// ConvertGRUSequenceOpConverter
//

class ConvertSubGRUSequenceToConvPass ::GRUSequenceOpConverter final :
        public mlir::OpRewritePattern<IE::GRUSequenceOp> {
public:
    GRUSequenceOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GRUSequenceOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GRUSequenceOp gruSequenceOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/*
    input iniHidState weights reWeights biases
       \         \       |       /       /
           \      \      |      /     /
               \   \     |     /   /
                   GRUSequence
                         |
                  reuslt0 result1

                        ||
                       \  /
                        \/

      input weights
        |      |
    reshape  reshape
        \      /
       convolution
            |
         reshape
            |
       permuteCast
            |
          result iniHidState reWeights biases
             \        |          |       /
                \     |          |     /
                  GRUSequenceLastPart
                          |
                   reuslt0 result1
*/

mlir::LogicalResult ConvertSubGRUSequenceToConvPass::GRUSequenceOpConverter::matchAndRewrite(
        IE::GRUSequenceOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertGRUSequenceOp(origOp, _log)) {
        return mlir::failure();
    }

    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto ctx = rewriter.getContext();
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputData().getType());
    const auto inputShape = inputType.getShape().raw();
    const auto inputShapeDim = inputShape.size();
    if (inputShapeDim != 3) {
        _log.trace("Expected dimension of input shape equals 3, but got '{0}'", inputShapeDim);
        return mlir::failure();
    }
    const auto batchSize = inputShape[0];
    const auto inputSize = inputShape[2];

    const auto seqLength = origOp.getSeqLength();

    const auto weightsType = mlir::cast<vpux::NDTypeInterface>(origOp.getWeights().getType());
    const auto weightsShape = weightsType.getShape().raw();

    auto newInputShape = getIntArrayAttr(ctx, SmallVector<int64_t>({batchSize * seqLength, 1, 1, inputSize}));
    auto newWeightsShape = getIntArrayAttr(ctx, SmallVector<int64_t>({1, 1, weightsShape[1], inputSize}));

    auto inputReshapeOp =
            IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "in_reshape"), origOp.getInputData(), newInputShape);
    auto weightsReshapeOp =
            IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "weights_reshape"), origOp.getWeights(), newWeightsShape);

    auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    auto padsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto padsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    auto convolutionOp = IE::ConvolutionOp::create(rewriter, origOp.getLoc(), weightsReshapeOp.getOutput(),
                                                   inputReshapeOp.getOutput(), strides, padsBegin, padsEnd, dilations);
    auto newResultShape = getIntArrayAttr(ctx, SmallVector<int64_t>({batchSize, seqLength, weightsShape[1], 1}));

    auto resultReshapeOp = IE::ReshapeOp::create(rewriter, takeOpLoc(origOp, "out_reshape"), convolutionOp.getOutput(),
                                                 newResultShape);

    auto memPerm = mlir::AffineMapAttr::get(DimsOrder::NCHW.toAffineMap(ctx));
    auto dstOrder = mlir::AffineMapAttr::get(DimsOrder::NHWC.toAffineMap(ctx));
    auto permuteCastOp = IE::PermuteCastOp::create(rewriter, takeOpLoc(origOp, "out_permute"),
                                                   resultReshapeOp.getOutput(), dstOrder, memPerm);

    auto gruSequenceLastPartOp = IE::GRUSequenceLastPartOp::create(
            rewriter, takeOpLoc(origOp, "gru_last_part"), permuteCastOp.getOutput(), origOp.getInitialHiddenState(),
            origOp.getRecurrenceWeights(), origOp.getBiases(), origOp.getHiddenSizeAttr(), origOp.getSeqLengthAttr(),
            origOp.getDirectionAttr(), origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());

    rewriter.replaceOp(origOp,
                       {gruSequenceLastPartOp.getMiddleHiddenState(), gruSequenceLastPartOp.getOutputHiddenState()});

    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertSubGRUSequenceToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GRUSequenceOpConverter>(&ctx, _log);

    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createConvertSubGRUSequenceToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSubGRUSequenceToConvPass(Logger log) {
    return std::make_unique<ConvertSubGRUSequenceToConvPass>(log);
}
