//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEGRUSEQUENCE
#define GEN_PASS_DEF_DECOMPOSEGRUSEQUENCE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// GRUSequenceOpConverter
//

class GRUSequenceOpConverter final : public mlir::OpRewritePattern<IE::GRUSequenceOp> {
public:
    GRUSequenceOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GRUSequenceOp>(ctx, benefit), _log(log) {
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
        \      /
         MatMul
            |
           Add (biasses, 3 of them)
            |
          result iniHidState reWeights biases(last one)
             \        |          |       /
                \     |          |     /
                  GRUSequenceLastPart
                          |
                   reuslt0 result1
*/

mlir::LogicalResult GRUSequenceOpConverter::matchAndRewrite(IE::GRUSequenceOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());
    const auto shouldLinearBeforeReset = origOp.getShouldLinearBeforeReset();
    auto ctx = rewriter.getContext();
    const auto loc = origOp.getLoc();
    const auto minHiddenSize = 32;  // Splitting in multiple Ops this small amount of processing is not optimal.
    if ((origOp.getHiddenSize() < minHiddenSize)) {
        _log.trace("GRUSequence not split. HiddenSize={0} is smaller that '{1}'; full shave op is faster.",
                   origOp.getHiddenSize(), minHiddenSize);
        return mlir::failure();
    }

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInputData().getType());
    auto inputData = origOp.getInputData();
    const auto weights = origOp.getWeights();
    const auto inputShape = inputType.getShape().raw();
    const auto inputShapeDim = inputShape.size();
    if (inputShapeDim != 3) {
        _log.trace("Expected dimension of input shape equals 3, but got '{0}'", inputShapeDim);
        return mlir::failure();
    }

    const auto axisOne = 1;
    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{axisOne});

    auto newInputData = rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_inputDataUnsqueeze"), inputData, nullptr,
                                                         axisOneArrayAttr);

    mlir::Value newWeights =
            rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_weightsUnsqueeze"), weights, nullptr, axisZeroArrayAttr);

    auto matMulInputOp =
            rewriter.create<IE::MatMulOp>(appendLoc(loc, "_matMul"), newInputData, newWeights, false, true, nullptr);
    auto newInputForGru = matMulInputOp.getOutput();

    auto biases = origOp.getBiases();
    auto biasesSqueezeOp = rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_sqeenze_bias_sequence"), biases, nullptr,
                                                          axisZeroArrayAttr);
    biases = biasesSqueezeOp.getOutput();

    // include biases in pre-processing, 1 op for all sequence.
    if (!shouldLinearBeforeReset) {
        auto numpyBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
        auto bZrOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
        auto bZrSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{3 * origOp.getHiddenSize()});
        auto bZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_bZr_slice"), biases, bZrOffsets, bZrSizes);
        auto zrt = rewriter.create<IE::AddOp>(appendLoc(loc, "_zrt_add0"), newInputForGru, bZr, numpyBroadcastTypeAttr,
                                              nullptr, nullptr, nullptr, nullptr);
        newInputForGru = zrt.getOutput();
    }
    auto gruSequenceLastPartOp = rewriter.create<IE::GRUSequenceLastPartOp>(
            takeOpLoc(origOp, "gru_last_part"), newInputForGru, origOp.getInitialHiddenState(),
            origOp.getRecurrenceWeights(), biases, origOp.getHiddenSizeAttr(), origOp.getSeqLengthAttr(),
            origOp.getDirectionAttr(), origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());
    _log.trace("At Decompose '{0}' produce '{1}'", origOp->getName(), gruSequenceLastPartOp);

    rewriter.replaceOp(origOp,
                       {gruSequenceLastPartOp.getMiddleHiddenState(), gruSequenceLastPartOp.getOutputHiddenState()});
    return mlir::success();
}

//
// UnrollGRUSequenceLastPartToGRUCellsRewriter
//

// Convert an GRUSequenceLastPart operator to seqLength GRUCell operators.
// Every GRUCell will be split in 1 MatMul and remaining GruGates operator if shouldLinearBeforeReset is true.
// If shouldLinearBeforeReset is false, Cell will be decompose in all basic operation components.
class UnrollGRUSequenceLastPartToGRUCellsRewriter final : public mlir::OpRewritePattern<IE::GRUSequenceLastPartOp> {
public:
    UnrollGRUSequenceLastPartToGRUCellsRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::GRUSequenceLastPartOp>(ctx, benefit), _log(std::move(log)) {
        this->setDebugName("UnrollGRUSequenceLastPartToGRUCellsRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GRUSequenceLastPartOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;

    mlir::Value matchAndRewriteGRUCellCutOp(mlir::Location loc, ::mlir::Value inputData, ::mlir::Value hiddenState,
                                            ::mlir::Value recurrenceWeights, ::mlir::Value biasses,
                                            const bool shouldLinearBeforeReset, mlir::PatternRewriter& rewriter) const;
};

mlir::Value UnrollGRUSequenceLastPartToGRUCellsRewriter::matchAndRewriteGRUCellCutOp(
        mlir::Location loc, ::mlir::Value inputData, ::mlir::Value hiddenState, ::mlir::Value recurrenceWeights,
        ::mlir::Value biasses, const bool shouldLinearBeforeReset, mlir::PatternRewriter& rewriter) const {
    _log.trace("Unroll GruCell at '{0}'", loc);
    auto* ctx = rewriter.getContext();
    auto hiddenStateShape = getShape(hiddenState);
    VPUX_THROW_UNLESS(hiddenStateShape.size() == 2, "initial_hidden_state rank expected to be 2, got {0}",
                      hiddenStateShape.size());
    const auto batchSize = hiddenStateShape[Dim(0)];
    const auto hiddenSize = hiddenStateShape[Dim(1)];

    if (shouldLinearBeforeReset) {
        // hidenData = H * (R^T) -> [batch_size, 3 * hidden_size]
        auto hr = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_hidden_matmul"), hiddenState, recurrenceWeights, false,
                                                true, nullptr);
        auto gruGates = rewriter.create<IE::GRUGatesOp>(appendLoc(loc, "_gates"), inputData, hiddenState,
                                                        hr.getOutput(), biasses);
        return gruGates.getOutputHiddenState();
    }
    auto xw = inputData;

    // hr = H * (R^T) -> [batch_size, 3 * hidden_size]
    auto hr = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_hr_matmul"), hiddenState, recurrenceWeights, false, true,
                                            nullptr);

    auto xwZrOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto xwZrSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, 2 * hiddenSize});
    auto xwZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_xwZr_slice"), xw, xwZrOffsets, xwZrSizes);
    auto hrZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_hrZr_slice"), hr, xwZrOffsets, xwZrSizes);

    auto noneOrExplicitBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    auto numpyBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);

    // zrt = sigmoid(xwZr + hrZr + bZr) -> [batch_size, 2 * hidden_size]
    mlir::Value zrt = rewriter.create<IE::AddOp>(appendLoc(loc, "_zrt_add1"), xwZr, hrZr,
                                                 noneOrExplicitBroadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);
    zrt = rewriter.create<IE::SigmoidOp>(appendLoc(loc, "_zrt_sigmoid"), zrt);
    auto zrtSplitOp = rewriter.create<IE::SplitOp>(appendLoc(loc, "_zrt_split"), zrt, nullptr,
                                                   /*numSplits=*/getIntAttr(ctx, 2), /*axisValue=*/getIntAttr(ctx, 1));
    auto zt = zrtSplitOp.getResult(0);
    auto rt = zrtSplitOp.getResult(1);

    auto xwHOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 2 * hiddenSize});
    auto xwHSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, hiddenSize});
    auto xwH = rewriter.create<IE::SliceOp>(appendLoc(loc, "_xwH_slice"), xw, xwHOffsets, xwHSizes);

    mlir::Value ht;

    auto rHOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{2 * hiddenSize, 0});
    auto rHSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{hiddenSize, hiddenSize});
    auto rH = rewriter.create<IE::SliceOp>(appendLoc(loc, "_rh_slice"), recurrenceWeights, rHOffsets, rHSizes);

    // ht = tanh(xwH + (rt (.) H) * (Rh^T) + bH) -> [batch_size, hidden_size]
    ht = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_ht_mul"), rt, hiddenState, noneOrExplicitBroadcastTypeAttr,
                                         nullptr, nullptr, nullptr, nullptr);
    ht = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_ht_matMul"), ht, rH, false, true, nullptr);

    ht = rewriter.create<IE::AddOp>(appendLoc(loc, "_ht_add1"), xwH, ht, noneOrExplicitBroadcastTypeAttr, nullptr,
                                    nullptr, nullptr, nullptr);
    ht = rewriter.create<IE::TanhOp>(appendLoc(loc, "_ht_tanh"), ht);

    // Ht = (1 - zt) (.) ht + zt (.) H -> [batch_size, hidden_size]
    auto elemType = mlir::cast<vpux::NDTypeInterface>(zt.getType()).getElementType();
    auto one = Const::createConst(rewriter, loc, mlir::RankedTensorType::get({1}, elemType), ArrayRef({1.0f}));
    auto sub = rewriter.create<IE::SubtractOp>(appendLoc(loc, "_sub"), one, zt, numpyBroadcastTypeAttr, nullptr,
                                               nullptr, nullptr, nullptr);
    auto mul1 = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_mul1"), zt, hiddenState,
                                                noneOrExplicitBroadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);
    auto mul2 = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_mul2"), sub, ht, noneOrExplicitBroadcastTypeAttr,
                                                nullptr, nullptr, nullptr, nullptr);
    auto nextHiddenState = rewriter.create<IE::AddOp>(
            appendLoc(loc, "_add"), mul1, mul2, noneOrExplicitBroadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);

    return nextHiddenState.getOutput();
}

mlir::LogicalResult UnrollGRUSequenceLastPartToGRUCellsRewriter::matchAndRewrite(
        IE::GRUSequenceLastPartOp op, mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", op->getName(), op->getLoc());
    _log.trace("UnrollGRUSequenceLastPartToGRUCellsRewriter '{0}'", op);
    const auto direction = op.getDirection();
    VPUX_THROW_WHEN(direction != IE::RNNSequenceDirection::FORWARD && direction != IE::RNNSequenceDirection::REVERSE,
                    "Expected direction to be FORWARD or REVERSE, got {0}", direction);
    const auto isReverseDirection = direction == IE::RNNSequenceDirection::REVERSE;

    const auto ctx = rewriter.getContext();
    const auto loc = op.getLoc();

    const auto axisZeroArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
    const auto axisOneArrayAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1});

    int squeezeIdx = 0;
    const auto squeezeOnDim = [&](mlir::Value input, const mlir::ArrayAttr& axis) -> mlir::Value {
        if (!input) {
            return nullptr;
        }
        return rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_sqeenze_{0}", squeezeIdx++), input, nullptr, axis);
    };

    const mlir::Value inputData = squeezeOnDim(op.getFirstPartOutput(), axisOneArrayAttr);
    mlir::Value hiddenState = squeezeOnDim(op.getInitialHiddenState(), axisOneArrayAttr);
    const mlir::Value recurrenceWeights = squeezeOnDim(op.getRecurrenceWeights(), axisZeroArrayAttr);
    mlir::Value biases = op.getBiases();

    const auto inputDataShape = getShape(inputData).raw();
    VPUX_THROW_UNLESS(inputDataShape.size() == 3, "inputData expected to be of rank 3, got {0}", inputDataShape.size());
    const auto sequenceLenght = op.getSeqLength();

    SmallVector<int64_t> sliceOffsets(inputDataShape.size(), 0);
    SmallVector<int64_t> sliceSizes(inputDataShape);
    sliceSizes[1] = 1;
    const auto sliceSizesAttr = getIntArrayAttr(ctx, sliceSizes);

    SmallVector<mlir::Value> gruCellResults;

    for (int i = 0; i < sequenceLenght; i++) {
        sliceOffsets[1] = isReverseDirection ? sequenceLenght - 1 - i : i;
        auto sliceOp = rewriter.create<IE::SliceOp>(appendLoc(loc, "_slice_{0}", i), inputData,
                                                    getIntArrayAttr(ctx, sliceOffsets), sliceSizesAttr);
        auto sqeezeOp =
                rewriter.create<IE::SqueezeOp>(appendLoc(loc, "_squeeze_{0}", i), sliceOp, nullptr, axisOneArrayAttr);

        auto GRUCellCutOp =
                matchAndRewriteGRUCellCutOp(appendLoc(loc, "_gruCell_{0}", i), sqeezeOp.getOutput(), hiddenState,
                                            recurrenceWeights, biases, op.getShouldLinearBeforeReset(), rewriter);

        auto unsqueezeOp = rewriter.create<IE::UnsqueezeOp>(appendLoc(loc, "_unsqueeze_{0}", i), GRUCellCutOp, nullptr,
                                                            axisOneArrayAttr);

        gruCellResults.push_back(unsqueezeOp.getOutput());
        hiddenState = GRUCellCutOp;
    }

    if (isReverseDirection) {
        std::reverse(gruCellResults.begin(), gruCellResults.end());
    }

    mlir::Value newOutputHiddenValues = rewriter.create<IE::ConcatOp>(takeOpLoc(op, "_concat"), gruCellResults, Dim(1));
    newOutputHiddenValues = rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), newOutputHiddenValues,
                                                             nullptr, axisOneArrayAttr);
    const mlir::Value newHiddenState =
            rewriter.create<IE::UnsqueezeOp>(takeOpLoc(op, "_unsqueeze"), hiddenState, nullptr, axisOneArrayAttr);

    const SmallVector<mlir::Value> newResults{newOutputHiddenValues, newHiddenState};
    rewriter.replaceOp(op, newResults);

    return mlir::success();
}

//
// DecomposeGRUSequencePass
//

class DecomposeGRUSequencePass final : public IE::impl::DecomposeGRUSequenceBase<DecomposeGRUSequencePass> {
public:
    explicit DecomposeGRUSequencePass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeGRUSequencePass::safeRunOnFunc() {
    auto& ctx = getContext();
    const uint32_t levelCount = 2;
    const auto benefitLevels = getBenefitLevels(levelCount);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GRUSequenceOpConverter>(&ctx, benefitLevels[0], _log);
    patterns.add<UnrollGRUSequenceLastPartToGRUCellsRewriter>(&ctx, benefitLevels[1], _log);
    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeGRUSequencePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeGRUSequencePass(Logger log) {
    return std::make_unique<DecomposeGRUSequencePass>(std::move(log));
}
