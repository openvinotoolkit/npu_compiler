//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <optional>
#include <utility>

namespace vpux::IE {
#define GEN_PASS_DECL_DECOMPOSEGRUCELL
#define GEN_PASS_DEF_DECOMPOSEGRUCELL
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// GRUCellRewriter
//

class GRUCellRewriter final : public mlir::OpRewritePattern<IE::GRUCellOp> {
public:
    GRUCellRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GRUCellOp>(ctx), _log(std::move(log)) {
        this->setDebugName("GRUCellRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GRUCellOp gruCell, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// GRUCell formula:
//  zt = f(X*(Wz^T) + H*(Rz^T) + Wbz + Rbz)
//  rt = f(X*(Wr^T) + H*(Rr^T) + Wbr + Rbr)
//  ht = g(X*(Wh^T) + (rt (.) H)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
//  ht = g(X*(Wh^T) + (rt (.) (H*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
//  Ht+1 = (1 - zt) (.) ht + zt (.) H
//
// Variables:
//  X       - inputData
//  H       - initialHiddenState
//  W       - weights
//  R       - recurrenceWeights
//  Wb, Rb  - biases (optional)
//  Ht+1    - nextHiddenState
//
// Functions and operators:
//   *  - matrix multiplication
//  (.) - Hadamard product (element-wise multiplication)
//  [,] - concatenation
//   f  - sigmoid
//   g  - tanh

mlir::LogicalResult GRUCellRewriter::matchAndRewrite(IE::GRUCellOp gruCell, mlir::PatternRewriter& rewriter) const {
    auto loc = gruCell.getLoc();
    auto* ctx = gruCell.getContext();

    auto inputData = gruCell.getInputData();
    auto hiddenState = gruCell.getInitialHiddenState();
    auto weights = gruCell.getWeights();
    auto recurrenceWeights = gruCell.getRecurrenceWeights();
    auto hasBiases = gruCell.getBiases() != nullptr;
    const auto biases = hasBiases ? std::optional<mlir::Value>{gruCell.getBiases()} : std::nullopt;

    auto hiddenStateShape = getShape(hiddenState);
    VPUX_THROW_UNLESS(hiddenStateShape.size() == 2, "initial_hidden_state rank expected to be 2, got {0}",
                      hiddenStateShape.size());
    const auto batchSize = hiddenStateShape[Dim(0)];
    const auto hiddenSize = hiddenStateShape[Dim(1)];
    const auto shouldLinearBeforeReset = gruCell.getShouldLinearBeforeReset();

    // xw = X * (W^T) -> [batch_size, 3 * hidden_size]
    auto xw = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_xw_matmul"), inputData, weights, false, true, nullptr);

    // hr = H * (R^T) -> [batch_size, 3 * hidden_size]
    auto hr = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_hr_matmul"), hiddenState, recurrenceWeights, false, true,
                                            nullptr)
                      .getResult();

    auto numpyBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
    if (hasBiases) {
        // place bZr (and rbh) ahead to make it fused as bias of previous conv
        SmallVector<mlir::Value> biasInputs;

        auto bZrOffstes = getIntArrayAttr(ctx, SmallVector<int64_t>{0});
        auto bZrSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{2 * hiddenSize});
        auto bZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_bZr_slice"), biases.value(), bZrOffstes, bZrSizes);
        biasInputs.push_back(bZr.getResult());

        if (shouldLinearBeforeReset) {
            // Rbh = biases[3 * hiddenSize :] when shouldLinearBeforeReset is true
            auto rbhOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{3 * hiddenSize});
            auto rbhSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{hiddenSize});
            auto rbh = rewriter.create<IE::SliceOp>(appendLoc(loc, "_rbh_slice"), biases.value(), rbhOffsets, rbhSizes);
            biasInputs.push_back(rbh.getResult());
        } else {
            const Shape biasShape = {hiddenSize};
            std::vector<float> biasValue(hiddenSize, .0f);

            const auto biasType = mlir::RankedTensorType::get(
                    biasShape.raw(), mlir::cast<vpux::NDTypeInterface>(biases.value().getType()).getElementType(),
                    getTensorAttr(rewriter.getContext(), DimsOrder::C, nullptr));

            biasInputs.push_back(
                    Const::buildWeightsConst(rewriter, appendLoc(loc, "_rbh_slice"), biasType, ArrayRef(biasValue)));
        }

        auto hrBias = rewriter.create<IE::ConcatOp>(appendLoc(loc, "_hr_bias"), biasInputs, Dim(0));
        hr = rewriter.create<IE::AddOp>(appendLoc(loc, "_hr_add0"), hr, hrBias.getResult(), numpyBroadcastTypeAttr,
                                        nullptr, nullptr, nullptr, nullptr)
                     .getResult();
    }

    auto xwZrOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    auto xwZrSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, 2 * hiddenSize});
    auto xwZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_xwZr_slice"), xw, xwZrOffsets, xwZrSizes);
    auto hrZr = rewriter.create<IE::SliceOp>(appendLoc(loc, "_hrZr_slice"), hr, xwZrOffsets, xwZrSizes);

    auto noneOrExplicitBroadcastTypeAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);

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
    if (!shouldLinearBeforeReset) {
        auto rHOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{2 * hiddenSize, 0});
        auto rHSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{hiddenSize, hiddenSize});
        auto rH = rewriter.create<IE::SliceOp>(appendLoc(loc, "_rh_slice"), recurrenceWeights, rHOffsets, rHSizes);

        // ht = tanh(xwH + (rt (.) H) * (Rh^T) + bH) -> [batch_size, hidden_size]
        ht = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "_ht_mul"), rt, hiddenState,
                                             noneOrExplicitBroadcastTypeAttr, nullptr, nullptr, nullptr, nullptr);
        ht = rewriter.create<IE::MatMulOp>(appendLoc(loc, "_ht_matMul"), ht, rH, false, true, nullptr);
        if (hasBiases) {
            auto bHOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{2 * hiddenSize});
            auto bHSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{hiddenSize});
            auto bH = rewriter.create<IE::SliceOp>(appendLoc(loc, "_bh_slice"), biases.value(), bHOffsets, bHSizes);
            ht = rewriter.create<IE::AddOp>(appendLoc(loc, "_ht_add0"), ht, bH, numpyBroadcastTypeAttr, nullptr,
                                            nullptr, nullptr, nullptr);
        }
        ht = rewriter.create<IE::AddOp>(appendLoc(loc, "_ht_add1"), xwH, ht, noneOrExplicitBroadcastTypeAttr, nullptr,
                                        nullptr, nullptr, nullptr);
        ht = rewriter.create<IE::TanhOp>(appendLoc(loc, "_ht_tanh"), ht);
    } else {
        auto hrHOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 2 * hiddenSize});
        auto hrHSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{batchSize, hiddenSize});
        auto hrH = rewriter.create<IE::SliceOp>(appendLoc(loc, "_hrH_slice"), hr, hrHOffsets, hrHSizes);

        // ht = tanh(xwH + (rt (.) (hrH + Rbh)) + Wbh) -> [batch_size, hidden_size]
        ht = hrH;
        ht = rewriter.create<IE::MultiplyOp>(appendLoc(loc, "ht_mul"), rt, ht, noneOrExplicitBroadcastTypeAttr, nullptr,
                                             nullptr, nullptr, nullptr);
        if (hasBiases) {
            // Wbh = biases[2 * hiddenSize : 3 * hiddenSize] when shouldLinearBeforeReset is true
            auto wbhOffsets = getIntArrayAttr(ctx, SmallVector<int64_t>{2 * hiddenSize});
            auto wbhSizes = getIntArrayAttr(ctx, SmallVector<int64_t>{hiddenSize});
            auto wbh = rewriter.create<IE::SliceOp>(appendLoc(loc, "_wbh_slice"), biases.value(), wbhOffsets, wbhSizes);
            ht = rewriter.create<IE::AddOp>(appendLoc(loc, "ht_add1"), ht, wbh, numpyBroadcastTypeAttr, nullptr,
                                            nullptr, nullptr, nullptr);
        }
        ht = rewriter.create<IE::AddOp>(appendLoc(loc, "_ht_add2"), xwH, ht, noneOrExplicitBroadcastTypeAttr, nullptr,
                                        nullptr, nullptr, nullptr);
        ht = rewriter.create<IE::TanhOp>(appendLoc(loc, "_ht_tanh"), ht);
    }

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

    rewriter.replaceOp(gruCell, nextHiddenState);
    return mlir::success();
}

//
// GRUSequenceToCellRewriter
//

class GRUSequenceToCellRewriter final : public mlir::OpRewritePattern<IE::GRUSequenceOp> {
public:
    GRUSequenceToCellRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GRUSequenceOp>(ctx), _log(std::move(log)) {
        this->setDebugName("GRUSequenceToCellRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GRUSequenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GRUSequenceToCellRewriter::matchAndRewrite(IE::GRUSequenceOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto getNewReshapeOut = [&](mlir::Value input) -> mlir::Value {
        const auto shape = getShape(input).raw();
        SmallVector<int64_t> newShape(shape.begin() + 1, shape.end());

        const auto newInputShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
        return rewriter.create<IE::ReshapeOp>(origOp.getLoc(), input, nullptr, false, newInputShapeAttr).getResult();
    };

    const auto inputData = getNewReshapeOut(origOp.getInputData());
    const auto initialHiddenState = getNewReshapeOut(origOp.getInitialHiddenState());
    const auto weights = getNewReshapeOut(origOp.getWeights());
    const auto recurrenceWeights = getNewReshapeOut(origOp.getRecurrenceWeights());
    const auto biases = getNewReshapeOut(origOp.getBiases());
    auto newGRUCellOp = rewriter.create<IE::GRUCellOp>(origOp.getLoc(), inputData, initialHiddenState, weights,
                                                       recurrenceWeights, biases, origOp.getHiddenSizeAttr(),
                                                       origOp.getShouldLinearBeforeResetAttr(), origOp.getClipAttr());

    auto newOutReshape1 =
            rewriter.create<IE::ReshapeOp>(origOp.getLoc(), newGRUCellOp.getResult(), nullptr, false,
                                           getIntArrayAttr(rewriter.getContext(), getShape(origOp->getResult(0))));
    auto newOutReshape2 =
            rewriter.create<IE::ReshapeOp>(origOp.getLoc(), newGRUCellOp.getResult(), nullptr, false,
                                           getIntArrayAttr(rewriter.getContext(), getShape(origOp->getResult(1))));

    rewriter.replaceOp(origOp, {newOutReshape1.getResult(), newOutReshape2.getResult()});

    _log.trace("[{0}] Replaced with 'IE::GRUCellOp'", getDebugName());
    return mlir::success();
}

//
// DecomposeGRUCellPass
//

class DecomposeGRUCellPass final : public IE::impl::DecomposeGRUCellBase<DecomposeGRUCellPass> {
public:
    explicit DecomposeGRUCellPass(Logger log) {
        Base::initLogger(std::move(log), Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void DecomposeGRUCellPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<IE::IEDialect, Const::ConstDialect>();
    target.addIllegalOp<IE::GRUCellOp>();
    target.addDynamicallyLegalOp<IE::GRUSequenceOp>([&](IE::GRUSequenceOp origOp) {
        auto inputShape = getShape(origOp.getInputData());
        auto hidenShape = getShape(origOp.getInitialHiddenState());

        if (inputShape.empty() || hidenShape.empty() || getShape(origOp.getWeights()).empty() ||
            getShape(origOp.getRecurrenceWeights()).empty() || getShape(origOp.getBiases()).empty()) {
            return true;
        }

        // When input's batch&seq_len and hidden's batch&directions are both one, it can be transfered to GRUCell
        return inputShape[Dim(0)] != 1 || inputShape[Dim(1)] != 1 || hidenShape[Dim(0)] != 1 || hidenShape[Dim(1)] != 1;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GRUSequenceToCellRewriter>(&ctx, _log);
    patterns.add<GRUCellRewriter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createDecomposeGRUCellPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDecomposeGRUCellPass(Logger log) {
    return std::make_unique<DecomposeGRUCellPass>(std::move(log));
}
