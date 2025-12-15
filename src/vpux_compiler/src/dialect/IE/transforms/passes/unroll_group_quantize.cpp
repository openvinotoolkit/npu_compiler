//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_UNROLLGROUPQUANTIZE
#define GEN_PASS_DEF_UNROLLGROUPQUANTIZE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

template <typename ConcreteOp>
class GenericUnrollBase : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericUnrollBase(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;
    mlir::Value getValue(const mlir::ValueRange values, const int64_t idx) const;
    SmallVector<mlir::Value> splitValue(const mlir::Value val, const int64_t axis,
                                        mlir::PatternRewriter& rewriter) const;

private:
    virtual SmallVector<mlir::Value> splitInputs(ConcreteOp origOp, const int64_t axis,
                                                 mlir::PatternRewriter& rewriter) const = 0;
    virtual std::set<int64_t> findAxes(ConcreteOp origOp) const = 0;

    virtual bool isBeneficialForUnroll(ConcreteOp origOp) const;

private:
    Logger _log;
};

// Dispatch the case when the shapes of quantization parameters don't match.
// For example fqLow = 1x1x1, fqHigh = 16x1x64.
// In that case fqHigh is split into an array of 16 tensors.
// fqLow array will contain only one value.
template <typename ConcreteOp>
mlir::Value GenericUnrollBase<ConcreteOp>::getValue(const mlir::ValueRange values, const int64_t idx) const {
    if (values.size() == 1) {
        return values[0];
    }
    VPUX_THROW_UNLESS(idx < checked_cast<int64_t>(values.size()), "Out of bounds access: index: {0}, values: {1}", idx,
                      values.size());
    return values[idx];
}

// Split a value by specified axis.
// 1x3x1x16 with axis 1 will be split in three 1x1x1x16 values
// 1x1x2x16 with axis 2 will be split in two 1x1x1x16 values
// 1x3x1x16 with axis 2 won't be split and will return a vector with only one 1x3x1x16 element
template <typename ConcreteOp>
SmallVector<mlir::Value> GenericUnrollBase<ConcreteOp>::splitValue(const mlir::Value val, const int64_t axis,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto valShape = getShape(val);
    VPUX_THROW_UNLESS(axis < checked_cast<int64_t>(valShape.size()), "Cannot split shape {0} by axis {1}", valShape,
                      axis);
    const auto groups = valShape[Dim(axis)];
    if (groups == 1) {
        return SmallVector<mlir::Value>{val};
    }
    auto staticSizes = to_small_vector(valShape);
    staticSizes[axis] = 1;
    const auto staticSizesAttr = getIntArrayAttr(rewriter.getContext(), staticSizes);
    SmallVector<mlir::Value> inputChunks;
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(val.getLoc(), "_slice_{0}", idx);
        SmallVector<int64_t> offsets(valShape.size(), 0);
        offsets[axis] = idx;
        const auto offsetsAttr = getIntArrayAttr(rewriter.getContext(), offsets);
        auto slice = rewriter.create<IE::SliceOp>(loc, val, offsetsAttr, staticSizesAttr);

        inputChunks.push_back(slice.getResult());
    }
    return inputChunks;
}

template <typename ConcreteOp>
bool GenericUnrollBase<ConcreteOp>::isBeneficialForUnroll(ConcreteOp) const {
    return true;
}

template <typename ConcreteOp>
mlir::LogicalResult GenericUnrollBase<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (mlir::isa_and_nonnull<IE::GroupConvolutionOp>(*origOp.getResult().user_begin())) {
        return mlir::failure();
    }

    const auto axes = findAxes(origOp);
    // Cases when there's only one axis or there are no axes at all are fine. Nothing to do.
    if (axes.size() <= 1) {
        return mlir::failure();
    }

    if (!isBeneficialForUnroll(origOp)) {
        return mlir::failure();
    }
    // Unroll the eligible axis.
    // If an affine reshape collapses two dimensions together, the compiler must unroll the first dimension of the two
    // e.g.: axis C in [C, 128, W] -> [C * 128, W], axis H in [C, H, 128] -> [C, H * 128].
    auto axis = *axes.begin();

    if (auto child = mlir::dyn_cast<IE::AffineReshapeOp>(*origOp->getUsers().begin())) {
        auto affineInShape = getShape(child.getInput());
        auto affineOutShape = getShape(child.getOutput());
        for (const auto axisIt : axes) {
            if (axisIt >= checked_cast<int64_t>(affineOutShape.size()) ||
                affineOutShape[Dim(axisIt)] != affineInShape[Dim(axisIt)]) {
                axis = axisIt;
                break;
            }
        }
    }

    const auto fqChunks = splitInputs(origOp, axis, rewriter);
    auto concatOp = rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "output_concat"), fqChunks, axis);
    rewriter.replaceOp(origOp, concatOp.getResult());
    return mlir::success();
}

class UnrollFakeQuantize final : public GenericUnrollBase<IE::FakeQuantizeOp> {
public:
    UnrollFakeQuantize(mlir::MLIRContext* ctx, Logger log): GenericUnrollBase<IE::FakeQuantizeOp>(ctx, log) {
    }

private:
    SmallVector<mlir::Value> splitInputs(IE::FakeQuantizeOp origOp, const int64_t axis,
                                         mlir::PatternRewriter& rewriter) const override;
    std::set<int64_t> findAxes(IE::FakeQuantizeOp origOp) const override;
};

// Split the inputs (data, input_high, input_low, output_high, output_low) by specified axis.
// The method returns a vector of fake quantize operations.
// IE.FakeQuantize with data = 1x2x8x16, in_low = in_high = 1x1x1x1, out_low = out_high = 1x2x1x16
// will be split by channels into 2 FQ operations:
// IE.FakeQuantize with data = 1x1x8x16, in_low = in_high = 1x1x1x1, out_low = out_high = 1x1x1x16
SmallVector<mlir::Value> UnrollFakeQuantize::splitInputs(IE::FakeQuantizeOp fqOp, const int64_t axis,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto data = splitValue(fqOp.getInput(), axis, rewriter);
    const auto inLow = splitValue(fqOp.getInputLow(), axis, rewriter);
    const auto inHigh = splitValue(fqOp.getInputHigh(), axis, rewriter);
    const auto outLow = splitValue(fqOp.getOutputLow(), axis, rewriter);
    const auto outHigh = splitValue(fqOp.getOutputHigh(), axis, rewriter);
    SmallVector<mlir::Value> fqResults;
    const auto groups = data.size();
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(fqOp.getLoc(), "_slice_{0}", idx);
        auto reducedFq = rewriter.create<IE::FakeQuantizeOp>(
                loc, data[idx], getValue(inLow, idx), getValue(inHigh, idx), getValue(outLow, idx),
                getValue(outHigh, idx), fqOp.getLevelsAttr(), fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcast());
        fqResults.push_back(reducedFq.getResult());
    }
    return fqResults;
}

std::set<int64_t> UnrollFakeQuantize::findAxes(IE::FakeQuantizeOp origOp) const {
    return IE::findAxes(origOp);
}

class UnrollDynamicDequantize final : public GenericUnrollBase<IE::DynamicDequantizeOp> {
public:
    UnrollDynamicDequantize(mlir::MLIRContext* ctx, Logger log): GenericUnrollBase<IE::DynamicDequantizeOp>(ctx, log) {
    }

private:
    SmallVector<mlir::Value> splitInputs(IE::DynamicDequantizeOp origOp, const int64_t axis,
                                         mlir::PatternRewriter& rewriter) const override;
    std::set<int64_t> findAxes(IE::DynamicDequantizeOp origOp) const override;
    bool isBeneficialForUnroll(IE::DynamicDequantizeOp origOp) const override;
};

SmallVector<mlir::Value> UnrollDynamicDequantize::splitInputs(IE::DynamicDequantizeOp origOp, const int64_t axis,
                                                              mlir::PatternRewriter& rewriter) const {
    const auto input = splitValue(origOp.getInput(), axis, rewriter);
    const auto scale = splitValue(origOp.getScale(), axis, rewriter);
    bool hasZeroPoint = false;
    SmallVector<mlir::Value> zeroPoint;
    if (origOp.getZp() != nullptr) {
        hasZeroPoint = true;
        zeroPoint = splitValue(origOp.getZp(), axis, rewriter);
    }
    SmallVector<mlir::Value> deQuantizeResults;
    const auto groups = input.size();
    for (const auto& idx : irange(groups)) {
        const auto loc = appendLoc(origOp.getLoc(), "_slice_{0}", idx);
        auto reduceDequantize = rewriter.create<IE::DynamicDequantizeOp>(
                loc, getValue(input, idx), getValue(scale, idx), hasZeroPoint ? getValue(zeroPoint, idx) : nullptr,
                origOp.getDstElemType());
        deQuantizeResults.push_back(reduceDequantize.getResult());
    }
    return deQuantizeResults;
}

std::set<int64_t> UnrollDynamicDequantize::findAxes(IE::DynamicDequantizeOp origOp) const {
    return IE::findAxes(origOp);
}

bool isValidFullyConnectedOp(IE::FullyConnectedOp fcOp) {
    if (fcOp == nullptr) {
        return false;
    }

    auto outShape = getShape(fcOp.getOutput());
    return outShape.size() == 2 && outShape[Dim(0)] == 1;
}

bool isValidAffineReshape(IE::AffineReshapeOp reshape) {
    if (!reshape->hasOneUse()) {
        return false;
    }

    const auto reshapeInputDims = getShape(reshape.getInput());
    const auto reshapeOutputDims = getShape(reshape.getOutput());
    return reshapeInputDims.size() == 3 && reshapeOutputDims.size() == 2;
}

bool UnrollDynamicDequantize::isBeneficialForUnroll(IE::DynamicDequantizeOp origOp) const {
    // Benefit when: KV cache model, the first dim is one.
    // For prefill mode, it's more performant to keep the op unrolled.
    if (!origOp->hasOneUse()) {
        return false;
    }

    auto propagateTranspose = [](mlir::Operation* userOp) {
        if (!mlir::isa_and_nonnull<IE::TransposeOp>(userOp) || !userOp->hasOneUse()) {
            return userOp;
        }
        return *userOp->getUsers().begin();
    };

    auto user = propagateTranspose(*origOp->getUsers().begin());

    auto reshape = mlir::dyn_cast<IE::AffineReshapeOp>(user);
    if (reshape == nullptr) {
        auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(user);
        return isValidFullyConnectedOp(fcOp);
    }

    if (!isValidAffineReshape(reshape)) {
        return false;
    }

    user = propagateTranspose(*reshape->getUsers().begin());

    auto fcOp = mlir::dyn_cast<IE::FullyConnectedOp>(user);
    return isValidFullyConnectedOp(fcOp);
}

class UnrollGroupQuantizePass final : public IE::impl::UnrollGroupQuantizeBase<UnrollGroupQuantizePass> {
public:
    explicit UnrollGroupQuantizePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void UnrollGroupQuantizePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnrollFakeQuantize>(&ctx, _log);
    patterns.add<UnrollDynamicDequantize>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createUnrollGroupQuantizePass(Logger log) {
    return std::make_unique<UnrollGroupQuantizePass>(log);
}
