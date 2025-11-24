//
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <tuple>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESTATICSCALE
#define GEN_PASS_DEF_FUSESTATICSCALE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
// Returns an "origin" operation of the specified type (one of Ops), ignoring
// pure view ops, for the given operation. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename Op>
Op findOriginOp(const Logger& log, mlir::Operation* current) {
    // skip "pure view ops"
    while (current && IE::isPureViewOp(current)) {
        // Note: for the sake of this pass, only single-use op chains are
        // considered
        auto operands = current->getOperands();
        if (operands.size() != 1) {
            log.trace("{0} at {1} has unexpected number of operands {2}, expected 1", current->getName(),
                      current->getLoc(), operands.size());
            return nullptr;
        }

        // ViewOp should have single user
        if (!current->hasOneUse()) {
            return nullptr;
        }

        current = operands[0].getDefiningOp();
    }

    auto originOp = mlir::dyn_cast_or_null<Op>(current);
    if (originOp == nullptr || originOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
        originOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
        return nullptr;
    }

    return originOp;
}

// Returns defining op of the specified type for some operand of the op ignoring IE.FakeQuantize ops
template <typename Op>
Op findDefiningOp(mlir::Operation* op) {
    for (const auto& operand : op->getOperands()) {
        auto parent = operand.getDefiningOp();
        while (mlir::isa_and_nonnull<IE::FakeQuantizeOp>(parent)) {
            parent = parent->getOperand(0).getDefiningOp();
        }
        if (mlir::isa_and_nonnull<Op>(parent)) {
            return mlir::cast<Op>(parent);
        }
    }
    return nullptr;
}

// Returns a "target" operation of the specified type (one of Ops), ignoring
// pure view ops, for operations that use the given operation's result. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename Op>
Op findTargetOp(const Logger&, mlir::Operation* current) {
    // Only consider operations with single use
    if (!current->hasOneUse()) {
        return nullptr;
    }

    auto user = *current->getUsers().begin();

    // skip "pure view ops"
    while (user != nullptr && IE::isPureViewOp(user)) {
        // ViewOp should have single user
        if (!user->hasOneUse()) {
            return nullptr;
        }
        user = *user->getUsers().begin();
    }

    auto targetOp = mlir::dyn_cast_or_null<Op>(user);
    if (targetOp == nullptr || targetOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
        targetOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
        return nullptr;
    }

    return targetOp;
}

bool isSuitableConstant(Const::DeclareOp op) {
    return op && op.getContentAttr().isSplat() &&
           mlir::isa<mlir::FloatType>(op.getContentAttr().getType().getElementType()) &&
           // Currently can't support negative scale, see E#138352
           !Const::hasNegativeValues(op.getContent());
}

// Checks if an operand is IE.FakeQuantize op with constOp input
auto checkFQOperand(Const::DeclareOp constOp, mlir::Value input) {
    if (auto fqOp = mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(input.getDefiningOp())) {
        return fqOp.getInput().getDefiningOp() == constOp;
    }
    return false;
};

// Returns IE.FakeQuantize op with constOp input as an operand of the op
mlir::FailureOr<IE::FakeQuantizeOp> findFQOperand(Const::DeclareOp constOp, mlir::Operation* op) {
    for (const auto& operand : op->getOperands()) {
        if (checkFQOperand(constOp, operand)) {
            return mlir::cast<IE::FakeQuantizeOp>(operand.getDefiningOp());
        }
    }
    return mlir::failure();
}

mlir::FailureOr<float> getConstSplatValue(Const::DeclareOp constOp, IE::MultiplyOp multOp) {
    auto maybeFqInput = findFQOperand(constOp, multOp);
    if (mlir::failed(maybeFqInput)) {
        return vpux::Const::template getSplatValue<float>(constOp);
    }
    auto fqOp = maybeFqInput.value();

    if (!IE::isPerTensorFQ({fqOp})) {
        return mlir::failure();
    }

    const auto inLowSplat = vpux::Const::template getSplatValue<float>(fqOp.getInputLow());
    const auto inHighSplat = vpux::Const::template getSplatValue<float>(fqOp.getInputHigh());
    const auto outLowSplat = vpux::Const::template getSplatValue<float>(fqOp.getOutputLow());
    const auto outHighSplat = vpux::Const::template getSplatValue<float>(fqOp.getOutputHigh());

    if (mlir::failed(inLowSplat) || mlir::failed(inHighSplat) || mlir::failed(outLowSplat) ||
        mlir::failed(outHighSplat)) {
        return mlir::failure();
    }
    const auto inLowVal = inLowSplat.value();
    const auto inHighVal = inHighSplat.value();
    const auto outLowVal = outLowSplat.value();
    const auto outHighVal = outHighSplat.value();

    if (!fqOp.getLevels().has_value()) {
        return mlir::failure();
    }
    const auto levels = fqOp.getLevels().value();

    auto constSplatValue = vpux::Const::template getSplatValue<float>(constOp);
    if (mlir::failed(constSplatValue)) {
        return mlir::failure();
    }
    return vpux::fakeQuantize(constSplatValue.value(), inLowVal, inHighVal, outLowVal, outHighVal, levels);
}

Const::DeclareOp getRescaledConst(Const::DeclareOp constOp, mlir::PatternRewriter& rewriter,
                                  const float& constSplatValue) {
    auto constOutputType = mlir::cast<vpux::NDTypeInterface>(constOp.getOutput().getType());
    auto contentAttr = constOp.transformContentAttr().rescale(constSplatValue).get();
    return rewriter.create<Const::DeclareOp>(constOp.getLoc(), constOutputType, std::move(contentAttr));
};

IE::FakeQuantizeOp getRescaledFQ(IE::FakeQuantizeOp fqOp, Const::DeclareOp outputLowConstOp,
                                 Const::DeclareOp outputHighConstOp, mlir::PatternRewriter& rewriter,
                                 const float& constSplatValue) {
    auto newOutputLow = getRescaledConst(outputLowConstOp, rewriter, constSplatValue);
    auto newOutputHigh = getRescaledConst(outputHighConstOp, rewriter, constSplatValue);
    return rewriter.create<IE::FakeQuantizeOp>(fqOp->getLoc(), fqOp.getType(), fqOp.getInput(), fqOp.getInputLow(),
                                               fqOp.getInputHigh(), newOutputLow, newOutputHigh, fqOp.getLevelsAttr(),
                                               fqOp.getLowFpTypeAttr(), fqOp.getAutoBroadcastAttr());
};

// E#122893: consider moving this rewriter into
// InsertReorderBetweenLayerAndConcat pass
struct InsertMultiplyBeforeConcat : public mlir::OpRewritePattern<IE::MultiplyOp> {
    InsertMultiplyBeforeConcat(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitHigh), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

mlir::LogicalResult InsertMultiplyBeforeConcat::matchAndRewrite(IE::MultiplyOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Found IE.Multiply at {0}", origOp->getLoc());
    if (origOp.getPostOpAttr() != nullptr) {
        _log.trace("Ignore: IE.Multiply is not simple (has ppe)");
        return mlir::failure();
    }

    auto constOp = findDefiningOp<Const::DeclareOp>(origOp);
    if (!isSuitableConstant(constOp)) {
        _log.trace("Ignore: IE.Multiply is not floating-point constant scalar multiplication");
        return mlir::failure();
    }
    auto concatOp = findDefiningOp<IE::ConcatOp>(origOp);
    if (concatOp == nullptr || !concatOp->hasOneUse()) {
        _log.trace("Ignore: IE.Multiply does not have suitable IE.Concat");
        return mlir::failure();
    }
    // Note: only put IE.Multiply "inside" concat block when it is feasible:
    // *each* concat branch should have a convolution (so we could actually fuse
    // multiply in all the cases)
    for (auto [index, operand] : llvm::enumerate(concatOp->getOperands())) {
        auto convOp = findOriginOp<IE::ConvolutionOp>(_log, operand.getDefiningOp());
        if (convOp == nullptr) {
            _log.trace("Ignore: IE.Multiply does not have preceding IE.Convolution in concat branch #{0} - no "
                       "possibility to fuse",
                       index);
            return mlir::failure();
        }
    }

    _log.trace("Reordering IE.Multiply at {0} and IE.Concat at {1}", origOp->getLoc(), concatOp->getLoc());
    // create a new constant suitable for usage inside concat blocks
    const auto singleConstResult = [&]() {
        VPUX_THROW_WHEN(!constOp.getContentAttr().isSplat(), "Expected splat constant in IE.Multiply");

        const auto concatInputShape = getShape(concatOp.getInputs().front());
        const SmallVector<int64_t> newShape(concatInputShape.size(), 1);  // 1x1x1x1

        // Note: in order to preserve the original constant's splat value,
        // always create an FP32 storage and then convert the resulting content
        // attr to a suitable type (either FP32 or FP16). this way, the actual
        // splat value is preserved (FP32 -> FP32) or upcasted (FP16 -> FP32)
        // internally.
        auto maybeConstSplatValue = getConstSplatValue(constOp, origOp);
        if (mlir::failed(maybeConstSplatValue)) {
            return mlir::Value();
        }
        auto constSplatValue = maybeConstSplatValue.value();

        const auto fp32TensorType = mlir::RankedTensorType::get(newShape, mlir::Float32Type::get(getContext()));
        const auto newConst = Const::createFloatConst(rewriter, appendLoc(constOp->getLoc(), "_to_mult_scalar"),
                                                      fp32TensorType, constSplatValue);
        const auto actualElemType = mlir::cast<NDTypeInterface>(constOp.getType()).getElementType();
        const bool floatConstantHasCorrectType = (actualElemType == fp32TensorType.getElementType());
        if (floatConstantHasCorrectType) {
            return newConst;
        }
        const auto actualType = mlir::RankedTensorType::get(newShape, actualElemType);
        auto patchedContentAttr = newConst.getDefiningOp<Const::DeclareOp>()
                                          .getContentAttr()
                                          .transform()
                                          .castElemType(actualElemType)
                                          .get();
        return rewriter.create<Const::DeclareOp>(newConst.getLoc(), actualType, std::move(patchedContentAttr))
                .getOutput();
    }();

    if (singleConstResult == nullptr) {
        _log.trace("Ignore: Cannot get const value");
        return mlir::failure();
    }

    SmallVector<mlir::Value> newMultiplyResults;
    for (const auto& concatInput : concatOp.getInputs()) {
        // create IE.Multiply with the new constant
        static_assert(IE::MultiplyOp::hasTrait<mlir::OpTrait::IsCommutative>(),
                      "The order of operands does not matter for IE.Multiply.");
        // Note: since we use 1x1x1x1 constant, the broadcast type has to be
        // overwritten just in case
        const auto numpyBroadcast = IE::AutoBroadcastTypeAttr::get(origOp.getContext(), IE::AutoBroadcastType::NUMPY);
        auto newMultiply = rewriter.create<IE::MultiplyOp>(
                concatOp->getLoc(), concatInput, singleConstResult, numpyBroadcast, origOp.getPostOpAttr(),
                origOp.getClampAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
        newMultiplyResults.push_back(newMultiply.getOutput());
    }

    auto newConcatOp =
            rewriter.create<IE::ConcatOp>(concatOp->getLoc(), concatOp.getOutput().getType(), newMultiplyResults,
                                          concatOp.getPerAxisAttr(), concatOp.getStaticOffsetsAttr());

    rewriter.replaceOp(concatOp, newConcatOp);
    rewriter.replaceAllUsesWith(origOp.getOutput(), newConcatOp.getOutput());

    return mlir::success();
}

// Helper for common Multiply validation and extraction
bool validateAndExtract(IE::MultiplyOp origOp, const Logger& log, float& constSplatValue,
                        mlir::Value& nonConstMultiplyOperand) {
    log.trace("Found IE.Multiply at {0} for static scale fusion", origOp->getLoc());
    if (origOp.getPostOpAttr() != nullptr) {
        log.trace("Ignore: IE.Multiply is not simple (has ppe)");
        return false;
    }
    auto constOp = findDefiningOp<Const::DeclareOp>(origOp);
    if (!isSuitableConstant(constOp)) {
        log.trace("Ignore: IE.Multiply is not floating-point constant scalar multiplication");
        return false;
    }
    auto maybeConstSplatValue = getConstSplatValue(constOp, origOp);
    if (mlir::failed(maybeConstSplatValue)) {
        log.trace("Ignore: Cannot get const value");
        return false;
    }
    constSplatValue = maybeConstSplatValue.value();
    nonConstMultiplyOperand =
            origOp.getInput1().getDefiningOp() == constOp || checkFQOperand(constOp, origOp.getInput1())
                    ? origOp.getInput2()
                    : origOp.getInput1();
    return true;
}

// Backward fusion: fuse Multiply into previous SupportedOp's PPE
template <class SupportedOp>
struct FuseStaticScaleToPpeBackward : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseStaticScaleToPpeBackward(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final {
        float constSplatValue = 1.0f;
        mlir::Value nonConstMultiplyOperand;
        if (!validateAndExtract(origOp, _log, constSplatValue, nonConstMultiplyOperand)) {
            return mlir::failure();
        }
        auto backwardTargetOp = findOriginOp<SupportedOp>(_log, nonConstMultiplyOperand.getDefiningOp());
        if (backwardTargetOp != nullptr && backwardTargetOp->hasOneUse()) {
            _log.trace("Backward fusing IE.Multiply at {0} into operation at {1}", origOp->getLoc(),
                       backwardTargetOp->getLoc());
            const auto originalScale = backwardTargetOp.getStaticScaleAttr()
                                               ? backwardTargetOp.getStaticScaleAttr().getValueAsDouble()
                                               : 1.0;
            const auto newScale = originalScale * constSplatValue;
            backwardTargetOp.setStaticScaleAttr(
                    mlir::FloatAttr::get(mlir::Float32Type::get(origOp.getContext()), newScale));
            rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);
            return mlir::success();
        }
        return mlir::failure();
    }

private:
    const Logger& _log;
};

// Forward fusion: fuse Multiply into following SupportedOp's PPE (no scale check)
template <class SupportedOp>
struct FuseStaticScaleToPpeForward : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseStaticScaleToPpeForward(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final {
        float constSplatValue = 1.0f;
        mlir::Value nonConstMultiplyOperand;
        if (!validateAndExtract(origOp, _log, constSplatValue, nonConstMultiplyOperand)) {
            return mlir::failure();
        }
        auto forwardTargetOp = findTargetOp<SupportedOp>(_log, origOp);
        if (forwardTargetOp == nullptr) {
            _log.trace("Ignore: IE.Multiply has no valid preceding or following operation - cannot fuse");
            return mlir::failure();
        }
        _log.trace("Forward fusing IE.Multiply at {0} into operation at {1}", origOp->getLoc(),
                   forwardTargetOp->getLoc());
        const auto originalScale =
                forwardTargetOp.getStaticScaleAttr() ? forwardTargetOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
        const auto newScale = originalScale * constSplatValue;
        forwardTargetOp.setStaticScaleAttr(mlir::FloatAttr::get(mlir::Float32Type::get(origOp.getContext()), newScale));
        rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);
        return mlir::success();
    }

private:
    const Logger& _log;
};

// Wrapper for non-pooling ops to check scale < 1.0 before delegating to FuseStaticScaleToPpeForward
template <class SupportedOp>
struct FuseStaticScaleToPpeForwardWithScaleCheck : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseStaticScaleToPpeForwardWithScaleCheck(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log), _delegate(ctx, log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final {
        float constSplatValue = 1.0f;
        mlir::Value nonConstMultiplyOperand;
        if (!validateAndExtract(origOp, _log, constSplatValue, nonConstMultiplyOperand)) {
            return mlir::failure();
        }
        if (constSplatValue < 1.0f) {
            _log.trace("Ignore: IE.Multiply has scale < 1.0 - do not fuse forward to avoid accuracy degradation");
            return mlir::failure();
        }
        // Delegate to the standard pattern
        return _delegate.matchAndRewrite(origOp, rewriter);
    }

private:
    const Logger& _log;
    FuseStaticScaleToPpeForward<SupportedOp> _delegate;
};

// Forward fusion: fuse Multiply into following SupportedOp's weights/filter if possible
template <class SupportedOp>
struct FuseStaticScaleToWeights : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseStaticScaleToWeights(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitHigh), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final {
        float constSplatValue = 1.0f;
        mlir::Value nonConstMultiplyOperand;
        if (!validateAndExtract(origOp, _log, constSplatValue, nonConstMultiplyOperand)) {
            return mlir::failure();
        }
        auto forwardTargetOp = findTargetOp<SupportedOp>(_log, origOp);
        if (forwardTargetOp == nullptr) {
            _log.trace("Ignore: IE.Multiply has no valid preceding or following operation - cannot fuse");
            return mlir::failure();
        }
        auto weightVal = forwardTargetOp.getFilter();
        auto weightConstOp = weightVal.template getDefiningOp<Const::DeclareOp>();
        if (weightConstOp) {
            auto newConst = getRescaledConst(weightConstOp, rewriter, constSplatValue);
            rewriter.replaceAllUsesWith(weightConstOp.getOutput(), newConst.getOutput());
            rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);
            return mlir::success();
        }
        auto fqOp = weightVal.template getDefiningOp<IE::FakeQuantizeOp>();
        if (!fqOp) {
            _log.trace("Ignore: SupportedOp has no constant weight or valid FakeQuantizeOp - cannot fuse");
            return mlir::failure();
        }
        auto outputLowConstOp = fqOp.getOutputLow().template getDefiningOp<Const::DeclareOp>();
        auto outputHighConstOp = fqOp.getOutputHigh().template getDefiningOp<Const::DeclareOp>();
        if (!outputLowConstOp || !outputHighConstOp) {
            _log.trace("Ignore: FakeQuantizeOp has no constant output low or high - cannot fuse");
            return mlir::failure();
        }
        auto newFQ = getRescaledFQ(fqOp, outputLowConstOp, outputHighConstOp, rewriter, constSplatValue);
        rewriter.replaceAllUsesWith(fqOp.getOutput(), newFQ.getOutput());
        rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);
        return mlir::success();
    }

private:
    const Logger& _log;
};

class FuseStaticScalePass final : public IE::impl::FuseStaticScaleBase<FuseStaticScalePass> {
public:
    explicit FuseStaticScalePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseStaticScalePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertMultiplyBeforeConcat>(&ctx, _log);
    patterns.add<FuseStaticScaleToPpeBackward<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<FuseStaticScaleToPpeBackward<IE::AvgPoolOp>>(&ctx, _log);
    patterns.add<FuseStaticScaleToWeights<IE::ConvolutionOp>>(&ctx, _log);
    // For a pooling operation, use standard pattern (no scale check)
    patterns.add<FuseStaticScaleToPpeForward<IE::AvgPoolOp>>(&ctx, _log);
    // If the supported operation is not a pooling operation and the scale is less than 1.0,
    // do not fuse forward to avoid accuracy degradation. This is because the input values
    // before multiplication are larger than the values after multiplication, which can
    // cause the supported operation to overflow during accumulation.
    patterns.add<FuseStaticScaleToPpeForwardWithScaleCheck<IE::ConvolutionOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseStaticScalePass(Logger log) {
    return std::make_unique<FuseStaticScalePass>(log);
}
