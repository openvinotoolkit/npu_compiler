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
    if (originOp == nullptr || originOp.getPostOpAttr() != nullptr || originOp.getClampAttr() != nullptr) {
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

template <class ParentOp>
struct FuseStaticScale : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseStaticScale(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log) {
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

template <class ParentOp>
mlir::LogicalResult FuseStaticScale<ParentOp>::matchAndRewrite(IE::MultiplyOp origOp,
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

    auto maybeConstSplatValue = getConstSplatValue(constOp, origOp);
    if (mlir::failed(maybeConstSplatValue)) {
        _log.trace("Ignore: Cannot get const value");
        return mlir::failure();
    }
    auto constSplatValue = maybeConstSplatValue.value();

    const auto nonConstMultiplyOperand =
            origOp.getInput1().getDefiningOp() == constOp || checkFQOperand(constOp, origOp.getInput1())
                    ? origOp.getInput2()
                    : origOp.getInput1();
    auto targetOp = findOriginOp<ParentOp>(_log, nonConstMultiplyOperand.getDefiningOp());
    if (targetOp == nullptr || !targetOp->hasOneUse()) {
        _log.trace("Ignore: IE.Multiply has no valid preceding operation - cannot fuse");
        return mlir::failure();
    }

    _log.trace("Fusing IE.Multiply at {0} into operation at {1}", origOp->getLoc(), targetOp->getLoc());
    // fuse IE.Multiply by specifying the static scale attribute
    const auto originalScale = targetOp.getStaticScaleAttr() ? targetOp.getStaticScaleAttr().getValueAsDouble() : 1.0;
    const auto newScale = originalScale * constSplatValue;
    targetOp.setStaticScaleAttr(mlir::FloatAttr::get(mlir::Float32Type::get(origOp.getContext()), newScale));

    rewriter.replaceAllUsesWith(origOp.getOutput(), nonConstMultiplyOperand);

    return mlir::success();
}

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
    patterns.add<FuseStaticScale<IE::ConvolutionOp>>(&ctx, _log);
    patterns.add<FuseStaticScale<IE::AvgPoolOp>>(&ctx, _log);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseStaticScalePass(Logger log) {
    return std::make_unique<FuseStaticScalePass>(log);
}
