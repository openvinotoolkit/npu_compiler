//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/types/quantile_float/types.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/convolution_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/mpe_engine_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <tuple>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSESCALE
#define GEN_PASS_DEF_FUSESCALE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {
// Returns an "origin" operation of the specified type (one of Ops), ignoring
// pure view ops, for the given operation. This procedure assumes IR in question
// is a chain of single-use operations.
template <typename Op>
Op findOriginOp(const Logger& log, mlir::Operation* current, bool checkPostOp = true, bool skipTranspose = false) {
    // skip "pure view ops"
    while (current && (IE::isPureViewOp(current) || (skipTranspose && mlir::isa<IE::TransposeOp>(current)))) {
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
    if (checkPostOp) {
        if (originOp == nullptr || originOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
            originOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
            return nullptr;
        }
    }

    return originOp;
}

// Returns defining op of the specified type for some operand of the op ignoring IE.FakeQuantize ops
template <typename Op>
Op findDefiningOp(mlir::Operation* op, const bool ignoreReshapeOp = false) {
    for (const auto& operand : op->getOperands()) {
        auto parent = operand.getDefiningOp();
        while (mlir::isa_and_nonnull<IE::FakeQuantizeOp>(parent) ||
               (ignoreReshapeOp && mlir::isa_and_nonnull<IE::AffineReshapeOp, IE::TransposeOp>(parent))) {
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
Op findTargetOp(const Logger&, mlir::Operation* current, bool checkPostOp = true) {
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
    if (checkPostOp) {
        if (targetOp == nullptr || targetOp->hasAttr(vpux::OperationAttrName::POST_OP) ||
            targetOp->hasAttr(vpux::OperationAttrName::CLAMP)) {
            return nullptr;
        }
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
    if (origOp.getPostOpAttr() != nullptr || origOp.getClampAttr() != nullptr) {
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
        const auto newConst = Const::createFloatConst(rewriter, appendLoc(constOp->getLoc(), "to_mult_scalar"),
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
    if (origOp.getPostOpAttr() != nullptr || origOp.getClampAttr() != nullptr) {
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
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitMid), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final {
        float constSplatValue = 1.0f;
        mlir::Value nonConstMultiplyOperand;
        if (!validateAndExtract(origOp, _log, constSplatValue, nonConstMultiplyOperand)) {
            return mlir::failure();
        }
        auto backwardTargetOp = findOriginOp<SupportedOp>(_log, nonConstMultiplyOperand.getDefiningOp());
        if (backwardTargetOp != nullptr && backwardTargetOp->hasOneUse()) {
            if (auto convOp = mlir::dyn_cast_or_null<IE::ConvolutionOp>(backwardTargetOp.getOperation())) {
                if (convOp.getScale() != nullptr) {
                    return mlir::failure();
                }
            }
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
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitMid), _log(log) {
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
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitMid), _log(log), _delegate(ctx, log) {
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
        if (auto convOp = mlir::dyn_cast_or_null<IE::ConvolutionOp>(forwardTargetOp.getOperation())) {
            if (convOp.getScale() != nullptr) {
                return mlir::failure();
            }
        }

        auto weightVal = forwardTargetOp.getFilter();
        auto weightConstOp = weightVal.template getDefiningOp<Const::DeclareOp>();
        if (weightConstOp) {
            auto newConst = getRescaledConst(weightConstOp, rewriter, constSplatValue);
            // Only replace the weight operand of this specific convolution
            rewriter.modifyOpInPlace(forwardTargetOp, [&]() {
                forwardTargetOp->setOperand(1, newConst.getOutput());
            });
            rewriter.replaceOp(origOp, nonConstMultiplyOperand);
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
        // Only replace the weight operand of this specific convolution
        rewriter.modifyOpInPlace(forwardTargetOp, [&]() {
            forwardTargetOp->setOperand(1, newFQ.getOutput());
        });
        rewriter.replaceOp(origOp, nonConstMultiplyOperand);
        return mlir::success();
    }

private:
    const Logger& _log;
};

template <class SupportedOp>
struct FuseChannelWiseScales : public mlir::OpRewritePattern<IE::MultiplyOp> {
    FuseChannelWiseScales(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefitLow), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const Logger& _log;
};

template <class SupportedOp>
bool isOpEligibleForFusion(SupportedOp origOp, const Logger& log) {
    if (!origOp->hasOneUse()) {
        return false;
    }

    if (VPU::NCEInvariant::isSupported(origOp).failed()) {
        return false;
    }

    if (!VPU::MPEEngineConfig::useNewWeightTableFormat(origOp, /*isCompressConv*/ false)) {
        return false;
    }

    auto layerWithPostOp = mlir::dyn_cast_or_null<IE::LayerWithPostOpInterface>(origOp.getOperation());
    if (layerWithPostOp == nullptr || layerWithPostOp.getPostOp()) {
        return false;
    }

    if (origOp.getStaticScaleAttr() != nullptr) {
        return false;
    }

    mlir::Type filterElemType;
    auto dequantizeOp = findOriginOp<IE::DequantizeOp>(log, origOp.getFilter().getDefiningOp(), false);
    if (dequantizeOp == nullptr) {
        auto convertOp = findOriginOp<IE::ConvertOp>(log, origOp.getFilter().getDefiningOp(), false);
        if (convertOp == nullptr) {
            return false;
        }
        filterElemType = mlir::cast<vpux::NDTypeInterface>(convertOp.getInput().getType()).getElementType();
    } else {
        filterElemType = mlir::cast<vpux::NDTypeInterface>(dequantizeOp.getInput().getType()).getElementType();
    }
    // Check if dequantize scale is 1.0f. If not, we need to add a new multiply operation to multiply the scale value
    // with dequantize scale in lowerring to VPU pass. The new multiply operation is fp32, and it will greatly degrade
    // the performance, so here we need to avoid the case.
    if (auto uniformQuantizedType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(filterElemType)) {
        if (uniformQuantizedType.getScale() != 1.0f) {
            return false;
        }
    } else if (!mlir::isa<vpux::type::QuantileType>(filterElemType)) {
        return false;
    }

    // Check input type
    auto inElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    if ((!inElemType.isF16() && !inElemType.isF32())) {
        return false;
    }
    auto inFakeQuantOp = findOriginOp<IE::FakeQuantizeOp>(log, origOp.getInput().getDefiningOp(), /*checkPostOp*/ false,
                                                          /*skipTranspose*/ true);

    return !inFakeQuantOp;
}

template <class SupportedOp>
struct MultiplyToSupportedOpViewChain {
    SupportedOp nceOp = nullptr;
    mlir::Value nonScaleInput = nullptr;
    mlir::Value scaleInput = nullptr;
    SmallVector<mlir::Operation*> viewLikeOps;
};

// For each input of a MultiplyOp, try to find an eligible SupportedOp by
// walking backward through view-like ops (AffineReshape / Transpose), then
// forward-validate the channel axis is preserved through the chain.
template <class SupportedOp>
mlir::FailureOr<MultiplyToSupportedOpViewChain<SupportedOp>> findSupportedOpFromMultiply(IE::MultiplyOp multiplyOp,
                                                                                         const Logger& log) {
    for (auto [targetSideValue, scaleInput] : {std::pair{multiplyOp.getInput1(), multiplyOp.getInput2()},
                                               std::pair{multiplyOp.getInput2(), multiplyOp.getInput1()}}) {
        MultiplyToSupportedOpViewChain<SupportedOp> chain;
        chain.scaleInput = scaleInput;

        // Walk backward through view-like ops to find the SupportedOp.
        mlir::Value currentValue = targetSideValue;
        while (true) {
            auto* defOp = currentValue.getDefiningOp();
            if (defOp == nullptr) {
                break;
            }
            if (auto nceOp = mlir::dyn_cast<SupportedOp>(defOp)) {
                chain.nceOp = nceOp;
                chain.nonScaleInput = targetSideValue;
                break;
            }
            if (!mlir::isa<IE::AffineReshapeOp, IE::TransposeOp>(defOp) || !defOp->hasOneUse()) {
                break;
            }
            chain.viewLikeOps.push_back(defOp);
            currentValue = defOp->getOperand(0);
        }

        if (chain.nceOp == nullptr || !chain.nceOp->hasOneUse()) {
            continue;
        }

        // Reverse to get forward order: conv output → multiply.
        std::reverse(chain.viewLikeOps.begin(), chain.viewLikeOps.end());

        // Forward-track the target op output channel axis through the view-like chain,
        // then verify the scale input has the matching shape.
        int64_t currentChannelAxis = Dims4D::Act::C.ind();
        const auto scaleValue = getShape(chain.nceOp.getOutput())[Dim(currentChannelAxis)];
        bool axisValid = true;

        for (auto* viewLikeOp : chain.viewLikeOps) {
            if (auto transposeOp = mlir::dyn_cast<IE::TransposeOp>(viewLikeOp)) {
                if (!transposeOp.getOrderValue().has_value()) {
                    axisValid = false;
                    break;
                }
                const auto orderMap = transposeOp.getOrderValue().value();
                if (currentChannelAxis >= checked_cast<int64_t>(orderMap.getNumInputs())) {
                    axisValid = false;
                    break;
                }
                const auto order = DimsOrder::fromAffineMap(orderMap);
                bool found = false;
                for (const auto outAxis : irange(orderMap.getNumInputs())) {
                    if (order.dimAt(outAxis) == Dim(currentChannelAxis)) {
                        currentChannelAxis = checked_cast<int64_t>(outAxis);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    log.trace("Skip: cannot map channel axis '{0}' through Transpose", currentChannelAxis);
                    axisValid = false;
                    break;
                }
            } else if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(viewLikeOp)) {
                const auto inShape = getShape(affineReshapeOp.getInput());
                const auto outShape = getShape(affineReshapeOp.getOutput());
                const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());
                if (currentChannelAxis >= checked_cast<int64_t>(dimMapping.size())) {
                    axisValid = false;
                    break;
                }
                const mlir::DenseSet<int64_t> modifiedAxes{currentChannelAxis};
                if (IE::areModifiedAxesSplitOrMerged(dimMapping, inShape, outShape, modifiedAxes,
                                                     /*swapOrder=*/true, log.nest())) {
                    log.trace("Skip: channel axis '{0}' is split/merged through AffineReshape", currentChannelAxis);
                    axisValid = false;
                    break;
                }
                bool found = false;
                for (const auto outAxis : dimMapping[currentChannelAxis]) {
                    if (outShape[Dim(outAxis)] == inShape[Dim(currentChannelAxis)]) {
                        currentChannelAxis = outAxis;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    log.trace("Skip: cannot map channel axis '{0}' through AffineReshape", currentChannelAxis);
                    axisValid = false;
                    break;
                }
            }
        }

        if (!axisValid) {
            continue;
        }

        // Verify scale input shape: channel axis must equal scaleValue, all others must be 1.
        const auto scaleInputShape = getShape(chain.scaleInput);
        if (scaleInputShape.size() != getShape(multiplyOp.getOutput()).size()) {
            continue;
        }
        const auto channelAxis = checked_cast<size_t>(currentChannelAxis);
        bool scaleShapeValid = true;
        for (const auto i : irange(scaleInputShape.size())) {
            if (i == channelAxis && scaleValue != scaleInputShape[Dim(i)]) {
                scaleShapeValid = false;
                break;
            }
            if (i != channelAxis && scaleInputShape[Dim(i)] != 1) {
                scaleShapeValid = false;
                break;
            }
        }
        if (!scaleShapeValid) {
            continue;
        }

        return chain;
    }

    return mlir::failure();
}

template <class SupportedOp>
mlir::LogicalResult FuseChannelWiseScales<SupportedOp>::matchAndRewrite(IE::MultiplyOp origOp,
                                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("Try to fuse channel-wise scales for '{0}' at {1}", origOp->getName(), origOp->getLoc());
    auto ctx = origOp->getContext();

    if (origOp.getPostOpAttr() != nullptr || origOp.getClampAttr() != nullptr) {
        _log.nest().trace("Skip: IE.Multiply has post-op or clamp");
        return mlir::failure();
    }

    auto outFakeQuantOp = findTargetOp<IE::FakeQuantizeOp>(_log, origOp, false);
    if (outFakeQuantOp) {
        _log.nest().trace("Skip: IE.Multiply has FakeQuantize user");
        return mlir::failure();
    }

    const auto maybeChain = findSupportedOpFromMultiply<SupportedOp>(origOp, _log.nest());
    if (mlir::failed(maybeChain)) {
        _log.nest().trace("Skip: failed to match Multiply <- (AffineReshape|Transpose)* <- SupportedOp chain");
        return mlir::failure();
    }

    const auto& chain = maybeChain.value();
    auto nceOp = chain.nceOp;
    auto scaleInput = chain.scaleInput;
    auto nonScaleInput = chain.nonScaleInput;

    if (!isOpEligibleForFusion<SupportedOp>(nceOp, _log)) {
        _log.nest().trace("Skip: Not an eligible operation");
        return mlir::failure();
    }

    const auto nceOutShape = getShape(nceOp.getOutput());
    const auto nceChannel = nceOutShape[Dims4D::Act::C];

    const auto scaleTableShape = VPU::NCESparsity::inferWeightsTableShape(nceChannel, /*newFormat=*/true);
    const auto scaleShapeAttr = getIntArrayAttr(ctx, scaleTableShape.raw());
    rewriter.setInsertionPointAfter(origOp);

    // Reshape scale tensor into weights-table layout expected by IE.Convolution scale input.
    mlir::Value scales =
            rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_Reshape"), scaleInput, scaleShapeAttr)
                    .getOutput();

    // scales input need to be FP32
    scales = rewriter.createOrFold<IE::ConvertOp>(appendLoc(origOp->getLoc(), "_Convert"), scales,
                                                  mlir::TypeAttr::get(mlir::Float32Type::get(ctx)));

    // If the target op already has a scale tensor, multiply the two scale tensors together so that
    // the combined scale replaces both.  This keeps scale-only data out of the target op output path.
    if (nceOp.getScale() != nullptr) {
        scales = rewriter.create<IE::MultiplyOp>(appendLoc(origOp->getLoc(), "_ScaleMerge"), nceOp.getScale().getType(),
                                                 nceOp.getScale(), scales, IE::AutoBroadcastType::NUMPY,
                                                 /*post_op=*/nullptr,
                                                 /*clamp=*/nullptr,
                                                 /*output_channels=*/nullptr,
                                                 /*input_channels=*/nullptr)
                         .getOutput();
    }

    const auto nceOpOutType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto outType = mlir::cast<vpux::NDTypeInterface>(nceOp.getOutput().getType());
    outType = outType.changeElemType(nceOpOutType.getElementType());

    mlir::Operation* newNceOp;
    if (auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(nceOp.getOperation())) {
        newNceOp = IE::cloneConvolutionOp(rewriter, convOp, outType, convOp.getInput(), convOp.getFilter(),
                                          convOp.getBias(), scales, convOp.getZeroPoints());
    } else {
        VPUX_THROW("FuseChannelWiseScales: We don't support other op type now '{0}'", nceOp->getName());
    }

    rewriter.replaceOp(origOp, nonScaleInput);
    rewriter.replaceOp(nceOp, newNceOp->getResult(0));

    // New op is inserted after multiply, so move the whole matched
    // AffineReshape/Transpose chain after it to preserve dominance and original order.
    mlir::Operation* insertionTail = newNceOp;
    for (auto* viewLikeOp : chain.viewLikeOps) {
        viewLikeOp->moveAfter(insertionTail);
        insertionTail = viewLikeOp;
    }

    _log.trace("Fused channel-wise scales to operation");
    return mlir::success();
}

class FuseScalePass final : public IE::impl::FuseScaleBase<FuseScalePass> {
public:
    explicit FuseScalePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FuseScalePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    {
        mlir::RewritePatternSet patterns(&ctx);
        patterns.add<InsertMultiplyBeforeConcat>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }

    {
        // NOTE: InsertMultiplyBeforeConcat inserts Multiply Ops which get optimized out
        // by FuseStaticScale patterns, therefore require a separate func walk
        mlir::RewritePatternSet patterns(&ctx);
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
        patterns.add<FuseChannelWiseScales<IE::ConvolutionOp>>(&ctx, _log);
        collectOpsAndApplyPatterns(func, std::move(patterns));
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createFuseScalePass(Logger log) {
    return std::make_unique<FuseScalePass>(log);
}
