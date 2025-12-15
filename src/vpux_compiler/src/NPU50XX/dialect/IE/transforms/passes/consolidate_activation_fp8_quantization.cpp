//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch50xx {
#define GEN_PASS_DECL_CONSOLIDATEACTIVATIONFP8QUANTIZATION
#define GEN_PASS_DEF_CONSOLIDATEACTIVATIONFP8QUANTIZATION
#include "vpux/compiler/NPU50XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch50xx

namespace vpux {

//
// FakeConvertRewriter
//
class FakeConvertRewriter final : public mlir::OpRewritePattern<IE::FakeConvertOp> {
private:
    Logger _log;

public:
    FakeConvertRewriter(mlir::MLIRContext* ctx, const Logger& log)
            : mlir::OpRewritePattern<IE::FakeConvertOp>(ctx), _log(log) {
        setDebugName("FakeConvertRewriter");
    }

    // Decompose FakeConvert to DWConv->Quantize->Dequantize->Divide
    mlir::LogicalResult matchAndRewrite(IE::FakeConvertOp origOp, mlir::PatternRewriter& rewriter) const final {
        _log.trace("Got {0} at `{1}`.", origOp->getName(), origOp->getLoc());

        if (!isFloat8(origOp.getDstType())) {
            return matchFailed(_log, rewriter, origOp, "Destination type is not FP8");
        }

        auto scale = origOp.getScale();
        if (auto constScale = scale.getDefiningOp<Const::DeclareOp>()) {
            return matchFailed(_log, rewriter, origOp, "FakeConvert with const scale not supported");
        }

        if (getShape(origOp.getScale()).totalSize() != 1) {
            return matchFailed(_log, rewriter, origOp, "FakeConvert with scale size > 1 not supported");
        }

        if (origOp.getShift() != nullptr) {
            auto shiftCst = mlir::dyn_cast_if_present<Const::DeclareOp>(origOp.getShift().getDefiningOp());
            if (shiftCst == nullptr) {
                return matchFailed(_log, rewriter, origOp, "FakeConvert with non-const shift not supported");
            }

            const auto content = shiftCst.getContent();
            if (!content.isSplat() || content.getSplatValue<float>() != 0.0f) {
                return matchFailed(_log, rewriter, origOp, "FakeConvert with non-zero shift not supported");
            }
        }

        auto ctx = rewriter.getContext();
        auto loc = origOp.getLoc();

        // Create DepthWiseConv
        size_t targetRank = 4;
        auto fakeConvertInput = origOp.getInput();
        auto fakeConvertInputShape = getShape(fakeConvertInput);
        auto newFakeConvertInputShape = to_small_vector(fakeConvertInputShape);
        newFakeConvertInputShape.insert(newFakeConvertInputShape.begin(), targetRank - fakeConvertInputShape.size(), 1);

        auto activationReshapeOp =
                rewriter.create<IE::ReshapeOp>(appendLoc(loc, "_act_reshape"), fakeConvertInput, nullptr, false,
                                               getIntArrayAttr(ctx, newFakeConvertInputShape));

        auto newFakeConvertScaleShape = SmallVector<size_t>{1, 1, 1, 1};
        auto filterReshapeOp =
                rewriter.create<IE::ReshapeOp>(appendLoc(loc, "_scale_reshape"), origOp.getScale(), nullptr, false,
                                               getIntArrayAttr(ctx, newFakeConvertScaleShape));

        // Broadcast filter
        auto targetShape = Shape{newFakeConvertInputShape[Dims4D::Filter::IC.ind()], 1, 1, 1};
        auto broadCastOp = IE::createBroadcast(rewriter, appendLoc(loc, "_filter_broadcast"),
                                               filterReshapeOp.getOutput(), targetShape);

        const SmallVector<int32_t> strides = {1, 1};
        const SmallVector<int32_t> padBegin = {0, 0};
        const SmallVector<int32_t> padEnd = {0, 0};
        const SmallVector<int32_t> dilations = {1, 1};

        auto dilationsAttr = getIntArrayAttr(origOp.getContext(), dilations);
        auto stridesAttr = getIntArrayAttr(origOp.getContext(), strides);
        auto padBeginAttr = getIntArrayAttr(origOp.getContext(), padBegin);
        auto padEndAttr = getIntArrayAttr(origOp.getContext(), padEnd);
        auto groupAttr = getIntAttr(origOp.getContext(), newFakeConvertInputShape[Dims4D::Act::C.ind()]);

        auto depthWiseConv = rewriter.create<IE::GroupConvolutionOp>(
                appendLoc(loc, "_conv"), activationReshapeOp, broadCastOp,
                /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
                /*postOp=*/nullptr, /*clamp=*/nullptr,
                /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

        auto convOutReshape =
                rewriter.create<IE::ReshapeOp>(appendLoc(loc, "_conv_reshape"), depthWiseConv.getOutput(), nullptr,
                                               false, getIntArrayAttr(ctx, getShape(origOp.getOutput())));

        // Create a pair of dummy Quantize->Dequantize operations to ensure the data actually gets converted
        const auto dequantType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType()).getElementType();
        const auto minMax = vpux::getFp8Range(origOp.getDstType());
        VPUX_THROW_WHEN(mlir::failed(minMax), "Unexpected data type {0}", origOp.getDstType());

        const auto qMin = std::get<0>(*minMax), qMax = std::get<1>(*minMax);
        auto quantType =
                mlir::quant::UniformQuantizedType::get(mlir::quant::QuantizationFlags::Signed, origOp.getDstType(),
                                                       dequantType, /*scales=*/1.0, /*zeroPoints=*/0, qMin, qMax);

        auto quantizeOp =
                rewriter.create<IE::QuantizeOp>(appendLoc(loc, "_quantize"), convOutReshape.getOutput(), quantType);
        auto dequantizeOp =
                rewriter.create<IE::DequantizeOp>(appendLoc(loc, "_dequantize"), quantizeOp.getOutput(), dequantType);

        rewriter.replaceOp(origOp, dequantizeOp.getOutput());

        // Create Divide
        rewriter.setInsertionPointAfter(dequantizeOp);
        auto divideOp = rewriter.create<IE::DivideOp>(appendLoc(loc, "_divide"), dequantizeOp.getOutput(), scale,
                                                      IE::AutoBroadcastType::NUMPY);
        rewriter.replaceAllUsesExcept(dequantizeOp.getOutput(), divideOp.getOutput(), {divideOp});

        return mlir::success();
    }
};

//
// MoveDividePost
//

template <typename ConcreteOp>
class MoveDividePost final : public mlir::OpRewritePattern<IE::DivideOp> {
private:
    bool isLegalTransformation(ConcreteOp) const;

private:
    Logger _log;

public:
    MoveDividePost(mlir::MLIRContext* ctx, const Logger& log): mlir::OpRewritePattern<IE::DivideOp>(ctx), _log(log) {
        setDebugName("MoveDividePost");
    }

    mlir::LogicalResult matchAndRewrite(IE::DivideOp origOp, mlir::PatternRewriter& rewriter) const final;
};

template <typename ConcreteOp>
bool MoveDividePost<ConcreteOp>::isLegalTransformation(ConcreteOp op) const {
    // IE::MatMulOp should not have post op
    if constexpr (std::is_same_v<ConcreteOp, IE::MatMulOp>) {
        return op.getPostOpAttr() == nullptr;
    }
    // IE::FullyConnectedOp should not have bias
    else if constexpr (std::is_same_v<ConcreteOp, IE::FullyConnectedOp>) {
        return op.getBias() == nullptr;
    }

    return false;
}

// Move Divide post MatMul/FC
template <typename ConcreteOp>
mlir::LogicalResult MoveDividePost<ConcreteOp>::matchAndRewrite(IE::DivideOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got divide layer at '{1}'", origOp->getName(), origOp->getLoc());

    if (!origOp->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "divide has more than one user");
    }

    auto maybeMatMulOp = *origOp.getOutput().getUsers().begin();
    if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(maybeMatMulOp)) {
        if (!multiplyOp->hasOneUse()) {
            return matchFailed(_log, rewriter, origOp, "multiply has more than one user");
        }
        if (multiplyOp.getPostOp() || multiplyOp.getClamp()) {
            return matchFailed(_log, rewriter, origOp, "multiply has post or clamp");
        }
        auto constInput =
                multiplyOp.getInput1() == origOp.getOutput() ? multiplyOp.getInput2() : multiplyOp.getInput1();
        auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(constInput.getDefiningOp());
        if (constOp && IE::isBaseContentSplat(constOp)) {
            maybeMatMulOp = *multiplyOp.getOutput().getUsers().begin();
        } else {
            return matchFailed(_log, rewriter, origOp, "multiply need to have splat const input");
        }
    }

    while (mlir::isa<IE::ViewLikeOpInterface, IE::ConvertOp, IE::TransposeOp>(maybeMatMulOp)) {
        if (!maybeMatMulOp->hasOneUse()) {
            return matchFailed(_log, rewriter, origOp, "Result of {0} has multiple uses", maybeMatMulOp->getName());
        }
        maybeMatMulOp = *maybeMatMulOp->getResult(0).getUsers().begin();
    }

    auto layerOp = mlir::dyn_cast<ConcreteOp>(maybeMatMulOp);
    if (layerOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "invalid layerOp");
    }

    if (!isLegalTransformation(layerOp)) {
        return matchFailed(_log, rewriter, origOp, "illegal to swap divide with layerOp");
    }

    auto scale = origOp.getInput2();
    auto scaleSize = getShape(scale).totalSize();
    if (scaleSize > 1) {
        return matchFailed(_log, rewriter, origOp, "divide with scale size > 1 is not supported");
    }

    auto producer = origOp.getInput1().getDefiningOp<IE::DequantizeOp>();
    if (producer == nullptr || !producer->hasOneUse()) {
        return matchFailed(_log, rewriter, origOp, "producer must be a dequantize with single use");
    }

    // Remove original Divide
    rewriter.replaceAllUsesWith(origOp.getOutput(), producer->getResult(0));

    // Insert new Divide
    rewriter.setInsertionPointAfter(layerOp);
    auto newDivide = rewriter.create<IE::DivideOp>(appendLoc(origOp->getLoc(), "_post_fc"), layerOp->getResult(0),
                                                   scale, IE::AutoBroadcastType::NUMPY);
    rewriter.replaceAllUsesExcept(layerOp->getResult(0), newDivide.getOutput(), {newDivide});

    return mlir::success();
}

class ConsolidateActivationFP8QuantizationPass final :
        public IE::arch50xx::impl::ConsolidateActivationFP8QuantizationBase<ConsolidateActivationFP8QuantizationPass> {
public:
    explicit ConsolidateActivationFP8QuantizationPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConsolidateActivationFP8QuantizationPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FakeConvertRewriter>(&ctx, _log);
    patterns.add<MoveDividePost<IE::FullyConnectedOp>>(&ctx, _log);
    patterns.add<MoveDividePost<IE::MatMulOp>>(&ctx, _log);
    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace vpux

//
// createConsolidateActivationFP8QuantizationPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch50xx::createConsolidateActivationFP8QuantizationPass(const Logger& log) {
    return std::make_unique<ConsolidateActivationFP8QuantizationPass>(log);
}
