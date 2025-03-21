//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes/convert_to_mixed_precision.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE::arch37xx {
#define GEN_PASS_DECL_OPTIMIZENETWORKINPUTCONVERT
#define GEN_PASS_DEF_OPTIMIZENETWORKINPUTCONVERT
#include "vpux/compiler/NPU37XX/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE::arch37xx

using namespace vpux;

namespace {

//
// OptimizeNetworkInputConvertPass
//

class OptimizeNetworkInputConvertPass final :
        public IE::arch37xx::impl::OptimizeNetworkInputConvertBase<OptimizeNetworkInputConvertPass> {
public:
    explicit OptimizeNetworkInputConvertPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

// Search for FloatInput in the following patterns
// BlockArg -> IE.Quantize -> NCE
mlir::Value findFloatInput(mlir::Value nceOpInput) {
    auto maybeQuantize = nceOpInput.getDefiningOp<IE::QuantizeOp>();
    if (maybeQuantize == nullptr) {
        return nullptr;
    }

    // So far, only the first NCE task should be executed in fp16/u8 mode.
    // The main problem with this mode is that FakeQuantize is removed from the input completely.
    // Without FakeQuantize the information about data clamping is lost.
    // It makes sense to omit clamping only when the input data fits the range required for a given NCE task.
    // For some models performance gain is worth the risk of losing the clamping information.
    if (!maybeQuantize.getInput().isa<mlir::BlockArgument>()) {
        return nullptr;
    }

    return maybeQuantize.getInput();
}

bool isFloatInputSupported(mlir::Operation* origOp, const bool onlyPerTensorQuant, Logger log) {
    if (!IE::arch37xx::isMixPrecisionSupported(origOp, false, log)) {
        return false;
    }

    // Supporting only fusing of Convert into fully quantized operation.
    const auto outElemType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    if (!outElemType.isa<mlir::quant::QuantizedType>()) {
        return false;
    }

    if (onlyPerTensorQuant && outElemType.isa<mlir::quant::UniformQuantizedPerAxisType>()) {
        return false;
    }

    if (!IE::checkQuantApproximation(origOp)) {
        return false;
    }

    // The input of the operation must be quantized.
    const auto inElemType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
    return inElemType.isa<mlir::quant::QuantizedType>();
}

class NetworkInputConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    NetworkInputConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NetworkInputConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, false, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    const auto dstElemType = maybeFloatInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dequantFilter =
            rewriter.createOrFold<IE::DequantizeOp>(origOp->getLoc(), origOp.getFilter(), dstElemType);
    VPUX_THROW_UNLESS(dequantFilter != nullptr, "Failed to de-quantize given filter");
    rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origOp, origOp.getType(), maybeFloatInput, dequantFilter,
                                                   origOp.getBias(), origOp.getStrides(), origOp.getPadsBegin(),
                                                   origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(),
                                                   origOp.getClampAttr(), origOp.getStaticScaleAttr(),
                                                   origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

    return mlir::success();
}

class NetworkInputGroupConvRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    NetworkInputGroupConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NetworkInputGroupConvRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, false, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    const auto dstElemType = maybeFloatInput.getType().cast<vpux::NDTypeInterface>().getElementType();
    const auto dequantFilter =
            rewriter.createOrFold<IE::DequantizeOp>(origOp->getLoc(), origOp.getFilter(), dstElemType);
    VPUX_THROW_UNLESS(dequantFilter != nullptr, "Failed to de-quantize given filter");
    rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.getType(), maybeFloatInput, dequantFilter, origOp.getBias(), origOp.getStrides(),
            origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(), origOp.getGroupsAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputChannelsAttr(),
            origOp.getInputChannelsAttr());

    return mlir::success();
}

class NetworkInputAvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    NetworkInputAvgPoolRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NetworkInputAvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, true, _log)) {
        return mlir::failure();
    }

    auto maybeFloatInput = findFloatInput(origOp->getOperand(0));
    if (maybeFloatInput == nullptr) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
            origOp, origOp.getType(), maybeFloatInput, origOp.getKernelSize(), origOp.getStrides(),
            origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getRoundingTypeAttr(), origOp.getExcludePadsAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputChannelsAttr(),
            origOp.getInputChannelsAttr());

    return mlir::success();
}

class NetworkInputAddRewriter final : public mlir::OpRewritePattern<IE::AddOp> {
public:
    NetworkInputAddRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AddOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult NetworkInputAddRewriter::matchAndRewrite(IE::AddOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!isFloatInputSupported(origOp, true, _log)) {
        return mlir::failure();
    }

    // Check that both inputs of IE.Add have float source.
    SmallVector<mlir::Value> floatInputs;
    for (unsigned idx = 0; idx < 2; idx++) {
        floatInputs.push_back(findFloatInput(origOp->getOperand(idx)));
    }
    const auto nullptrPredicate = [](const mlir::Value operand) -> bool {
        return operand == nullptr;
    };
    if (std::any_of(floatInputs.begin(), floatInputs.end(), nullptrPredicate)) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::AddOp>(origOp, origOp.getType(), floatInputs[0], floatInputs[1],
                                           origOp.getAutoBroadcast(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                           origOp.getOutputChannelsAttr(), origOp.getInputChannelsAttr());

    return mlir::success();
}

void OptimizeNetworkInputConvertPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);

    // Max pooling is omitted intentionally.
    // When we do floating point maxpool the activation datatype appears into the PPE.
    // However, the PPE has only conversion functions from float32, not float16.
    patterns.add<NetworkInputConvRewriter>(&ctx, _log);
    patterns.add<NetworkInputGroupConvRewriter>(&ctx, _log);
    patterns.add<NetworkInputAvgPoolRewriter>(&ctx, _log);
    patterns.add<NetworkInputAddRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeNetworkInputConvertPass
//

std::unique_ptr<mlir::Pass> vpux::IE::arch37xx::createOptimizeNetworkInputConvertPass(Logger log) {
    return std::make_unique<OptimizeNetworkInputConvertPass>(log);
}
