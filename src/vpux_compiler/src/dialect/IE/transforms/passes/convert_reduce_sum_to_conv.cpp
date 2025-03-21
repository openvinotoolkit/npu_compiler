//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTREDUCESUMTOCONV
#define GEN_PASS_DEF_CONVERTREDUCESUMTOCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// ReduceSumToConvRewriter
//

class ReduceSumToConvRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumToConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("ReduceSumToConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isValidShape(vpux::ShapeRef inputShape, Logger log) const;
    bool isSupportedReduceSum(IE::ReduceSumOp origOp, Logger log) const;

    IE::ConvolutionOp createConvolution(mlir::Value activation, mlir::Value weights, mlir::Location newLoc,
                                        mlir::PatternRewriter& rewriter) const;

    mlir::Value createConvFilter(mlir::Value activation, mlir::PatternRewriter& rewriter) const;

    Logger _log;
};

bool ReduceSumToConvRewriter::isValidShape(vpux::ShapeRef inputShape, Logger log) const {
    if (inputShape.size() != 4) {
        log.trace("Only support 4D ReduceSum");
        return false;
    }

    if (inputShape[Dims4D::Act::N] != 1) {
        log.trace("Batch must be equal to 1");
        return false;
    }

    return true;
}

// We have two optimization pass for ReduceSum on DimC. 1. convert to convolution
// in this pass. 2. convert to avgpool in the coming pass. Let's assume the case
// is 1x32x64x128[NCHW], reduce to 1x1x64x128.
// For option1: 32 is channel, need to be the lowest dim, so there is transpose needed
// to convert from NCHW to NHWC.
// For option2: we can permute cast to NHWC layout, then H = 32, and the avgpool happen
// on DimH, then we actually don't need the transpose.
// So here we add isBeneficial to convert when:
// 1. there is a NCE parent or child
// 2. W is not aligned.

bool isBeneficialToConvert(IE::ReduceSumOp origOp, Logger log) {
    auto outShape = getShape(origOp.getOutput());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto alignment = VPU::NCEInvariant::getAlignment(outType.getElementType());
    if (outShape[Dims4D::Act::W] % alignment != 0) {
        return true;
    }

    auto parentOp = origOp.getInput().getDefiningOp();
    if (parentOp != nullptr && mlir::succeeded(VPU::NCEInvariant::isSupported(parentOp, log))) {
        return true;
    }

    for (auto user : origOp.getOutput().getUsers()) {
        if (mlir::succeeded(VPU::NCEInvariant::isSupported(user, log))) {
            return true;
        }
    }
    return false;
}

bool ReduceSumToConvRewriter::isSupportedReduceSum(IE::ReduceSumOp origOp, Logger log) const {
    // Check shape
    const auto inputShape = getShape(origOp.getInput());
    if (!isValidShape(inputShape, log)) {
        log.trace("Shape is invalid {0} at {1}", origOp->getName(), origOp->getLoc());
        return false;
    }

    // Check reduce axis
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    if (axes.size() != 1) {
        log.trace("Only support ReduceSum reduce on one dimension");
        return false;
    }

    auto reduceAxis = axes[0];
    if (reduceAxis != Dims4D::Act::C.ind()) {
        log.trace("Only support ReduceSum reduce on channel");
        return false;
    }

    // Check keep_dims
    if (!origOp.getKeepDims()) {
        log.trace("Only support ReduceSum when keep_dims is true");
        return false;
    }

    return true;
}

IE::ConvolutionOp ReduceSumToConvRewriter::createConvolution(mlir::Value activation, mlir::Value weights,
                                                             mlir::Location newLoc,
                                                             mlir::PatternRewriter& rewriter) const {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    return rewriter.create<IE::ConvolutionOp>(newLoc, activation, weights, nullptr, strides, kernelPadsBegin,
                                              kernelPadsEnd, dilations, nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// For example, a ReduceSum operation with 1x16x8x8@NCHW input tensor
// Create 1x16x1x1 convolution filter, the weights value should be:
// 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
//
mlir::Value ReduceSumToConvRewriter::createConvFilter(mlir::Value activation, mlir::PatternRewriter& rewriter) const {
    const auto IC = getShape(activation)[Dims4D::Act::C];
    const auto KX = 1;
    const auto KY = 1;
    const auto OC = 1;

    const Shape weightShape = {OC, IC, KX, KY};

    SmallVector<float> weights(weightShape.totalSize(), .0f);

    // assign values
    for (auto i = 0; i < IC; ++i) {
        weights[i] = 1.0f;
    }

    const DimsOrder weightOrder = DimsOrder::OIYX;
    const auto weightType = mlir::RankedTensorType::get(
            weightShape.raw(), mlir::cast<NDTypeInterface>(activation.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightType, ArrayRef(weights));
}

mlir::LogicalResult ReduceSumToConvRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    if (!isSupportedReduceSum(origOp, _log)) {
        return mlir::failure();
    }

    if (!isBeneficialToConvert(origOp, _log)) {
        return mlir::failure();
    }

    const auto origLoc = origOp->getLoc();
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origLoc);

    // Create convolution filiter
    auto weights = createConvFilter(origOp.getInput(), rewriter);

    // Create convolution
    const auto convLoc = appendLoc(origLoc, "as_convolution");
    auto conv = createConvolution(origOp.getInput(), weights, convLoc, rewriter);

    rewriter.replaceOp(origOp, conv.getOutput());

    _log.trace("[{0}] Successfully convert ReduceSum to Convolution '{1}'", getDebugName(), origLoc);
    return mlir::success();
}

//
// ConvertReduceSumToConvPass
//

class ConvertReduceSumToConvPass final : public IE::impl::ConvertReduceSumToConvBase<ConvertReduceSumToConvPass> {
public:
    explicit ConvertReduceSumToConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertReduceSumToConvPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    // Convert ReduceSum to Convolution operation is optimum solution in case reduce axis is C
    mlir::RewritePatternSet pattern(&ctx);
    pattern.add<ReduceSumToConvRewriter>(&ctx, _log);

    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(pattern), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReduceSumToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceSumToConvPass(Logger log) {
    return std::make_unique<ConvertReduceSumToConvPass>(log);
}
