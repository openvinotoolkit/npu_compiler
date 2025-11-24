//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
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

bool isSupportedElemType(IE::ReduceSumOp origOp) {
    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputElemType = inputType.getElementType();
    return mlir::isa<mlir::quant::QuantizedType, mlir::Float16Type>(inputElemType);
}

IE::ConvolutionOp createConvolution(mlir::Value activation, mlir::Value weights, mlir::Location newLoc,
                                    mlir::PatternRewriter& rewriter, NDTypeInterface outType = nullptr) {
    const auto ctx = rewriter.getContext();
    const auto strides = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto kernelPadsBegin = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto kernelPadsEnd = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
    const auto dilations = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});

    return (outType == nullptr) ? rewriter.create<IE::ConvolutionOp>(newLoc, activation, weights, nullptr, strides,
                                                                     kernelPadsBegin, kernelPadsEnd, dilations, nullptr,
                                                                     nullptr, nullptr, nullptr, nullptr)
                                : rewriter.create<IE::ConvolutionOp>(newLoc, outType, activation, weights, nullptr,
                                                                     strides, kernelPadsBegin, kernelPadsEnd, dilations,
                                                                     nullptr, nullptr, nullptr, nullptr, nullptr);
}

//
// For example, a ReduceSum operation with 1x16x8x8@NCHW input tensor
// Create 1x16x1x1 convolution filter, the weights value should be:
// 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
//
mlir::Value createConvFilter(mlir::Value activation, mlir::PatternRewriter& rewriter) {
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
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightType, ArrayRef(weights));
}

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
    if (!isSupportedElemType(origOp)) {
        return false;
    }
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
// InnerDimReduceSumToConvRewriter
//

class InnerDimReduceSumToConvRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    InnerDimReduceSumToConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("InnerDimReduceSumToConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    bool isValidReduceSum(IE::ReduceSumOp origOp) const;

    bool isBeneficialTransformation(IE::ReduceSumOp origOp) const;

    mutable int64_t _reduceAxis = -1;

    Logger _log;
};

bool InnerDimReduceSumToConvRewriter::isValidReduceSum(IE::ReduceSumOp origOp) const {
    if (!isSupportedElemType(origOp)) {
        return false;
    }

    auto axesAttr = origOp.getAxesValue();
    if (!axesAttr.has_value()) {
        return false;
    }

    auto axes = parseIntArrayAttr<int64_t>(axesAttr.value());
    if (axes.size() != 1) {
        _log.trace("Only support ReduceSum reduce on one dimension");
        return false;
    }

    _reduceAxis = axes[0];
    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    auto order = inputType.getDimsOrder();
    if (order.dimAt(order.numDims() - 1) != Dim(_reduceAxis)) {
        _log.trace("Only support ReduceSum reduce on the inner most dimension");
        return false;
    }

    return true;
}

bool InnerDimReduceSumToConvRewriter::isBeneficialTransformation(IE::ReduceSumOp origOp) const {
    const auto inputShape = getShape(origOp.getInput());
    constexpr int64_t THRESHOLD_FOR_BENEFICIAL_TRANSFORMATION = 4096;
    return inputShape.totalSize() >= THRESHOLD_FOR_BENEFICIAL_TRANSFORMATION;
}

mlir::LogicalResult InnerDimReduceSumToConvRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (config::isReduceOpSupportedOnNCE(origOp) && VPU::isNCEReduceSupported(origOp, logCb)) {
        return mlir::failure();
    }

    if (!isValidReduceSum(origOp) || !isBeneficialTransformation(origOp)) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    // Calculate the total dim size on the non-reduction axes
    int64_t mergedShape = 1;
    auto inputShape = inputType.getShape();
    for (int64_t i = 0; i < inputType.getRank(); i++) {
        if (i == _reduceAxis) {
            continue;
        }
        mergedShape *= inputShape[Dim(i)];
    }

    auto ctx = rewriter.getContext();
    const auto origLoc = origOp->getLoc();

    // Reshape input to handle the case where the original shape is not 4D
    auto newH = mergedShape;
    auto newW = 1;
    auto newC = inputShape[Dim(_reduceAxis)];

    auto newInputShape = Shape({1, newH, newW, newC});
    const auto newInShapeAttr = getIntArrayAttr(getContext(), newInputShape);
    auto inReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "input_reshape"), origOp->getOperand(0),
                                                      nullptr, false, newInShapeAttr);

    // Cast input to NHWC as Conv activation
    auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(checked_cast<uint32_t>(newInputShape.size()), ctx);
    auto inPermuteCastOp =
            rewriter.create<IE::PermuteCastOp>(appendLoc(origLoc, "input_permute_cast"), inReshapeOp.getOutput(),
                                               DimsOrder::NHWC.toAffineMap(ctx), identityMap);

    // Create filter and Conv
    auto weights = createConvFilter(inPermuteCastOp.getOutput(), rewriter);

    const auto convLoc = appendLoc(origLoc, "reducesum_as_conv");
    auto convOutputType = mlir::cast<NDTypeInterface>(inPermuteCastOp.getOutput().getType());
    convOutputType = convOutputType.changeShape(ShapeRef({1, 1, newH, newW}));
    auto conv = createConvolution(inPermuteCastOp.getOutput(), weights, convLoc, rewriter, convOutputType);

    // Cast Conv output to the original order and shape
    auto outPermuteCast = rewriter.create<IE::PermuteCastOp>(
            appendLoc(origLoc, "output_permute_cast"), conv.getOutput(), DimsOrder::NCHW.toAffineMap(ctx), identityMap);
    auto outReshapeOp =
            rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "output_reshape"), outPermuteCast.getOutput(), nullptr,
                                           false, getIntArrayAttr(ctx, getShape(origOp.getOutput())));

    rewriter.replaceOp(origOp, outReshapeOp);

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
    pattern.add<InnerDimReduceSumToConvRewriter>(&ctx, _log);

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
