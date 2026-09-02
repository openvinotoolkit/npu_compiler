//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"

#include <limits>
#include <optional>

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
#include "vpux/compiler/utils/walk_utils.hpp"

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

    return (outType == nullptr) ? rewriter.create<IE::ConvolutionOp>(newLoc, activation, weights, strides,
                                                                     kernelPadsBegin, kernelPadsEnd, dilations)
                                : rewriter.create<IE::ConvolutionOp>(newLoc, outType, activation, weights, strides,
                                                                     kernelPadsBegin, kernelPadsEnd, dilations);
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

    const DimsOrder weightOrder = DimsOrder::OIYX;
    const auto weightType = mlir::RankedTensorType::get(
            weightShape.raw(), mlir::cast<NDTypeInterface>(activation.getType()).getElementType(),
            getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    return Const::buildWeightsConst(rewriter, activation.getLoc(), weightType, ArrayRef<float>{1.0f});
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

// Check whether a ReduceSum has an NCE-supported parent or child, without
// considering W-alignment of the output tensor.
bool hasNCEParentOrChild(IE::ReduceSumOp origOp, Logger log) {
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

bool isBeneficialToConvert(IE::ReduceSumOp origOp, Logger log) {
    auto outShape = getShape(origOp.getOutput());
    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto alignment = VPU::NCEInvariant::getAlignment(outType.getElementType());

    if (outShape.size() == 4) {
        if (outShape[Dims4D::Act::W] % alignment != 0) {
            return true;
        }
    }

    return hasNCEParentOrChild(origOp, log);
}

// Returns the single reduce axis (normalized, 0-based) or std::nullopt if the op
// cannot be handled (no axes attr, multiple axes, or out-of-range value).
std::optional<int64_t> getSingleReduceAxis(IE::ReduceSumOp origOp) {
    auto axesAttr = origOp.getAxesValue();
    auto axes = parseIntArrayAttr<int64_t>(axesAttr);
    if (axes.size() != 1) {
        return std::nullopt;
    }
    const int64_t rank = mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getRank();
    std::ignore = rank;
    const int64_t axis = axes[0];
    assert(axis >= 0 && axis < rank && "Axis must be normalized in the operation");
    return axis;
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
    auto reduceAxis = getSingleReduceAxis(origOp);
    if (!reduceAxis.has_value()) {
        log.trace("Only support ReduceSum reduce on one dimension");
        return false;
    }

    if (*reduceAxis != Dims4D::Act::C.ind()) {
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
// OuterDimReduceSumToConvRewriter
//
// Handles ReduceSum on the outermost non-unit dimension of non-4D tensors.
// For axis=0: reshapes [A, D1, D2, ...] to [1, A, D1*D2*..., 1].
// For axis=1 when dim(0)=1: reshapes [1, A, D1, D2, ...] to [1, A, D1*D2*..., 1].
// Both cases preserve row-major memory layout and apply Conv with IC=A to reduce
// A channels to 1 in a single operation.
// Without this, these reductions fall through to multi-stage AvgPool decomposition
// with C=1, generating expensive DMA padding operations for channel alignment.
//

class OuterDimReduceSumToConvRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    OuterDimReduceSumToConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("OuterDimReduceSumToConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult OuterDimReduceSumToConvRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    if (config::isReduceOpSupportedOnNCE(origOp) && VPU::isNCEReduceSupported(origOp, logCb)) {
        return mlir::failure();
    }

    if (!isSupportedElemType(origOp)) {
        return mlir::failure();
    }

    // Do not rewrite ReduceSum with explicit input/output padding attributes.
    // The Convolution replacement does not preserve the semantics of ignoring
    // padded elements in reduction.
    if (origOp.getInputPaddingAttr() != nullptr || origOp.getOutputPaddingAttr() != nullptr) {
        return mlir::failure();
    }

    const auto inputType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto inputRank = inputShape.size();

    // Only handle non-4D tensors; 4D is handled by ReduceSumToConvRewriter
    if (inputRank < 2 || inputRank == 4) {
        return mlir::failure();
    }

    // Handle outermost non-unit dimension reduction:
    //   axis=0: [A, D1, D2, ...] -> Conv with IC=A
    //   axis=1 when dim(0)=1: [1, A, D1, D2, ...] -> Conv with IC=A
    // Both preserve row-major memory layout when mapped to [1, A, rest, 1].
    auto normalizedReduceAxis = getSingleReduceAxis(origOp);
    if (!normalizedReduceAxis.has_value()) {
        return mlir::failure();
    }

    const int64_t reduceAxis = *normalizedReduceAxis;
    // Accept axis=0, or axis=1 when the batch dimension is 1
    const bool isAxis0 = (reduceAxis == 0);
    const bool isAxis1WithUnitBatch = (reduceAxis == 1) && (inputShape[Dim(0)] == 1);
    if (!isAxis0 && !isAxis1WithUnitBatch) {
        return mlir::failure();
    }

    // Require canonical (row-major) memory layout so that mapping the reduce
    // dimension into the convolution channel dimension does not change semantics.
    if (inputType.getDimsOrder() != DimsOrder::fromNumDims(inputRank)) {
        return mlir::failure();
    }

    const auto reduceSize = inputShape[Dim(reduceAxis)];
    if (reduceSize == mlir::ShapedType::kDynamic || reduceSize <= 0) {
        return mlir::failure();
    }

    auto outType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto alignment = VPU::NCEInvariant::getAlignment(outType.getElementType());

    // For axis=1 reductions, only convert when the reduction dimension fits within
    // a single DPU channel group (alignment). Larger reductions produce different
    // FP16 accumulation results compared to the staged AvgPool decomposition,
    // which causes accuracy regressions on models with large IC (e.g. IC=64).
    if (isAxis1WithUnitBatch && reduceSize > alignment) {
        return mlir::failure();
    }

    // For axis=0: Conv is beneficial when the reduction dimension is large enough
    // to exceed the NCE alignment threshold.  Small reductions are only converted
    // when an NCE parent/child makes the Conv worthwhile.
    // For axis=1: the accuracy guard above already restricts to small reduceSize,
    // and converting avoids the multi-stage AvgPool decomposition overhead.
    if (isAxis0 && reduceSize <= alignment && !hasNCEParentOrChild(origOp, _log)) {
        return mlir::failure();
    }

    // Calculate merged spatial dimension from all non-reduce dimensions.
    // Use checked multiplication to guard against overflow on extreme shapes.
    int64_t mergedRest = 1;
    for (size_t i = 0; i < inputRank; ++i) {
        if (static_cast<int64_t>(i) == reduceAxis) {
            continue;
        }
        const auto dimSize = inputShape[Dim(i)];
        if (dimSize == mlir::ShapedType::kDynamic || dimSize <= 0) {
            return mlir::failure();
        }
        if (mergedRest > std::numeric_limits<int64_t>::max() / dimSize) {
            return mlir::failure();
        }
        mergedRest *= dimSize;
    }

    auto ctx = rewriter.getContext();
    const auto origLoc = origOp->getLoc();
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origLoc);

    // Reshape to 4D: [1, A, D1*D2*..., 1] mapping the reduce dimension to IC.
    const auto newInputShape = Shape({1, reduceSize, mergedRest, 1});
    auto inReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "reshape_to_4d"), origOp.getInput(),
                                                      getIntArrayAttr(ctx, newInputShape));

    // Create Conv filter [1, A, 1, 1] with all-ones weights
    auto weights = createConvFilter(inReshapeOp.getOutput(), rewriter);

    // Create Conv: [1, A, mergedRest, 1] -> [1, 1, mergedRest, 1]
    const auto convLoc = appendLoc(origLoc, "as_convolution");
    auto conv = createConvolution(inReshapeOp.getOutput(), weights, convLoc, rewriter);

    // Reshape Conv output to the original ReduceSum output shape
    const auto outShape = getShape(origOp.getOutput());
    auto outReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "reshape_output"), conv.getOutput(),
                                                       getIntArrayAttr(ctx, outShape));

    rewriter.replaceOp(origOp, outReshapeOp);

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

    auto axis = getSingleReduceAxis(origOp);
    if (!axis.has_value()) {
        _log.trace("Only support ReduceSum reduce on one dimension");
        return false;
    }

    _reduceAxis = *axis;
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
    auto inReshapeOp =
            rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "input_reshape"), origOp->getOperand(0), newInShapeAttr);

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
    auto outReshapeOp = rewriter.create<IE::ReshapeOp>(appendLoc(origLoc, "output_reshape"), outPermuteCast.getOutput(),
                                                       getIntArrayAttr(ctx, getShape(origOp.getOutput())));

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

    // Three patterns cover all ReduceSum-to-Conv cases:
    //   ReduceSumToConvRewriter         - 4D input, reduce on channel dim (C)
    //   OuterDimReduceSumToConvRewriter - non-4D input, reduce on outermost dim (axis=0)
    //   InnerDimReduceSumToConvRewriter - non-4D input, reduce on innermost dim
    mlir::RewritePatternSet pattern(&ctx);
    pattern.add<ReduceSumToConvRewriter>(&ctx, _log);
    pattern.add<OuterDimReduceSumToConvRewriter>(&ctx, _log);
    pattern.add<InnerDimReduceSumToConvRewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(pattern));
}

}  // namespace

//
// createConvertReduceSumToConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceSumToConvPass(Logger log) {
    return std::make_unique<ConvertReduceSumToConvPass>(log);
}
