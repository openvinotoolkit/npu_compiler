//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTREDUCETOPOOLING
#define GEN_PASS_DEF_CONVERTREDUCETOPOOLING
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

struct PoolingAttr {
    mlir::ArrayAttr kernelAttr;
    mlir::ArrayAttr stridesAttr;
    mlir::ArrayAttr padBeginAttr;
    mlir::ArrayAttr padEndAttr;
    IE::RoundingTypeAttr roundingAttr;
    mlir::UnitAttr excludePadsAttr;
};

bool isReduceOneDim(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> axes) {
    for (const auto& axis : axes) {
        if (inputShape[axis] != 1) {
            return false;
        }
    }
    return true;
}

bool isSpatialDimsReduction(ArrayRef<int64_t> axes) {
    for (const auto& axis : axes) {
        if (axis <= 1) {
            return false;
        }
    }
    return true;
}

bool isBatchDimReduction(ArrayRef<int64_t> axes) {
    for (const auto& axis : axes) {
        if (axis == 0) {
            return true;
        }
    }
    return false;
}

bool isAxesConsecutive(ArrayRef<int64_t> axes) {
    for (size_t i = 1; i < axes.size(); i++) {
        if (axes[i] - axes[i - 1] != 1) {
            return false;
        }
    }

    return true;
}

void constructAvgOpParams(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> outputShape, ArrayRef<int64_t> axes,
                          bool avoidExpandCase, SmallVector<int64_t>& kernel, SmallVector<int64_t>& strides,
                          SmallVector<int64_t>& padBegin, SmallVector<int64_t>& padEnd,
                          SmallVector<int64_t>& shapeBegin, SmallVector<int64_t>& shapeEnd) {
    /*
     * Prepare default attributes for Pooling operation
     *      padBegin/padEnd - should be zeros as we don't need any padding
     *      strides - should be filled with ones
     *      kernel  - depends on Reduction operation axes
     *
     * Also here we decide should we use Reshapes before and after Pooling
     *      shapeBegin - if not empty indicates that we need a Reshape before Pooling
     *      shapeEnd   - if not empty indicates that we need a Reshape after Pooling
     */

    shapeEnd = to_small_vector(outputShape);
    strides.assign({1, 1});
    padBegin.assign({0, 0});
    padEnd.assign({0, 0});

    if (avoidExpandCase) {
        // Reduction is on channel dimension, so we ensure that we leave at least 16 channels after flattening data for
        // hardware requirements
        const int64_t spatialSizeProd = inputShape[Dims4D::Act::H.ind()] * inputShape[Dims4D::Act::W.ind()];
        const int64_t newC = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        const auto origC = inputShape[Dims4D::Act::C.ind()];
        shapeBegin.assign({1, inputShape[Dims4D::Act::N.ind()] * newC, origC, spatialSizeProd / newC});
        kernel.assign({origC, 1});
    } else if (!isSpatialDimsReduction(axes) || inputShape.size() != 4) {
        // In case if reduction applies not to spatial dimensions
        // we have to fit it into 4D Pooling
        int64_t dimsProd = 1;
        int64_t dimsBegin = 1;
        int64_t dimsEnd = 1;
        for (int64_t i = 0; static_cast<size_t>(i) < inputShape.size(); ++i) {
            if (i < axes.front()) {
                dimsBegin *= checked_cast<size_t>(inputShape[i]);
            } else if (i >= axes.front() && i <= axes.back()) {
                dimsProd *= inputShape[i];
            } else {
                dimsEnd *= inputShape[i];
            }
        }
        // The batch dimension is repositioned in the shape
        // only in case of batch dimension reduction
        if (isBatchDimReduction(axes)) {
            shapeBegin.assign({dimsBegin, 1, dimsProd, dimsEnd});
        } else {
            shapeBegin.assign({1, dimsBegin, dimsProd, dimsEnd});
        }

        kernel.assign({dimsProd, 1});
    } else {
        // When input shape size is 4 and axis is not 0 or 1
        shapeBegin = to_small_vector(inputShape);
        kernel.assign({1, 1});
        for (const auto& axis : axes) {
            kernel[axis - 2] = inputShape[axis];
        }
    }

    // In case both spatial dimensions are reduced, we should try to make them balanced
    // For example, global pooling kernel [77, 1] should be converted to [11, 7]
    // For input tensors having C dim > VPU_DIMENSION_LIMIT it seems like not having them balanced is better
    // so data can be splitted in ReshapeMaxPool pass
    if (shapeBegin[Dims4D::Act::H.ind()] == kernel[Dims4D::Kernel::Y.ind()] &&
        shapeBegin[Dims4D::Act::W.ind()] == kernel[Dims4D::Kernel::X.ind()] &&
        shapeBegin[Dims4D::Act::C.ind()] <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        const int64_t spatialSize = kernel[Dims4D::Kernel::Y.ind()] * kernel[Dims4D::Kernel::X.ind()];
        const auto factor = getFactorsList(spatialSize).back();
        if (shapeBegin[Dims4D::Act::H.ind()] != factor.second) {
            shapeBegin[Dims4D::Act::H.ind()] = factor.first;
            shapeBegin[Dims4D::Act::W.ind()] = factor.second;
            kernel.assign({factor.first, factor.second});
        }
    }
}

mlir::LogicalResult generalReduceRewrite(
        mlir::Operation* origOp, mlir::PatternRewriter& rewriter, llvm::SmallVector<int64_t> axes,
        FuncRef<mlir::Operation*(mlir::Location, mlir::Value, PoolingAttr)> makeOperations) {
    auto inputShape = getShape(origOp->getOperand(0)).raw();
    const auto outputShape = getShape(origOp->getResult(0)).raw();
    auto* ctx = origOp->getContext();
    // If axis is negative, it indicates indexing from the end.
    // For example -1 represents the last dimension, -2 represents the penultimate dimension
    // Adding shape size of tensor to the negative axis and do sorting can convert it to a normal positive axis case.
    for (size_t i = 0; i < axes.size(); i++) {
        if (axes[i] < 0) {
            axes[i] += inputShape.size();
        }
    }
    std::sort(axes.begin(), axes.end());

    // If Reduce op reduces only 1 dims we replace it with Reshape
    if (isReduceOneDim(inputShape, axes)) {
        auto newResult = origOp->getOperand(0);
        if (inputShape != outputShape) {
            const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
            newResult = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in_0"), origOp->getOperand(0),
                                                       outputShapeAttr);
        }
        rewriter.replaceOp(origOp, newResult);
        return mlir::success();
    }

    // Optimization for N-axis (batch) reduction: when H=1 and W is channel-aligned,
    // two Transpose ops bracket the pool so that AdjustLayouts can later rewrite each
    // Transpose into a zero-copy PermuteCast with NHWC layout. In NHWC terms, the
    // reduction axis becomes the spatial H dimension and W becomes the channel dimension
    // (C_nhwc=W >= 16), avoiding Expand/Slice overhead.
    //
    //   NxCx1xW NCHW (axes=[0])
    //       |
    //   Reshape -> 1xNxCxW NCHW  (valid since H=1, flat index unchanged)
    //       |
    //   Transpose [N,W,C,H] -> 1xWxNxC NCHW  (AdjustLayouts rewrites to PermuteCast NHWC)
    //       |
    //   Pool kernel [N,1]  (W >= 16 ensures channel alignment in NHWC, no Expand/Slice)
    //       |
    //   Transpose [N,H,W,C] -> 1x1xCxW NCHW  (AdjustLayouts rewrites to PermuteCast)
    //       |
    //   Reshape -> output shape
    if (inputShape.size() == 4 && axes.size() == 1 && axes.front() == Dims4D::Act::N.ind() &&
        inputShape[Dims4D::Act::H.ind()] == 1 &&
        inputShape[Dims4D::Act::W.ind()] % VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT == 0 &&
        inputShape[Dims4D::Act::N.ind()] > 1) {
        const auto origN = inputShape[Dims4D::Act::N.ind()];
        const auto origC = inputShape[Dims4D::Act::C.ind()];
        const auto origW = inputShape[Dims4D::Act::W.ind()];

        auto input = origOp->getOperand(0);

        // Reshape NxCx1xW -> 1xNxCxW (valid since H=1, flat index unchanged)
        const auto reshapeShapeAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, origN, origC, origW});
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), input, reshapeShapeAttr);

        // Transpose [N,W,C,H]: 1xNxCxW NCHW -> 1xWxNxC NCHW
        // AdjustLayouts will later reorder this Transpose to NHWC layout, enabling a zero-copy
        // PermuteCast. In NHWC terms: C=origW (channel-aligned >= 16), H=origN (pool target), W=origC.
        DimArr permFwd{Dims4D::Act::N, Dims4D::Act::W, Dims4D::Act::C, Dims4D::Act::H};
        auto orderFwd = DimsOrder::fromPermutation(ArrayRef(permFwd));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), input, nullptr,
                                                 mlir::AffineMapAttr::get(orderFwd.toAffineMap(ctx)));

        const auto poolKernelAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{origN, 1});
        const auto poolStridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
        const auto poolPadBeginAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto poolPadEndAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{0, 0});
        const auto poolRoundingAttr = IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR);
        const auto poolExcludePadsAttr = mlir::UnitAttr::get(ctx);

        auto* newOp = makeOperations(origOp->getLoc(), input,
                                     {poolKernelAttr, poolStridesAttr, poolPadBeginAttr, poolPadEndAttr,
                                      poolRoundingAttr, poolExcludePadsAttr});
        input = newOp->getResult(0);

        // Transpose [N,H,W,C]: 1xWx1xC NCHW -> 1x1xCxW NCHW
        // Inverse of transpose_in; AdjustLayouts will convert this to a PermuteCast as well.
        DimArr permBwd{Dims4D::Act::N, Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
        auto orderBwd = DimsOrder::fromPermutation(ArrayRef(permBwd));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_out"), input, nullptr,
                                                 mlir::AffineMapAttr::get(orderBwd.toAffineMap(ctx)));

        if (getShape(input).raw() != outputShape) {
            const auto outputShapeAttr = getIntArrayAttr(ctx, ArrayRef(outputShape));
            input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), input, outputShapeAttr);
        }

        rewriter.replaceOp(origOp, input);
        return mlir::success();
    }

    const auto reduceChannels = axes.size() == 1 && axes.front() == Dims4D::Act::C.ind();
    const auto heightAligned = (inputShape.size() == 4)
                                       ? (inputShape[Dims4D::Act::H.ind()] * inputShape[Dims4D::Act::W.ind()] %
                                                  VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT ==
                                          0)
                                       : false;

    auto avoidExpandCase = reduceChannels && heightAligned;

    SmallVector<int64_t> kernel, strides, padBegin, padEnd, shapeBegin, shapeEnd;
    constructAvgOpParams(inputShape, outputShape, axes, avoidExpandCase, kernel, strides, padBegin, padEnd, shapeBegin,
                         shapeEnd);

    /*
     *  ReduceMean => AvgPool
     *                AvgPool->Reshape (in case if keep_dims=False)
     *                Reshape->AvgPool->Reshape (in case when axes don't match spatial dims)
     *  ReduceMax  => MaxPool
     *                MaxPool->Reshape (in case if keep_dims=False)
     *                Reshape->MaxPool->Reshape (in case when axes don't match spatial dims)
     *  ReduceSum  => AvgPool->Multiply
     *                AvgPool->Multiply->Reshape (in case if keep_dims=False)
     *                Reshape->AvgPool->Multiply->Reshape (in case when axes don't match spatial dims)
     *  ReduceMin  => Negative->MaxPool->Negative
     *                Negative->MaxPool->->Negative->Reshape (in case if keep_dims=False)
     *                Reshape->Negative->MaxPool->Negative->Reshape (in case when axes don't match spatial dims)
     */

    auto input = origOp->getOperand(0);

    if (avoidExpandCase) {
        auto newN = inputShape[Dims4D::Act::N.ind()];
        auto newC = inputShape[Dims4D::Act::C.ind()];
        auto newW = VPU::NCEInvariant::VPU_CHANNEL_ALIGNMENT;
        auto newH = inputShape[Dims4D::Act::W.ind()] * inputShape[Dims4D::Act::H.ind()] / newW;

        /*
         *  Keep W dim alignment, so that we can convert the Transpose to permuteCast
         *  Eg:  Convert reduceMax 1x256x24x24 -> 1x1x24x24 to
         *          input 1x256x24x24 NCHW
         *            |
         *         reshape to 1x256x36x16 NCHW
         *            |
         *        permuteCast to 1x16x256x36 NHWC
         *            |
         *         maxPool to 1x16x1x36 NHWC
         *            |
         *         permuteCast to 1x1x36x16 NCHW
         *            |
         *         reshape to 1x1x24x24 NCHW
         */

        Shape newShape{newN, newC, newH, newW};
        const auto newShapeAttr = getIntArrayAttr(ctx, ShapeRef(newShape));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), input, newShapeAttr);

        DimArr perm{Dims4D::Act::N, Dims4D::Act::W, Dims4D::Act::C, Dims4D::Act::H};
        auto order = DimsOrder::fromPermutation(ArrayRef(perm));
        auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), input, nullptr, orderAttr);
        inputShape = getShape(input).raw();
    }

    if (shapeBegin != inputShape) {
        const auto shapeBeginAttr = getIntArrayAttr(ctx, ArrayRef(shapeBegin));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in_2"), input, shapeBeginAttr);
    }

    const auto kernelAttr = getIntArrayAttr(ctx, ArrayRef(kernel));
    const auto stridesAttr = getIntArrayAttr(ctx, ArrayRef(strides));
    const auto padBeginAttr = getIntArrayAttr(ctx, ArrayRef(padBegin));
    const auto padEndAttr = getIntArrayAttr(ctx, ArrayRef(padEnd));
    const auto roundingAttr = IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR);
    const auto excludePadsAttr = mlir::UnitAttr::get(ctx);

    auto* newOp = makeOperations(origOp->getLoc(), input,
                                 {kernelAttr, stridesAttr, padBeginAttr, padEndAttr, roundingAttr, excludePadsAttr});
    input = newOp->getResult(0);

    // when N>1, we create reshape_in_2 based on fused NC after transpose_in, so we need to split NC first before
    // transpose_out/reshape_out
    const auto origReduceInShape = to_small_vector(getShape(origOp->getOperand(0)));
    const int64_t origInN = (origReduceInShape.size() == 4) ? origReduceInShape[Dims4D::Act::N.ind()] : 1;
    const auto newPoolOutShape = to_small_vector(getShape(input));
    if (shapeEnd != newPoolOutShape && origInN != 1) {
        const auto newOutN = newPoolOutShape[Dims4D::Act::N.ind()] * origInN;
        const auto newOutC = newPoolOutShape[Dims4D::Act::C.ind()] / origInN;
        const auto newOutH = newPoolOutShape[Dims4D::Act::H.ind()];
        const auto newOutW = newPoolOutShape[Dims4D::Act::W.ind()];
        const auto splitNCShape = getIntArrayAttr(ctx, SmallVector<int64_t>({newOutN, newOutC, newOutH, newOutW}));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out_2"), input, splitNCShape);
    }

    if (avoidExpandCase) {
        DimArr perm{Dims4D::Act::N, Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
        auto order = DimsOrder::fromPermutation(ArrayRef(perm));
        auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_out"), input, nullptr, orderAttr);
    }

    if (shapeEnd != to_small_vector(getShape(input))) {
        const auto shapeEndAttr = getIntArrayAttr(ctx, ArrayRef(shapeEnd));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), input, shapeEndAttr);
    }

    rewriter.replaceOp(origOp, input);
    return mlir::success();
}

bool isValidOperationForConversion(mlir::Operation* op, Logger log,
                                   IE::ChannelAxisReductionWithDPUParentCheckerBase* channelReductionChecker,
                                   bool checkConsecutiveAxes = true) {
    const auto isLegalOp = [&](mlir::Operation* op) {
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };

        if (!mlir::isa<IE::ReduceMeanOp, IE::ReduceSumOp, IE::ReduceMinOp, IE::ReduceMaxOp>(op)) {
            return false;
        }
        /*To do: remove this check and implement E#129361*/
        if (mlir::isa<IE::ReduceMeanOp, IE::ReduceSumOp>(op)) {
            if (config::isReduceOpSupportedOnNCE(op) && VPU::isNCEReduceSupported(op, logCb)) {
                return false;
            }
        }

        const auto inputShape = getShape(op->getOperand(0)).raw();
        int64_t mergedDim = 1;

        // Check that axes are consecutive otherwise this conversion is not applicable
        llvm::SmallVector<int64_t> axes = {0};
        if (op->hasAttr("axes_value")) {
            if (auto axesValue = mlir::dyn_cast_or_null<mlir::ArrayAttr>(op->getAttr("axes_value"))) {
                axes = parseIntArrayAttr<int64_t>(axesValue);
            }
        } else {
            VPUX_THROW("Operator has no axes_value attribute");
        }

        for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] < 0) {
                axes[i] += inputShape.size();
            }
            mergedDim *= inputShape[axes[i]];
        }
        // For ReduceMin/ReduceMax on C axis, keep fusion opportunity only when the parent can run on DPU.
        if (channelReductionChecker->isChannelAxisReductionWithDPUParent(op, axes, log)) {
            return false;
        }

        std::sort(axes.begin(), axes.end());

        if (checkConsecutiveAxes && !isAxesConsecutive(axes)) {
            return false;
        }

        const auto inputElementType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
        const auto outputElementType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
        const auto isDataTypeDpuCompatible = inputElementType.isF16() && outputElementType.isF16();

        const auto maxKernelSize = config::getNPUConstraints(op->getContext()).maxKernelSize;
        // Check that handleLargeKernels supports this op
        const auto isKernelSizeDpuCompatible =
                (axes.size() == 2) ? (vpux::IE::isPoolingKernelSizeValid(inputShape[axes[0]], maxKernelSize) &&
                                      vpux::IE::isPoolingKernelSizeValid(inputShape[axes[1]], maxKernelSize))
                                   : vpux::IE::isPoolingKernelSizeValid(mergedDim, maxKernelSize);
        const auto dpuCompatible = isKernelSizeDpuCompatible && isDataTypeDpuCompatible;

        const bool isHWCompilationMode = config::getCompilationMode(op) != config::CompilationMode::ReferenceSW;

        // Apply the conversion only if the resulting Pooling operation can run on DPU
        return dpuCompatible && isHWCompilationMode;
    };

    if (mlir::isa<IE::ReduceMinOp>(op)) {
        const auto maxKernelSize = config::getNPUConstraints(op->getContext()).maxKernelSize;
        if ((getShape(op->getResult(0)).totalSize() == 1) &&
            (getShape(op->getOperand(0)).totalSize() > std::pow(maxKernelSize, 2))) {
            return false;
        }
    }

    return isLegalOp(op);
}

//
// ReduceMeanRewriter
//

class ReduceMeanRewriter final : public mlir::OpRewritePattern<IE::ReduceMeanOp> {
public:
    ReduceMeanRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log,
                       IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<IE::ReduceMeanOp>(ctx, benefit), _log(log), _channelReductionChecker(checker) {
        setDebugName("ReduceMeanRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMeanOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

mlir::LogicalResult ReduceMeanRewriter::matchAndRewrite(IE::ReduceMeanOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMean layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    if (!isValidOperationForConversion(origOp, _log, _channelReductionChecker)) {
        return mlir::failure();
    }
    return generalReduceRewrite(origOp, rewriter, std::move(axes),
                                [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                                    return rewriter.create<IE::AvgPoolOp>(
                                            appendLoc(loc, "as_avgpool"), input, attr.kernelAttr, attr.stridesAttr,
                                            attr.padBeginAttr, attr.padEndAttr, attr.roundingAttr, attr.excludePadsAttr,
                                            nullptr, nullptr, nullptr, nullptr, nullptr);
                                });
}

//
// ReduceMaxRewriter
//

class ReduceMaxRewriter final : public mlir::OpRewritePattern<IE::ReduceMaxOp> {
public:
    ReduceMaxRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log,
                      IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<IE::ReduceMaxOp>(ctx, benefit), _log(log), _channelReductionChecker(checker) {
        setDebugName("ReduceMaxRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

mlir::LogicalResult ReduceMaxRewriter::matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMax layer at '{1}'", getDebugName(), origOp->getLoc());
    if (!isValidOperationForConversion(origOp, _log, _channelReductionChecker)) {
        _log.trace("Operation at '{0}' is not valid for conversion to Pooling", origOp->getLoc());
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    return generalReduceRewrite(origOp, rewriter, std::move(axes),
                                [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                                    return rewriter.create<IE::MaxPoolOp>(
                                            appendLoc(loc, "as_maxpool"), input, attr.kernelAttr, attr.stridesAttr,
                                            attr.padBeginAttr, attr.padEndAttr, attr.roundingAttr, nullptr, nullptr,
                                            nullptr, nullptr, nullptr);
                                });
}

//
// ReduceSumRewriter
//

class ReduceSumRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log,
                      IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx, benefit), _log(log), _channelReductionChecker(checker) {
        setDebugName("ReduceSumRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

mlir::LogicalResult ReduceSumRewriter::matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    if (!isValidOperationForConversion(origOp, _log, _channelReductionChecker)) {
        return mlir::failure();
    }

    return generalReduceRewrite(
            origOp, rewriter, std::move(axes),
            [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                input = rewriter.create<IE::AvgPoolOp>(appendLoc(loc, "as_avgpool"), input, attr.kernelAttr,
                                                       attr.stridesAttr, attr.padBeginAttr, attr.padEndAttr,
                                                       attr.roundingAttr, attr.excludePadsAttr, nullptr, nullptr,
                                                       nullptr, nullptr, nullptr);

                mlir::MLIRContext* ctx = origOp->getContext();
                const auto dataStorageTensor = mlir::RankedTensorType::get({1}, mlir::Float16Type::get(ctx));
                const auto inputShape = getShape(origOp.getInput()).raw();

                auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
                for (size_t i = 0; i < axes.size(); i++) {
                    if (axes[i] < 0) {
                        axes[i] += inputShape.size();
                    }
                }
                std::sort(axes.begin(), axes.end());

                float reductionDimsCount = 1;
                for (const auto& axis : axes) {
                    reductionDimsCount *= inputShape[axis];
                }
                const auto cstOutput = Const::createFloatConst(rewriter, loc, dataStorageTensor, reductionDimsCount);

                const auto broadCastAttr = IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NUMPY);
                return rewriter.create<IE::MultiplyOp>(loc, input, cstOutput, broadCastAttr, nullptr, nullptr, nullptr,
                                                       nullptr);
            });
}

//
// ReduceMinRewriter
//

class ReduceMinRewriter final : public mlir::OpRewritePattern<IE::ReduceMinOp> {
public:
    ReduceMinRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log,
                      IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<IE::ReduceMinOp>(ctx, benefit), _log(log), _channelReductionChecker(checker) {
        setDebugName("ReduceMinRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

mlir::LogicalResult ReduceMinRewriter::matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMin layer at '{1}'", getDebugName(), origOp->getLoc());
    if (!isValidOperationForConversion(origOp, _log, _channelReductionChecker)) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    return generalReduceRewrite(
            origOp, rewriter, std::move(axes),
            [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                auto scale1 = rewriter.create<IE::NegativeOp>(appendLoc(loc, "inv_in"), input.getType(), input);
                auto maxPool = rewriter.create<IE::MaxPoolOp>(appendLoc(loc, "as_maxpool"), scale1.getOutput(),
                                                              attr.kernelAttr, attr.stridesAttr, attr.padBeginAttr,
                                                              attr.padEndAttr, attr.roundingAttr, nullptr, nullptr,
                                                              nullptr, nullptr, nullptr);
                return rewriter.create<IE::NegativeOp>(appendLoc(loc, "inv_out"), maxPool.getOutput().getType(),
                                                       maxPool.getOutput());
            });
}

//
// ReduceSimplifyAxesRewriter
//
// Simplifies Reduce op by removing trivial reduction axes (where input dim is 1)
// and forcing keepDims=true. A trailing Reshape restores the original output shape
// when keepDims was false or trivial axes were dropped.
// After simplification, some Reduce cases become eligible for conversion to
// pooling operations.
// Example: ReduceMean(1x1x3136x64, axes=[1,3], keepDims=false)
//       -> ReduceMean(1x1x3136x64, axes=[3], keepDims=true) -> Reshape
//

template <class ConcreteOp>
class ReduceSimplifyAxesRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReduceSimplifyAxesRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log,
                               IE::ChannelAxisReductionWithDPUParentCheckerBase* checker)
            : mlir::OpRewritePattern<ConcreteOp>(ctx, benefit), _log(log), _channelReductionChecker(checker) {
        this->setDebugName("ReduceSimplifyAxesRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    IE::ChannelAxisReductionWithDPUParentCheckerBase* _channelReductionChecker;
};

template <class ConcreteOp>
mlir::LogicalResult ReduceSimplifyAxesRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Reduce layer at '{1}'", this->getDebugName(), origOp->getLoc());

    if (isValidOperationForConversion(origOp, _log, _channelReductionChecker)) {
        return mlir::failure();
    }

    auto axesAttr = origOp.getAxesValue();
    if (!axesAttr.has_value()) {
        return mlir::failure();
    }
    auto axes = parseIntArrayAttr<int64_t>(axesAttr.value());

    const auto inputShape = getShape(origOp.getInput());
    const auto rank = static_cast<int64_t>(inputShape.size());

    // Normalize negative axes
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
    }
    std::sort(axes.begin(), axes.end());

    // Remove axes where input dim is 1 (trivial reduction)
    SmallVector<int64_t> simplifiedAxes;
    for (const auto& axis : axes) {
        if (inputShape[Dim(axis)] != 1) {
            simplifiedAxes.push_back(axis);
        }
    }

    // Nothing to simplify when all axes are non-trivial
    if (simplifiedAxes.size() == axes.size() || simplifiedAxes.empty()) {
        return mlir::failure();
    }

    if (!isAxesConsecutive(simplifiedAxes) ||
        !isValidOperationForConversion(origOp, _log, _channelReductionChecker, false)) {
        return mlir::failure();
    }

    auto* ctx = origOp->getContext();
    // Create new Reduce op with keepDims=true and only non-trivial axes
    const auto keepDimsAttr = mlir::UnitAttr::get(ctx);
    const auto newAxesAttr = getIntArrayAttr(ctx, simplifiedAxes);

    auto* newOp = rewriter.clone(*origOp.getOperation());
    newOp->setAttr("axes_value", newAxesAttr);
    newOp->setAttr("keep_dims", keepDimsAttr);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::SHAPE | vpux::InferShapedTypeMode::LAYOUT);
    mlir::Value output = newOp->getResult(0);

    // Reshape to restore original output shape (accounts for removed trivial axes and keepDims=false)
    const auto origOutputShape = to_small_vector(getShape(origOp.getOutput()));
    const auto newOutputShape = to_small_vector(getShape(output));
    if (origOutputShape != newOutputShape) {
        const auto outputShapeAttr = getIntArrayAttr(ctx, origOutputShape);
        output = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), output, outputShapeAttr);
    }

    rewriter.replaceOp(origOp, output);
    _log.trace("[{0}] done", this->getDebugName());

    return mlir::success();
}

//
// ReduceSumBatchToChannelRewriter
//

class ReduceSumBatchToChannelRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumBatchToChannelRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ReduceSumOp>(ctx, benefit), _log(log) {
        setDebugName("ReduceSumBatchToChannelRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceSumBatchToChannelRewriter::matchAndRewrite(IE::ReduceSumOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    const auto inputShape = getShape(origOp.getInput());

    if (inputShape.size() != 4) {
        return mlir::failure();
    }

    // Only handle reduce sum on batch dim
    if (axes.size() != 1 || axes[0] != Dims4D::Act::N.ind()) {
        return mlir::failure();
    }

    if (inputShape[Dims4D::Act::C] != 1 || inputShape[Dims4D::Act::N] == 1) {
        return mlir::failure();
    }

    // Swap N and C dims
    const auto newShape = SmallVector<int64_t>{inputShape[Dims4D::Act::C], inputShape[Dims4D::Act::N],
                                               inputShape[Dims4D::Act::H], inputShape[Dims4D::Act::W]};
    const auto newShapeAttr = getIntArrayAttr(getContext(), newShape);
    auto reshapeOp = rewriter.create<IE::ReshapeOp>(origOp->getLoc(), origOp.getInput(), newShapeAttr);

    const auto newAxesAttr = getIntArrayAttr(getContext(), SmallVector<int64_t>{Dims4D::Act::C.ind()});
    auto newReduceOp = rewriter.create<IE::ReduceSumOp>(origOp->getLoc(), reshapeOp.getOutput(), nullptr, newAxesAttr,
                                                        origOp.getKeepDimsAttr());

    rewriter.replaceOp(origOp, newReduceOp.getOutput());
    return mlir::success();
}

//
// ConvertReduceToPoolingPass
//

class ConvertReduceToPoolingPass final : public IE::impl::ConvertReduceToPoolingBase<ConvertReduceToPoolingPass> {
public:
    ConvertReduceToPoolingPass(bool enableFuseReduceMinMaxToDpu, Logger log) {
        this->enableFuseReduceMinMaxToDpu = enableFuseReduceMinMaxToDpu;
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertReduceToPoolingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    const auto channelReductionChecker =
            strategyFactory->getChannelAxisReductionWithDPUParentChecker(enableFuseReduceMinMaxToDpu);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReduceSimplifyAxesRewriter<IE::ReduceMeanOp>>(&ctx, benefitLevels[0], _log,
                                                               channelReductionChecker.get());
    patterns.add<ReduceSimplifyAxesRewriter<IE::ReduceMaxOp>>(&ctx, benefitLevels[0], _log,
                                                              channelReductionChecker.get());
    patterns.add<ReduceSimplifyAxesRewriter<IE::ReduceMinOp>>(&ctx, benefitLevels[0], _log,
                                                              channelReductionChecker.get());
    patterns.add<ReduceSimplifyAxesRewriter<IE::ReduceSumOp>>(&ctx, benefitLevels[0], _log,
                                                              channelReductionChecker.get());
    patterns.add<ReduceSumBatchToChannelRewriter>(&ctx, benefitLevels[0], _log);
    patterns.add<ReduceMeanRewriter>(&ctx, benefitLevels[1], _log, channelReductionChecker.get());
    patterns.add<ReduceMaxRewriter>(&ctx, benefitLevels[1], _log, channelReductionChecker.get());
    patterns.add<ReduceSumRewriter>(&ctx, benefitLevels[1], _log, channelReductionChecker.get());
    patterns.add<ReduceMinRewriter>(&ctx, benefitLevels[1], _log, channelReductionChecker.get());

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReduceToPoolingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceToPoolingPass(bool enableFuseReduceMinMaxToDpu, Logger log) {
    return std::make_unique<ConvertReduceToPoolingPass>(enableFuseReduceMinMaxToDpu, log);
}
