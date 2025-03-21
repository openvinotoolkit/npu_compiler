//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
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
        shapeBegin.assign({1, newC, origC, spatialSizeProd / newC});
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
                                                       nullptr, false, outputShapeAttr);
        }
        rewriter.replaceOp(origOp, newResult);
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
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), input, nullptr, false, newShapeAttr);

        DimArr perm{Dims4D::Act::N, Dims4D::Act::W, Dims4D::Act::C, Dims4D::Act::H};
        auto order = DimsOrder::fromPermutation(ArrayRef(perm));
        auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), input, nullptr, orderAttr);
        inputShape = getShape(input).raw();
    }

    if (shapeBegin != inputShape) {
        const auto shapeBeginAttr = getIntArrayAttr(ctx, ArrayRef(shapeBegin));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in_2"), input, nullptr, false,
                                               shapeBeginAttr);
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

    if (avoidExpandCase) {
        DimArr perm{Dims4D::Act::N, Dims4D::Act::H, Dims4D::Act::W, Dims4D::Act::C};
        auto order = DimsOrder::fromPermutation(ArrayRef(perm));
        auto orderAttr = mlir::AffineMapAttr::get(order.toAffineMap(ctx));
        input = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_out"), input, nullptr, orderAttr);
    }

    if (shapeEnd != to_small_vector(getShape(input))) {
        const auto shapeEndAttr = getIntArrayAttr(ctx, ArrayRef(shapeEnd));
        input = rewriter.create<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_out"), input, nullptr, false, shapeEndAttr);
    }

    rewriter.replaceOp(origOp, input);
    return mlir::success();
}

//
// ReduceMeanRewriter
//

class ReduceMeanRewriter final : public mlir::OpRewritePattern<IE::ReduceMeanOp> {
public:
    ReduceMeanRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMeanOp>(ctx), _log(log) {
        setDebugName("ReduceMeanRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMeanOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMeanRewriter::matchAndRewrite(IE::ReduceMeanOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMean layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
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
    ReduceMaxRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMaxOp>(ctx), _log(log) {
        setDebugName("ReduceMaxRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMaxRewriter::matchAndRewrite(IE::ReduceMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMax layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    return generalReduceRewrite(origOp, rewriter, std::move(axes),
                                [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                                    return rewriter.create<IE::MaxPoolOp>(
                                            appendLoc(loc, "as_maxpool"), input, attr.kernelAttr, attr.stridesAttr,
                                            attr.padBeginAttr, attr.padEndAttr, attr.roundingAttr, nullptr, nullptr,
                                            nullptr, nullptr);
                                });
}

//
// ReduceSumRewriter
//

class ReduceSumRewriter final : public mlir::OpRewritePattern<IE::ReduceSumOp> {
public:
    ReduceSumRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceSumOp>(ctx), _log(log) {
        setDebugName("ReduceSumRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceSumRewriter::matchAndRewrite(IE::ReduceSumOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceSum layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
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
    ReduceMinRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ReduceMinOp>(ctx), _log(log) {
        setDebugName("ReduceMinRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ReduceMinRewriter::matchAndRewrite(IE::ReduceMinOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got ReduceMin layer at '{1}'", getDebugName(), origOp->getLoc());
    auto axes = parseIntArrayAttr<int64_t>(origOp.getAxesValue().value());
    return generalReduceRewrite(
            origOp, rewriter, std::move(axes),
            [&](mlir::Location loc, mlir::Value input, PoolingAttr attr) -> mlir::Operation* {
                auto scale1 = rewriter.create<IE::NegativeOp>(appendLoc(loc, "inv_in"), input.getType(), input);
                auto maxPool = rewriter.create<IE::MaxPoolOp>(
                        appendLoc(loc, "as_maxpool"), scale1.getOutput(), attr.kernelAttr, attr.stridesAttr,
                        attr.padBeginAttr, attr.padEndAttr, attr.roundingAttr, nullptr, nullptr, nullptr, nullptr);
                return rewriter.create<IE::NegativeOp>(appendLoc(loc, "inv_out"), maxPool.getOutput().getType(),
                                                       maxPool.getOutput());
            });
}

//
// ConvertReduceToPoolingPass
//

class ConvertReduceToPoolingPass final : public IE::impl::ConvertReduceToPoolingBase<ConvertReduceToPoolingPass> {
public:
    explicit ConvertReduceToPoolingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertReduceToPoolingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalOp = [&](mlir::Operation* op) {
        const auto logCb = [&](const formatv_object_base& msg) {
            _log.trace("{0}", msg.str());
        };

        /*To do: remove this check and implement E#129361*/
        if (mlir::isa<IE::ReduceMeanOp, IE::ReduceSumOp>(op)) {
            if (VPU::isReduceOpSupportedOnNCE(op) && VPU::isNCEReduceSupported(op, logCb)) {
                return true;
            }
        }

        const auto inputShape = getShape(op->getOperand(0)).raw();
        int64_t mergedDim = 1;

        // Check that axes are consecutive otherwise this conversion is not applicable
        llvm::SmallVector<int64_t> axes = {0};
        if (op->hasAttr("axes_value")) {
            if (auto axesValue = op->getAttr("axes_value").dyn_cast_or_null<mlir::ArrayAttr>()) {
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

        std::sort(axes.begin(), axes.end());
        for (size_t i = 1; i < axes.size(); i++) {
            if (axes[i] - axes[i - 1] != 1) {
                return true;
            }
        }

        const auto inputElementType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        const auto outputElementType = op->getResult(0).getType().cast<vpux::NDTypeInterface>().getElementType();
        const auto isDataTypeDpuCompatible = inputElementType.isF16() && outputElementType.isF16();

        const auto maxKernelSize = VPU::getMaxKernelSize(op);
        // Check that handleLargeKernels supports this op
        const auto isKernelSizeDpuCompatible =
                (axes.size() == 2) ? (vpux::IE::isPoolingKernelSizeValid(inputShape[axes[0]], maxKernelSize) &&
                                      vpux::IE::isPoolingKernelSizeValid(inputShape[axes[1]], maxKernelSize))
                                   : vpux::IE::isPoolingKernelSizeValid(mergedDim, maxKernelSize);
        const auto dpuCompatible = isKernelSizeDpuCompatible && isDataTypeDpuCompatible;

        const bool isHWCompilationMode = VPU::getCompilationMode(op) == VPU::CompilationMode::DefaultHW;

        // Apply the conversion only if the resulting Pooling operation can run on DPU
        return !(dpuCompatible && isHWCompilationMode);
    };

    // For ReduceMin, in case all axes are reduced and the kernel size exceeds maxKernelSize, the
    // SHAVE kernel offers more performance since the NCE solution generates pyramidal MaxPool operations based on the
    // kernel size restriction
    const auto isLegalReduceMin = [&](IE::ReduceMinOp op) {
        if (isLegalOp(op)) {
            return true;
        }

        const auto maxKernelSize = VPU::getMaxKernelSize(op);

        if ((getShape(op->getResult(0)).totalSize() == 1) &&
            (getShape(op->getOperand(0)).totalSize() > std::pow(maxKernelSize, 2))) {
            return true;
        }

        return false;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::AvgPoolOp>();
    target.addLegalOp<IE::MaxPoolOp>();
    target.addLegalOp<IE::ReshapeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::MultiplyOp>();
    target.addLegalOp<IE::NegativeOp>();
    target.addLegalOp<IE::TransposeOp>();
    target.addDynamicallyLegalOp<IE::ReduceMeanOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMaxOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceSumOp>(isLegalOp);
    target.addDynamicallyLegalOp<IE::ReduceMinOp>(isLegalReduceMin);

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ReduceMeanRewriter>(&ctx, _log);
    patterns.add<ReduceMaxRewriter>(&ctx, _log);
    patterns.add<ReduceSumRewriter>(&ctx, _log);
    patterns.add<ReduceMinRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertReduceToPoolingPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertReduceToPoolingPass(Logger log) {
    return std::make_unique<ConvertReduceToPoolingPass>(log);
}
