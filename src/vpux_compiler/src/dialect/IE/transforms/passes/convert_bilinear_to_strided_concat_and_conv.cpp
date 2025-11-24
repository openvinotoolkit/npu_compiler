//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTBILINEARTOSTRIDEDCONCATANDCONV
#define GEN_PASS_DEF_CONVERTBILINEARTOSTRIDEDCONCATANDCONV
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t SMALL_CHANNEL_SUPPORT_SCALER = 4;

mlir::Value createAverageConv(mlir::Value input, ShapeRef kernelShape, mlir::Location loc, int32_t convStrideH,
                              ShapeRef outputShape, mlir::PatternRewriter& rewriter, Logger log) {
    log.nest().trace("Create conv {0}: kernel {1}", loc, kernelShape);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{convStrideH, 1});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});

    const auto OC = inShape[Dims4D::Act::C];
    const auto IC = inShape[Dims4D::Act::C];
    const auto KY = kernelShape[Dim(0)];
    const auto KX = kernelShape[Dim(1)];

    const Shape weightShape = {OC, IC, KY, KX};
    SmallVector<float> weights(weightShape.totalSize(), .0f);
    const float weightRealVal = 1.0f / static_cast<float>(kernelShape[Dim(0)] * kernelShape[Dim(1)]);

    for (auto i = 0; i < OC; ++i) {
        auto beginIndex = i * KY * KX + i * IC * KY * KX;
        auto endIndex = beginIndex + KY * KX;
        for (auto j = beginIndex; j < endIndex; ++j) {
            weights[j] = weightRealVal;
        }
    }

    const DimsOrder weightOrder = DimsOrder::OYXI;

    VPUX_THROW_UNLESS(!mlir::isa<Core::BoundedTensorType>(input.getType()), "{0} doesn't support dynamic shapes",
                      IE::ConvolutionOp::getOperationName());
    const auto weightType = mlir::RankedTensorType::get(weightShape.raw(),
                                                        mlir::cast<NDTypeInterface>(input.getType()).getElementType(),
                                                        getTensorAttr(rewriter.getContext(), weightOrder, nullptr));
    auto weight = Const::buildWeightsConst(rewriter, input.getLoc(), weightType, ArrayRef(weights));

    auto newLoc = appendLoc(loc, "interpolate_conv");

    const auto convInType = mlir::cast<vpux::NDTypeInterface>(input.getType());
    const auto convOutType = convInType.changeShape(outputShape);

    auto averageConv = rewriter.create<IE::ConvolutionOp>(
            newLoc, convOutType, input, weight, /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr,
            /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr,
            /*staticScaleAttr=*/nullptr, /*outputPaddingAttr=*/nullptr, /*inputPaddingAttr=*/nullptr);

    return averageConv.getOutput();
}

auto createAverageDWConv(mlir::Value input, ShapeRef kernelShape, mlir::Location loc, IE::FakeQuantizeOp inputFQ,
                         int32_t convStrideH, int32_t convStrideW, mlir::PatternRewriter& rewriter, Logger log) {
    log.nest().trace("Create dw conv {0}: kernel {1}", loc, kernelShape);
    auto inShape = getShape(input);
    auto dilationsAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{1, 1});
    auto stridesAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{convStrideH, convStrideW});
    auto padBeginAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto padEndAttr = getIntArrayAttr(rewriter, SmallVector<int32_t>{0, 0});
    auto groupAttr = getIntAttr(rewriter, inShape[Dims4D::Act::C]);

    const auto elemType = mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType();

    // OC is equal with IC
    const auto weightShape = Shape{inShape[Dims4D::Act::C], 1, kernelShape[Dim(0)], kernelShape[Dim(1)]};
    const float weightsScaleFactor = 1.0f / static_cast<float>(kernelShape[Dim(0)] * kernelShape[Dim(1)]);
    const float weightRealVal = (inputFQ != nullptr) ? 1.0f : weightsScaleFactor;
    const auto dataStorageType = mlir::RankedTensorType::get(to_small_vector(weightShape), elemType);
    auto weights = Const::createFloatConst(rewriter, loc, dataStorageType, weightRealVal);

    // Add fakeQuan after kernel if needed
    if (inputFQ != nullptr) {
        weights = vpux::IE::createFQScaling(loc, weights, weightsScaleFactor, elemType, inputFQ.getLevels(),
                                            inputFQ.getLowFpType(), inputFQ.getAutoBroadcastAttr(), rewriter);
    }

    auto newLoc = appendLoc(loc, "_interpolate_GroupConv_{0}_{1}", kernelShape[Dim(0)], kernelShape[Dim(1)]);
    auto averageDWConv = rewriter.create<IE::GroupConvolutionOp>(
            newLoc, input, weights, /*bias=*/nullptr, stridesAttr, padBeginAttr, padEndAttr, dilationsAttr, groupAttr,
            /*post_opAttr=*/nullptr, /*clampAttr=*/nullptr, /*outputPadding=*/nullptr, /*inputPadding=*/nullptr);

    return averageDWConv.getOutput();
}

mlir::Value createMaxPool(mlir::Value input, mlir::Location loc, mlir::PatternRewriter& rewriter) {
    const SmallVector<int64_t> maxPoolStrides = {1, 1};
    const SmallVector<int64_t> maxPoolKernels = {1, 1};
    const SmallVector<int64_t> pads = {0, 0};
    const auto padsAttr = getIntArrayAttr(rewriter, pads);
    auto ctx = rewriter.getContext();

    return rewriter
            .create<IE::MaxPoolOp>(loc, input, getIntArrayAttr(rewriter, maxPoolKernels),
                                   getIntArrayAttr(rewriter, maxPoolStrides), padsAttr, padsAttr,
                                   vpux::IE::RoundingTypeAttr::get(ctx, vpux::IE::RoundingType::FLOOR), nullptr,
                                   nullptr, nullptr, nullptr)
            .getOutput();
}

//
// ConvertBilinearToStridedConcatAndConvPass
//

class ConvertBilinearToStridedConcatAndConvPass final :
        public IE::impl::ConvertBilinearToStridedConcatAndConvBase<ConvertBilinearToStridedConcatAndConvPass> {
public:
    explicit ConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp, Logger log)
            : _interpolateAsSEOp(interpolateAsSEOp) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

public:
    class BilinearInterpolateOpConverter;
    class BilinearInterpolateOpConverterV2;
    class SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter;

private:
    void safeRunOnFunc() final;

private:
    bool _interpolateAsSEOp;
};

mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // When this parameter has a value, it probably comes from LIT test.
    // Override the default
    if (interpolateAsSEOp.hasValue()) {
        _interpolateAsSEOp = interpolateAsSEOp.getValue();
    }

    return mlir::success();
}

// BilinearInterpolateOpConverter
class ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// BilinearInterpolateOpConverterV2
class ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverterV2 final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    BilinearInterpolateOpConverterV2(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx, benefit), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/// @brief Replace NxN bilinear interpolate as a cascaded structure of one NxN nearest interpolate with padding and
/// one depthwise convolution with NXN kernel.
/// @details Nearest part is implemented by strided concats over width and height,
/// padding part is implemented by slice-concat. See ticket: E43217.
/// @reminder Current solution can be optimized further by SEP feature in future
mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get bilinear Interpolate Op {0}", origOp);
    const auto inputShape = getShape(origOp.getInput());
    const auto attrs = origOp.getAttr();
    const auto outShape = getShape(origOp.getOutput());
    auto coordMode = attrs.getCoordMode().getValue();

    bool isAlignCorners = 0;
    int32_t scaleW;
    int32_t scaleH;

    bool isBothUpscale = (outShape[Dims4D::Act::W] > inputShape[Dims4D::Act::W]) &&
                         (outShape[Dims4D::Act::H] > inputShape[Dims4D::Act::H]);
    bool isBothDownscale = (outShape[Dims4D::Act::W] < inputShape[Dims4D::Act::W]) &&
                           (outShape[Dims4D::Act::H] < inputShape[Dims4D::Act::H]);

    if (isBothUpscale) {
        if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
            isAlignCorners = 1;
            scaleW = (outShape[Dims4D::Act::W] - 1) / (inputShape[Dims4D::Act::W] - 1);
            scaleH = (outShape[Dims4D::Act::H] - 1) / (inputShape[Dims4D::Act::H] - 1);
        } else {
            scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
            scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
        }

    } else if (isBothDownscale) {
        // E#95440: Support for downscaling cases will be provided once the permuteQuantize issue has been resolved.
        scaleW = inputShape[Dims4D::Act::W] / outShape[Dims4D::Act::W];
        scaleH = inputShape[Dims4D::Act::H] / outShape[Dims4D::Act::H];
    } else {
        VPUX_THROW("Interpolate H and W must be both upscale or downscale");
    }

    int32_t strideParamsW = 1, forwardPadParamsW = 1, kernelParamsW = 1, strideParamsH = 1, forwardPadParamsH = 1,
            kernelParamsH = 1;
    int32_t forwardPadW = 1, backPadW = 1, forwardPadH = 1, backPadH = 1;
    int32_t convKernelShapeW = 1, convKernelShapeH = 1;
    int32_t convStrideW = 1, convStrideH = 1;
    int32_t upSampleW = 1, upSampleH = 1;

    if ((coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) ||
        (coordMode == IE::InterpolateCoordMode::HALF_PIXEL)) {
        if (scaleW % 2 == 0) {
            strideParamsW = 2;
            forwardPadParamsW = 1 - scaleW;
            kernelParamsW = 2 * scaleW;
        } else {
            strideParamsW = 1;
            forwardPadParamsW = (1 - scaleW) / 2;
            kernelParamsW = scaleW;
        };

        if (scaleH % 2 == 0) {
            strideParamsH = 2;
            forwardPadParamsH = 1 - scaleH;
            kernelParamsH = 2 * scaleH;
        } else {
            strideParamsH = 1;
            forwardPadParamsH = (1 - scaleH) / 2;
            kernelParamsH = scaleH;
        }
    }

    if ((coordMode == IE::InterpolateCoordMode::ASYMMETRIC) || (isAlignCorners)) {
        strideParamsW = 1;
        forwardPadParamsW = 0;
        kernelParamsW = scaleW;

        strideParamsH = 1;
        forwardPadParamsH = 0;
        kernelParamsH = scaleH;
    }

    forwardPadW = -forwardPadParamsW;
    backPadW = forwardPadParamsW + kernelParamsW - strideParamsW;
    forwardPadH = -forwardPadParamsH;
    backPadH = forwardPadParamsH + kernelParamsH - strideParamsH;
    upSampleW = kernelParamsW;
    upSampleH = kernelParamsH;

    convKernelShapeH = kernelParamsH;
    convKernelShapeW = kernelParamsW;
    convStrideH = strideParamsH;
    convStrideW = strideParamsW;

    const auto inputFQ = origOp.getInput().getDefiningOp<IE::FakeQuantizeOp>();

    SmallVector<mlir::Value> widthSlices(upSampleW, origOp.getInput());
    auto newOp = widthSlices.size() != 1 ? rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "wslices_concat"),
                                                                         widthSlices, Dims4D::Act::W, 1, upSampleW)
                                                   .getOutput()
                                         : widthSlices.front();

    SmallVector<mlir::Value> heightSlices(upSampleH, newOp);
    auto nearestInterpolateOut = heightSlices.size() != 1
                                         ? rewriter.create<IE::ConcatOp>(takeOpLoc(origOp, "hslices_concat"),
                                                                         heightSlices, Dims4D::Act::H, 1, upSampleH)
                                                   .getOutput()
                                         : heightSlices.front();

    auto tensorPadded = nearestInterpolateOut;
    if (!isAlignCorners) {
        auto tensorPaddedWidth = (scaleW - 1 > 0) ? IE::createPadding(rewriter, origOp, nearestInterpolateOut,
                                                                      Dims4D::Act::W, forwardPadW, backPadW, "pad_w")
                                                  : nearestInterpolateOut;
        tensorPadded = (scaleH - 1 > 0) ? IE::createPadding(rewriter, origOp, tensorPaddedWidth, Dims4D::Act::H,
                                                            forwardPadH, backPadH, "pad_wh")
                                        : tensorPaddedWidth;
    }

    // Create depthwise convolution
    auto dwConv = createAverageDWConv(tensorPadded, {convKernelShapeH, convKernelShapeW}, origOp.getLoc(), inputFQ,
                                      convStrideH, convStrideW, rewriter, _log);
    rewriter.replaceOp(origOp, dwConv);

    return mlir::success();
}

/// @brief Replace 2x2 bilinear interpolate as three depthwise convolutions and one maxpool or six depthwise
/// convolutions if use cmx-concat, and a cascaded structure of strided concat over W and H
/// @details It is a faster solutoin than the original one above. See ticket: E#49791
mlir::LogicalResult ConvertBilinearToStridedConcatAndConvPass::BilinearInterpolateOpConverterV2::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("DPU Interp solution V2: optimize bilinear Interpolate Op {0}", origOp);
    if (!origOp->hasOneUse()) {
        return mlir::failure();
    }

    auto input = origOp.getInput();
    const auto inputShape = getShape(input);
    const auto outShape = getShape(origOp.getOutput());

    const auto scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
    const auto scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
    const auto attrs = origOp.getAttr();
    // This is a fast dpu solution only for 2x2 upsampling
    if (scaleW != vpux::IE::CONVERT_BILINEAR_TO_STRIDED_CONCAT_CONVOLUTION_V2_SUPPORTED_SCALE ||
        scaleH != vpux::IE::CONVERT_BILINEAR_TO_STRIDED_CONCAT_CONVOLUTION_V2_SUPPORTED_SCALE ||
        (attrs.getCoordMode().getValue() != IE::InterpolateCoordMode::ASYMMETRIC)) {
        return mlir::failure();
    }

    // This solution introduces many dw convs (four) so that channel alignment will lead
    // extra computation overhead and strided dmas by slice ops than original solution
    const auto elemType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getElementType();
    const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
    if (inputShape[Dims4D::Act::C] % alignment != 0) {
        return mlir::failure();
    }

    const auto inputFQ = input.getDefiningOp<IE::FakeQuantizeOp>();
    const auto outputFQ = mlir::dyn_cast<IE::FakeQuantizeOp>(*(origOp.getOutput().user_begin()));
    const mlir::Location location = origOp.getLoc();
    int32_t forwardPad = 0;

    // Right slice and padding
    auto concatW = IE::createPadding(rewriter, origOp, input, Dims4D::Act::W, forwardPad, scaleW - 1, "concat_width");

    // Bottom slice and padding
    auto concatWH = IE::createPadding(rewriter, origOp, concatW, Dims4D::Act::H, forwardPad, scaleH - 1, "concat_wh");

    // Left slice
    auto concatWHShape = getShape(concatWH);
    auto offsets = SmallVector<int64_t>(concatWHShape.size(), 0);
    auto sizes = SmallVector<int64_t>(concatWHShape.begin(), concatWHShape.end());
    if (sizes[Dims4D::Act::W.ind()] <= 1) {
        return mlir::failure();
    }
    sizes[Dims4D::Act::W.ind()] -= 1;
    mlir::Value leftSlice = rewriter.create<IE::SliceOp>(appendLoc(location, "left_slice"), concatWH,
                                                         getIntArrayAttr(origOp.getContext(), offsets),
                                                         getIntArrayAttr(origOp.getContext(), sizes));
    int32_t convStrideH = 1, convStrideW = 1;
    auto getAverageDWConv = [&](auto input, ShapeRef kernelShape, StringRef locSuffix) {
        auto DWConv = createAverageDWConv(input, kernelShape, appendLoc(location, locSuffix), inputFQ, convStrideH,
                                          convStrideW, rewriter, _log);
        if (outputFQ != nullptr) {
            DWConv = vpux::IE::createFQ(rewriter, DWConv, outputFQ,
                                        takeOpLoc(outputFQ, StringLiteral("fq_{0}_out"), locSuffix));
        }
        return DWConv;
    };

    auto type = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto module = origOp->getParentOfType<mlir::ModuleOp>();

    // The interpolate was divided into 4 conv. for the first concat, output is 1/2 of original output
    auto canUseCmxConcat = [&]() -> bool {
        // if it can be convert to CMX concat, the performance can be improved, but if not,
        // the extra convolution will decrease the performance, so here add the size check with output. input,
        // weights, and weights table to make it more accurate
        const auto arch = config::getArch(origOp);
        const auto weightTable = vpux::VPU::NCEInvariant::getWeightsTableSize(inputShape[Dims4D::Act::C]);
        int64_t outSize = type.getTotalAllocSize().count() / scaleH;
        int64_t inSize = outSize / scaleW;
        int64_t weightsSize = inputShape[Dims4D::Act::C] * Byte(type.getElemTypeSize()).count();

        int64_t typeSizeFactor = 2;
        if (inputFQ != nullptr) {
            inSize = inSize / typeSizeFactor;
            weightsSize = weightsSize / typeSizeFactor;
        }
        if (outputFQ != nullptr) {
            outSize = outSize / typeSizeFactor;
        }

        SmallVector<Byte> buffers = {Byte(inSize), Byte(outSize), Byte(weightsSize), weightTable};
        return vpux::VPU::calculateAlignedBuffersMemoryRequirement(arch, buffers) <= vpux::VPU::getTotalCMXSize(module);
    };

    // MaxPool has a faster execution speed than DWConv, so if cmxconcat cannot be used,
    // MaxPool is used. When cmxconcat is available, cmxconcat requires that the DistributionAttrs
    // of the dpu tasks before and after the concat are compatible. So we insert DWConv here so that
    // all dpu tasks before and after concat are of the same type and size,which can ensure that they
    // are assigned the same attributes and that concat can be converted to cmxconcat.

    auto originCopy = canUseCmxConcat() ? getAverageDWConv(input, {1, 1}, "center")
                                        : createMaxPool(input, appendLoc(location, "as_maxpool"), rewriter);

    // Average over concatW, kernel_h = 1, kernel_w = 2
    auto averagePoolOverW = getAverageDWConv(concatW, {1, 2}, "over_w");

    // Average over leftSlice, kernel_h = 2, kernel_w = 1
    auto averagePoolOverH = getAverageDWConv(leftSlice, {2, 1}, "over_h");

    // Average over concatWH, kernel_h = 2, kernel_w = 2
    auto averagePoolOverWH = getAverageDWConv(concatWH, {2, 2}, "over_wh");

    // Join over W
    SmallVector<mlir::Value> widthSlices{originCopy, averagePoolOverW};
    auto joinOverW0 = rewriter.create<IE::ConcatOp>(appendLoc(location, "concat_w0"), widthSlices, Dims4D::Act::W, 1, 2)
                              .getOutput();

    widthSlices.clear();
    widthSlices = {averagePoolOverH, averagePoolOverWH};
    auto joinOverW1 = rewriter.create<IE::ConcatOp>(appendLoc(location, "concat_w1"), widthSlices, Dims4D::Act::W, 1, 2)
                              .getOutput();

    if (canUseCmxConcat()) {
        joinOverW0 = getAverageDWConv(joinOverW0, {1, 1}, "over_w0");
        joinOverW1 = getAverageDWConv(joinOverW1, {1, 1}, "over_w1");
        if (outputFQ != nullptr) {
            joinOverW1 = vpux::IE::createFQ(rewriter, joinOverW1, outputFQ, takeOpLoc(outputFQ, "fq_w1")).getOutput();
            joinOverW0 = vpux::IE::createFQ(rewriter, joinOverW0, outputFQ, takeOpLoc(outputFQ, "fq_w0")).getOutput();
        }
    }

    // Join over H
    SmallVector<mlir::Value> heightSlices{joinOverW0, joinOverW1};
    auto joinOverH = rewriter.create<IE::ConcatOp>(appendLoc(location, "concat_h"), heightSlices, Dims4D::Act::H, 1, 2)
                             .getOutput();

    rewriter.replaceOp(origOp, joinOverH);

    return mlir::success();
}

// SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter
class ConvertBilinearToStridedConcatAndConvPass::SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter final :
        public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

/* An interpolate will be convert to:

    Reorder(NCHW -> NHWC)
        |
    Interpolate on H
        |
    Mempermute (exchange H & W)
        |
    Interpolate on H
        |
    Mempermute (exchange H & W)
        |
    Reorder(NHWC -> NCHW)

    The Interpolate on H will be convert to:

             Input
             ||||
            Concat
            /  |  \
        Slice  |  Slice
          \    |   /
            Concat
              |
          Convolution

    Eg input is 123
           123
            |
          Concat
            |
        111122223333
            |
          Slice
            |
     1 111122223333 3
            |
          conv - kernel [0.25, 0.25, 0.25, 0.25] stride = 2
            |
  1 1.25 1.75 2.25 2.75 3

    This conversion will bring 3 benefits:

    1. All convolution can avoid expand channel through shape-cast in AdjustConvolutionShapePass.
        For conv build in BilinearInterpolateOpConverter, it can't keep both input and output shapeW * shapeC align
        to 16. This convert can ensure that the shapeW remains unchanged before and after conv, so that we can avoid
        expand through shape-cast in AdjustConvolutionShapePass.
    2. All slice and concat were on H dim(NHWC layout), which can avoid stride DMA.
        In pass BilinearInterpolateOpConverter, it will perform concat and slice simultaneously in H and W, which
        will definitely introduce stride DMA.
    3. Split the interpolate will reduce the kernel size of conv.
        In pass BilinearInterpolateOpConverter, a 2x scale PYTORCH_HALF_PIXEL interpolate will build a 4x4
        kernel conv, after split we will build 2 4x1 kernel conv.

*/

mlir::LogicalResult
ConvertBilinearToStridedConcatAndConvPass::SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter::matchAndRewrite(
        IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("Get interp {0} on {1}", origOp->getName(), origOp->getLoc());

    // The conversion logic is the same as pass BilinearInterpolateOpConverter, which is a concretization of pass
    // BilinearInterpolateOpConverter's 2x scaling of a single dimension for PYTORCH_HALF_PIXEL mode. There are some
    // different: 1.The concat and slice is on H dim for NHWC layout. 2.Use convolution to simulate DW-convolution
    // 3.Only handle the H dim interpolate
    const auto inputShape = getShape(origOp.getInput());
    const auto outputShape = getShape(origOp.getOutput());
    const auto scaleH = outputShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
    int32_t strideParamsH = 1, forwardPadParamsH = 1, kernelParamsH = 1;
    int32_t forwardPadH = 1, backPadH = 1;
    int32_t convKernelShapeH = 1;
    int32_t convStrideH = 1;
    int32_t upSampleH = 1;

    if (scaleH % 2 == 0) {
        strideParamsH = 2;
        forwardPadParamsH = 1 - scaleH;
        kernelParamsH = 2 * scaleH;
    } else {
        strideParamsH = 1;
        forwardPadParamsH = (1 - scaleH) / 2;
        kernelParamsH = scaleH;
    }

    forwardPadH = -forwardPadParamsH;
    backPadH = forwardPadParamsH + kernelParamsH - strideParamsH;
    upSampleH = kernelParamsH;

    convKernelShapeH = kernelParamsH;
    convStrideH = strideParamsH;

    auto createSliceConcatConv = [&](mlir::Value intput, StringRef locSuffix) {
        SmallVector<mlir::Value> heightSlices(upSampleH, intput);

        auto nearestInterpolateOut =
                heightSlices.size() != 1
                        ? rewriter.create<IE::ConcatOp>(
                                          takeOpLoc(origOp, StringLiteral("{0}_slices_concat"), locSuffix),
                                          heightSlices, Dims4D::Act::H, 1, upSampleH)
                                  .getOutput()
                        : heightSlices.front();

        auto tensorPadded = IE::createPadding(rewriter, origOp, nearestInterpolateOut, Dims4D::Act::H, forwardPadH,
                                              backPadH, "pad_h_" + locSuffix.str());

        const auto inputShape = getShape(intput);
        const Shape outputShapeOnH = {inputShape[vpux::Dims4D::Act::N], inputShape[vpux::Dims4D::Act::C],
                                      inputShape[vpux::Dims4D::Act::H] * scaleH, inputShape[vpux::Dims4D::Act::W]};

        return createAverageConv(tensorPadded, {convKernelShapeH, 1},
                                 takeOpLoc(origOp, StringLiteral("{0}_avgconv"), locSuffix), convStrideH,
                                 outputShapeOnH, rewriter, _log);
    };

    auto dstOrder = DimsOrder::NHWC.toAffineMap(getContext());
    auto memPermAttr = getPermutationFromOrders(DimsOrder::NHWC, DimsOrder::NWHC, origOp->getContext());

    auto inputReorder =
            rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder_in_to_NWHC"), origOp.getInput(), dstOrder);

    auto interpolateH1 = createSliceConcatConv(inputReorder.getOutput(), "H1");
    auto memPermute1 =
            rewriter.create<IE::MemPermuteOp>(takeOpLoc(origOp, "mem_perm_H1"), interpolateH1, dstOrder, memPermAttr);
    auto interpolateH2 = createSliceConcatConv(memPermute1.getOutput(), "H2");
    auto memPermute2 =
            rewriter.create<IE::MemPermuteOp>(takeOpLoc(origOp, "mem_perm_H2"), interpolateH2, dstOrder, memPermAttr);

    auto outputReorder =
            rewriter.create<IE::ReorderOp>(takeOpLoc(origOp, "reorder_out_to_NCHW"), memPermute2.getOutput(),
                                           DimsOrder::NCHW.toAffineMap(getContext()));

    rewriter.replaceOp(origOp, outputReorder.getOutput());
    _log.trace("Split {0} successful", origOp->getName());
    return mlir::success();
}

//
// safeRunOnFunc
//

void ConvertBilinearToStridedConcatAndConvPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        if (_interpolateAsSEOp) {
            if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                                   /*checkBatch=*/false)) {
                return true;
            }
        }

        const auto attrs = op.getAttr();
        const auto interpMode = attrs.getMode().getValue();
        const auto antiAlias = attrs.getAntialias().getValue();
        const auto coordMode = attrs.getCoordMode().getValue();
        const auto inputShape = getShape(op.getInput());
        const auto outShape = getShape(op.getOutput());
        int64_t scaleW = 1;
        int64_t scaleH = 1;

        if ((interpMode != IE::InterpolateMode::LINEAR_ONNX && interpMode != IE::InterpolateMode::LINEAR) ||
            antiAlias) {
            return true;
        }

        // Only support 4D Input shape
        if (inputShape.size() != 4) {
            return true;
        }

        if ((inputShape[Dims4D::Act::N] != outShape[Dims4D::Act::N]) ||
            (inputShape[Dims4D::Act::C] != outShape[Dims4D::Act::C])) {
            return true;
        }

        // Small-channel models WA: when the channel size is smaller than the channel alignment
        // The alignment causes worse performance than SHAVE interpolation
        // For channel size is smaller than channel alignement, it's partially resolved by
        // SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter.
        const auto elemType = mlir::cast<vpux::NDTypeInterface>(op.getOutput().getType()).getElementType();
        const auto alignment = VPU::NCEInvariant::getAlignment(elemType);
        if (inputShape[Dims4D::Act::C] < alignment) {
            return true;
        }

        // Runtime already has a efficient implementation for this case
        // And also current solution for this case will produce lots of DMAs, which is not efficient
        if (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1) {
            return true;
        }

        bool isBothUpscale = (outShape[Dims4D::Act::W] > inputShape[Dims4D::Act::W]) &&
                             (outShape[Dims4D::Act::H] > inputShape[Dims4D::Act::H]);

        if (isBothUpscale) {
            // E46240: only this kind of align_corners is accurate and is supported
            if (coordMode == IE::InterpolateCoordMode::ALIGN_CORNERS) {
                if ((outShape[Dims4D::Act::W] - 1) % (inputShape[Dims4D::Act::W] - 1) == 0 &&
                    (outShape[Dims4D::Act::H] - 1) % (inputShape[Dims4D::Act::H] - 1) == 0) {
                    scaleW = (outShape[Dims4D::Act::W] - 1) / (inputShape[Dims4D::Act::W] - 1);
                    scaleH = (outShape[Dims4D::Act::H] - 1) / (inputShape[Dims4D::Act::H] - 1);
                    return (scaleW > 4 && scaleH > 4);
                } else {
                    return true;
                }
            }

            // E#95440: Support for TF_HALF_PIXEL_FOR_NN upsampling will be provided once the permuteQuantize issue
            // has been resolved.
            else if (coordMode == IE::InterpolateCoordMode::TF_HALF_PIXEL_FOR_NN) {
                return true;
            } else if ((coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) &&
                       (outShape[Dims4D::Act::W] == 1 || outShape[Dims4D::Act::H] == 1)) {
                return true;
            } else if (outShape[Dims4D::Act::W] % inputShape[Dims4D::Act::W] == 0 &&
                       outShape[Dims4D::Act::H] % inputShape[Dims4D::Act::H] == 0) {
                // Support N times upsampling of asymmetric, half_pixel, pytorch_half_pixel
                // modes
                scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
                scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
                if (coordMode == IE::InterpolateCoordMode::ASYMMETRIC) {
                    // Deeplab-v3 WA: SHAVE implementation may be better for some big bilinear interpolates
                    // Current solution will produce some extra DMAs as we need do padding by slice-concat, which may
                    // cause some performance loss especially for big interpolates. In future, SEP may help to solve
                    // this issue. Details see ticket: E43217 The scaleW 4 and scaleH 4 is more efficient on VPUX37XX.
                    // Details see ticket: E56905
                    return (scaleW > 4 && scaleH > 4);
                } else if ((coordMode == IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL) ||
                           (coordMode == IE::InterpolateCoordMode::HALF_PIXEL)) {
                    scaleW = outShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
                    scaleH = outShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
                    // Similar to the asymmetric mode, ensure that the convolutional kernel size does not exceed 4.
                    if ((scaleW % 2 == 0) && (scaleH % 2 == 0)) {
                        return ((scaleW > 2) || (scaleH > 2));
                    } else if ((scaleW % 2 != 0) && (scaleH % 2 != 0)) {
                        return ((scaleW > 3) || (scaleH > 3));
                    } else {
                        return true;
                    };
                };
            } else {
                return true;
            }
        }
        // E#95440: Support for downscaling cases will be provided once the permuteQuantize issue has been
        // resolved.
        return true;
    });

    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::MaxPoolOp>();
    target.addLegalOp<IE::FakeQuantizeOp>();
    target.addLegalOp<Const::DeclareOp>();
    target.addLegalOp<IE::GroupConvolutionOp>();

    mlir::RewritePatternSet patterns(&ctx);
    /// @warning The insert order for patterns can't decide the real execuation order
    /// So we use the explicit declaration (PatternBenefit) in their class constrution function
    /// to control them. Here pattern V2 executed first
    patterns.insert<BilinearInterpolateOpConverterV2>(&ctx, vpux::benefitMid, _log);
    patterns.insert<BilinearInterpolateOpConverter>(&ctx, vpux::benefitLow, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }

    mlir::ConversionTarget smallChannelTarget(ctx);
    smallChannelTarget.addDynamicallyLegalOp<IE::InterpolateOp>([&](IE::InterpolateOp op) {
        if (_interpolateAsSEOp) {
            if (VPU::NCEInterpolateOp::isSupported(op, logCb, /*checkLayout=*/false, /*checkChannelAlignment=*/false,
                                                   /*checkBatch=*/false)) {
                return true;
            }
        }

        const auto attrs = op.getAttr();
        const auto interpMode = attrs.getMode().getValue();
        const auto antiAlias = attrs.getAntialias().getValue();
        const auto coordMode = attrs.getCoordMode().getValue();
        const auto inputShape = getShape(op.getInput());
        const auto outputShape = getShape(op.getOutput());
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());

        if ((interpMode != IE::InterpolateMode::LINEAR_ONNX && interpMode != IE::InterpolateMode::LINEAR) ||
            antiAlias) {
            return true;
        }

        // Use ExecutorOpInterface to determine if SHAVE is preferred
        if (auto iface = mlir::dyn_cast<IE::ExecutorOpInterface>(op.getOperation())) {
            auto execs = iface.getPreferredExecutors();
            if (!execs.empty() && execs[0] == VPU::ExecutorKind::SHAVE_ACT) {
                return true;
            }
        }

        const auto inputElemType = inputType.getElementType();
        if (mlir::isa<mlir::quant::QuantizedType>(inputElemType)) {
            // Support of quantized case will be open after E#104698 fix AC issue.
            return true;
        }

        if (inputShape[Dims4D::Act::N] != 1) {
            return true;
        }

        if ((inputShape[Dims4D::Act::N] != outputShape[Dims4D::Act::N]) ||
            (inputShape[Dims4D::Act::C] != outputShape[Dims4D::Act::C])) {
            return true;
        }

        if (auto alignInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op.getOperation())) {
            const auto alignment = alignInterface.getInputChannelAlignment();
            // Ensure that the converted convolution can avoid expand through shapeCast in AdjustConvolutionShapePass
            if ((inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::W]) % alignment != 0) {
                return true;
            }
            // After first permutation outputShapeH will become the real W of second conv, so we need keep it can avoid
            // expand through shapeCast in AdjustConvolutionShapePass.
            if ((inputShape[Dims4D::Act::C] * outputShape[Dims4D::Act::H]) % alignment != 0) {
                return true;
            }
        }

        // For channel >= 8 case, we can use SEP-interp to get better performance.
        if (inputShape[Dims4D::Act::C] >= 8) {
            return true;
        }

        // Runtime already has a efficient implementation for this case
        // And also current solution for this case will produce lots of DMAs, which is not efficient.
        if (inputShape[Dims4D::Act::H] == 1 && inputShape[Dims4D::Act::W] == 1) {
            return true;
        }

        if (coordMode != vpux::IE::InterpolateCoordMode::PYTORCH_HALF_PIXEL &&
            coordMode != vpux::IE::InterpolateCoordMode::HALF_PIXEL) {
            return true;
        }

        // This conversion does not necessarily bring performance improvement for scaling greater than
        // SMALL_CHANNEL_SUPPORT_SCALER times.
        const auto scalerHeight = outputShape[Dims4D::Act::H] / inputShape[Dims4D::Act::H];
        const auto scalerWidth = outputShape[Dims4D::Act::W] / inputShape[Dims4D::Act::W];
        if (scalerHeight != scalerWidth || scalerHeight > SMALL_CHANNEL_SUPPORT_SCALER) {
            return true;
        }
        // This check ensure the scaler is an integer, otherwise it will case accuracy issue.
        if (inputShape[Dims4D::Act::H] * scalerHeight != outputShape[Dims4D::Act::H] ||
            inputShape[Dims4D::Act::W] * scalerWidth != outputShape[Dims4D::Act::W]) {
            return true;
        }

        return false;
    });

    smallChannelTarget.addLegalOp<IE::SliceOp>();
    smallChannelTarget.addLegalOp<IE::ConcatOp>();
    smallChannelTarget.addLegalOp<IE::ReorderOp>();
    smallChannelTarget.addLegalOp<IE::MemPermuteOp>();
    smallChannelTarget.addLegalOp<Const::DeclareOp>();
    smallChannelTarget.addLegalOp<IE::ConvolutionOp>();

    mlir::RewritePatternSet smallChannelPatterns(&ctx);

    // For channel 16 align case, BilinearInterpolateOpConverter and BilinearInterpolateOpConverterV2 will handle. (If
    // SEP-interpolate is enable, prioritize using SEP-interpolate)
    // For channel < 8 case, SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter will get a better performance.
    // For 8 <= channel < 16 case, we perfer using SEP-interpolate, it will get better performance.

    smallChannelPatterns.add<SmallChannelPytorchHalfPixelBilinearInterpolateOpConverter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, smallChannelTarget, std::move(smallChannelPatterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertBilinearToStridedConcatAndConvPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertBilinearToStridedConcatAndConvPass(const bool interpolateAsSEOp,
                                                                                      Logger log) {
    return std::make_unique<ConvertBilinearToStridedConcatAndConvPass>(interpolateAsSEOp, log);
}
