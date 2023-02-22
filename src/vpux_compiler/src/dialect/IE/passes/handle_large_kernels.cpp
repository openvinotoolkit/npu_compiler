//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/VPU/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/IE/loop.hpp"

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace vpux;

namespace {

constexpr int64_t PADDING_RIGHT = 1;
constexpr int64_t PADDING_BOT = 3;

//
// Assign parameter `stride = 1` rather than firstOpKernel or sequencedOpKernel if axis of sequenced pools is global.
//
mlir::ArrayAttr getGlobalOrOrigStride(mlir::MLIRContext* ctx, mlir::Value input, std::array<int64_t, 2> origKernel,
                                      std::array<int64_t, 2> origStride, mlir::ArrayAttr padBeginAttr,
                                      mlir::ArrayAttr padEndAttr) {
    std::array<int64_t, 2> newOpStride = {1, 1};
    auto origInputShape = getShape(input).raw();

    auto padBegin = parseIntArrayAttr<int64_t>(padBeginAttr);
    auto padEnd = parseIntArrayAttr<int64_t>(padEndAttr);

    if (origKernel[Dims4D::Kernel::Y.ind()] !=
        (origInputShape[Dims4D::Act::H.ind()] + padBegin[Dims4D::PadsBegin::Top.ind()] +
         padEnd[Dims4D::PadsEnd::Bottom.ind()]))
        newOpStride[Dims4D::Strides::Y.ind()] = origStride[Dims4D::Strides::Y.ind()];

    if (origKernel[Dims4D::Kernel::X.ind()] !=
        (origInputShape[Dims4D::Act::W.ind()] + padBegin[Dims4D::PadsBegin::Left.ind()] +
         padEnd[Dims4D::PadsEnd::Right.ind()]))
        newOpStride[Dims4D::Strides::X.ind()] = origStride[Dims4D::Strides::X.ind()];

    return getIntArrayAttr(ctx, makeArrayRef(newOpStride));
}

void getFactorsForSecondDimension(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                  std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                  ArrayRef<int64_t> kernelSize) {
    int64_t padValue = 1;
    const auto factorsResult =
            vpux::IE::getFactors(kernelSize[smallDim], padValue);  // toggling between the two kernel sizes

    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factorsSecondDim = factorsResult.getValue();

    log.trace("Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}", smallDim,
              kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);

    VPUX_THROW_UNLESS((factorsSecondDim.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE) &&
                              (factorsSecondDim.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE),
                      "Second dimension factors ({1}, {2})  are larger than MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factorsSecondDim.larger, factorsSecondDim.smaller);
    firstOpKernel[smallDim] = factorsSecondDim.larger;
    sequencedOpKernel[smallDim] = factorsSecondDim.smaller;
    auto multipliedFactors = firstOpKernel[smallDim] * sequencedOpKernel[smallDim];

    padding[PADDING_BOT] = (multipliedFactors > kernelSize[smallDim]) ? 1 : 0;
}

void getFactorsForSecondDimensionWithLimit(std::array<int64_t, 4>& padding, std::array<int64_t, 2>& firstOpKernel,
                                           std::array<int64_t, 2>& sequencedOpKernel, int32_t smallDim, Logger log,
                                           ArrayRef<int64_t> kernelSize) {
    int64_t padValue = 1;
    auto factorsResult = vpux::IE::getFactors(kernelSize[smallDim], padValue);  // toggling between the two kernel sizes
    if (!factorsResult.hasValue()) {
        factorsResult = vpux::IE::getFactorsWithSupportedLarger(kernelSize[smallDim]);
    }

    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factorsSecondDim = factorsResult.getValue();

    if (factorsSecondDim.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE) {
        log.trace("Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}", smallDim,
                  kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);
    } else {
        log.trace(
                "Second Dimension kernel[{0}]= {1}, larger factor: {2} , smaller factor: {3}(Required further split!)",
                smallDim, kernelSize[smallDim], factorsSecondDim.larger, factorsSecondDim.smaller);
    }

    VPUX_THROW_UNLESS(factorsSecondDim.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE,
                      "Second dimension factors ({1}, {2})  are larger than MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factorsSecondDim.larger, factorsSecondDim.smaller);
    firstOpKernel[smallDim] = factorsSecondDim.larger;
    sequencedOpKernel[smallDim] = factorsSecondDim.smaller;
    auto multipliedFactors = firstOpKernel[smallDim] * sequencedOpKernel[smallDim];

    padding[PADDING_BOT] = (multipliedFactors > kernelSize[smallDim]) ? 1 : 0;
}

bool checkKernelSizeSupported(ArrayRef<int64_t> kernelSize) {
    if ((kernelSize[Dims4D::Kernel::Y.ind()] > VPU::NCEInvariant::MAX_KERNEL_SIZE) ||
        (kernelSize[Dims4D::Kernel::X.ind()] > VPU::NCEInvariant::MAX_KERNEL_SIZE)) {
        return false;
    }
    return true;
}

void calculateKernelsAndPadding(ArrayRef<int64_t> kernelSize, std::array<int64_t, 4>& padding,
                                std::array<int64_t, 2>& firstOpKernel, std::array<int64_t, 2>& sequencedOpKernel,
                                bool supportMultipleSplitting, Logger log) {
    const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
    const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

    // figure out the bigger kernel dimension width or height when having an asymmetric kernel
    auto largerKernelSize = KX;
    auto largeDim = Dims4D::Kernel::X.ind();
    auto smallDim = Dims4D::Kernel::Y.ind();
    auto asymmetricCase = (KX != KY);
    auto asymmetricBothKernelsLarge =
            (asymmetricCase && (KX > VPU::NCEInvariant::MAX_KERNEL_SIZE) && (KY > VPU::NCEInvariant::MAX_KERNEL_SIZE));

    // deal with asymmetric kernels when one dim is larger than MAX_KERNEL_SIZE
    if (asymmetricCase && (KX < KY)) {
        largerKernelSize = KY;
        largeDim = Dims4D::Kernel::Y.ind();
        smallDim = Dims4D::Kernel::X.ind();
    }
    int64_t padValue = 1;
    auto factorsResult = vpux::IE::getFactors(largerKernelSize, padValue);
    if (!factorsResult.hasValue() && supportMultipleSplitting) {
        factorsResult = vpux::IE::getFactorsWithSupportedLarger(largerKernelSize);
    }
    VPUX_THROW_UNLESS(
            factorsResult.hasValue(),
            "Failed to get valid factors when splitting kernel! Large padding value would lead to accuracy drop.");

    const auto factors = factorsResult.getValue();

    if (factors.smaller <= VPU::NCEInvariant::MAX_KERNEL_SIZE) {
        log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}", largeDim,
                  largerKernelSize, factors.larger, factors.smaller);
    } else {
        log.trace("Large Dimension kernelSize[{0}] = {1}, larger factor: {2} , smaller factor: {3}"
                  "(Required further split!)",
                  largeDim, largerKernelSize, factors.larger, factors.smaller);
    }
    VPUX_THROW_UNLESS(factors.larger <= VPU::NCEInvariant::MAX_KERNEL_SIZE,
                      "Large dimension factors ({1}, {2})  are larger the MAX_KERNEL_SIZE {0}",
                      VPU::NCEInvariant::MAX_KERNEL_SIZE, factors.larger, factors.smaller);

    // cascading supported ops
    // first op kernel [factors.larger, factorsSecondDim.larger] - firstOpKernel
    // sequenced op kernel [factors.smaller, factorsSecondDim.smaller] - sequencedOpKernel
    // Padding quantity relationship is (input size + pad) / k = output size, padding config is TRUE, FALSE
    firstOpKernel[largeDim] = factors.larger;  // first was the large dimension
    sequencedOpKernel[largeDim] = factors.smaller;
    auto multipliedFactors = firstOpKernel[largeDim] * sequencedOpKernel[largeDim];

    if (asymmetricCase) {
        if (asymmetricBothKernelsLarge) {
            if (factors.smaller > VPU::NCEInvariant::MAX_KERNEL_SIZE) {
                getFactorsForSecondDimensionWithLimit(padding, firstOpKernel, sequencedOpKernel, smallDim, log,
                                                      kernelSize);
            } else {
                getFactorsForSecondDimension(padding, firstOpKernel, sequencedOpKernel, smallDim, log, kernelSize);
            }
        } else {
            firstOpKernel[smallDim] = kernelSize[smallDim];
            sequencedOpKernel[smallDim] =
                    1;  // the smallDim was not factorized, the multiplication kSize*1 covers the second op

            padding[PADDING_BOT] = 0;
        }
        // factors multiplied > kernel, we need padding
        padding[PADDING_RIGHT] = (multipliedFactors > kernelSize[largeDim]) ? padValue : 0;

        if (largeDim != Dims4D::Kernel::X.ind()) {
            // change the padding on the other dimensions as largeDim was not on the width dimension - PADD_RIGHT
            std::swap(padding[PADDING_RIGHT], padding[PADDING_BOT]);
        }
    } else {
        firstOpKernel[smallDim] = factors.larger;  // largeDim has the same kernel size as smallDim
        sequencedOpKernel[smallDim] = factors.smaller;
        padding[PADDING_RIGHT] = padding[PADDING_BOT] = (multipliedFactors > kernelSize[largeDim]) ? padValue : 0;
    }
}

//
// AveragePoolRewriter
//

class AveragePoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AveragePoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AveragePoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

    mlir::FailureOr<mlir::Value> splitAvgOperationSlicing(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const;

private:
    Logger _log;
};

mlir::LogicalResult AveragePoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AveragePool layer at '{1}'", getDebugName(), origOp->getLoc());

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel = {1, 1}, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    auto* ctx = origOp->getContext();

    auto curOpInput = origOp.input();
    auto curKernelSize = kernelSize;
    ::mlir::Value curAvgPoolOutput;

    // Support multiple splitting for larger kernel size (> 11 * 11)
    // For example, kernel = 128, it will return [8, 16] factors in first round splitting
    // The supported factor 8: it will be used for current AvgPool kernel
    // The unsupported factor 16 (>11): it will be splitted to [4, 4] in the next round splitting
    // FP16/FP32 input: multiple splitting will introduce accuracy loss as Min/Max changed for FP16-INT8 model
    bool supportMultipleSplitting = false;
    const auto inDataType = origOp.input().getType().cast<vpux::NDTypeInterface>();
    const auto inputElemType = inDataType.getElementType();
    if (inputElemType.isF16() || inputElemType.isF32()) {
        supportMultipleSplitting = true;
    }

    while (!checkKernelSizeSupported(curKernelSize)) {
        calculateKernelsAndPadding(curKernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel,
                                   supportMultipleSplitting, _log.nest(2));

        if (!checkKernelSizeSupported(sequencedOpKernel)) {
            const auto curFirstOpPadBegin =
                    getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
            const auto curFirstOpPadEnd =
                    getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
            const auto curFirstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
            const auto curFirstOpStrideAttr = getGlobalOrOrigStride(ctx, curOpInput, firstOpKernel, firstOpKernel,
                                                                    curFirstOpPadBegin, curFirstOpPadEnd);

            auto curFirstOp = rewriter.create<IE::AvgPoolOp>(
                    origOp->getLoc(), curOpInput, curFirstOpKernelAttr, curFirstOpStrideAttr, curFirstOpPadBegin,
                    curFirstOpPadEnd, origOp.rounding_typeAttr(), origOp.exclude_padsAttr(), origOp.post_opAttr());

            curAvgPoolOutput = curFirstOp.output();
            curOpInput = curAvgPoolOutput;

            // VPUX3720 support different XY strides
            const auto arch = VPU::getArch(origOp);
            if (arch != VPU::ArchKind::VPUX37XX) {
                auto checkStrideRelation = [](const int64_t strideLeft, const int64_t strideRight) -> bool {
                    return strideLeft > strideRight && strideLeft % strideRight == 0;
                };

                bool useSplitAvgOperationSlicing = checkStrideRelation(firstOpKernel[Dims4D::Strides::X.ind()],
                                                                       firstOpKernel[Dims4D::Strides::Y.ind()]) ||
                                                   checkStrideRelation(firstOpKernel[Dims4D::Strides::Y.ind()],
                                                                       firstOpKernel[Dims4D::Strides::X.ind()]);
                if (useSplitAvgOperationSlicing) {
                    const auto concatOp = splitAvgOperationSlicing(curFirstOp, rewriter);
                    if (mlir::failed(concatOp)) {
                        return mlir::failure();
                    }
                    curOpInput = concatOp.getValue();
                }
            }
        } else {
            const auto KY = curKernelSize[Dims4D::Kernel::Y.ind()];
            const auto KX = curKernelSize[Dims4D::Kernel::X.ind()];
            const auto inShape = getShape(curOpInput);
            if ((KY == KX) && (inShape[Dims4D::Act::H] == KY && inShape[Dims4D::Act::W] == KX) &&
                (firstOpKernel[Dims4D::Kernel::Y.ind()] > VPU::NCEInvariant::MAX_STRIDE)) {
                // The first kernel has stride size same as kernel size. For a global AveragePool with symmetric kernel,
                // if the first kernel is larger than MAX_STRIDE, it can't be converted to NCE task. However, we can
                // reverse kernel order to avoid large stride on the first kernel. For example, when we split a large
                // kernel of 65x65, the first kernel would be 11x11 with stride 11 and the second kernel would be 6x6
                // with stride 1. The first kernel has a large stride so it can't be converted to NCE task. However, if
                // we reverse kernel order, the first kernel would be 6x6 with stride 6 and the second kernel would be
                // 11x11 with stride 1. Both of them can be converted to NCE task.
                auto tmp = firstOpKernel;
                firstOpKernel = sequencedOpKernel;
                sequencedOpKernel = tmp;
            }

            const auto firstOpPadBegin =
                    getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
            const auto firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
            const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
            const auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));
            const auto firstOpStrideAttr = getGlobalOrOrigStride(ctx, curOpInput, firstOpKernel, firstOpKernel,
                                                                 firstOpPadBegin, firstOpPadEnd);

            auto firstOp = rewriter.create<IE::AvgPoolOp>(
                    origOp->getLoc(), curOpInput, firstOpKernelAttr, firstOpStrideAttr, firstOpPadBegin, firstOpPadEnd,
                    origOp.rounding_typeAttr(), origOp.exclude_padsAttr(), origOp.post_opAttr());

            const auto firstOpOutputShapeType = firstOp.output().getType().cast<vpux::NDTypeInterface>();
            const auto firstOpOutputShape = firstOpOutputShapeType.getShape().raw();
            auto firstAvgPoolOutput = firstOp.output();

            // VPUX3720 support different XY strides
            const auto arch = VPU::getArch(origOp);
            if (arch != VPU::ArchKind::VPUX37XX) {
                auto checkStrideRelation = [](const int64_t strideLeft, const int64_t strideRight) -> bool {
                    return strideLeft > strideRight && strideLeft % strideRight == 0;
                };

                bool useSplitAvgOperationSlicing = checkStrideRelation(firstOpKernel[Dims4D::Strides::X.ind()],
                                                                       firstOpKernel[Dims4D::Strides::Y.ind()]) ||
                                                   checkStrideRelation(firstOpKernel[Dims4D::Strides::Y.ind()],
                                                                       firstOpKernel[Dims4D::Strides::X.ind()]);
                if (useSplitAvgOperationSlicing) {
                    const auto concatOp = splitAvgOperationSlicing(firstOp, rewriter);
                    if (mlir::failed(concatOp)) {
                        return mlir::failure();
                    }
                    firstAvgPoolOutput = concatOp.getValue();
                }
            }

            auto globalAvgOverH = firstOpOutputShape[Dims4D::Act::H.ind()] == sequencedOpKernel[0];
            auto globalAvgOverW = firstOpOutputShape[Dims4D::Act::W.ind()] == sequencedOpKernel[1];

            std::array<int64_t, 2> sequencedOpStrides = {1, 1};
            if (!globalAvgOverH) {
                sequencedOpStrides[0] = sequencedOpKernel[0];
            }
            if (!globalAvgOverW) {
                sequencedOpStrides[1] = sequencedOpKernel[1];
            }

            calculatedPadding = {0, 0, 0, 0};
            const auto sequencedOpPadBegin =
                    getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
            const auto sequencedOpPadEnd =
                    getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));

            const auto sequencedOpStridesAttr =
                    getGlobalOrOrigStride(ctx, firstAvgPoolOutput, sequencedOpKernel, sequencedOpStrides,
                                          sequencedOpPadBegin, sequencedOpPadEnd);

            rewriter.replaceOpWithNewOp<IE::AvgPoolOp>(
                    origOp, origOp.getType(), firstAvgPoolOutput, sequencedOpKernelAttr, sequencedOpStridesAttr,
                    sequencedOpPadBegin, sequencedOpPadEnd, origOp.rounding_typeAttr(), origOp.exclude_padsAttr(),
                    origOp.post_opAttr());
        }

        curKernelSize[0] = sequencedOpKernel[0];
        curKernelSize[1] = sequencedOpKernel[1];
    }

    return mlir::success();
}

mlir::FailureOr<mlir::Value> AveragePoolRewriter::splitAvgOperationSlicing(IE::AvgPoolOp origOp,
                                                                           mlir::PatternRewriter& rewriter) const {
    auto ctx = origOp.getContext();
    const auto inputShape = getShape(origOp.input());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.strides());
    if (strides[0] <= 0 || strides[1] <= 0) {
        return errorAt(origOp->getLoc(), "Invalid stride value");
    }
    const auto minStride = std::min(strides[0], strides[1]);
    const auto maxStride = std::max(strides[0], strides[1]);
    auto paddingEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());

    // calculate the new stride for avg pooling
    const auto newStrides = getIntArrayAttr(ctx, makeArrayRef({maxStride, maxStride}));

    // try to slice the tensor into maxStride/minStride pieces on the dim with minStride, and don't need slice on the
    // other dim
    int64_t stepsH = (strides[1] + strides[0] - 1) / strides[0];  // the slice number on the height axis
    int64_t stepsW = (strides[0] + strides[1] - 1) / strides[1];  // the slice number on the width axis

    mlir::SmallVector<mlir::Value> wSliced;
    for (auto i : irange(stepsW)) {  // slicing on the horizontal axis
        mlir::SmallVector<mlir::Value> hSliced;
        for (auto j : irange(stepsH)) {  // slicing on the vertical axis
            Shape offsets(inputShape.size());
            SmallVector<int64_t> slicePaddingEnd(2);

            // calculate the offset for the slice
            offsets[Dims4D::Act::H] = j * minStride;
            offsets[Dims4D::Act::W] = i * minStride;
            if (inputShape[Dims4D::Act::H] <= offsets[Dims4D::Act::H] ||
                inputShape[Dims4D::Act::W] <= offsets[Dims4D::Act::W]) {
                continue;
            }

            // calculate the shape of the slice
            SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C],
                                            inputShape[Dims4D::Act::H] - offsets[Dims4D::Act::H],
                                            inputShape[Dims4D::Act::W] - offsets[Dims4D::Act::W]};

            const auto loc = appendLoc(origOp->getLoc(), "slice {0}, {1}", i, j);

            auto slicedInput = rewriter.create<IE::SliceOp>(
                    loc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            // create avg pooling for this slice with new symmetric stride
            auto roundingTypeAttr = IE::RoundingTypeAttr::get(ctx, IE::RoundingType::FLOOR);
            auto newOp = rewriter.create<IE::AvgPoolOp>(loc, slicedInput.result(), origOp.kernel_size(), newStrides,
                                                        origOp.pads_begin(), origOp.pads_end(), roundingTypeAttr,
                                                        origOp.exclude_padsAttr(), origOp.post_opAttr());
            hSliced.push_back(newOp->getResult(0));
        }
        if (!hSliced.empty()) {
            // concatenate the slices if there are more than one slice on vertical axis, and store it in wSliced
            wSliced.push_back(hSliced.size() != 1 ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), hSliced,
                                                                                  Dims4D::Act::H, 1, stepsH)
                                                  : hSliced.front());
        }
    }
    if (wSliced.empty()) {
        return errorAt(origOp->getLoc(), "Empty slice for avgpool");
    }

    // concatenate the slices if there are more than one slice on horizontal axis
    const auto concatOp = wSliced.size() != 1
                                  ? rewriter.create<IE::ConcatOp>(origOp->getLoc(), wSliced, Dims4D::Act::W, 1, stepsW)
                                  : wSliced.front();
    rewriter.replaceOp(origOp, concatOp);
    return concatOp;
}

//
// MaxPoolRewriter
//

class MaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    MaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("MaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    std::array<int64_t, 4> calculatedPadding = {0, 0, 0, 0};
    std::array<int64_t, 2> firstOpKernel, sequencedOpKernel = {1, 1};

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.kernel_size());
    calculateKernelsAndPadding(kernelSize, calculatedPadding, firstOpKernel, sequencedOpKernel, false, _log.nest(2));
    mlir::MLIRContext* ctx = origOp->getContext();

    const auto origStridesAttr = origOp.strides();
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.pads_begin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.pads_end());
    std::array<int64_t, 4> origPadding = {padsBegin[1], padsEnd[1], padsBegin[0], padsEnd[0]};
    std::array<int64_t, 4> inputPadding = calculatedPadding;

    auto stridesAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    const auto origStrides = parseIntArrayAttr<int64_t>(origStridesAttr);

    auto outShape = getShape(origOp.output());

    if ((origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
         origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) &&
        !(outShape[Dims4D::Act::H] == 1 && outShape[Dims4D::Act::W] == 1)) {
        inputPadding = origPadding;
        stridesAttr = origStridesAttr;
    }

    mlir::ArrayAttr firstOpPadBegin, firstOpPadEnd;
    bool unsuportedPad = false;
    bool isSupportedYPadding = (inputPadding[0] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2) &&
                               (inputPadding[1] < firstOpKernel[Dims4D::Kernel::Y.ind()] / 2);
    bool isSupportedXPadding = (inputPadding[2] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2) &&
                               (inputPadding[3] < firstOpKernel[Dims4D::Kernel::X.ind()] / 2);
    bool allPaddingsEqual = std::all_of(inputPadding.cbegin(), inputPadding.cend(), [&inputPadding](int64_t inPad) {
        return inPad == inputPadding[0];
    });

    if (!isSupportedXPadding && !isSupportedYPadding && allPaddingsEqual) {
        unsuportedPad = true;
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2] / 2, inputPadding[0] / 2}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3] / 2, inputPadding[1] / 2}));
    } else {
        firstOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({inputPadding[2], inputPadding[0]}));
        firstOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({inputPadding[3], inputPadding[1]}));
    }
    const auto firstOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(firstOpKernel));
    auto sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    auto strides = parseIntArrayAttr<int64_t>(stridesAttr);
    stridesAttr = getGlobalOrOrigStride(ctx, origOp.input(), firstOpKernel,
                                        {strides[Dims4D::Strides::Y.ind()], strides[Dims4D::Strides::X.ind()]},
                                        firstOpPadBegin, firstOpPadEnd);

    auto firstOp = rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), origOp.input(), firstOpKernelAttr, stridesAttr,
                                                  firstOpPadBegin, firstOpPadEnd, origOp.rounding_type(),
                                                  origOp.post_opAttr());
    stridesAttr = sequencedOpKernelAttr;

    if ((origStrides[Dims4D::Strides::X.ind()] != kernelSize[Dims4D::Kernel::X.ind()] ||
         origStrides[Dims4D::Strides::Y.ind()] != kernelSize[Dims4D::Kernel::Y.ind()]) &&
        !(outShape[Dims4D::Act::H] == 1 && outShape[Dims4D::Act::W] == 1)) {
        // in this case stride shall be taken into account and pyramid cascading does not work
        // use expression orig_kernel = sum (k1, k2, ..., ki)
        sequencedOpKernel[Dims4D::Kernel::X.ind()] =
                kernelSize[Dims4D::Kernel::X.ind()] - firstOpKernel[Dims4D::Kernel::X.ind()] + 1;
        sequencedOpKernel[Dims4D::Kernel::Y.ind()] =
                kernelSize[Dims4D::Kernel::Y.ind()] - firstOpKernel[Dims4D::Kernel::Y.ind()] + 1;
        stridesAttr = origStridesAttr;
        calculatedPadding = {0, 0, 0, 0};
    }
    if (unsuportedPad) {
        calculatedPadding[0] = inputPadding[0] - inputPadding[0] / 2;
        calculatedPadding[1] = inputPadding[1] - inputPadding[1] / 2;
        calculatedPadding[2] = inputPadding[2] - inputPadding[2] / 2;
        calculatedPadding[3] = inputPadding[3] - inputPadding[3] / 2;
    }

    const auto sequencedOpPadBegin = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[2], calculatedPadding[0]}));
    const auto sequencedOpPadEnd = getIntArrayAttr(ctx, makeArrayRef({calculatedPadding[3], calculatedPadding[1]}));
    sequencedOpKernelAttr = getIntArrayAttr(ctx, makeArrayRef(sequencedOpKernel));

    strides = parseIntArrayAttr<int64_t>(stridesAttr);
    stridesAttr = getGlobalOrOrigStride(ctx, firstOp.output(), sequencedOpKernel,
                                        {strides[Dims4D::Strides::Y.ind()], strides[Dims4D::Strides::X.ind()]},
                                        sequencedOpPadBegin, sequencedOpPadEnd);
    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origOp, firstOp.output(), sequencedOpKernelAttr, stridesAttr,
                                               sequencedOpPadBegin, sequencedOpPadEnd, origOp.rounding_type(),
                                               origOp.post_opAttr());
    return mlir::success();
}

//
// HandleLargeKernelsPass
//

class HandleLargeKernelsPass final : public IE::HandleLargeKernelsBase<HandleLargeKernelsPass> {
public:
    explicit HandleLargeKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleLargeKernelsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    const auto hasSupportedKernels = [](const SmallVector<int64_t>& kernelSize) {
        const auto KY = kernelSize[Dims4D::Kernel::Y.ind()];
        const auto KX = kernelSize[Dims4D::Kernel::X.ind()];

        return KY <= VPU::NCEInvariant::MAX_KERNEL_SIZE && KX <= VPU::NCEInvariant::MAX_KERNEL_SIZE;
    };

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([&](IE::AvgPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        if (hasSupportedKernels(kernelSize)) {
            return true;
        }
        const auto inDataType = op.input().getType().cast<vpux::NDTypeInterface>();
        const auto inDataShape = inDataType.getShape().raw();
        const auto strides = parseIntArrayAttr<int64_t>(op.strides());
        const auto inputElemType = inDataType.getElementType();
        auto unsupportedKernelCheck = [&](int32_t kernelInd, int32_t actInd, int32_t strideInd) {
            // Support multiple splitting for larger kernel size (> 11 * 11) with FP16/FP32 input as no drop in accuracy
            if (inputElemType.isF16() || inputElemType.isF32()) {
                return (kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]);
            } else {
                const auto maxKernelSizeSupported =
                        VPU::NCEInvariant::MAX_KERNEL_SIZE *
                        VPU::NCEInvariant::MAX_KERNEL_SIZE;  // we can only get 2 factors
                                                             // and max kernel should be 11 * 11 = 121
                return ((kernelSize[kernelInd] < inDataShape[actInd] && kernelSize[kernelInd] != strides[strideInd]) ||
                        kernelSize[kernelInd] > maxKernelSizeSupported);
            }
        };

        if (unsupportedKernelCheck(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        if (unsupportedKernelCheck(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
            _log.trace("AvgPool operation unsupported by HandleLargeKernel pass");
            return true;
        }
        // In these cases, more performant to execute this AvgPool on shave
        // leave it on for KMB as soon as VPUX3720 has HW AVG
        const auto arch = VPU::getArch(op);
        if ((arch == VPU::ArchKind::VPUX30XX || arch == VPU::ArchKind::VPUX311X) &&
            (kernelSize[Dims4D::Kernel::X.ind()] == 1 || kernelSize[Dims4D::Kernel::Y.ind()] == 1)) {
            _log.trace("AvgPool operation ignored by HandleLargeKernel pass for performance");
            return true;
        }

        const auto padsBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(op.pads_end());
        const auto zeros = SmallVector<int64_t>{0, 0};
        if ((padsBegin != zeros || padsEnd != zeros) && op.exclude_pads()) {
            return true;
        }
        return false;
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([&](IE::MaxPoolOp op) {
        const auto kernelSize = parseIntArrayAttr<int64_t>(op.kernel_size());
        return hasSupportedKernels(kernelSize);
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AveragePoolRewriter>(&ctx, _log);
    patterns.add<MaxPoolRewriter>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createHandleLargeKernelsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createHandleLargeKernelsPass(Logger log) {
    return std::make_unique<HandleLargeKernelsPass>(log);
}
