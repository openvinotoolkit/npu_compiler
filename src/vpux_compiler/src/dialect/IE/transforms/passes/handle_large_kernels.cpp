//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/handle_kernels_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/factors.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>

namespace vpux::IE {
#define GEN_PASS_DECL_HANDLELARGEKERNELS
#define GEN_PASS_DEF_HANDLELARGEKERNELS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 4;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

SmallVector<mlir::Value> sliceFilter(const mlir::Value filterToSplit, const int64_t numXSlices,
                                     const int64_t numYSlices, const int64_t targetKernelSize,
                                     const mlir::Location location, mlir::PatternRewriter& rewriter) {
    auto ctx = rewriter.getContext();

    SmallVector<mlir::Value> slicedFilters;

    const auto filterShape = getShape(filterToSplit);
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto KY = filterShape[Dims4D::Filter::KY];

    for (int64_t j = 0; j < numYSlices; j++) {
        for (int64_t i = 0; i < numXSlices; i++) {
            int64_t slicedKX = std::min(KX, targetKernelSize);
            if (i == (numXSlices - 1)) {
                slicedKX = KX - (numXSlices - 1) * targetKernelSize;
            }

            int64_t slicedKY = std::min(KY, targetKernelSize);
            if (j == (numYSlices - 1)) {
                slicedKY = KY - (numYSlices - 1) * targetKernelSize;
            }

            const auto IC = filterShape[Dims4D::Filter::IC];
            const auto OC = filterShape[Dims4D::Filter::OC];
            SmallVector<int64_t> sliceShape{OC, IC, slicedKY, slicedKX};

            Shape offsets(filterShape.size());
            offsets[Dims4D::Filter::KX] = i * targetKernelSize;
            offsets[Dims4D::Filter::KY] = j * targetKernelSize;
            auto slice =
                    rewriter.create<IE::SliceOp>(appendLoc(location, "slice_{0}_{1}", j, i), filterToSplit,
                                                 getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));
            slicedFilters.push_back(slice);
        }
    }
    return slicedFilters;
}

mlir::Value getExtendedActivation(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) {
    SmallVector<mlir::Value> extendedActivation;

    auto activation = origOp->getOperand(0);
    const auto inputShape = getShape(activation);
    int64_t newWidth = inputShape[Dims4D::Act::W];

    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));

    auto const extendActivationOnWidth = [&](int64_t dim) {
        Shape zeroConstShape(to_small_vector(inputShape));
        zeroConstShape[Dims4D::Act::W] = dim;
        const auto zeroType = mlir::RankedTensorType::get(
                zeroConstShape.raw(), mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getElementType());
        auto constZeros = Const::createZerosConst(rewriter, takeOpLoc(origOp, "zeros"), zeroType);
        extendedActivation.push_back(constZeros);
        newWidth += dim;
    };

    if (padStart[Dims4D::PadsBegin::Left] > 0) {
        extendActivationOnWidth(padStart[Dims4D::PadsBegin::Left]);
    }

    extendedActivation.push_back(activation);

    if (padEnd[Dims4D::PadsEnd::Right] > 0) {
        extendActivationOnWidth(padEnd[Dims4D::PadsEnd::Right]);
    }

    auto tempActivation =
            rewriter.createOrFold<IE::ConcatOp>(origOp.getLoc(), mlir::ValueRange(extendedActivation), Dims4D::Act::W);

    extendedActivation.clear();

    auto const extendActivationOnHeight = [&](int64_t dim) {
        Shape zeroConstShape(to_small_vector(inputShape));
        zeroConstShape[Dims4D::Act::H] = dim;
        zeroConstShape[Dims4D::Act::W] = newWidth;
        const auto zeroType = mlir::RankedTensorType::get(
                zeroConstShape.raw(), mlir::cast<NDTypeInterface>(origOp.getInput().getType()).getElementType());
        auto constZeros = Const::createZerosConst(rewriter, origOp->getLoc(), zeroType);
        extendedActivation.push_back(constZeros);
    };
    if (padStart[Dims4D::PadsBegin::Top] > 0) {
        extendActivationOnHeight(padStart[Dims4D::PadsBegin::Top]);
    }

    extendedActivation.push_back(tempActivation);

    if (padEnd[Dims4D::PadsEnd::Bottom] > 0) {
        extendActivationOnHeight(padEnd[Dims4D::PadsEnd::Bottom]);
    }

    return rewriter.createOrFold<IE::ConcatOp>(takeOpLoc(origOp, "concat_res"), mlir::ValueRange(extendedActivation),
                                               Dims4D::Act::H);
}

mlir::Value getTransformValue(mlir::PatternRewriter& rewriter, ShapeRef newShape, const mlir::Value newValue,
                              ArrayRef<uint32_t> perm) {
    const auto newShapeAttr = getIntArrayAttr(rewriter.getContext(), newShape);
    auto reshapeValue = rewriter.createOrFold<IE::ReshapeOp>(appendLoc(newValue.getLoc(), "_reshape"), newValue,
                                                             nullptr, false, newShapeAttr);
    const auto orderAttr = mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(perm, rewriter.getContext()));
    auto newTransOp = rewriter.create<IE::TransposeOp>(appendLoc(newValue.getLoc(), "_transpose"), reshapeValue,
                                                       nullptr, orderAttr);
    return newTransOp.getOutput();
};

void rewriteSubGraph(IE::ConvolutionOp origOp, ArrayRef<mlir::Value> slicedFilters, mlir::Value extendedActivation,
                     int64_t numXSlices, const int64_t numYSlices, const int64_t targetKernelSize,
                     mlir::PatternRewriter& rewriter, Logger log) {
    auto ctx = rewriter.getContext();

    const auto inputShape = getShape(origOp->getOperand(0));
    const auto filterShape = getShape(origOp.getFilter());
    const auto origKX = filterShape[Dims4D::Filter::KX];
    const auto origKY = filterShape[Dims4D::Filter::KY];
    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto broadcastType = vpux::IE::AutoBroadcastTypeAttr::get(ctx, IE::AutoBroadcastType::NONE_OR_EXPLICIT);
    const auto extendedActivationShape = getShape(extendedActivation);

    SmallVector<mlir::Value> accumulativeOutputTensors;
    for (int64_t j = 0; j < numYSlices; j++) {
        for (int64_t i = 0; i < numXSlices; i++) {
            int64_t startW = i * targetKernelSize;
            VPUX_THROW_WHEN(startW >= extendedActivationShape[Dims4D::Act::W], "dimension W out of range");
            int64_t startH = j * targetKernelSize;
            VPUX_THROW_WHEN(startH >= extendedActivationShape[Dims4D::Act::H], "dimension H out of range");

            auto slicedFilterShape = getShape(slicedFilters[j * numXSlices + i]);
            // Calculate activation slice shape
            int64_t newActivationWidth =
                    ((extendedActivationShape[Dims4D::Act::W] - origKX) / strides[Dims4D::Strides::X]) *
                            strides[Dims4D::Strides::X] +
                    slicedFilterShape[Dims4D::Act::W];
            int64_t newActivationHeight =
                    ((extendedActivationShape[Dims4D::Act::H] - origKY) / strides[Dims4D::Strides::Y]) *
                            strides[Dims4D::Strides::Y] +
                    slicedFilterShape[Dims4D::Act::H];
            if (newActivationWidth > extendedActivationShape[Dims4D::Act::W]) {
                newActivationWidth = extendedActivationShape[Dims4D::Act::W];
            }
            if (newActivationHeight > extendedActivationShape[Dims4D::Act::H]) {
                newActivationHeight = extendedActivationShape[Dims4D::Act::H];
            }

            mlir::Value convInput;
            SmallVector<int64_t> sliceShape{extendedActivationShape[Dims4D::Act::N],
                                            extendedActivationShape[Dims4D::Act::C], newActivationHeight,
                                            newActivationWidth};
            Shape offsets(inputShape.size());
            offsets[Dims4D::Act::W] = startW;
            offsets[Dims4D::Act::H] = startH;
            log.trace("Activation slice shape {0}, slice offsets {1}", sliceShape, offsets);

            convInput = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("slice_in_{0}_{1}"), j, i),
                                                     extendedActivation, getIntArrayAttr(ctx, offsets.raw()),
                                                     getIntArrayAttr(ctx, sliceShape));

            // Add bias and post process for the last convolution and eltwise.
            auto isLastSlice = i == (numXSlices - 1) && j == (numYSlices - 1);
            auto conv = rewriter.create<IE::ConvolutionOp>(
                    takeOpLoc(origOp, StringLiteral("conv_slice_{0}_{1}"), j, i), convInput,
                    slicedFilters[j * numXSlices + i],
                    isLastSlice ? origOp.getBias() : mlir::TypedValue<mlir::RankedTensorType>{nullptr},
                    origOp.getStrides(), getIntArrayAttr(origOp->getContext(), ArrayRef({0, 0})),
                    getIntArrayAttr(origOp->getContext(), ArrayRef({0, 0})), origOp.getDilationsAttr(), nullptr,
                    nullptr, origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

            if (!accumulativeOutputTensors.empty()) {
                auto add = rewriter.create<IE::AddOp>(
                        takeOpLoc(origOp, StringLiteral("res_add_{0}"), accumulativeOutputTensors.size()),
                        accumulativeOutputTensors.back(), conv, broadcastType, isLastSlice ? origOp.getClampAttr(),
                        origOp.getPostOpAttr() : nullptr, nullptr, origOp.getOutputPaddingAttr(),
                        origOp.getInputPaddingAttr());
                accumulativeOutputTensors.push_back(add);
            } else {
                accumulativeOutputTensors.push_back(conv);
            }
        }
    }

    log.trace("Successfully replace large convolution at {0}", origOp->getLoc());

    rewriter.replaceOp(origOp, accumulativeOutputTensors.back());
}

//
// GeneralPoolingBaseRewriter
//

template <class ConcreteOp>
class GeneralPoolingBaseRewriter : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GeneralPoolingBaseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("GeneralPoolingBaseRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

    virtual bool isLegalGeneralPoolingOp(ConcreteOp origOp) const final;
    virtual mlir::Value createFirstSplitOp(ConcreteOp origOp, IE::KernelsInfo& kernelsInfoVal,
                                           mlir::PatternRewriter& rewriter) const final;
    virtual mlir::Value createSecondSplitOp(ConcreteOp origOp, mlir::Value input, IE::KernelsInfo& kernelsInfoVal,
                                            mlir::PatternRewriter& rewriter) const final;
    virtual mlir::Operation* mappingNewOp(ConcreteOp origOp, mlir::Value input, mlir::ArrayAttr kernelsAttr,
                                          mlir::ArrayAttr padBeginAttr, mlir::ArrayAttr padEndAttr,
                                          mlir::ArrayAttr stridesAttr, mlir::PatternRewriter& rewriter,
                                          StringRef locSuffix) const final;

    virtual mlir::Value handlePoolWithPadding(ConcreteOp origOp, IE::KernelsInfo& kernelsInfo,
                                              mlir::PatternRewriter& rewriter) const = 0;

    virtual void removeStaticScale(mlir::Operation* newOp) const = 0;
    virtual void updateStaticScale(ConcreteOp origOp, mlir::Operation* newOp) const = 0;

protected:
    Logger _log;
};

// General Pooling Op is global pooling or the kernel size equal stride size.
// - Global pooling case:
//   Pooling Op with kernel [24, 24], pads_begin = [0, 0], pads_end = [0, 0], strides = [24, 24]
//   Input shape is 1x16x24x24 and the output shape will be 1x16x1x1
// - Kernel size equal stride size case:
//   Pooling Op with kernel [24, 24], pads_begin = [0, 0], pads_end = [0, 0], strides = [24, 24]
//   Input shape is 1x16x96x96 and the output shape will be 1x16x4x4
// For those scenarios, MaxPool and AvgPool have the same split logic.
template <class ConcreteOp>
bool GeneralPoolingBaseRewriter<ConcreteOp>::isLegalGeneralPoolingOp(ConcreteOp origOp) const {
    const auto inDataType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inDataShape = inDataType.getShape().raw();
    if (inDataShape.size() != 4) {
        _log.trace("Pooling Op should with input shape rank equal 4");
        return true;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStrides());
    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    if (vpux::IE::hasSupportedKernels(ShapeRef(kernelSize), maxKernelSize)) {
        _log.trace("Kernel size of general Pooling Op is legal");
        return true;
    }

    // Split large kernel for genaral pooling Op has below limitation:
    // Should not has data overlapped. It should be global pooling or stride equal to kernel size.

    auto isDataHasOverlapped = [&](int32_t kernelIdx, int32_t actIdx, int32_t strideIdx) {
        return (kernelSize[kernelIdx] < inDataShape[actIdx] && kernelSize[kernelIdx] != strides[strideIdx]);
    };
    if (isDataHasOverlapped(Dims4D::Kernel::X.ind(), Dims4D::Act::W.ind(), Dims4D::Strides::X.ind()) ||
        isDataHasOverlapped(Dims4D::Kernel::Y.ind(), Dims4D::Act::H.ind(), Dims4D::Strides::Y.ind())) {
        _log.trace("General Pooling Op should with global pooling or stride equal to kernel size");
        return true;
    }

    return false;
}

template <class ConcreteOp>
mlir::Operation* GeneralPoolingBaseRewriter<ConcreteOp>::mappingNewOp(
        ConcreteOp origOp, mlir::Value input, mlir::ArrayAttr kernelsAttr, mlir::ArrayAttr padBeginAttr,
        mlir::ArrayAttr padEndAttr, mlir::ArrayAttr stridesAttr, mlir::PatternRewriter& rewriter,
        StringRef locSuffix) const {
    mlir::IRMapping mapper;
    mapper.map(origOp->getOperands(), SmallVector<mlir::Value>{input});
    auto* newPoolingOp = rewriter.clone(*origOp.getOperation(), mapper);
    extendOpLoc(newPoolingOp, StringLiteral("{0}"), locSuffix);

    VPUX_THROW_UNLESS(newPoolingOp->hasAttr("pads_begin") && newPoolingOp->hasAttr("pads_end") &&
                              newPoolingOp->hasAttr("strides") && newPoolingOp->hasAttr("kernel_size"),
                      "Cannot get all attributions");
    newPoolingOp->setAttr("kernel_size", kernelsAttr);
    newPoolingOp->setAttr("pads_begin", padBeginAttr);
    newPoolingOp->setAttr("pads_end", padEndAttr);
    newPoolingOp->setAttr("strides", stridesAttr);
    vpux::inferReturnTypes(newPoolingOp, vpux::InferShapedTypeMode::ALL);

    return newPoolingOp;
}

template <class ConcreteOp>
mlir::Value GeneralPoolingBaseRewriter<ConcreteOp>::createFirstSplitOp(ConcreteOp origOp, IE::KernelsInfo& kernelsInfo,
                                                                       mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();
    const auto firstOpKernelAttr = getIntArrayAttr(ctx, kernelsInfo.firstKernel);
    const auto kernelPadEnd = kernelsInfo.padEnd;
    const auto firstOpPadBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto firstOpPadEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    firstOpPadEndAttr[Dims4D::PadsEnd::Right.ind()] += kernelPadEnd[Dims4D::PadsEnd::Right];
    firstOpPadEndAttr[Dims4D::PadsEnd::Bottom.ind()] += kernelPadEnd[Dims4D::PadsEnd::Bottom];

    // If it is global pooling, set the stride with 1
    // Elsewise set stride equal to kernel size
    auto getStrides = [](ShapeRef inShape, ShapeRef kernel, ShapeRef padBegin, ShapeRef padEnd) {
        auto strides = Shape{1, 1};

        if (kernel[Dims4D::Kernel::Y] !=
            (inShape[Dims4D::Act::H] + padBegin[Dims4D::PadsBegin::Top] + padEnd[Dims4D::PadsEnd::Bottom])) {
            strides[Dims4D::Strides::Y] = kernel[Dims4D::Strides::Y];
        }

        if (kernel[Dims4D::Kernel::X] !=
            (inShape[Dims4D::Act::W] + padBegin[Dims4D::PadsBegin::Left] + padEnd[Dims4D::PadsEnd::Right])) {
            strides[Dims4D::Strides::X] = kernel[Dims4D::Strides::X];
        }

        return strides;
    };

    const auto firstOpStrideAttr =
            getIntArrayAttr(ctx, getStrides(getShape(origOp.getInput()), kernelsInfo.firstKernel,
                                            ShapeRef(firstOpPadBeginAttr), ShapeRef(firstOpPadEndAttr)));

    const auto firstOp =
            mappingNewOp(origOp, origOp.getInput(), firstOpKernelAttr, origOp.getPadsBegin(),
                         getIntArrayAttr(ctx, firstOpPadEndAttr), firstOpStrideAttr, rewriter, "first_split");

    // When the original pool operation is split into several cascading AvgPool operations, only the last one retains
    // the static scale.
    // Remove static_scale attr in the first split operation.
    removeStaticScale(firstOp);

    _log.trace("Create firstSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
               firstOpKernelAttr, firstOpStrideAttr, firstOpPadBeginAttr, firstOpPadEndAttr);
    return firstOp->getResult(0);
}

template <class ConcreteOp>
mlir::Value GeneralPoolingBaseRewriter<ConcreteOp>::createSecondSplitOp(ConcreteOp origOp, mlir::Value input,
                                                                        IE::KernelsInfo& kernelsInfo,
                                                                        mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();
    const auto secondOpInShape = mlir::cast<vpux::NDTypeInterface>(input.getType()).getShape();
    const auto secondOpKernel = kernelsInfo.secondKernel;
    auto globalAvgOverH = secondOpInShape[Dims4D::Act::H] == secondOpKernel[Dims4D::Kernel::Y];
    auto globalAvgOverW = secondOpInShape[Dims4D::Act::W] == secondOpKernel[Dims4D::Kernel::X];
    auto secondOpStride = Shape{globalAvgOverH ? 1 : secondOpKernel[Dims4D::Kernel::Y],
                                globalAvgOverW ? 1 : secondOpKernel[Dims4D::Kernel::X]};

    const auto secondOpKernelAttr = getIntArrayAttr(ctx, secondOpKernel);
    const auto secondOpStridesAttr = getIntArrayAttr(ctx, secondOpStride);
    const auto secondOpPadBeginAttr = getIntArrayAttr(ctx, ArrayRef({0, 0}));
    const auto secondOpPadEndAttr = getIntArrayAttr(ctx, ArrayRef({0, 0}));

    const auto secondOp = mappingNewOp(origOp, input, secondOpKernelAttr, secondOpPadBeginAttr, secondOpPadEndAttr,
                                       secondOpStridesAttr, rewriter, "second_split");

    // When the original pool operation is split into several cascading AvgPool operations, only the last one retains
    // the static scale.
    // Update static_scale attr in the second split operation.
    updateStaticScale(origOp, secondOp);

    _log.trace("Create secondSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
               secondOpKernelAttr, secondOpStridesAttr, secondOpPadBeginAttr, secondOpPadEndAttr);
    return secondOp->getResult(0);
}

template <class ConcreteOp>
mlir::LogicalResult GeneralPoolingBaseRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' layer at '{2}'", this->getDebugName(), origOp.getOperationName(), origOp->getLoc());

    if (isLegalGeneralPoolingOp(origOp)) {
        return mlir::failure();
    }

    const auto origKernel = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto maxKernelSize = config::getMaxKernelSize(origOp);

    const auto kernelsInfo = IE::calculateKernelsInfo(ShapeRef(origKernel), maxKernelSize, _log.nest(2));
    if (!kernelsInfo.has_value()) {
        _log.trace("[{0}] Cannot spilt large kernel of '{1}' layer at '{2}'", this->getDebugName(),
                   origOp.getOperationName(), origOp->getLoc());
        return mlir::failure();
    }
    auto kernelsInfoVal = kernelsInfo.value();

    // Step 1: Handle the first split Op
    auto firstOpOutput = createFirstSplitOp(origOp, kernelsInfoVal, rewriter);

    // Step 2: Handle padding to avoid accuracy issue
    // Some scenarios need pad for first split Op.
    // For example: Pooling Op with kernel [23, 23], pads_begin = [0, 0], pads_end = [0, 0], strides = [23, 23]
    // - Split 1: Kernel: [8, 8], Stride: [8, 8], PadBegin: [0, 0], PadEnd: [1, 1] (need padding)
    // - Split 2: Kernel: [3, 3], Stride: [1, 1], PadBegin: [0, 0], PadEnd: [0, 0]
    // Avoid accuracy regression caused by this padding. Need specific handle:
    // MaxPool: If the input is all negative value the maximum value should also be negative
    //          If with an extra pad, the maximum value of the last row and column will be changed to 0
    // - The solution is to slice and concat input tensor instead of padding 0
    // AvgPool: The average weight for each value should be 1 / (KX * KY)
    //          If with an extra pad, the weight of last row and column will change to 1 / ((KX + padX) * (KY + padY))
    // - The solution is to use GroupConv with specific weight value instead of AvgPool
    const auto inShape = getShape(origOp.getInput());
    auto hasPadValue = llvm::any_of(kernelsInfoVal.padEnd, [](const int64_t padVal) {
        return padVal > 0;
    });
    const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    if (hasPadValue) {
        // Only support global pooling with extra padding
        // For example: Pooling Op with kernel [23, 23], pads_begin = [0, 0], pads_end = [0, 0], strides = [23, 23]
        // Input shape: 1x16x46x46, Output shape: 1x16x2x2
        // TODO(E#90030): concat four global pooling with inshape 1x16x23x23 and strides [23, 23]
        if ((kernelsInfoVal.padEnd[Dims4D::PadsEnd::Bottom] > 0 &&
             (inShape[Dims4D::Act::H] + padsBegin[Dims4D::PadsBegin::Top.ind()] +
              padsEnd[Dims4D::PadsEnd::Bottom.ind()]) != origKernel[Dims4D::Kernel::Y.ind()]) ||
            (kernelsInfoVal.padEnd[Dims4D::PadsEnd::Right] > 0 &&
             (inShape[Dims4D::Act::W] + padsBegin[Dims4D::PadsBegin::Left.ind()] +
              padsEnd[Dims4D::PadsEnd::Right.ind()]) != origKernel[Dims4D::Kernel::X.ind()])) {
            rewriter.eraseOp(firstOpOutput.getDefiningOp());
            _log.trace("Cannot handle padding of '{0}' layer at '{1}'", origOp.getOperationName(), origOp->getLoc());
            return mlir::failure();
        }

        auto firstOp = mlir::dyn_cast<ConcreteOp>(firstOpOutput.getDefiningOp());
        VPUX_THROW_UNLESS(firstOp != nullptr, "Cannot get the first split op");
        firstOpOutput = handlePoolWithPadding(firstOp, kernelsInfoVal, rewriter);
    }

    // Step 3: Handle the second split Op
    auto secondOpOutput = createSecondSplitOp(origOp, firstOpOutput, kernelsInfoVal, rewriter);
    rewriter.replaceOp(origOp, secondOpOutput);
    return mlir::success();
}

//
// GeneralAvgPoolRewriter
//

class GeneralAvgPoolRewriter final : public GeneralPoolingBaseRewriter<IE::AvgPoolOp> {
public:
    GeneralAvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): GeneralPoolingBaseRewriter<IE::AvgPoolOp>(ctx, log) {
        setDebugName("AvgPoolRewriter");
    }

    mlir::Value handlePoolWithPadding(IE::AvgPoolOp origOp, IE::KernelsInfo& kernelsInfo,
                                      mlir::PatternRewriter& rewriter) const override;

    void removeStaticScale(mlir::Operation* newPoolingOp) const override;
    void updateStaticScale(IE::AvgPoolOp origOp, mlir::Operation* newPoolingOp) const override;
};

void GeneralAvgPoolRewriter::removeStaticScale(mlir::Operation* newPoolingOp) const {
    const auto staticScaleAttrName = IE::AvgPoolOp::getStaticScaleAttrName(
            mlir::OperationName(llvm::StringLiteral("IE.AvgPool"), newPoolingOp->getContext()));
    if (!newPoolingOp->hasAttr(staticScaleAttrName)) {
        return;
    }
    newPoolingOp->removeAttr(staticScaleAttrName);
}

void GeneralAvgPoolRewriter::updateStaticScale(IE::AvgPoolOp origOp, mlir::Operation* newPoolingOp) const {
    const auto staticScaleAttrName = origOp.getStaticScaleAttrName();
    if (!origOp->hasAttr(staticScaleAttrName)) {
        return;
    }
    newPoolingOp->setAttr(staticScaleAttrName, origOp.getStaticScaleAttr());
}

mlir::Value GeneralAvgPoolRewriter::handlePoolWithPadding(IE::AvgPoolOp origOp, IE::KernelsInfo& kernelsInfo,
                                                          mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();

    const auto inShapeType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto inShape = inShapeType.getShape();
    const auto outShape = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType()).getShape();
    const auto kernelPadEnd = kernelsInfo.padEnd;
    const auto padsBegin = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    auto padsEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    padsEnd[Dims4D::PadsEnd::Right] -= kernelPadEnd[Dims4D::PadsEnd::Right];
    padsEnd[Dims4D::PadsEnd::Bottom] -= kernelPadEnd[Dims4D::PadsEnd::Bottom];
    int64_t inShapeHWithPad = inShape[Dims4D::Act::H];
    int64_t inShapeWWithPad = inShape[Dims4D::Act::W];
    if (!origOp.getExcludePads()) {
        inShapeHWithPad =
                inShape[Dims4D::Act::H] + padsBegin[Dims4D::PadsBegin::Top] + padsEnd[Dims4D::PadsEnd::Bottom];
        inShapeWWithPad =
                inShape[Dims4D::Act::W] + padsBegin[Dims4D::PadsBegin::Left] + padsEnd[Dims4D::PadsEnd::Right];
    }

    const auto weightsScaleVal = static_cast<float>(outShape[Dims4D::Act::H] * outShape[Dims4D::Act::W]) /
                                 static_cast<float>(inShapeHWithPad * inShapeWWithPad);

    const auto kernel = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto weightShape = SmallVector<int64_t>{inShape[Dims4D::Act::C], 1, kernel[Dims4D::Kernel::Y.ind()],
                                                  kernel[Dims4D::Kernel::X.ind()]};
    const auto dataStorageType = mlir::RankedTensorType::get(weightShape, inShapeType.getElementType());
    const auto weights =
            Const::createFloatConst(rewriter, takeOpLoc(origOp, "weights"), dataStorageType, weightsScaleVal);

    const auto dilationsAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto groupAttr = getIntAttr(ctx, inShape[Dims4D::Act::C]);

    _log.trace("Replace AvgPoolOp has Padding value with GroupConv to avoid accuracy issue");
    auto gConvOp = rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
            origOp, origOp.getInput(), weights,
            /*bias=*/nullptr, origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(),
            dilationsAttr, groupAttr,
            /*post_opAttr=*/nullptr, /*clamp=*/nullptr, origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    extendOpLoc(gConvOp, "as_groupconv");
    return gConvOp.getOutput();
}

//
// GeneralMaxPoolRewriter
//

class GeneralMaxPoolRewriter final : public GeneralPoolingBaseRewriter<IE::MaxPoolOp> {
public:
    GeneralMaxPoolRewriter(mlir::MLIRContext* ctx, Logger log): GeneralPoolingBaseRewriter<IE::MaxPoolOp>(ctx, log) {
        setDebugName("GeneralMaxPoolRewriter");
    }

    mlir::Value handlePoolWithPadding(IE::MaxPoolOp origOp, IE::KernelsInfo& kernelsInfo,
                                      mlir::PatternRewriter& rewriter) const override;
    void removeStaticScale(mlir::Operation* newPoolingOp) const override;
    void updateStaticScale(IE::MaxPoolOp origOp, mlir::Operation* newPoolingOp) const override;
};

void GeneralMaxPoolRewriter::removeStaticScale(mlir::Operation* newPoolingOp) const {
    VPUX_THROW_WHEN(newPoolingOp->hasAttr("static_scale"), "Unexpected static_scale attribute");
}

void GeneralMaxPoolRewriter::updateStaticScale(IE::MaxPoolOp origOp, mlir::Operation*) const {
    VPUX_THROW_WHEN(origOp->hasAttr("static_scale"), "Unexpected static_scale attribute");
}

mlir::Value GeneralMaxPoolRewriter::handlePoolWithPadding(IE::MaxPoolOp origOp, IE::KernelsInfo& kernelsInfo,
                                                          mlir::PatternRewriter& rewriter) const {
    auto* ctx = origOp->getContext();

    const auto inShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getShape();
    auto paddingActivation = [&](mlir::Value input, const int64_t padSize, const Dim padDim, StringRef locSuffix) {
        auto offsets = SmallVector<int64_t>(inShape.size(), 0);
        auto sliceShape = to_small_vector(inShape.raw());
        sliceShape[padDim.ind()] = padSize;
        auto sliceOp = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("slice_{0}"), locSuffix), input,
                                                    getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, sliceShape));

        return rewriter
                .create<IE::ConcatOp>(takeOpLoc(origOp, StringLiteral("concat_{0}"), locSuffix),
                                      SmallVector<mlir::Value>{input, sliceOp.getResult()}, padDim)
                .getOutput();
    };

    const auto kernelPadEnd = kernelsInfo.padEnd;
    auto inputVal = origOp.getInput();
    if (kernelPadEnd[Dims4D::PadsEnd::Right] > 0) {
        inputVal = paddingActivation(inputVal, kernelPadEnd[Dims4D::PadsEnd::Right], Dims4D::Act::W, "right");
    }

    if (kernelPadEnd[Dims4D::PadsEnd::Bottom] > 0) {
        inputVal = paddingActivation(inputVal, kernelPadEnd[Dims4D::PadsEnd::Bottom], Dims4D::Act::H, "bottom");
    }
    auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    padsEnd[Dims4D::PadsEnd::Right.ind()] -= kernelPadEnd[Dims4D::PadsEnd::Right];
    padsEnd[Dims4D::PadsEnd::Bottom.ind()] -= kernelPadEnd[Dims4D::PadsEnd::Bottom];

    _log.trace("Replace MaxPoolOp has Padding value with Slice and Concat to avoid accuracy issue");
    return rewriter
            .replaceOpWithNewOp<IE::MaxPoolOp>(origOp, inputVal, origOp.getKernelSizeAttr(), origOp.getStridesAttr(),
                                               origOp.getPadsBeginAttr(), getIntArrayAttr(ctx, padsEnd),
                                               origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                               origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr())
            .getOutput();
}

//
// OverlappedMaxPoolRewriter
//

class OverlappedMaxPoolRewriter final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    OverlappedMaxPoolRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("OverlappedMaxPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOverlappedMaxPool(IE::MaxPoolOp origOp, int64_t maxKernelSize) const;

private:
    Logger _log;
};

// Currently only support specific MaxPool with overlapped data
// For example: MaxPool Op with kernel [13, 13], pads_begin = [6, 6], pads_end = [6, 6], strides = [1, 1]
// - Split 1: Kernel: [7, 7], Stride: [1, 1], PadBegin: [3, 3], PadEnd: [3, 3]
// - Split 2: Kernel: [7, 7], Stride: [1, 1], PadBegin: [3, 3], PadEnd: [3, 3]
// If there are more related cases in the future. This part can be considered a more general implementation.
bool OverlappedMaxPoolRewriter::isLegalOverlappedMaxPool(IE::MaxPoolOp origOp, int64_t maxKernelSize) const {
    const auto inShape = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getShape();
    if (inShape.size() != 4) {
        _log.trace("Overlapped MaxPool Op should with input shape rank equal 4");
        return true;
    }

    const auto kernelSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    if (vpux::IE::hasSupportedKernels(ShapeRef(kernelSize), maxKernelSize)) {
        _log.trace("Kernel size of Overlapped MaxPool Op is legal");
        return true;
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    const auto strideSize = parseIntArrayAttr<int64_t>(origOp.getStrides());

    const auto isLargerThanOne = [](auto stride) {
        return stride > 1;
    };

    if (llvm::any_of(strideSize, isLargerThanOne)) {
        _log.trace("Overlapped MaxPool should with strides equal 1");
        return true;
    }

    const auto isEvenNumb = [](const int64_t val) {
        return val % 2 == 0;
    };

    if (llvm::any_of(kernelSize, isEvenNumb) || !llvm::all_of(padBegin, isEvenNumb)) {
        _log.trace("Overlapped MaxPool should with Kernel is even number and Pad is odd number");
        return true;
    }

    auto isSupportedRelation = [](int64_t kernelVal, int64_t padVal) {
        return kernelVal == padVal * 2 + 1;
    };

    auto isSupportedOverlappedMaxPool =
            isSupportedRelation(kernelSize[Dims4D::Kernel::X.ind()], padBegin[Dims4D::PadsBegin::Left.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::X.ind()], padBegin[Dims4D::PadsEnd::Right.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::Y.ind()], padBegin[Dims4D::PadsBegin::Top.ind()]) &&
            isSupportedRelation(kernelSize[Dims4D::Kernel::Y.ind()], padBegin[Dims4D::PadsEnd::Bottom.ind()]);

    if (!isSupportedOverlappedMaxPool) {
        _log.trace("Overlapped MaxPool should with Kernel size and Pad size: kernelVal -1 = 2 * padVal");
        return true;
    }

    return false;
}

mlir::LogicalResult OverlappedMaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' layer at '{2}'", getDebugName(), origOp.getOperationName(), origOp->getLoc());

    auto* ctx = origOp->getContext();
    const auto maxKernelSize = config::getMaxKernelSize(origOp);

    if (isLegalOverlappedMaxPool(origOp, maxKernelSize)) {
        return mlir::failure();
    }

    const auto kernelsSize = parseIntArrayAttr<int64_t>(origOp.getKernelSize());
    const auto padsBeginVal = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padsEndVal = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    const auto newKernelsAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{(kernelsSize[Dims4D::Kernel::Y.ind()] + 1) / 2,
                                                      (kernelsSize[Dims4D::Kernel::X.ind()] + 1) / 2});
    const auto newPadsBeginAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{padsBeginVal[Dims4D::PadsBegin::Top.ind()] / 2,
                                                      padsBeginVal[Dims4D::PadsBegin::Left.ind()] / 2});
    const auto newPadsEndAttr =
            getIntArrayAttr(ctx, SmallVector<int64_t>{padsEndVal[Dims4D::PadsEnd::Bottom.ind()] / 2,
                                                      padsEndVal[Dims4D::PadsEnd::Right.ind()] / 2});

    auto firstOp = rewriter.create<IE::MaxPoolOp>(
            takeOpLoc(origOp, "first_split"), origOp.getInput(), newKernelsAttr, origOp.getStridesAttr(),
            newPadsBeginAttr, newPadsEndAttr, origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());
    _log.nest(2).trace("Create firstSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
                       newKernelsAttr, origOp.getStridesAttr(), newPadsBeginAttr, newPadsEndAttr);

    _log.nest(2).trace("Create secondSplitOp with Kernel: '{0}', Stride: '{1}', PadBegin: '{2}', PadEnd: '{3}'",
                       newKernelsAttr, origOp.getStridesAttr(), newPadsBeginAttr, newPadsEndAttr);
    auto secondOp = rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            origOp, firstOp.getOutput(), newKernelsAttr, origOp.getStridesAttr(), newPadsBeginAttr, newPadsEndAttr,
            origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());
    extendOpLoc(secondOp, "second_split");

    return mlir::success();
}

//
// PadLargeKernelRewriter
//

/* For 1D convolution NCE ops where the kernel size exceeds maxKernelSize, this rewriter applies a GCD-based
stride factorization algorithm to optimize large kernel handling through strategic padding and restructuring.

Algorithm Overview:
The core idea is to factorize the original stride S into S = newStride × gcdFactor, where newStride ≤ MAX_STRIDE (8).
Then padding both input and kernel to be multiples of gcdFactor, enabling parallel processing.

Mathematical Foundation:
1. Stride Factorization: S = S₁ × S₂, where S₁ ≤ 8 and S₂ = S/S₁
2. Padding Strategy:
   - paddedInputSize = ⌈inputSize / gcdFactor⌉ × gcdFactor
   - paddedKernelSize = ⌈kernelSize / gcdFactor⌉ × gcdFactor
3. Parallel Processing: After padding, the convolution can process gcdFactor parallel slices

Example Transformation:
Original: Input[1,1,1,49964], Filter[1025,1,1,2048], Stride[1,242]
Step 1: Factorize stride 242 = 1 × 242 (newStride=1, gcdFactor=242)
Step 2: Pad input: 49964 → 50006 (⌈49964/242⌉×242 = 207×242)
Step 3: Pad kernel: 2048 → 2178 (⌈2048/242⌉×242 = 9×242)

Transformation Flow:
                    Original Convolution
                 Input[1,1,1,I]  Filter[OC,1,1,K]
                      |              |
                   Pad to            Pad to
              ⌈I/GCD⌉×GCD      ⌈K/GCD⌉×GCD
                      |              |
                Input_padded    Filter_padded
                      \              /
                    Convolution (optimized)
                          |
                    Output (GCD parallel slices)
                          |
                    Slice to original size
                          |
                    Final Output
*/

class PadLargeKernelRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    PadLargeKernelRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("PadLargeKernelRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOpToConvert(IE::ConvolutionOp origOp, int64_t& paddedInputSize, int64_t& paddedKernelSize,
                            int64_t& outputSize) const;

private:
    Logger _log;
};

bool PadLargeKernelRewriter::isLegalOpToConvert(IE::ConvolutionOp origOp, int64_t& paddedInputSize,
                                                int64_t& paddedKernelSize, int64_t& outputSize) const {
    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    const auto strideSize = parseIntArrayAttr<int64_t>(origOp.getStrides());

    _log.trace("Input shape: {0}, Filter shape: {1}, Strides: {2}", inputShape, filterShape, strideSize);

    // Basic validation - ensure we have 4D tensors
    if (inputShape.size() != 4 || filterShape.size() != 4 || strideSize.size() != 2) {
        _log.trace("Invalid tensor dimensions");
        return false;
    }

    // Check if this is a 1D convolution (either H-dimensional or W-dimensional)
    bool isWDimensional = (inputShape[Dims4D::Act::H] == 1 && filterShape[Dims4D::Filter::KY] == 1);
    bool isHDimensional = (inputShape[Dims4D::Act::W] == 1 && filterShape[Dims4D::Filter::KX] == 1);

    if (!isWDimensional && !isHDimensional) {
        _log.trace("Not a 1D convolution, skipping");
        return false;
    }

    int64_t inputSize, kernelSize, stride;
    if (isWDimensional) {
        inputSize = inputShape[Dims4D::Act::W];
        kernelSize = filterShape[Dims4D::Filter::KX];
        stride = strideSize[Dims4D::Strides::X.ind()];
        _log.trace("W-dimensional 1D conv: input={0}, kernel={1}, stride={2}", inputSize, kernelSize, stride);
    } else {
        inputSize = inputShape[Dims4D::Act::H];
        kernelSize = filterShape[Dims4D::Filter::KY];
        stride = strideSize[Dims4D::Strides::Y.ind()];
        _log.trace("H-dimensional 1D conv: input={0}, kernel={1}, stride={2}", inputSize, kernelSize, stride);
    }

    const auto maxKernelSize = config::getMaxKernelSize(origOp);

    if (kernelSize <= maxKernelSize) {
        _log.trace("Kernel size {0} <= max {1}, no padding needed", kernelSize, maxKernelSize);
        return false;
    }

    // Apply stride factorization algorithm
    auto findOptimalStrideFactorization = [](int64_t stride) -> std::pair<int64_t, int64_t> {
        // Special case: stride = 1 is always factorizable as 1 × 1
        if (stride == 1) {
            return {1, 1};
        }

        // Find the smallest factor that is <= MAX_STRIDE
        for (int64_t factor1 = 1; factor1 <= VPU::NCEInvariant::MAX_STRIDE; ++factor1) {
            if (factor1 <= stride && stride % factor1 == 0) {
                int64_t factor2 = stride / factor1;
                return {factor1, factor2};
            }
        }
        return {0, 0};
    };

    auto [newStride, gcdFactor] = findOptimalStrideFactorization(stride);

    if (newStride == 0 || gcdFactor == 0) {
        _log.trace("No valid stride factorization found");
        return false;
    }

    // Calculate padded sizes based on GCD
    paddedInputSize = ((inputSize + gcdFactor - 1) / gcdFactor) * gcdFactor;
    paddedKernelSize = ((kernelSize + gcdFactor - 1) / gcdFactor) * gcdFactor;

    // Check if padded kernel size is within limits after division by GCD
    int64_t kernelSliceSize = paddedKernelSize / gcdFactor;
    if (kernelSliceSize > maxKernelSize) {
        _log.trace("Kernel slice size {0} > max {1} after GCD division", kernelSliceSize, maxKernelSize);
        return false;
    }

    // IMPORTANT: Check if this convolution was already processed
    // If the current kernel size divided by GCD is already <= maxKernelSize,
    // it means this convolution was likely already processed by our algorithm
    int64_t currentKernelSliceSize = kernelSize / gcdFactor;
    if (currentKernelSliceSize <= maxKernelSize && kernelSize % gcdFactor == 0) {
        _log.trace("Kernel already appears to be processed (sliceSize={0} <= max={1})", currentKernelSliceSize,
                   maxKernelSize);
        return false;
    }

    // Calculate correct output size:
    // After GCD division, we have inputSliceSize blocks of size kernelSliceSize
    // Each block processes inputSliceSize elements with stride newStride
    int64_t inputSliceSize = paddedInputSize / gcdFactor;
    int64_t outputSliceSize = (inputSliceSize - kernelSliceSize) / newStride + 1;
    outputSize = outputSliceSize * gcdFactor;

    return true;
}

mlir::LogicalResult PadLargeKernelRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    int64_t paddedInputSize = 0, paddedKernelSize = 0, outputSize = 0;
    if (!isLegalOpToConvert(origOp, paddedInputSize, paddedKernelSize, outputSize)) {
        return mlir::failure();
    }

    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    const auto outputShape = getShape(origOp.getOutput());

    // Determine if this is H-dimensional or W-dimensional 1D convolution
    bool isWDimensional = (inputShape[Dims4D::Act::H] == 1 && filterShape[Dims4D::Filter::KY] == 1);

    int64_t currentInputSize, currentKernelSize;
    int64_t inputDimIndex, kernelDimIndex;

    if (isWDimensional) {
        currentInputSize = inputShape[Dims4D::Act::W];
        currentKernelSize = filterShape[Dims4D::Filter::KX];
        inputDimIndex = Dims4D::Act::W.ind();
        kernelDimIndex = Dims4D::Filter::KX.ind();
    } else {
        // H-dimensional convolution
        currentInputSize = inputShape[Dims4D::Act::H];
        currentKernelSize = filterShape[Dims4D::Filter::KY];
        inputDimIndex = Dims4D::Act::H.ind();
        kernelDimIndex = Dims4D::Filter::KY.ind();
    }

    const auto inputPadAmount = paddedInputSize - currentInputSize;
    const auto kernelPadAmount = paddedKernelSize - currentKernelSize;

    _log.trace("[{0}] Padding input by {1} and kernel by {2}", getDebugName(), inputPadAmount, kernelPadAmount);

    auto createPaddingConstant = [&](mlir::PatternRewriter& rewriter, mlir::Location loc,
                                     ArrayRef<int64_t> paddingShape, mlir::Type elementType) -> mlir::Value {
        auto paddingConstType = mlir::RankedTensorType::get(paddingShape, elementType);
        auto paddingConstAttr = Const::ContentAttr::get(
                mlir::DenseElementsAttr::get(paddingConstType, mlir::FloatAttr::get(elementType, 0.0)));
        return rewriter.create<Const::DeclareOp>(loc, paddingConstType, paddingConstAttr).getOutput();
    };

    // Step 1: Pad input if needed
    mlir::Value paddedInput = origOp.getInput();
    if (inputPadAmount > 0) {
        SmallVector<int64_t> paddingShapeVec = to_small_vector(inputShape.raw());
        paddingShapeVec[inputDimIndex] = inputPadAmount;

        auto inputElementType = mlir::cast<mlir::RankedTensorType>(origOp.getInput().getType()).getElementType();
        auto paddingConst = createPaddingConstant(rewriter, origOp->getLoc(), paddingShapeVec, inputElementType);

        // Concatenate input with padding
        paddedInput = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_pad_input_for_large_kernel"),
                                                    mlir::ValueRange{origOp.getInput(), paddingConst},
                                                    getIntAttr(rewriter.getContext(), inputDimIndex))
                              .getOutput();
    }

    // Step 2: Pad kernel if needed
    mlir::Value paddedFilter = origOp.getFilter();
    if (kernelPadAmount > 0) {
        SmallVector<int64_t> filterPaddingShapeVec = to_small_vector(filterShape.raw());
        filterPaddingShapeVec[kernelDimIndex] = kernelPadAmount;

        auto filterElementType = mlir::cast<mlir::RankedTensorType>(origOp.getFilter().getType()).getElementType();
        auto filterPaddingConst =
                createPaddingConstant(rewriter, origOp->getLoc(), filterPaddingShapeVec, filterElementType);

        // Concatenate filter with padding
        paddedFilter = rewriter.create<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_pad_filter_for_large_kernel"),
                                                     mlir::ValueRange{origOp.getFilter(), filterPaddingConst},
                                                     getIntAttr(rewriter.getContext(), kernelDimIndex))
                               .getOutput();
    }

    // Step 3: Create new convolution with padded inputs
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            appendLoc(origOp->getLoc(), "_padded_conv_for_large_kernel"), paddedInput, paddedFilter, origOp.getBias(),
            origOp.getStridesAttr(), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(),
            origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    // Step 4: Slice output to match original output size if needed
    mlir::Value finalOutput = newConvOp.getOutput();
    int64_t currentOutputSize = isWDimensional ? outputShape[Dims4D::Act::W] : outputShape[Dims4D::Act::H];
    if (outputSize != currentOutputSize) {
        finalOutput =
                rewriter.create<IE::SliceOp>(appendLoc(origOp->getLoc(), "_slice_output_for_large_kernel"),
                                             newConvOp.getOutput(),
                                             getIntArrayAttr(rewriter.getContext(), SmallVector<int64_t>{0, 0, 0, 0}),
                                             getIntArrayAttr(rewriter.getContext(), outputShape.raw()))
                        .getOutput();
    }

    rewriter.replaceOp(origOp, finalOutput);
    _log.trace("[{0}] Successfully replaced conv with padded version", getDebugName());
    return mlir::success();
}

//
// SliceLargeConvRewriter
//

class SliceLargeConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SliceLargeConvRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("SliceLargeConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOpToConvert(IE::ConvolutionOp origOp, int64_t maxKernelSize) const;

private:
    Logger _log;
};

bool SliceLargeConvRewriter::isLegalOpToConvert(IE::ConvolutionOp origOp, int64_t maxKernelSize) const {
    auto activationRank = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank();
    auto filterRank = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType()).getRank();
    if (activationRank != 4 || filterRank != 4) {
        return false;
    }

    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    if (dilations[Dims4D::Dilation::X] > 1 || dilations[Dims4D::Dilation::Y] > 1) {
        return false;
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    return KY > maxKernelSize || KX > maxKernelSize;
}

mlir::LogicalResult SliceLargeConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto targetKernelSize = config::getMaxKernelSize(origOp);
    if (targetKernelSize <= 0) {
        return mlir::failure();
    }
    if (!isLegalOpToConvert(origOp, targetKernelSize)) {
        return mlir::failure();
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto KY = filterShape[Dims4D::Filter::KY];

    auto numXSlices = checked_cast<int64_t>(llvm::alignTo(KX, targetKernelSize) / targetKernelSize);
    auto numYSlices = checked_cast<int64_t>(llvm::alignTo(KY, targetKernelSize) / targetKernelSize);

    // Slice filter
    SmallVector<mlir::Value> slicedFilters =
            sliceFilter(origOp.getFilter(), numXSlices, numYSlices, targetKernelSize, origOp->getLoc(), rewriter);
    _log.trace("[{0}] Split kernel into {1} small kernels {2} at {3}", getDebugName(), slicedFilters.size(),
               slicedFilters, origOp->getLoc());

    // Pad activation
    auto extendedActivation = getExtendedActivation(origOp, rewriter);
    _log.trace("[{0}] Pad on activation, new shape {1}, new activation {2} at {3}", getDebugName(),
               getShape(extendedActivation), extendedActivation, origOp->getLoc());

    // Create new sub graph and replace origOp
    rewriteSubGraph(origOp, slicedFilters, extendedActivation, numXSlices, numYSlices, targetKernelSize, rewriter,
                    _log);
    return mlir::success();
}

//
// ReshapeLargeConvRewriter
//

/* For convolution NCE ops, if IC and height or width equals to 1 and KX or KY is bigger than maxKernelSize.
The utilization may be low if slicing it directly. Add AffineReshape to change the shape will not involve extra memory
copy and increase utilization. After large height or width being split, we will tranpose lowest-dimension continuous
data to IC to make result data correctness. Original shape only one dim is valid, after affineReshape and transpose, IC
of activation is factors.second, so strides should div by factors.second.
Case 1: stride can be divided by factor
                                Convert convolution from
                                    input        filter
                                [1, 1, 1, W]   [OC, 1, 1, KX]
                                    \             /
                                      convolution
                                           |
                                        output
                                           |
                                      [1, C, 1, OW]
=>
                  input                                    filter
                    |                                        |
                AffineReshape                            AffineReshape
    [1, W/factors.first, 1, factors.second]      [OC, factors.first, 1, KX/factors.second]
                    |                                        |
                transpose                                transpose
                    |                                        |
    [1, factors.second, 1, W/factors.first]      [OC, KX/factors.second, 1, factors.first]
                                 \                   /
                                       convolution
                                            |
                                       [1, C, 1, OW]
Case 2: stride is 1
If kernel > maxKernelSize and stride is 1, we need pad kernel and copy filters to make stride bigger.
Such as filter [1, 1, 1, 54] and stride [1, 1]:
            filter(1x1x1x54)                  new Filter(6x1x1x60)
            1, 1, ..., 1           ->         1, 1, ..., 1, 0, 0, 0, 0, 0, 0
                                              0, 1, 1, ..., 1, 0, 0, 0, 0, 0
                                              0, 0, 1, 1, ..., 1, 0, 0, 0, 0
                                              0, 0, 0, 1, 1, ..., 1, 0, 0, 0
                                              0, 0, 0, 0, 1, 1, ..., 1, 0, 0
                                              0, 0, 0, 0, 0, 1, 1, ..., 1, 0
The new filter is [6, 1, 1, 60] and new strides is [1, 6], the new output is [1, 6, 1, output_width / 6].
Then transpose and reshape new output to [1, 1, 1, (output_width / 6) * 6] to get the same result as original output.
After that, call the same function as Case1 to fold KY to IC channel to handle large kernel.
*/
class ReshapeLargeConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ReshapeLargeConvRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("ReshapeLargeConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOpToConvert(IE::ConvolutionOp origOp, std::optional<Factors>& factorsResult, Shape inputShape,
                            bool& strideOne) const;

    mlir::Value getOutputFromLargeStride(mlir::PatternRewriter& rewriter, IE::ConvolutionOp origOp,
                                         const vpux::Factors& factors, ShapeRef newInputShape, ArrayRef<uint32_t> perm,
                                         bool factorsOnKX) const;
    mlir::Value getOutputFromStrideOne(mlir::PatternRewriter& rewriter, IE::ConvolutionOp origOp,
                                       const vpux::Factors& factors, ShapeRef newInputShape, ShapeRef paddedInputShape,
                                       ArrayRef<uint32_t> perm, bool factorsOnKX, int64_t padValue) const;

private:
    Logger _log;
};

bool ReshapeLargeConvRewriter::isLegalOpToConvert(IE::ConvolutionOp origOp, std::optional<Factors>& factorsResult,
                                                  Shape inputShape, bool& strideOne) const {
    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    const auto inputHeight = inputShape[Dims4D::Act::H];
    const auto inputWidth = inputShape[Dims4D::Act::W];

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    auto padBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto padEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto sx = strides[Dims4D::Strides::X.ind()];
    auto sy = strides[Dims4D::Strides::Y.ind()];

    if (sx == 1 && sy == 1) {
        strideOne = true;
        if (padBeginAttr[Dims4D::PadsBegin::Left.ind()] != 0 || padEndAttr[Dims4D::PadsEnd::Right.ind()] != 0 ||
            padBeginAttr[Dims4D::PadsBegin::Top.ind()] != 0 || padEndAttr[Dims4D::PadsEnd::Bottom.ind()] != 0) {
            return false;
        }
    }

    if (filterShape[Dims4D::Filter::IC] != 1) {
        return false;
    }

    auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
        return false;
    }

    if (!factorsResult.has_value()) {
        return false;
    }

    auto factors = factorsResult.value();
    auto factor = strideOne ? factors.first : factors.second;

    if (KX > maxKernelSize && KY == 1 && inputHeight == 1) {
        if (inputWidth % factor) {
            return false;
        }

        if (sx != 1) {
            if (sx % factor) {
                return false;
            }
            if (padBeginAttr[Dims4D::PadsBegin::Left.ind()] % factor ||
                padEndAttr[Dims4D::PadsEnd::Right.ind()] % factor) {
                return false;
            }
        }

        return true;
    }

    if (KY > maxKernelSize && KX == 1 && inputWidth == 1) {
        if (inputHeight % factor) {
            return false;
        }

        if (sy != 1) {
            if (sy % factor) {
                return false;
            }
            if (padBeginAttr[Dims4D::PadsBegin::Top.ind()] % factor ||
                padEndAttr[Dims4D::PadsEnd::Bottom.ind()] % factor) {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool isPrime(int num) {
    if (num < 2) {
        return false;
    }
    int sqrtNum = static_cast<int>(std::sqrt(num));
    for (int i = 2; i <= sqrtNum; ++i) {
        if (num % i == 0) {
            return false;
        }
    }
    return true;
}

mlir::Value ReshapeLargeConvRewriter::getOutputFromLargeStride(mlir::PatternRewriter& rewriter,
                                                               IE::ConvolutionOp origOp, const vpux::Factors& factors,
                                                               ShapeRef newInputShape, ArrayRef<uint32_t> perm,
                                                               bool factorsOnKX) const {
    // factor is bigger one, new input channel is factor and new kernel(kx or ky) is factors.first
    auto factor = factors.second;

    mlir::Value inputValue = origOp.getInput();
    auto newInputValue = getTransformValue(rewriter, newInputShape, inputValue, perm);
    auto filterShape = getShape(origOp.getFilter());
    auto newFilterShape = Shape(SmallVector<int64_t>{filterShape[Dims4D::Filter::OC], factors.first,
                                                     factorsOnKX ? 1 : factor, factorsOnKX ? factor : 1});

    mlir::Value filterValue = origOp.getFilter();
    auto newFilterValue = getTransformValue(rewriter, newFilterShape, filterValue, perm);

    const auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    SmallVector<int64_t> newStrides = {factorsOnKX ? 1 : strides[Dims4D::Strides::Y.ind()] / factor,
                                       factorsOnKX ? strides[Dims4D::Strides::X.ind()] / factor : 1};
    const auto newStridesAttr = getIntArrayAttr(rewriter.getContext(), newStrides);

    auto padBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto padEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    padBeginAttr[Dims4D::PadsBegin::Left.ind()] = padBeginAttr[Dims4D::PadsBegin::Left.ind()] / factor;
    padBeginAttr[Dims4D::PadsBegin::Top.ind()] = padBeginAttr[Dims4D::PadsBegin::Top.ind()] / factor;
    padEndAttr[Dims4D::PadsEnd::Right.ind()] = padEndAttr[Dims4D::PadsEnd::Right.ind()] / factor;
    padEndAttr[Dims4D::PadsEnd::Bottom.ind()] = padEndAttr[Dims4D::PadsEnd::Bottom.ind()] / factor;
    auto newBeginAttr = getIntArrayAttr(rewriter.getContext(), padBeginAttr);
    auto newEndAttr = getIntArrayAttr(rewriter.getContext(), padEndAttr);

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp->getLoc(), newInputValue, newFilterValue, origOp.getBias(), newStridesAttr, newBeginAttr, newEndAttr,
            origOp.getDilationsAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    return rewriter
            .create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_reshape_out"), newConvOp.getOutput(), nullptr, false,
                                   getIntArrayAttr(rewriter.getContext(), getShape(origOp.getOutput())))
            .getOutput();
}

mlir::Value ReshapeLargeConvRewriter::getOutputFromStrideOne(mlir::PatternRewriter& rewriter, IE::ConvolutionOp origOp,
                                                             const vpux::Factors& factors, ShapeRef newInputShape,
                                                             ShapeRef paddedInputShape, ArrayRef<uint32_t> perm,
                                                             bool factorsOnKX, int64_t padValue) const {
    // factor is smaller one, new input channel & stride are both factor and new kernel(kx or ky) is factors.second
    auto factor = factors.first;

    auto origInputShape = getShape(origOp.getInput());
    auto padBegin = mlir::SmallVector<int64_t>(paddedInputShape.size(), 0);
    auto padEnd = mlir::SmallVector<int64_t>(paddedInputShape.size(), 0);
    if (factorsOnKX) {
        padEnd[vpux::Dims4D::Act::W.ind()] = paddedInputShape[Dims4D::Act::W] - origInputShape[Dims4D::Act::W];
    } else {
        padEnd[vpux::Dims4D::Act::H.ind()] = paddedInputShape[Dims4D::Act::H] - origInputShape[Dims4D::Act::H];
    }
    auto inputExpandOp = rewriter.create<IE::ExpandOp>(
            appendLoc(origOp->getLoc(), "_activation_expand"), origOp.getInput(),
            getIntArrayAttr(rewriter.getContext(), padBegin), getIntArrayAttr(rewriter.getContext(), padEnd));
    mlir::Value inputValue = inputExpandOp.getOutput();

    auto newInputValue = getTransformValue(rewriter, newInputShape, inputValue, perm);
    auto filterShape = getShape(origOp.getFilter());
    auto newOC = filterShape[Dims4D::Filter::OC] * factor;
    auto newFilterShape =
            Shape(SmallVector<int64_t>{newOC, factors.second, factorsOnKX ? 1 : factor, factorsOnKX ? factor : 1});

    mlir::Value filterValue = origOp.getFilter();
    mlir::Value biasValue = origOp.getBias();
    SmallVector<mlir::Value> newBias;
    // expand to the padded dimension
    auto weightConst = origOp.getFilter().getDefiningOp<Const::DeclareOp>();
    SmallVector<mlir::Value> newWeights;
    for (int i = 0; i < factors.first; i++) {
        int64_t padBeforeValue = i;
        int64_t padEndValue = padValue - i;
        auto weightConstAttr =
                weightConst.transformContentAttr()
                        .padWithZero({0, 0, factorsOnKX ? 0 : padBeforeValue, factorsOnKX ? padBeforeValue : 0},
                                     {0, 0, factorsOnKX ? 0 : padEndValue, factorsOnKX ? padEndValue : 0})
                        .get();
        auto newWeightLoc = appendLoc(origOp->getLoc(), "_weight_{0}", i);
        auto newConstValue =
                rewriter.create<Const::DeclareOp>(newWeightLoc, weightConstAttr.getType(), std::move(weightConstAttr))
                        .getResult();
        newWeights.push_back(newConstValue);
        if (biasValue) {
            newBias.push_back(biasValue);
        }
    }
    filterValue = rewriter.createOrFold<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_weight_concat"), newWeights,
                                                      Dims4D::Filter::OC);
    auto newFilterValue = getTransformValue(rewriter, newFilterShape, filterValue, perm);
    if (biasValue) {
        biasValue = rewriter.createOrFold<IE::ConcatOp>(appendLoc(origOp->getLoc(), "_bias_concat"), newBias,
                                                        Dims4D::Act::C);
    }

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp->getLoc(), newInputValue, newFilterValue, biasValue, origOp.getStridesAttr(),
            origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(), origOp.getDilationsAttr(), origOp.getPostOpAttr(),
            origOp.getClampAttr(), origOp.getStaticScaleAttr(), origOp.getOutputPaddingAttr(),
            origOp.getInputPaddingAttr());

    auto origOutputShape = getShape(origOp.getOutput());
    auto convOutputShape = getShape(newConvOp.getOutput());
    auto newOutputShape = Shape(
            SmallVector<int64_t>{origOutputShape[Dims4D::Act::N], factor, origOutputShape[Dims4D::Act::C],
                                 factorsOnKX ? convOutputShape[Dims4D::Act::W] : convOutputShape[Dims4D::Act::H]});

    auto outPerm = SmallVector<uint32_t>{0, 2, 3, 1};
    auto convOutTransValue = getTransformValue(rewriter, newOutputShape, newConvOp.getOutput(), outPerm);
    auto transOutShape = getShape(convOutTransValue);
    auto transOutHW = transOutShape[Dims4D::Act::H] * transOutShape[Dims4D::Act::W];
    auto newTransOutShape = SmallVector<int64_t>{transOutShape[Dims4D::Act::N], transOutShape[Dims4D::Act::C],
                                                 factorsOnKX ? 1 : transOutHW, factorsOnKX ? transOutHW : 1};
    auto outReshapeOp =
            rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_trans_out_reshape"), convOutTransValue,
                                           nullptr, false, getIntArrayAttr(rewriter.getContext(), newTransOutShape));

    auto outShape = getShape(outReshapeOp.getOutput());
    auto staticOffsets = SmallVector<int64_t>(outShape.size(), 0);
    SmallVector<int64_t> staticSizes(outShape.begin(), outShape.end());
    if (factorsOnKX) {
        staticSizes[Dims4D::Act::W.ind()] = origOutputShape[Dims4D::Act::W];
    } else {
        staticSizes[Dims4D::Act::H.ind()] = origOutputShape[Dims4D::Act::H];
    }

    return rewriter
            .create<IE::SliceOp>(appendLoc(origOp->getLoc(), "_slice_out"), outReshapeOp.getOutput(),
                                 getIntArrayAttr(rewriter.getContext(), staticOffsets),
                                 getIntArrayAttr(rewriter.getContext(), staticSizes))
            .getResult();
}

mlir::LogicalResult ReshapeLargeConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                              mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());
    Shape inputShape = Shape(getShape(origOp.getInput()));
    auto filterShape = getShape(origOp.getFilter());

    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    auto KX = filterShape[Dims4D::Filter::KX];
    auto KY = filterShape[Dims4D::Filter::KY];

    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    const auto limit = std::min(maxKernelSize, VPU::NCEInvariant::MAX_STRIDE);
    auto factorsResult =
            KX > maxKernelSize ? IE::getFactorsWithLimitation(KX, limit) : IE::getFactorsWithLimitation(KY, limit);

    bool factorsOnKX = KX > maxKernelSize ? true : false;
    bool strideOne = false;
    vpux::Factors factors;
    int64_t padValue = 0;

    if (!isLegalOpToConvert(origOp, factorsResult, inputShape, strideOne)) {
        if (strideOne) {
            auto getFactorsResult = [&](int64_t& kernel) -> std::optional<Factors> {
                if (isPrime(kernel)) {
                    // pad to make kernel not prime value
                    kernel = alignValUp(kernel, int64_t(2));
                }
                factorsResult = IE::getFactorsWithLimitation(kernel, limit);
                if (!factorsResult.has_value()) {
                    _log.trace("'{0}' can't get valid factor values", kernel);
                    return std::nullopt;
                }

                factors = factorsResult.value();
                // pad kernel value with factor.first to copy filter with factor.first times
                kernel += factors.first;
                factorsResult = IE::getFactorsWithLimitation(kernel, limit);
                if (!factorsResult.has_value()) {
                    _log.trace("Padded kernel size '{0}' can't get valid factor values", kernel);
                    return std::nullopt;
                }

                return factorsResult.value();
            };

            int64_t newKernel = factorsOnKX ? KX : KY;
            factorsResult = getFactorsResult(newKernel);
            if (!factorsResult.has_value()) {
                return mlir::failure();
            }
            factors = factorsResult.value();

            // new stride and kernel are both factors.first
            if (factors.first > limit || factors.second > maxKernelSize) {
                _log.trace("stride '{0}' > VPU::NCEInvariant::MAX_STRIDE or kernel size '{1}' > "
                           "VPU::NCEInvariant::MAX_KERNEL",
                           factors.first, factors.second);
                return mlir::failure();
            }

            if (!factorsOnKX) {
                // pad input shape on H
                padValue = newKernel - KY;
                inputShape[Dims4D::Act::H] += padValue;
                inputShape[Dims4D::Act::H] = alignValUp(inputShape[Dims4D::Act::H], factors.first);
            } else {
                // pad input shape on W
                padValue = newKernel - KX;
                inputShape[Dims4D::Act::W] += padValue;
                inputShape[Dims4D::Act::W] = alignValUp(inputShape[Dims4D::Act::W], factors.first);
            }
        } else {
            const auto stridesData = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
            if ((stridesData[Dims4D::Strides::X] <= 1 && stridesData[Dims4D::Strides::Y] <= 1) ||
                inputShape[Dims4D::Act::C] != 1) {
                return mlir::failure();
            }

            const auto newLimit =
                    factorsOnKX ? KX / stridesData[Dims4D::Strides::X] : KY / stridesData[Dims4D::Strides::Y];
            factorsResult = factorsOnKX ? IE::getFactorsWithLimitation(KX, newLimit)
                                        : IE::getFactorsWithLimitation(KY, newLimit);
        }

        if (!isLegalOpToConvert(origOp, factorsResult, inputShape, strideOne)) {
            _log.trace("ConvOp '{0}' can't handle large kernels by reshape", origOp->getLoc());
            return mlir::failure();
        }
    }

    factors = factorsResult.value();
    // When stride is 1, input channel and new_stride of padded conv are the same.
    // So it should be smaller one, factors.first.
    auto factor = strideOne ? factors.first : factors.second;
    // Tranpose lowest-dimension continuous data to IC to make result data correctness, transpose IC-Width default
    auto perm = factorsOnKX ? SmallVector<uint32_t>{0, 3, 2, 1} : SmallVector<uint32_t>{0, 2, 1, 3};
    auto newInputChannel = factorsOnKX ? inputShape[Dims4D::Act::W] / factor : inputShape[Dims4D::Act::H] / factor;
    auto newInputWidth = factorsOnKX ? factor : inputShape[Dims4D::Act::W];
    auto newInputHeight = factorsOnKX ? inputShape[Dims4D::Act::H] : factor;

    auto newInputShape =
            Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], newInputChannel, newInputHeight, newInputWidth});

    mlir::Value finalOutputValue =
            strideOne ? getOutputFromStrideOne(rewriter, origOp, factors, newInputShape, inputShape, perm, factorsOnKX,
                                               padValue)
                      : getOutputFromLargeStride(rewriter, origOp, factors, newInputShape, perm, factorsOnKX);
    rewriter.replaceOp(origOp, finalOutputValue);

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), finalOutputValue.getLoc());

    return mlir::success();
}

//
// SliceLargePrimeKernelRewriter
//

class SliceLargePrimeKernelRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    SliceLargePrimeKernelRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("SliceLargePrimeKernelRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOpToSlice(IE::ConvolutionOp origOp, int64_t maxKernelSize) const;

private:
    Logger _log;
};

bool SliceLargePrimeKernelRewriter::isLegalOpToSlice(IE::ConvolutionOp origOp, int64_t maxKernelSize) const {
    auto activationRank = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getRank();
    auto filterRank = mlir::cast<vpux::NDTypeInterface>(origOp.getFilter().getType()).getRank();
    if (activationRank != 4 || filterRank != 4) {
        return false;
    }

    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    if (dilations[Dims4D::Dilation::X] > 1 || dilations[Dims4D::Dilation::Y] > 1) {
        return false;
    }

    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];
    const auto inputShape = getShape(origOp.getInput());
    const auto inputHeight = inputShape[Dims4D::Act::H];
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto inputChannel = inputShape[Dims4D::Act::C];
    const auto stridesData = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));

    if ((KX > maxKernelSize && KY == 1 && inputHeight == 1 && stridesData[Dims4D::Strides::X] > 1 &&
         inputChannel == 1) ||
        (KY > maxKernelSize && KX == 1 && inputWidth == 1 && stridesData[Dims4D::Strides::Y] > 1 &&
         inputChannel == 1)) {
        return true;
    }

    return false;
}

mlir::LogicalResult SliceLargePrimeKernelRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());
    auto inputShape = getShape(origOp.getInput()).raw();
    auto filterShape = getShape(origOp.getFilter()).raw();

    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    auto KX = filterShape[Dims4D::Filter::KX.ind()];
    auto KY = filterShape[Dims4D::Filter::KY.ind()];
    const auto maxKernelSize = config::getMaxKernelSize(origOp);

    const auto limit = std::min(maxKernelSize, VPU::NCEInvariant::MAX_STRIDE);
    auto factorsResult =
            KX > maxKernelSize ? IE::getFactorsWithLimitation(KX, limit) : IE::getFactorsWithLimitation(KY, limit);
    if (factorsResult.has_value()) {
        return mlir::failure();
    }

    if (!isLegalOpToSlice(origOp, maxKernelSize)) {
        return mlir::failure();
    }

    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    auto targetKernelSize = 1;
    auto numXSlices = 1;
    auto numYSlices = 1;

    if (KX > maxKernelSize) {
        targetKernelSize = KX / strides[Dims4D::Strides::X] * strides[Dims4D::Strides::X];
        numXSlices = checked_cast<int64_t>(llvm::alignTo(KX, targetKernelSize) / targetKernelSize);
    } else {
        targetKernelSize = KY / strides[Dims4D::Strides::Y] * strides[Dims4D::Strides::Y];
        numYSlices = checked_cast<int64_t>(llvm::alignTo(KY, targetKernelSize) / targetKernelSize);
    }

    // Slice filter
    SmallVector<mlir::Value> slicedFilters =
            sliceFilter(origOp.getFilter(), numXSlices, numYSlices, targetKernelSize, origOp->getLoc(), rewriter);
    _log.trace("[{0}] Split kernel into {1} small kernels {2} at {3}", getDebugName(), slicedFilters.size(),
               slicedFilters, origOp->getLoc());

    // Pad activation
    auto extendedActivation = getExtendedActivation(origOp, rewriter);
    _log.trace("[{0}] Pad on activation, new shape {1}, new activation {2} at {3}", getDebugName(),
               getShape(extendedActivation), extendedActivation, origOp->getLoc());

    // Create new sub graph and replace origOp
    rewriteSubGraph(origOp, slicedFilters, extendedActivation, numXSlices, numYSlices, targetKernelSize, rewriter,
                    _log);

    return mlir::success();
}

//
// ReshapeLargeConvWithGCDRewriter
//

class ReshapeLargeConvWithGCDRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ReshapeLargeConvWithGCDRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("ReshapeLargeConvWithGCDRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;
    bool isLegalOpToConvert(IE::ConvolutionOp origOp, ShapeRef inputShape, int64_t factor) const;

private:
    Logger _log;
};

bool ReshapeLargeConvWithGCDRewriter::isLegalOpToConvert(IE::ConvolutionOp origOp, ShapeRef inputShape,
                                                         int64_t factor) const {
    const auto filterShape = getShape(origOp.getFilter());
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (factor == 1 || filterShape[Dims4D::Filter::IC] != 1) {
        return false;
    }

    auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
        return false;
    }

    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    const auto inputHeight = inputShape[Dims4D::Act::H];
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto padBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    const auto padEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());

    if (KX > maxKernelSize && KY == 1 && inputHeight == 1) {
        return !(inputWidth % factor || padBeginAttr[Dims4D::PadsBegin::Left.ind()] % factor ||
                 padEndAttr[Dims4D::PadsEnd::Right.ind()] % factor);
    }

    if (KY > maxKernelSize && KX == 1 && inputWidth == 1) {
        return !(inputHeight % factor || padBeginAttr[Dims4D::PadsBegin::Top.ind()] % factor ||
                 padEndAttr[Dims4D::PadsEnd::Bottom.ind()] % factor);
    }

    return false;
}

mlir::LogicalResult ReshapeLargeConvWithGCDRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());
    auto inputShape = getShape(origOp.getInput());
    auto filterShape = getShape(origOp.getFilter());

    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return mlir::failure();
    }

    auto KX = filterShape[Dims4D::Filter::KX];
    auto KY = filterShape[Dims4D::Filter::KY];

    auto strides = parseIntArrayAttr<int64_t>(origOp.getStridesAttr());
    auto sx = strides[Dims4D::Strides::X.ind()];
    auto sy = strides[Dims4D::Strides::Y.ind()];

    const auto kernel = std::max(KX, KY);
    const auto stride = std::max(sx, sy);

    const auto factor = std::gcd(kernel, stride);
    VPUX_THROW_WHEN(factor == 0, "division/modulo by zero issue");

    if (!isLegalOpToConvert(origOp, inputShape, factor)) {
        return mlir::failure();
    }

    bool factorsOnKX = KX > 1 ? true : false;
    auto perm = factorsOnKX ? SmallVector<uint32_t>{0, 3, 2, 1} : SmallVector<uint32_t>{0, 2, 1, 3};
    auto newInputChannel = factorsOnKX ? inputShape[Dims4D::Act::W] / factor : inputShape[Dims4D::Act::H] / factor;
    auto newInputWidth = factorsOnKX ? factor : inputShape[Dims4D::Act::W];
    auto newInputHeight = factorsOnKX ? inputShape[Dims4D::Act::H] : factor;

    auto newInputShape =
            Shape(SmallVector<int64_t>{inputShape[Dims4D::Act::N], newInputChannel, newInputHeight, newInputWidth});

    auto newInputValue = getTransformValue(rewriter, newInputShape, origOp.getInput(), perm);
    auto newFilterShape = Shape(SmallVector<int64_t>{filterShape[Dims4D::Filter::OC], kernel / factor,
                                                     factorsOnKX ? 1 : factor, factorsOnKX ? factor : 1});

    mlir::Value filterValue = origOp.getFilter();
    auto newFilterValue = getTransformValue(rewriter, newFilterShape, filterValue, perm);

    SmallVector<int64_t> newStrides = {factorsOnKX ? 1 : strides[Dims4D::Strides::Y.ind()] / factor,
                                       factorsOnKX ? strides[Dims4D::Strides::X.ind()] / factor : 1};
    const auto newStridesAttr = getIntArrayAttr(rewriter.getContext(), newStrides);

    auto padBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto padEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    padBeginAttr[Dims4D::PadsBegin::Left.ind()] = padBeginAttr[Dims4D::PadsBegin::Left.ind()] / factor;
    padBeginAttr[Dims4D::PadsBegin::Top.ind()] = padBeginAttr[Dims4D::PadsBegin::Top.ind()] / factor;
    padEndAttr[Dims4D::PadsEnd::Right.ind()] = padEndAttr[Dims4D::PadsEnd::Right.ind()] / factor;
    padEndAttr[Dims4D::PadsEnd::Bottom.ind()] = padEndAttr[Dims4D::PadsEnd::Bottom.ind()] / factor;
    auto newBeginAttr = getIntArrayAttr(rewriter.getContext(), padBeginAttr);
    auto newEndAttr = getIntArrayAttr(rewriter.getContext(), padEndAttr);

    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp->getLoc(), newInputValue, newFilterValue, origOp.getBias(), newStridesAttr, newBeginAttr, newEndAttr,
            origOp.getDilationsAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    auto outReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_reshape_out"), newConvOp.getOutput(), nullptr,
                                           false, getIntArrayAttr(rewriter.getContext(), getShape(origOp.getOutput())));
    rewriter.replaceOp(origOp, outReshape.getOutput());

    _log.trace("[{0}] Replaced with '{1}'", getDebugName(), outReshape.getLoc());
    return mlir::success();
}

//
// ReshapeGlobalConvRewriter
//

class ReshapeGlobalConvRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ReshapeGlobalConvRewriter(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx, benefit), _log(log) {
        setDebugName("ReshapeGlobalConvRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

    bool isLegalOpToConvert(IE::ConvolutionOp origOp) const;

private:
    Logger _log;
};

bool ReshapeGlobalConvRewriter::isLegalOpToConvert(IE::ConvolutionOp origOp) const {
    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    if (inputShape.size() != 4 || filterShape.size() != 4) {
        return false;
    }

    const auto inputHeight = inputShape[Dims4D::Act::H];
    const auto inputWidth = inputShape[Dims4D::Act::W];
    const auto KY = filterShape[Dims4D::Filter::KY];
    const auto KX = filterShape[Dims4D::Filter::KX];

    if (inputHeight != KY || inputWidth != KX) {
        return false;
    }

    auto padBeginAttr = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
    auto padEndAttr = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
    if (padBeginAttr[Dims4D::PadsBegin::Left.ind()] != 0 || padEndAttr[Dims4D::PadsEnd::Right.ind()] != 0 ||
        padBeginAttr[Dims4D::PadsBegin::Top.ind()] != 0 || padEndAttr[Dims4D::PadsEnd::Bottom.ind()] != 0) {
        return false;
    }

    auto dilations = parseIntArrayAttr<int64_t>(origOp.getDilations());
    if (dilations[Dims4D::Dilation::X.ind()] > 1 || dilations[Dims4D::Dilation::Y.ind()] > 1) {
        return false;
    }

    const auto maxKernelSize = config::getMaxKernelSize(origOp);
    return KX > maxKernelSize || KY > maxKernelSize;
}

mlir::LogicalResult ReshapeGlobalConvRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                               mlir::PatternRewriter& rewriter) const {
    if (!isLegalOpToConvert(origOp)) {
        return mlir::failure();
    }
    const auto inputShape = getShape(origOp.getInput());
    const auto filterShape = getShape(origOp.getFilter());
    auto newInShape = SmallVector<int64_t>{
            inputShape[Dims4D::Act::N],
            inputShape[Dims4D::Act::C] * inputShape[Dims4D::Act::H] * inputShape[Dims4D::Act::W], 1, 1};
    auto newFilterShape = SmallVector<int64_t>{
            filterShape[Dims4D::Act::N],
            filterShape[Dims4D::Act::C] * filterShape[Dims4D::Act::H] * filterShape[Dims4D::Act::W], 1, 1};

    auto inputReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_reshape_in"), origOp.getInput(), nullptr,
                                           false, getIntArrayAttr(rewriter.getContext(), newInShape));
    auto filterReshape =
            rewriter.create<IE::ReshapeOp>(appendLoc(origOp->getLoc(), "_reshape_filter"), origOp.getFilter(), nullptr,
                                           false, getIntArrayAttr(rewriter.getContext(), newFilterShape));
    SmallVector<int64_t> newStrides = {1, 1};
    auto newConvOp = rewriter.create<IE::ConvolutionOp>(
            origOp->getLoc(), inputReshape.getOutput(), filterReshape.getOutput(), origOp.getBias(),
            getIntArrayAttr(rewriter.getContext(), newStrides), origOp.getPadsBeginAttr(), origOp.getPadsEndAttr(),
            origOp.getDilationsAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getStaticScaleAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, newConvOp.getOutput());

    return mlir::success();
}

//
// HandleLargeKernelsPass
//

class HandleLargeKernelsPass final : public IE::impl::HandleLargeKernelsBase<HandleLargeKernelsPass> {
public:
    explicit HandleLargeKernelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void HandleLargeKernelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet convPatterns(&ctx);
    convPatterns.add<PadLargeKernelRewriter>(&ctx, benefitLevels[0], _log);
    convPatterns.add<ReshapeGlobalConvRewriter>(&ctx, benefitLevels[0], _log);
    convPatterns.add<SliceLargePrimeKernelRewriter>(&ctx, benefitLevels[1], _log);
    convPatterns.add<ReshapeLargeConvRewriter>(&ctx, benefitLevels[2], _log);
    convPatterns.add<ReshapeLargeConvWithGCDRewriter>(&ctx, benefitLevels[3], _log);
    convPatterns.add<SliceLargeConvRewriter>(&ctx, benefitLevels[3], _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(convPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    // Note 1: Each execution of matchAndRewrite will only split Pooling Op into two
    // In some cases, it need continue splitting until all the Op are legal
    // So used GreedyRewriteConfig here
    // Note 2: The reason to distinguish GeneralPool and OverlappedPool is the influence of padding Attr.
    // The pad will have different implementation methods. To simplify and split it.
    mlir::RewritePatternSet poolingPatterns(&ctx);
    poolingPatterns.add<GeneralAvgPoolRewriter>(&ctx, _log);
    poolingPatterns.add<GeneralMaxPoolRewriter>(&ctx, _log);
    poolingPatterns.add<OverlappedMaxPoolRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(poolingPatterns), getDefaultGreedyRewriteConfig()))) {
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
