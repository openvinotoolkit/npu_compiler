//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/config/constraints.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_LEGALIZEDILATEDCONVOLUTION
#define GEN_PASS_DEF_LEGALIZEDILATEDCONVOLUTION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t ADJUST_CONV_ALIGNMENT = 4;

struct SliceAndPaddingParameters {
    int64_t index;
    int64_t offsetX;
    int64_t sizeX;
    int64_t offsetY;
    int64_t sizeY;
    Shape padStart;
    Shape padEnd;
};

mlir::Value createNewOp(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                        ShapeRef padStart, ShapeRef padEnd, bool updatePad, bool clearPPE, StringRef locSuffix) {
    mlir::IRMapping mapper;
    mlir::Builder builder(origOp->getContext());
    mapper.map(origOp->getOperands(), operands);
    auto* newOp = rewriter.clone(*origOp, mapper);
    extendOpLoc(newOp, locSuffix);

    if (updatePad) {
        auto padBeginAttr =
                builder.getI64ArrayAttr({padStart[Dims4D::PadsBegin::Top], padStart[Dims4D::PadsBegin::Left]});
        auto padEndAttr = builder.getI64ArrayAttr({padEnd[Dims4D::PadsEnd::Bottom], padEnd[Dims4D::PadsEnd::Right]});
        VPUX_THROW_UNLESS(newOp->hasAttr("pads_end") && newOp->hasAttr("pads_begin"),
                          "operation does not have pad attribute");
        newOp->setAttr("pads_end", padEndAttr);
        newOp->setAttr("pads_begin", padBeginAttr);
    }

    if (clearPPE) {
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(newOp);
        VPUX_THROW_WHEN(layerWithPostOp == nullptr, "Cannot remove the post-op of a non-LayerWithPostOp operation: {0}",
                        newOp->getName());
        layerWithPostOp.clearPPE();
    }

    VPUX_THROW_UNLESS(newOp->hasAttr("dilations"), "operation does not have dilations attribute");
    auto dilationsAttr = builder.getI64ArrayAttr({1, 1});
    newOp->setAttr("dilations", dilationsAttr);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    return newOp->getResult(0);
}

// The func will return activation slice parmeter which included offset and size for both Height and width
// dimension. Also return the padding parameter for the new convolution
SmallVector<SliceAndPaddingParameters> getActivationSliceAndPaddingParameters(
        ArrayRef<int64_t> kernelOffsetsX, ArrayRef<int64_t> kernelSizesX, ArrayRef<int64_t> kernelOffsetsY,
        ArrayRef<int64_t> kernelSizesY, ShapeRef padStart, ShapeRef padEnd, ShapeRef inputShape, ShapeRef outputShape,
        ShapeRef dilations, ShapeRef strides) {
    SmallVector<SliceAndPaddingParameters> parameters;
    auto getParameters = [](int64_t start, int64_t end, int64_t padding, int64_t dimension) {
        int64_t padStart, padEnd, offset, size;
        start = start - padding;
        if (start < 0) {
            padStart = -start;
            start = 0;
        } else {
            padStart = 0;
        }

        end = end - padding;
        if (end > dimension) {
            padEnd = end - dimension;
            end = dimension;
        } else {
            padEnd = 0;
        }

        offset = start;
        size = end - start;
        return std::make_tuple(offset, size, padStart, padEnd);
    };

    for (size_t kxIndex = 0; kxIndex < kernelOffsetsX.size(); kxIndex++) {
        int64_t kx = kernelOffsetsX[kxIndex];
        int64_t kxSize = kernelSizesX[kxIndex];
        int64_t offsetX = 0, sizeX = 0;
        Shape newPadStart(padStart.size());
        Shape newPadEnd(padEnd.size());

        int64_t startW = kx * dilations[Dims4D::Dilation::X];
        int64_t expandedKernelX = (kxSize - 1) * dilations[Dims4D::Dilation::X] + 1;
        int64_t endW = outputShape[Dims4D::Act::W] * strides[Dims4D::Strides::X] + startW + expandedKernelX - 1;
        // in padding range, no need to compute, just continue
        if (startW >= inputShape[Dims4D::Act::W] + padStart[Dims4D::PadsBegin::Left]) {
            continue;
        }
        if (endW <= padStart[Dims4D::PadsBegin::Left]) {
            continue;
        }

        // 2. get activation slice parameter and padding parameter, for W
        std::tie(offsetX, sizeX, newPadStart[Dims4D::PadsBegin::Left], newPadEnd[Dims4D::PadsEnd::Right]) =
                getParameters(startW, endW, padStart[Dims4D::PadsBegin::Left], inputShape[Dims4D::Act::W]);

        for (size_t kyIndex = 0; kyIndex < kernelOffsetsY.size(); kyIndex++) {
            int64_t ky = kernelOffsetsY[kyIndex];
            int64_t kySize = kernelSizesY[kyIndex];
            int64_t offsetY = 0, sizeY = 0;

            int64_t startH = ky * dilations[Dims4D::Dilation::Y];
            int64_t expandedKernelY = (kySize - 1) * dilations[Dims4D::Dilation::Y] + 1;
            int64_t endH = outputShape[Dims4D::Act::H] * strides[Dims4D::Strides::Y] + startH + expandedKernelY - 1;
            // in padding range, no need to compute, just continue
            if (startH >= inputShape[Dims4D::Act::H] + padStart[Dims4D::PadsBegin::Top]) {
                continue;
            }
            if (endH <= padStart[Dims4D::PadsBegin::Top]) {
                continue;
            }

            // 2. get activation slice parameter and padding parameter, for H
            std::tie(offsetY, sizeY, newPadStart[Dims4D::PadsBegin::Top], newPadEnd[Dims4D::PadsEnd::Bottom]) =
                    getParameters(startH, endH, padStart[Dims4D::PadsBegin::Top], inputShape[Dims4D::Act::H]);

            int64_t index = kxIndex * kernelOffsetsY.size() + kyIndex;
            parameters.push_back(
                    SliceAndPaddingParameters{index, offsetX, sizeX, offsetY, sizeY, newPadStart, newPadEnd});
        }
    }
    return parameters;
}

/*
 Optimization for dilated convolution or group convolution where expanded kernels
 exceed the HW limitation (e.g., 11x11).

 1. If the dilated kernel fits into the hardware limit (e.g. 11x11), we simply expand the weights
    by inserting zeros (ExpandDilatedOp) and run a single convolution. This is generally the most efficient approach.

 2. If the dilated kernel is too large, we use a hybrid strategy:
    We split the original kernel into smaller "sub-kernels".
    Each sub-kernel is chosen such that when dilated, it fits within the hardware limit.

    Example: Kernel 4x4, Dilation 4, MaxKernel 11.
    Expanded size: (4-1)*4 + 1 = 13 > 11.
    Max SubKernel: (3-1)*4 + 1 = 9 <= 11.

    Instead of maximizing the chunk size (which would result in 3x3, 3x1, 1x3, 1x1),
    we use a balanced splitting strategy to distribute work evenly.
    So we split into 2x2 chunks.

        Convert Dilated Convolution
          Input        Filter (4x4, dil=4)
        [N, C, H, W]   [OC, IC, 4, 4]
             \             /
           Dilated Convolution
                    |
                  Output
    =>
                  Filter (4x4)
                      |
                 Split Filter
           (Balanced split, e.g. 2x2)
        /          |           |          \
    SubFilter1  SubFilter2  SubFilter3  SubFilter4
    (2x2,dil=4) (2x2,dil=4) (2x2,dil=4) (2x2,dil=4)
        |           |           |           |
      Input       Input       Input       Input
        |           |           |           |
    Slice Input Slice Input Slice Input Slice Input
        |           |           |           |
    Convolution Convolution Convolution Convolution
         \          |           |          /
                    AddOp (Chain)
                        |
                      Output
*/

//
// ConvGeneralRewriter
//

template <class ConcreteOp>
class ConvGeneralRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ConvGeneralRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
bool isExpandedSubKernelFitIntoCMX(ConcreteOp origOp, ShapeRef filterShape, ShapeRef dilations, int64_t subKernelSizeX,
                                   int64_t subKernelSizeY, Logger log) {
    const auto filterType = mlir::cast<NDTypeInterface>(origOp.getFilter().getType());
    const auto outputType = mlir::cast<NDTypeInterface>(origOp->getResult(0).getType());
    SmallVector<int64_t> expandedFilterShapeVec(filterShape.raw().begin(), filterShape.raw().end());
    // Calculate the shape of the expanded sub-kernel
    const auto expandedSubKernelX = (subKernelSizeX - 1) * dilations[Dims4D::Dilation::X] + 1;
    const auto expandedSubKernelY = (subKernelSizeY - 1) * dilations[Dims4D::Dilation::Y] + 1;

    // max kernel size after being split and applying ExpandDilatedOp
    expandedFilterShapeVec[Dims4D::Filter::KX.ind()] = expandedSubKernelX;
    expandedFilterShapeVec[Dims4D::Filter::KY.ind()] = expandedSubKernelY;
    // min OC after being tiled
    expandedFilterShapeVec[Dims4D::Filter::OC.ind()] = VPU::NCEInvariant::getAlignment(outputType.getElementType());

    // max IC when groupconv converted to conv
    if (auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(origOp.getOperation())) {
        if (groupConvOp.getGroups().has_value()) {
            expandedFilterShapeVec[Dims4D::Filter::IC.ind()] *= groupConvOp.getGroups().value();
        }
    }

    auto maxExpandedSubKernelShape = Shape(expandedFilterShapeVec);
    const auto expandedFilterType = filterType.changeShape(maxExpandedSubKernelShape);

    // Calculate memory requirement for the expanded weights
    SmallVector<Byte> buffers = {expandedFilterType.getTotalAllocSize()};
    auto expandedFilterSize =
            vpux::VPU::calculateAlignedBuffersMemoryRequirement(config::getArch(origOp), buffers).count();
    const auto cmxSize = vpux::VPU::getTotalCMXSize(origOp).count();

    if (expandedFilterSize > cmxSize) {
        log.trace("Force split to 1x1 because expanded sub-filter size {0} exceeds CMX size {1}", expandedFilterSize,
                  cmxSize);
        return false;
    }

    return true;
}

template <class ConcreteOp>
mlir::LogicalResult ConvGeneralRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got dilated '{0}'layer at '{1}'", origOp->getName(), origOp->getLoc());
    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    const auto filterShape = getShape(origOp.getFilter());

    const auto kernelY = filterShape[Dims4D::Filter::KY];
    const auto kernelX = filterShape[Dims4D::Filter::KX];
    const auto expandedKernelX = (kernelX - 1) * dilations[Dims4D::Dilation::X] + 1;
    const auto expandedKernelY = (kernelY - 1) * dilations[Dims4D::Dilation::Y] + 1;
    const auto maxKernelSize = config::getNPUConstraints(origOp->getContext()).maxKernelSize;

    const auto isKernelSupportedByNCE = (expandedKernelX <= maxKernelSize) && (expandedKernelY <= maxKernelSize);
    const auto isFilterConstant = mlir::succeeded(IE::getConstParentOp(origOp.getFilter()));

    _log.trace("Split dilated '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto inputShape = getShape(origOp->getOperand(0));
    const auto outputShape = getShape(origOp->getResult(0));
    mlir::MLIRContext* ctx = origOp->getContext();

    // 1. If the dilated kernel fits into the hardware limit, use ExpandDilatedOp and single convolution.
    if (isFilterConstant && isKernelSupportedByNCE) {
        _log.trace("Expand dilated '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
        auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(takeOpLoc(origOp, "dilatedfilter_in"),
                                                                  origOp.getFilter(), origOp.getDilations());
        const auto padsBegin = parseIntArrayAttr<int64_t>(origOp.getPadsBegin());
        const auto padsEnd = parseIntArrayAttr<int64_t>(origOp.getPadsEnd());
        auto newOp = createNewOp(rewriter, origOp, {origOp.getInput(), dilatedFilter.getResult(), origOp.getBias()},
                                 ShapeRef(padsBegin), ShapeRef(padsEnd), false, false, "expanded");
        rewriter.replaceOp(origOp, newOp);
        return mlir::success();
    }

    // 2. If the dilated kernel is too large, we use a hybrid strategy.
    //  Calculate the MAXIMUM sub-kernel size that fits into maxKernelSize after dilation.
    //  Formula: (subK - 1) * dilation + 1 <= maxKernelSize
    //        => subK - 1 <= (maxKernelSize - 1) / dilation
    //        => subK <= (maxKernelSize - 1) / dilation + 1
    int64_t maxSubKernelSizeX = (maxKernelSize - 1) / dilations[Dims4D::Dilation::X] + 1;
    int64_t maxSubKernelSizeY = (maxKernelSize - 1) / dilations[Dims4D::Dilation::Y] + 1;

    // Ensure at least 1x1
    maxSubKernelSizeX = std::max<int64_t>(maxSubKernelSizeX, 1);
    maxSubKernelSizeY = std::max<int64_t>(maxSubKernelSizeY, 1);

    // Calculate the balanced sub-kernel size to distribute the work more evenly.
    const auto numSplitsX = (kernelX + maxSubKernelSizeX - 1) / maxSubKernelSizeX;
    const auto numSplitsY = (kernelY + maxSubKernelSizeY - 1) / maxSubKernelSizeY;

    int64_t subKernelSizeX = (kernelX + numSplitsX - 1) / numSplitsX;
    int64_t subKernelSizeY = (kernelY + numSplitsY - 1) / numSplitsY;

    // Check if the expanded sub-kernel fits into CMX.
    // Even if the sub-kernel dimensions fit into the hardware kernel limit (e.g. 11x11),
    // the total memory required for the expanded weights (even with sparse tensor) might exceed CMX size.
    // If it exceeds CMX, we must force a 1x1 split to minimize memory usage.
    // TODO(E#195803): Split filter when it exceeds CMX even if tiled and sparse tensor applied. Then remove current
    // logic codes.
    const bool isFitIntoCMX =
            isExpandedSubKernelFitIntoCMX(origOp, filterShape, dilations, subKernelSizeX, subKernelSizeY, _log);

    // If filter is not constant, we cannot use ExpandDilatedOp (runtime expansion not supported easily).
    // So we must fallback to 1x1 split (subKernelSize = 1), which effectively implements the "Split" strategy.
    if (!isFilterConstant || !isFitIntoCMX) {
        subKernelSizeX = 1;
        subKernelSizeY = 1;
    }

    // Optimization for H=1 or W=1:
    // If the input has H=1 or W=1, subsequent optimizations (like AdjustConvInputShape) might require
    // the kernel size to be aligned (e.g. divisible by 4) to work correctly.
    // If the expanded sub-kernel size is not divisible by 4, we fallback to 1x1 split to avoid breaking those
    // optimizations.
    if (inputShape[Dims4D::Act::H] == 1 || inputShape[Dims4D::Act::W] == 1) {
        const auto expandedSubKernelX = (subKernelSizeX - 1) * dilations[Dims4D::Dilation::X] + 1;
        if (subKernelSizeX > 1 && expandedSubKernelX % ADJUST_CONV_ALIGNMENT != 0) {
            subKernelSizeX = 1;
        }
        const auto expandedSubKernelY = (subKernelSizeY - 1) * dilations[Dims4D::Dilation::Y] + 1;
        if (subKernelSizeY > 1 && expandedSubKernelY % ADJUST_CONV_ALIGNMENT != 0) {
            subKernelSizeY = 1;
        }
    }

    SmallVector<int64_t> kernelOffsetsX, kernelSizesX;
    for (int64_t kx = 0; kx < kernelX; kx += subKernelSizeX) {
        kernelOffsetsX.push_back(kx);
        kernelSizesX.push_back(std::min(subKernelSizeX, kernelX - kx));
    }

    SmallVector<int64_t> kernelOffsetsY, kernelSizesY;
    for (int64_t ky = 0; ky < kernelY; ky += subKernelSizeY) {
        kernelOffsetsY.push_back(ky);
        kernelSizesY.push_back(std::min(subKernelSizeY, kernelY - ky));
    }

    // 1. Slice Filters and Expand
    SmallVector<mlir::Value> slicedFilters;
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];

    for (size_t kxI = 0; kxI < kernelOffsetsX.size(); ++kxI) {
        for (size_t kyI = 0; kyI < kernelOffsetsY.size(); ++kyI) {
            int64_t kx = kernelOffsetsX[kxI];
            int64_t ky = kernelOffsetsY[kyI];
            int64_t sizeX = kernelSizesX[kxI];
            int64_t sizeY = kernelSizesY[kyI];

            Shape offsets(filterShape.size());
            offsets[Dims4D::Filter::KX] = kx;
            offsets[Dims4D::Filter::KY] = ky;
            SmallVector<int64_t> sliceShape{OC, IC, sizeY, sizeX};

            auto slice =
                    rewriter.create<IE::SliceOp>(takeOpLoc(origOp, "filter_slice_{0}_{1}", kx, ky), origOp.getFilter(),
                                                 getIntArrayAttr(ctx, offsets.raw()), getIntArrayAttr(ctx, sliceShape));

            if (sizeX == 1 && sizeY == 1) {
                slicedFilters.push_back(slice);
            } else {
                auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(
                        takeOpLoc(origOp, "dilatedfilter_{0}_{1}", kx, ky), slice, origOp.getDilations());
                slicedFilters.push_back(dilatedFilter);
            }
        }
    }

    // 2. get activation slice parameters and padding parameters
    auto activationSliceAndPaddingParameters =
            getActivationSliceAndPaddingParameters(kernelOffsetsX, kernelSizesX, kernelOffsetsY, kernelSizesY, padStart,
                                                   padEnd, inputShape, outputShape, dilations, strides);
    VPUX_THROW_UNLESS(activationSliceAndPaddingParameters.size() > 0, "no any activation slice");

    // 3. slice activation and create new convolution
    SmallVector<mlir::Value> newConvs;
    bool biasAdded = false;
    mlir::Value zeroBias;
    if (origOp.getBias() != nullptr) {
        biasAdded = true;
        const auto zeroType = mlir::RankedTensorType::get(
                getShape(origOp.getBias()).raw(),
                mlir::cast<NDTypeInterface>(origOp->getOperand(0).getType()).getElementType());
        zeroBias = Const::createZerosConst(rewriter, origOp->getLoc(), zeroType);
    }
    for (auto parameter : activationSliceAndPaddingParameters) {
        Shape offsets(inputShape.size());
        offsets[Dims4D::Act::W] = parameter.offsetX;
        offsets[Dims4D::Act::H] = parameter.offsetY;
        SmallVector<int64_t> sliceShape{inputShape[Dims4D::Act::N], inputShape[Dims4D::Act::C], parameter.sizeY,
                                        parameter.sizeX};
        const std::string locSuffixBase =
                llvm::formatv("slice_in_{0}_{1}_{2}_{3}_{4}", parameter.index, parameter.offsetY, parameter.offsetX,
                              parameter.sizeY, parameter.sizeX)
                        .str();
        auto sliceActLoc = takeOpLoc(origOp, locSuffixBase);
        auto slicedActivation =
                rewriter.create<IE::SliceOp>(sliceActLoc, origOp->getOperand(0), getIntArrayAttr(ctx, offsets.raw()),
                                             getIntArrayAttr(ctx, sliceShape));

        SmallVector<mlir::Value> operands;
        if (origOp.getBias() != nullptr) {
            operands = {slicedActivation, slicedFilters[parameter.index], biasAdded ? origOp.getBias() : zeroBias};
            biasAdded = false;
        } else {
            operands = {slicedActivation, slicedFilters[parameter.index]};
        }
        const auto locSuffix = "conv_" + locSuffixBase;
        newConvs.push_back(createNewOp(rewriter, origOp, operands, parameter.padStart, parameter.padEnd, true,
                                       IE::hasPPE(origOp), locSuffix));
    }

    // 4. add the new convolution one by one
    if (newConvs.empty()) {
        return matchFailed(rewriter, origOp, "no any new conv created.");
    }

    if (newConvs.size() > 1) {
        const auto broadcastType =
                vpux::IE::AutoBroadcastTypeAttr::get(origOp->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
        mlir::Value add = newConvs.front();
        for (size_t i = 1; i < newConvs.size(); i++) {
            const auto isLast = (i == newConvs.size() - 1);
            add = rewriter.create<IE::AddOp>(takeOpLoc(origOp, "add_{0}", i), add, newConvs[i], broadcastType,
                                             isLast ? origOp.getPostOpAttr() : nullptr,
                                             isLast ? origOp.getClampAttr() : nullptr, origOp.getOutputPaddingAttr(),
                                             origOp.getInputPaddingAttr())
                          ->getResult(0);
        }
        rewriter.replaceOp(origOp, add);

    } else {
        auto conv = newConvs.front().getDefiningOp();
        auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(conv);
        if (const auto postOp = origOp.getPostOpAttr()) {
            VPUX_THROW_WHEN(layerWithPostOp == nullptr, "Cannot bind post-op to non-LayerWithPostOp operation: {0}",
                            conv->getName());
            layerWithPostOp.setPostOpAttr(postOp);
        }
        if (const auto clamp = origOp.getClampAttr()) {
            VPUX_THROW_WHEN(layerWithPostOp == nullptr, "Cannot bind clamp to non-LayerWithPostOp operation: {0}",
                            conv->getName());
            layerWithPostOp.setClampAttr(clamp);
        }
        rewriter.replaceOp(origOp, conv->getResult(0));
    }

    return mlir::success();
}

class LegalizeDilatedConvolutionPass final :
        public IE::impl::LegalizeDilatedConvolutionBase<LegalizeDilatedConvolutionPass> {
public:
    explicit LegalizeDilatedConvolutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

bool isLegalGroupConvOpImpl(IE::GroupConvolutionOp op, bool enableDilatedGroupConv, Logger log) {
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());

    if (dilations[Dims4D::Dilation::X.ind()] == 1 && dilations[Dims4D::Dilation::Y.ind()] == 1) {
        return true;
    }
    if (!enableDilatedGroupConv) {
        return false;
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };

    auto seOp = mlir::dyn_cast<IE::SEOpInterface>(op.getOperation());
    if (!seOp) {
        log.trace("Group convolution with dilation is not supported.");
        return false;
    }
    return seOp.isSupported(logCb);
}

bool isLegalConvOp(IE::ConvolutionOp op) {
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    return dilations[Dims4D::Dilation::X.ind()] == 1 && dilations[Dims4D::Dilation::Y.ind()] == 1;
}

void LegalizeDilatedConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();
    const auto func = getOperation();
    const auto moduleOp = getModuleOp(func);
    const auto hasEnableExperimentalSEPtrsOps = config::hasEnableExperimentalSEPtrsOperations(moduleOp);

    auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp op) {
        return isLegalGroupConvOpImpl(op, hasEnableExperimentalSEPtrsOps, _log);
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>(isLegalGroupConvOp);
    target.addDynamicallyLegalOp<IE::ConvolutionOp>(&isLegalConvOp);

    target.addLegalOp<IE::ExpandDilatedOp>();
    target.addLegalOp<IE::ConcatOp>();
    target.addLegalOp<IE::SliceOp>();
    target.addLegalOp<IE::AddOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ConvGeneralRewriter<IE::GroupConvolutionOp>>(&ctx, _log);
    patterns.add<ConvGeneralRewriter<IE::ConvolutionOp>>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLegalizeDilatedConvolutionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeDilatedConvolutionPass(Logger log) {
    return std::make_unique<LegalizeDilatedConvolutionPass>(log);
}
