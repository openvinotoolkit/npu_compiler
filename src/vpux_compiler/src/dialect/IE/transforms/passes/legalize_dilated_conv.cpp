//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/max_kernel_size_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
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

struct SliceAndPaddingParameters {
    int64_t index;
    int64_t offsetX;
    int64_t sizeX;
    int64_t offsetY;
    int64_t sizeY;
    Shape padStart;
    Shape padEnd;
};

Shape getSlicedShape(int64_t dilationX, int64_t dilationY, int64_t kernelX, int64_t kernelY) {
    Shape slicedShape(2);
    slicedShape[Dims4D::Kernel::X] = dilationX > 1 ? kernelX : 1;
    slicedShape[Dims4D::Kernel::Y] = dilationY > 1 ? kernelY : 1;
    return slicedShape;
}

mlir::Value createNewOp(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, ArrayRef<mlir::Value> operands,
                        ShapeRef padStart, ShapeRef padEnd, bool updatePad, bool removePostOp, StringRef locSuffix) {
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

    if (removePostOp) {
        VPUX_THROW_UNLESS(newOp->hasAttr("post_op"), "operation does not have post op attribute");
        newOp->removeAttr("post_op");
    }

    VPUX_THROW_UNLESS(newOp->hasAttr("dilations"), "operation does not have dilations attribute");
    auto dilationsAttr = builder.getI64ArrayAttr({1, 1});
    newOp->setAttr("dilations", dilationsAttr);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);

    return newOp->getResult(0);
}

SmallVector<mlir::Value> getSlicedFilters(mlir::PatternRewriter& rewriter, mlir::Operation* origOp, mlir::Value input,
                                          ShapeRef slicedShape, ShapeRef filterShape) {
    SmallVector<mlir::Value> slicedFilters;
    const auto IC = filterShape[Dims4D::Filter::IC];
    const auto OC = filterShape[Dims4D::Filter::OC];
    const auto kernelY = filterShape[Dims4D::Filter::KY];
    const auto kernelX = filterShape[Dims4D::Filter::KX];
    mlir::MLIRContext* ctx = origOp->getContext();

    for (int64_t kx = 0; kx < slicedShape[Dims4D::Kernel::X]; kx++) {
        for (int64_t ky = 0; ky < slicedShape[Dims4D::Kernel::Y]; ky++) {
            Shape offsets(filterShape.size());
            offsets[Dims4D::Filter::KX] = kx;
            offsets[Dims4D::Filter::KY] = ky;
            SmallVector<int64_t> sliceShape{OC, IC, kernelY / slicedShape[Dims4D::Kernel::Y],
                                            kernelX / slicedShape[Dims4D::Kernel::X]};
            auto slice = rewriter.create<IE::SliceOp>(takeOpLoc(origOp, StringLiteral("filter_slice_{0}_{1}"), kx, ky),
                                                      input, getIntArrayAttr(ctx, offsets.raw()),
                                                      getIntArrayAttr(ctx, sliceShape));
            slicedFilters.push_back(slice);
        }
    }

    return slicedFilters;
}

// The func will return activation slice parmeter which included offset and size for both Height and width
// dimension. Also return the padding parameter for the new convolution
SmallVector<SliceAndPaddingParameters> getActivationSliceAndPaddingParameters(ShapeRef slicedShape, ShapeRef padStart,
                                                                              ShapeRef padEnd, ShapeRef inputShape,
                                                                              ShapeRef outputShape, ShapeRef dilations,
                                                                              ShapeRef strides) {
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

    for (int64_t kx = 0; kx < slicedShape[Dims4D::Kernel::X]; kx++) {
        int64_t offsetX = 0, sizeX = 0;
        Shape newPadStart(padStart.size());
        Shape newPadEnd(padEnd.size());

        if (slicedShape[Dims4D::Kernel::X] == 1) {
            // no slice, using the original parameter.
            offsetX = 0;
            sizeX = inputShape[Dims4D::Act::W];
            newPadStart[Dims4D::PadsBegin::Left] = padStart[Dims4D::PadsBegin::Left];
            newPadEnd[Dims4D::PadsEnd::Right] = padEnd[Dims4D::PadsEnd::Right];
        } else {
            int64_t startW = kx * dilations[Dims4D::Dilation::X];
            int64_t endW = outputShape[Dims4D::Act::W] * strides[Dims4D::Strides::X] + startW;
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
        }

        for (int64_t ky = 0; ky < slicedShape[Dims4D::Kernel::Y]; ky++) {
            int64_t offsetY = 0, sizeY = 0;
            if (slicedShape[Dims4D::Kernel::Y] == 1) {
                // no slice, using the original parameter.
                offsetY = 0;
                sizeY = inputShape[Dims4D::Act::H];
                newPadStart[Dims4D::PadsBegin::Top] = padStart[Dims4D::PadsBegin::Top];
                newPadEnd[Dims4D::PadsEnd::Bottom] = padEnd[Dims4D::PadsEnd::Bottom];
            } else {
                int64_t startH = ky * dilations[Dims4D::Dilation::Y];
                int64_t endH = outputShape[Dims4D::Act::H] * strides[Dims4D::Strides::Y] + startH;
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
            }

            int64_t index = kx * slicedShape[Dims4D::Kernel::Y] + ky;
            parameters.push_back(
                    SliceAndPaddingParameters{index, offsetX, sizeX, offsetY, sizeY, newPadStart, newPadEnd});
        }
    }
    return parameters;
}

// here is the optimization for dilated convolution or group convolution that expanded kernels
// are bigger than HW limitation(11x11). For example if kernel is 2x2, dilation is 15x15,
// we split the filter to 4 1x1 filter, then use IE.add to add each of them one by one.
// The detail step is :
// 1. slice the filter, from 2x2 to 4 1x1
// 2. get activation slice parameters(offset and size) and padding parameters of the small convolution
// 3. slice activation and create new convolution
// 4. add the new convolution one by one
//
//      [act]        [w]                     [act]       [w]         [act]        [w]
//        |           |           to           |          |            |           |
//       -(dilatedConv)-                    (slice)    (slice)      (slice)     (slice)
//                                             |          |            |           |
//                                               -(conv)-                -(conv)-
//                                                   |                       |
//                                                     ---- (eltwise) -----

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
mlir::LogicalResult ConvGeneralRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got dilated '{0}'layer at '{1}'", origOp->getName(), origOp->getLoc());
    const auto dilations = Shape(parseIntArrayAttr<int64_t>(origOp.getDilations()));
    const auto filterShape = getShape(origOp.getFilter());

    const auto kernelY = filterShape[Dims4D::Filter::KY];
    const auto kernelX = filterShape[Dims4D::Filter::KX];
    const auto expandKernelX = (kernelX - 1) * dilations[Dims4D::Dilation::X] + 1;
    const auto expandKernelY = (kernelY - 1) * dilations[Dims4D::Dilation::Y] + 1;
    const auto maxKernelSize = VPU::getMaxKernelSize(origOp);

    const auto isKernelSupportedByNCE = expandKernelX <= maxKernelSize && expandKernelY <= maxKernelSize;
    const auto isFilterConstant = mlir::succeeded(IE::getConstParentOp(origOp.getFilter()));

    // The dilation can be resolved by expanding the filter with zeroes by the dilation factor
    // This can only be done if the expanded filter still satisfies the kernel size limitation of the NCE, otherwise it
    // would execute on SHAVE and lose performance
    // Additionally, the filter has to be a constant to ensure that the IE.ExpandDilated operation gets folded into a
    // constant transformation since there is no full lowering path for this operation
    if (isKernelSupportedByNCE && isFilterConstant) {
        _log.trace("Expand dilated '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
        auto dilatedFilter = rewriter.create<IE::ExpandDilatedOp>(takeOpLoc(origOp, "dilatedfilter_in"),
                                                                  origOp.getFilter(), origOp.getDilations());
        auto newOp = createNewOp(rewriter, origOp, {origOp.getInput(), dilatedFilter.getResult(), origOp.getBias()},
                                 Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin())),
                                 Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd())), false, false, "expanded");
        rewriter.replaceOp(origOp, newOp);
        return mlir::success();
    }

    _log.trace("Split dilated '{0}' layer at '{1}'", origOp->getName(), origOp->getLoc());
    const auto padStart = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsBegin()));
    const auto padEnd = Shape(parseIntArrayAttr<int64_t>(origOp.getPadsEnd()));
    const auto strides = Shape(parseIntArrayAttr<int64_t>(origOp.getStrides()));
    const auto inputShape = getShape(origOp->getOperand(0));
    const auto outputShape = getShape(origOp->getResult(0));
    mlir::MLIRContext* ctx = origOp->getContext();

    auto slicedShape = getSlicedShape(dilations[Dims4D::Dilation::X], dilations[Dims4D::Dilation::Y], kernelX, kernelY);

    // 1. slice Filters
    SmallVector<mlir::Value> slicedFilters =
            getSlicedFilters(rewriter, origOp, origOp.getFilter(), slicedShape, filterShape);

    // 2. get activation slice parameters and padding parameters
    auto activationSliceAndPaddingParameters = getActivationSliceAndPaddingParameters(
            slicedShape, padStart, padEnd, inputShape, outputShape, dilations, strides);
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
                                       origOp.getPostOpAttr() != nullptr, locSuffix));
    }

    // 4. add the new convolution one by one
    if (newConvs.empty()) {
        return matchFailed(rewriter, origOp, "no any new conv created.");
    }

    if (newConvs.size() > 1) {
        const auto broadcastType =
                vpux::IE::AutoBroadcastTypeAttr::get(origOp->getContext(), IE::AutoBroadcastType::NONE_OR_EXPLICIT);
        mlir::Value add = newConvs.front();
        for (size_t i = 1; i < newConvs.size(); i++)
            add = rewriter.create<IE::AddOp>(takeOpLoc(origOp, StringLiteral("add_{0}"), i), add, newConvs[i],
                                             broadcastType, (i == newConvs.size() - 1) ? origOp.getClampAttr(),
                                             origOp.getPostOpAttr() : nullptr, nullptr, origOp.getOutputChannelsAttr(),
                                             origOp.getInputChannelsAttr())
                          ->getResult(0);
        rewriter.replaceOp(origOp, add);

    } else {
        auto conv = newConvs.front().getDefiningOp();
        if (origOp.getPostOpAttr() != nullptr) {
            conv->setAttr("post_op", origOp.getPostOpAttr());
        }
        rewriter.replaceOp(origOp, conv->getResult(0));
    }

    return mlir::success();
}

class LegalizeDilatedConvolutionPass final :
        public IE::impl::LegalizeDilatedConvolutionBase<LegalizeDilatedConvolutionPass> {
public:
    explicit LegalizeDilatedConvolutionPass(bool enableSEPDilatedGroupConv, Logger log)
            : _enableSEPDilatedGroupConv{enableSEPDilatedGroupConv} {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    bool _enableSEPDilatedGroupConv;
};

mlir::LogicalResult LegalizeDilatedConvolutionPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    // LIT test override
    if (enableSEPDilatedGroupConv.hasValue()) {
        _enableSEPDilatedGroupConv = enableSEPDilatedGroupConv.getValue();
    }

    return mlir::success();
}

bool isLegalGroupConvOpImpl(IE::GroupConvolutionOp op, bool enableDilatedGroupConv, Logger log) {
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    if ((dilations[Dims4D::Dilation::X.ind()] == 1 && dilations[Dims4D::Dilation::Y.ind()] == 1)) {
        return true;
    }
    auto isDilationSupported = enableDilatedGroupConv ? VPU::isSupportedSEPDilatedConv(op, logCb, /*checkLayout=*/false,
                                                                                       /*checkChannelAlignment=*/false)
                                                      : false;

    return isDilationSupported;
}

bool isLegalConvOp(IE::ConvolutionOp op) {
    const auto dilations = parseIntArrayAttr<int64_t>(op.getDilations());
    return dilations[Dims4D::Dilation::X.ind()] == 1 && dilations[Dims4D::Dilation::Y.ind()] == 1;
}

void LegalizeDilatedConvolutionPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto isLegalGroupConvOp = [&](IE::GroupConvolutionOp op) {
        return isLegalGroupConvOpImpl(op, _enableSEPDilatedGroupConv, _log);
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

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createLegalizeDilatedConvolutionPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createLegalizeDilatedConvolutionPass(bool enableDilatedGroupConv, Logger log) {
    return std::make_unique<LegalizeDilatedConvolutionPass>(enableDilatedGroupConv, log);
}
