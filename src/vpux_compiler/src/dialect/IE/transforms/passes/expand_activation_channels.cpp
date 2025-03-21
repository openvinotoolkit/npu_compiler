//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;

//
// generalRewrite
//

// Max/Avg Pooling and Convolution Ops should be handled there
//
// opCreator - function, which should place back operation, which being proceed, with new expanded input
// calcOutputSliceOffset - function, calcualte output slice offset, it's different for Conv and per-channel ops
//

mlir::LogicalResult IE::generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                       FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator,
                                       FuncRef<SmallVector<int64_t>(mlir::Operation*, ShapeRef)> calcOutputSliceOffset,
                                       FuncRef<void()> autopadAttributeModifier, Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    const auto inPadsEnd = IE::calcPadsEnd(inputType, iface.getInputChannelAlignment());
    const auto outPadsEnd = IE::calcPadsEnd(outputType, iface.getOutputChannelAlignment());

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[Dims4D::Act::C] == 0 && outPadsEnd[Dims4D::Act::C] == 0) {
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp) &&
            !origOp->hasAttr(VPU::outChanAttrName)) {
            autopadAttributeModifier();
            return mlir::success();
        }
        return matchFailed(log, rewriter, origOp, "Both input and output channels are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput = IE::paddingChannel(origOp, rewriter, origOp->getOperand(0), inPadsEnd, Dims4D::Act::C.ind());
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, outPadsEnd[Dims4D::Act::C]);

    if (outPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        auto offsets = calcOutputSliceOffset(origOp, outPadsEnd);

        auto sliceOp =
                rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                         getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
        extendOpLoc(sliceOp, "slice_out");
    }

    return mlir::success();
}

//
// MaxPoolRewriter
//

mlir::LogicalResult IE::MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::MaxPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getKernelSize(),
                                              origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                              origOp.getRoundingType(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                              outChanBeforeAttr, origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, autopadModifier, _log.nest());
}

//
// ConvolutionRewriter
//

mlir::LogicalResult IE::ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto filterShape = getShape(origOp.getFilter());

        const auto newInputShape = getShape(expandedInput);
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];

        const auto paddedFilter = IE::padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;
        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp.getBias(), std::nullopt,
                                                                   ShapeRef(biasPadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::ConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.getStrides(),
                origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(), origOp.getPostOpAttr(),
                origOp.getClampAttr(), origOp.getStaticScaleAttr(), outChanBeforeAttr, origOp.getInputChannelsAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, autopadModifier, _log.nest());
}

//
// MatMulRewriter
//

// This Rewriter relies on the fact that MatMul will eventually be replaced with Convolution later in the pipeline.
// MatMul's dimensions will be mapped in the following way:
// Before (MatMul): Input1 - [_, _, Row1, Col1], Input2 - [_, _, Col1, Row2]
// After (Convolution): Input1 - [1, Col1, Row1, 1], Input2 - [Row2, Col1, 1, 1] (Note: input2 eventually becomes a
// filter)                            ^                         ^     ^
//                                    |                         |     |
//                               Input Channel         Output Channel |
//                                                              Input Channel

mlir::LogicalResult IE::MatMulRewriter::matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MatMul layer at '{1}'", getDebugName(), origOp->getLoc());

    auto getPadsForChannels = [origOp]() mutable {
        const auto input2Type = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType());
        auto input2Dims = input2Type.getShape().toValues();
        auto input2Rank = input2Type.getRank();
        VPUX_THROW_UNLESS(input2Rank >= 2, "Matrix must have rows and columns. Got rank {0}", input2Rank);

        auto inputChannelDim = input2Dims[origOp.getTransposeB() ? Dim(input2Rank - 1) : Dim(input2Rank - 2)];
        auto outputChannelDim = input2Dims[origOp.getTransposeB() ? Dim(input2Rank - 2) : Dim(input2Rank - 1)];

        auto alignedChannelOpInterface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(origOp.getOperation());
        auto inputChannelAlignment = alignedChannelOpInterface.getInputChannelAlignment();
        auto outputChannelAlignment = alignedChannelOpInterface.getOutputChannelAlignment();
        auto inputChannelPad = alignValUp(inputChannelDim, inputChannelAlignment) - inputChannelDim;
        auto outputChannelPad = alignValUp(outputChannelDim, outputChannelAlignment) - outputChannelDim;

        return std::make_pair(inputChannelPad, outputChannelPad);
    };

    auto expandDimension = [origOp, &rewriter](auto dataToExpand, auto dimToExpand, auto pad, auto rank) mutable {
        const Shape padsBegin(rank, 0);
        Shape padsEnd(rank, 0);
        padsEnd[dimToExpand] = pad;

        return rewriter.createOrFold<IE::ExpandOp>(origOp.getLoc(), dataToExpand,
                                                   getIntArrayAttr(rewriter, ArrayRef(padsBegin.raw())),
                                                   getIntArrayAttr(rewriter, ArrayRef(padsEnd.raw())));
    };

    auto expandInputs = [origOp, &expandDimension](auto inputChannelPad, auto outputChannelPad) mutable {
        const auto input1Type = mlir::cast<vpux::NDTypeInterface>(origOp.getInput1().getType());
        const auto input2Type = mlir::cast<vpux::NDTypeInterface>(origOp.getInput2().getType());

        auto input1Rank = checked_cast<size_t>(input1Type.getRank());
        auto input2Rank = checked_cast<size_t>(input2Type.getRank());

        mlir::Value expandedInput1 = origOp.getInput1();
        mlir::Value expandedInput2 = origOp.getInput2();

        if (inputChannelPad != 0) {
            expandedInput1 =
                    expandDimension(expandedInput1, origOp.getTransposeA() ? Dim(input1Rank - 2) : Dim(input1Rank - 1),
                                    inputChannelPad, input1Rank);
            expandedInput2 =
                    expandDimension(expandedInput2, origOp.getTransposeB() ? Dim(input2Rank - 1) : Dim(input2Rank - 2),
                                    inputChannelPad, input2Rank);
        }

        if (outputChannelPad != 0) {
            expandedInput2 =
                    expandDimension(expandedInput2, origOp.getTransposeB() ? Dim(input2Rank - 2) : Dim(input2Rank - 1),
                                    outputChannelPad, input2Rank);
        }

        return std::make_pair(expandedInput1, expandedInput2);
    };

    auto inferOutputType = [origOp](auto outputChannelPad) mutable {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        const auto outputRank = checked_cast<size_t>(outputType.getRank());

        const Shape outPadsBegin(outputRank, 0);
        Shape outPadsEnd(outputRank, 0);
        outPadsEnd[Dim(outputRank - 1)] = outputChannelPad;

        return outputType.pad(outPadsBegin, outPadsEnd);
    };

    auto sliceOutput = [origOp, &rewriter](auto opToSlice) mutable {
        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
        const auto outShape = outputType.getShape();
        const auto sliceOffsets = SmallVector<int64_t>(outputType.getRank(), 0);
        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, opToSlice->getResult(0),
                                                 getIntArrayAttr(rewriter, sliceOffsets),
                                                 getIntArrayAttr(rewriter, outShape));
    };

    auto [inputChannelPad, outputChannelPad] = getPadsForChannels();
    auto [expandedInput1, expandedInput2] = expandInputs(inputChannelPad, outputChannelPad);
    auto newOutputType = inferOutputType(outputChannelPad);

    auto newOp = rewriter.create<IE::MatMulOp>(origOp.getLoc(), newOutputType, expandedInput1, expandedInput2,
                                               origOp.getTransposeA(), origOp.getTransposeB(), origOp.getPostOpAttr());

    sliceOutput(newOp);

    return mlir::success();
}

//
// GroupConvolutionRewriter
//

mlir::LogicalResult IE::GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolutionOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto filterShape = getShape(origOp.getFilter());

        mlir::Value paddedFilter;

        if (outChanPadEnd == 0) {
            paddedFilter = origOp.getFilter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;

            paddedFilter = IE::paddingChannel(origOp, rewriter, origOp.getFilter(), std::move(filterPadsEnd),
                                              Dims4D::Filter::OC.ind());
        }

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases =
                        IE::paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind());
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);
        const auto newConvOutShape = newOutputType.getShape().toValues();

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::GroupConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, paddedBiases, origOp.getStrides(),
                origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                getIntAttr(getContext(), newConvOutShape[Dims4D::Act::C]), origOp.getPostOpAttr(),
                origOp.getClampAttr(), outChanBeforeAttr, origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, autopadModifier, _log.nest());
}

//
// InterpolateRewriter
//

mlir::LogicalResult IE::InterpolateRewriter::matchAndRewrite(IE::InterpolateOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Interpolate layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto sizesInput = origOp.getSizes();
        auto sizesAttr = origOp.getSizesAttrAttr();
        const auto calcModeAttr = origOp.getAttr().getShapeCalcMode();
        if (calcModeAttr != nullptr && calcModeAttr.getValue() == IE::InterpolateCalcMode::SIZES) {
            const auto inType = mlir::cast<NDTypeInterface>(origOp.getInput().getType());
            const auto axesVal =
                    IE::getInterpAxesVal(origOp.getLoc(), origOp.getAxes(), origOp.getAxesAttrAttr(), inType);

            SmallVector<int64_t> newSizesVal(axesVal.size());
            const auto outputShape = newOutputType.getShape();
            for (const auto idx : irange(axesVal.size())) {
                newSizesVal[idx] = outputShape[Dim(axesVal[idx])];
            }
            sizesAttr = getIntArrayAttr(origOp.getContext(), newSizesVal);
            sizesInput = nullptr;
        }

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::InterpolateOp>(
                origOp.getLoc(), newOutputType, expandedInput, sizesInput, origOp.getScales(), origOp.getAxes(),
                sizesAttr, origOp.getScalesAttrAttr(), origOp.getAxesAttrAttr(), origOp.getTileOffsetAttrAttr(),
                origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttrAttr(),
                outChanBeforeAttr);
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, autopadModifier, _log.nest());
}

//
// TransposedConvolutionRewriter
//

mlir::LogicalResult IE::TransposedConvolutionRewriter::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Transposed Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadEnd) -> mlir::Operation* {
        const auto newInputShape = getShape(expandedInput);
        const auto filterShape = getShape(origOp.getFilter());
        const auto inChanPadEnd = newInputShape[Dims4D::Act::C] - filterShape[Dims4D::Filter::IC];
        auto paddedFilter = IE::padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases =
                        IE::paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind());
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = outputType.pad(outPadBefore, outPadAfter);

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), outputType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::TransposedConvolutionOp>(
                origOp.getLoc(), newOutputType, expandedInput, paddedFilter, origOp.getOutputShape(), paddedBiases,
                origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                origOp.getOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), outChanBeforeAttr,
                origOp.getInputChannelsAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, autopadModifier, _log.nest());
}

//
// PadRewriter
//

mlir::LogicalResult IE::PadRewriter::matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Pad layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::PadOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getPadsBegin(),
                                          origOp.getPadsEnd(), origOp.getPadValue(), origOp.getPadsBeginAttrAttr(),
                                          origOp.getPadsEndAttrAttr(), origOp.getPadValueAttrAttr(),
                                          origOp.getModeAttr(), outChanBeforeAttr);
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, autopadModifier, _log.nest());
}

//
// AvgPoolRewriter
//

mlir::LogicalResult IE::AvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AvgPoolRewriter layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput, int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = origOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        return rewriter.create<IE::AvgPoolOp>(origOp.getLoc(), newOutputType, expandedInput, origOp.getKernelSize(),
                                              origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                              origOp.getRoundingType(), origOp.getExcludePads(), origOp.getPostOpAttr(),
                                              origOp.getClampAttr(), origOp.getStaticScaleAttr(), outChanBeforeAttr,
                                              origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, autopadModifier, _log.nest());
}
