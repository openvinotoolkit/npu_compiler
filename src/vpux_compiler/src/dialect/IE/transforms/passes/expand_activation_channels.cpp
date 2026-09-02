//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes/expand_activation_channels.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/utils/interpolate_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"
#include "vpux/utils/core/range.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_EXPANDACTIVATIONCHANNELS
#define GEN_PASS_DEF_EXPANDACTIVATIONCHANNELS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

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
                                       FuncRef<mlir::Operation*(mlir::Value, int64_t, int64_t)> opCreator,
                                       FuncRef<SmallVector<int64_t>(mlir::Operation*, ShapeRef)> calcOutputSliceOffset,
                                       Logger log) {
    auto* ctx = origOp->getContext();

    auto iface = mlir::cast<IE::AlignedChannelsOpInterface>(origOp);

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());

    const auto inPadsEnd = IE::calcPadsEnd(inputType, iface.getInputChannelAlignment());
    const auto outPadsEnd = IE::calcPadsEnd(outputType, iface.getOutputChannelAlignment());

    log.trace("Input padding : {0}", inPadsEnd);
    log.trace("Output padding : {0}", outPadsEnd);

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Input channels are already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        log.trace("Expand input tensor");
        paddedInput = IE::paddingChannel(origOp, rewriter, origOp->getOperand(0), inPadsEnd, Dims4D::Act::C.ind(),
                                         "expand_input");
    }

    log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(paddedInput, inPadsEnd[Dims4D::Act::C], outPadsEnd[Dims4D::Act::C]);
    extendOpLoc(newOp, "expand_act_channels");

    if (outPadsEnd[Dims4D::Act::C] == 0) {
        log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        auto offsets = calcOutputSliceOffset(origOp, outPadsEnd);

        auto sliceOp = rewriter.create<IE::SliceOp>(takeOpLoc(newOp, "slice_out"), origOp->getResult(0).getType(),
                                                    newOp->getResult(0), getIntArrayAttr(ctx, offsets),
                                                    getIntArrayAttr(ctx, outShape));
        rewriter.replaceOp(origOp, sliceOp.getResult());
    }

    return mlir::success();
}

std::pair<mlir::ArrayAttr, mlir::ArrayAttr> IE::getPaddingAttributes(mlir::Operation* op, mlir::Value expandedInput,
                                                                     int64_t inChanPadEnd, ShapeRef outPadAfter) {
    if (!config::hasAutoPadding(getModuleOp(op))) {
        return {nullptr, nullptr};
    }
    Shape inPadAfter(checked_cast<size_t>(mlir::cast<NDTypeInterface>(expandedInput.getType()).getRank()), 0);
    inPadAfter[Dims4D::Act::C] = inChanPadEnd;
    return {getIntArrayAttr(op->getContext(), inPadAfter), getIntArrayAttr(op->getContext(), outPadAfter)};
}

//
// MaxPoolRewriter
//

mlir::LogicalResult IE::MaxPoolRewriter::matchAndRewrite(IE::MaxPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        mlir::Value paddedScale = nullptr;
        if (origOp.getScale() != nullptr) {
            if (outChanPadsEnd == 0) {
                paddedScale = origOp.getScale();
            } else {
                const auto scaleShape = getShape(origOp.getScale());

                Shape scalePadsEnd(scaleShape.size(), 0);
                scalePadsEnd[Dims4D::Act::N] = checked_cast<uint32_t>(outChanPadsEnd);

                paddedScale = rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "expand_scale"), origOp.getScale(),
                                                                  std::nullopt, ShapeRef(scalePadsEnd));
            }
        }

        return rewriter.create<IE::MaxPoolOp>(takeOpLoc(origOp, "expanded"), newOutputType, expandedInput, paddedScale,
                                              origOp.getKernelSize(), origOp.getStrides(), origOp.getPadsBegin(),
                                              origOp.getPadsEnd(), origOp.getRoundingType(), origOp.getPostOpAttr(),
                                              origOp.getClampAttr(), origOp.getStaticScaleAttr(), outputPaddingAttr,
                                              inputPaddingAttr);
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, _log.nest());
}

//
// ConvolutionRewriter
//

mlir::LogicalResult IE::ConvolutionRewriter::matchAndRewrite(IE::ConvolutionOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadEnd) -> mlir::Operation* {
        // We have to expand channels count for filter as well
        const auto paddedFilter = IE::padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;
        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "expand_bias"), origOp.getBias(),
                                                                   std::nullopt, ShapeRef(biasPadsEnd));
            }
        }

        mlir::Value paddedScale;
        if (origOp.getScale() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedScale = origOp.getScale();
            } else {
                const auto scaleShape = getShape(origOp.getScale());

                Shape scalePadsEnd(scaleShape.size(), 0);
                scalePadsEnd[Dims4D::Act::N] = checked_cast<uint32_t>(outChanPadEnd);

                paddedScale = rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "expand_scale"), origOp.getScale(),
                                                                  std::nullopt, ShapeRef(scalePadsEnd));
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        return rewriter.create<IE::ConvolutionOp>(takeOpLoc(origOp, "expanded"), newOutputType, expandedInput,
                                                  paddedFilter, paddedBiases, paddedScale, origOp.getZeroPoints(),
                                                  origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(),
                                                  origOp.getDilations(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                                  origOp.getStaticScaleAttr(), outputPaddingAttr, inputPaddingAttr);
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
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

    auto expandDimension = [&](auto dataToExpand, auto dimToExpand, auto pad, auto rank, bool isInput,
                               StringRef inputName) mutable {
        const auto dataType = mlir::cast<vpux::NDTypeInterface>(dataToExpand.getType());
        const auto quantizedType = mlir::dyn_cast_or_null<mlir::quant::UniformQuantizedType>(dataType.getElementType());
        if (quantizedType && isInput) {
            // For quantized type, we need to create a zero constant with the same type as the dataToExpand
            Shape padsEnd(getShape(dataToExpand));
            padsEnd[dimToExpand] = pad;

            SmallVector<mlir::Value> concatInputs;
            concatInputs.push_back(dataToExpand);
            concatInputs.push_back(generateZeroConst(origOp.getLoc(), dataType, ShapeRef(padsEnd), rewriter));

            return rewriter.createOrFold<IE::ConcatOp>(takeOpLoc(origOp, "concat_{0}", inputName), concatInputs,
                                                       dimToExpand);
        } else {
            if (!mlir::isa<mlir::FloatType>(dataType.getElementType())) {
                _log.trace("[{0}] Data type {1} is not float.", getDebugName(), dataType.getElementType());
            }

            const Shape padsBegin(rank, 0);
            Shape padsEnd(rank, 0);
            padsEnd[dimToExpand] = pad;

            return rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "expand_{0}", inputName), dataToExpand,
                                                       getIntArrayAttr(rewriter, ArrayRef(padsBegin.raw())),
                                                       getIntArrayAttr(rewriter, ArrayRef(padsEnd.raw())));
        }
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
                                    inputChannelPad, input1Rank, true, "in1");
            expandedInput2 =
                    expandDimension(expandedInput2, origOp.getTransposeB() ? Dim(input2Rank - 1) : Dim(input2Rank - 2),
                                    inputChannelPad, input2Rank, true, "in2_ic");
        }

        if (outputChannelPad != 0) {
            expandedInput2 =
                    expandDimension(expandedInput2, origOp.getTransposeB() ? Dim(input2Rank - 2) : Dim(input2Rank - 1),
                                    outputChannelPad, input2Rank, false, "in2_oc");
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
        auto newSlice = rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, opToSlice->getResult(0),
                                                                 getIntArrayAttr(rewriter, sliceOffsets),
                                                                 getIntArrayAttr(rewriter, outShape));
        extendOpLoc(newSlice, "matmul_slice_out");
    };

    auto [inputChannelPad, outputChannelPad] = getPadsForChannels();
    auto [expandedInput1, expandedInput2] = expandInputs(inputChannelPad, outputChannelPad);
    auto newOutputType = inferOutputType(outputChannelPad);

    auto newOp = cloneMatMulOp(rewriter, origOp, newOutputType, expandedInput1, expandedInput2);
    extendOpLoc(newOp, "expanded");

    sliceOutput(newOp);

    return mlir::success();
}

//
// GroupConvolutionRewriter
//

mlir::LogicalResult IE::GroupConvolutionRewriter::matchAndRewrite(IE::GroupConvolutionOp origOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolutionOp layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadEnd) -> mlir::Operation* {
        const auto filterShape = getShape(origOp.getFilter());

        mlir::Value paddedFilter;

        if (outChanPadEnd == 0) {
            paddedFilter = origOp.getFilter();
        } else {
            Shape filterPadsEnd(filterShape.size(), 0);
            filterPadsEnd[Dims4D::Filter::OC] = outChanPadEnd;

            paddedFilter = IE::paddingChannel(origOp, rewriter, origOp.getFilter(), filterPadsEnd,
                                              Dims4D::Filter::OC.ind(), "expand_filter");
        }

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = IE::paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind(),
                                                  "expand_bias");
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);
        const auto newConvOutShape = newOutputType.getShape().toValues();

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        return rewriter.create<IE::GroupConvolutionOp>(
                takeOpLoc(origOp, "expanded"), newOutputType, expandedInput, paddedFilter, paddedBiases,
                origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                getIntAttr(getContext(), newConvOutShape[Dims4D::Act::C]), origOp.getPostOpAttr(),
                origOp.getClampAttr(), outputPaddingAttr, inputPaddingAttr);
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, _log.nest());
}

//
// InterpolateRewriter
//

mlir::LogicalResult IE::InterpolateRewriter::matchAndRewrite(IE::InterpolateOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Interpolate layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
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

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        return rewriter.create<IE::InterpolateOp>(
                takeOpLoc(origOp, "expanded"), newOutputType, expandedInput, sizesInput, origOp.getScales(),
                origOp.getAxes(), sizesAttr, origOp.getScalesAttrAttr(), origOp.getAxesAttrAttr(),
                origOp.getTileOffsetAttrAttr(), origOp.getInitialInputDimsAttrAttr(),
                origOp.getInitialOutputDimsAttrAttr(), origOp.getAttrAttr(), outputPaddingAttr, inputPaddingAttr);
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        SmallVector<int64_t> offsets(outPadsEnd.size(), 0);

        return offsets;
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// TransposedConvolutionRewriter
//

mlir::LogicalResult IE::TransposedConvolutionRewriter::matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Transposed Convolution layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadEnd) -> mlir::Operation* {
        auto paddedFilter = IE::padConvFilter(rewriter, origOp, inChanPadEnd, outChanPadEnd, _log);

        mlir::Value paddedBiases;

        if (origOp.getBias() != nullptr) {
            if (outChanPadEnd == 0) {
                paddedBiases = origOp.getBias();
            } else {
                const auto biasShape = getShape(origOp.getBias());

                Shape biasPadsEnd(biasShape.size(), 0);
                biasPadsEnd[Dims4D::Act::C] = checked_cast<uint32_t>(outChanPadEnd);

                paddedBiases = IE::paddingChannel(origOp, rewriter, origOp.getBias(), biasPadsEnd, Dims4D::Act::C.ind(),
                                                  "expand_bias");
            }
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = outputType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        return rewriter.create<IE::TransposedConvolutionOp>(
                takeOpLoc(origOp, "expanded"), newOutputType, expandedInput, paddedFilter, origOp.getOutputShape(),
                paddedBiases, origOp.getStrides(), origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getDilations(),
                origOp.getSpatialOutputPaddingAttr(), origOp.getPostOpAttr(), origOp.getClampAttr(), outputPaddingAttr,
                inputPaddingAttr);
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// PadRewriter
//

mlir::LogicalResult IE::PadRewriter::matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Pad layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        return rewriter.create<IE::PadOp>(takeOpLoc(origOp, "expanded"), newOutputType, expandedInput,
                                          origOp.getPadsBegin(), origOp.getPadsEnd(), origOp.getPadValue(),
                                          origOp.getPadsBeginAttrAttr(), origOp.getPadsEndAttrAttr(),
                                          origOp.getPadValueAttrAttr(), origOp.getModeAttr(), outputPaddingAttr,
                                          inputPaddingAttr, origOp.getOutputShapeAttr(), origOp.getOutputBoundsAttr());
    };

    const auto calcOutputSliceOffset = [&](mlir::Operation*, ShapeRef outPadsEnd) -> SmallVector<int64_t> {
        return SmallVector<int64_t>(outPadsEnd.size(), 0);
    };

    return generalRewrite(origOp, rewriter, opCreator, calcOutputSliceOffset, _log.nest());
}

//
// AvgPoolRewriter
//

mlir::LogicalResult IE::AvgPoolRewriter::matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AvgPoolRewriter layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);
        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadsEnd;

        const auto ndType = mlir::cast<vpux::NDTypeInterface>(origOp.getType());
        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        auto [inputPaddingAttr, outputPaddingAttr] =
                getPaddingAttributes(origOp, expandedInput, inChanPadEnd, outPadAfter);

        mlir::Value paddedScale = nullptr;
        if (origOp.getScale() != nullptr) {
            if (outChanPadsEnd == 0) {
                paddedScale = origOp.getScale();
            } else {
                const auto scaleShape = getShape(origOp.getScale());

                Shape scalePadsEnd(scaleShape.size(), 0);
                scalePadsEnd[Dims4D::Act::N] = checked_cast<uint32_t>(outChanPadsEnd);

                paddedScale = rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "expand_scale"), origOp.getScale(),
                                                                  std::nullopt, ShapeRef(scalePadsEnd));
            }
        }

        return rewriter.create<IE::AvgPoolOp>(takeOpLoc(origOp, "expanded"), newOutputType, expandedInput, paddedScale,
                                              origOp.getKernelSize(), origOp.getStrides(), origOp.getPadsBegin(),
                                              origOp.getPadsEnd(), origOp.getRoundingType(), origOp.getExcludePads(),
                                              origOp.getPostOpAttr(), origOp.getClampAttr(),
                                              origOp.getStaticScaleAttr(), outputPaddingAttr, inputPaddingAttr);
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, _log.nest());
}

//
// SoftMaxRewriter
//

mlir::LogicalResult IE::SoftMaxRewriter::matchAndRewrite(IE::SoftMaxOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got SoftMaxRewriter layer at '{1}'", getDebugName(), origOp->getLoc());

    const auto opCreator = [&](mlir::Value expandedInput, int64_t inChanPadEnd,
                               int64_t outChanPadsEnd) -> mlir::Operation* {
        _log.trace("Expand SoftMax with pad {0} in {1} out", inChanPadEnd, outChanPadsEnd);
        return rewriter.create<IE::SoftMaxOp>(takeOpLoc(origOp, "expanded"), expandedInput, origOp.getAxisIndAttr(),
                                              getIntAttr(rewriter.getContext(), inChanPadEnd),
                                              origOp.getDstElemTypeAttr(), origOp.getMaskAwareAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, _log.nest());
}

//
// AttentionRewriter
//

mlir::LogicalResult IE::AttentionRewriter::matchAndRewrite(IE::AttentionOp origOp,
                                                           mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got AttentionRewriter layer at '{1}'", getDebugName(), origOp->getLoc());
    auto expandDimensions = [origOp, &rewriter](auto dataToExpand, auto dimsToExpand, ArrayRef<int64_t> pad, auto rank,
                                                const std::string& suffix) mutable {
        const Shape padsBegin(rank, 0);
        auto iterator = dataToExpand;
        for (auto i : dimsToExpand | indexed) {
            Shape padsEnd(rank, 0);
            padsEnd[Dim(i.value())] = pad[i.index()];
            if (pad[i.index()]) {
                iterator =
                        rewriter.createOrFold<IE::ExpandOp>(takeOpLoc(origOp, "{0}_{1}", suffix, i.index()), iterator,
                                                            getIntArrayAttr(rewriter, ArrayRef(padsBegin.raw())),
                                                            getIntArrayAttr(rewriter, ArrayRef(padsEnd.raw())));
            }
        }
        return iterator;
    };

    const auto inQType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto inKType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(1).getType());
    const auto inVType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(2).getType());

    const auto inQShape = inQType.getShape().toValues();
    const auto inKShape = inKType.getShape().toValues();
    const auto inVShape = inVType.getShape().toValues();

    auto inQRank = checked_cast<int64_t>(inQType.getRank());
    auto inKRank = checked_cast<int64_t>(inKType.getRank());
    auto inVRank = checked_cast<int64_t>(inVType.getRank());

    auto inEDim = inQShape[Dim(inQRank - 1)];
    auto inSDim = inKShape[Dim(inKRank - 2)];
    auto inEvDim = inVShape[Dim(inVRank - 2)];

    int64_t inputChannelAlignment = 16;
    auto inEPad = alignValUp(inEDim, inputChannelAlignment) - inEDim;
    auto inSPad = alignValUp(inSDim, inputChannelAlignment) - inSDim;

    int64_t inputChannelAlignmentEv = 16;
    auto inEvPad = alignValUp(inEvDim, inputChannelAlignmentEv) - inEvDim;

    mlir::Value expandedInQ = origOp->getOperand(0);
    mlir::Value expandedInK = origOp->getOperand(1);
    mlir::Value expandedInV = origOp->getOperand(2);
    expandedInQ = expandDimensions(expandedInQ, SmallVector<int64_t>{inQRank - 1}, SmallVector<int64_t>{inEPad},
                                   inQRank, "_expandedInQ");
    expandedInK = expandDimensions(expandedInK, SmallVector<int64_t>{inKRank - 1}, SmallVector<int64_t>{inEPad},
                                   inKRank, "_expandedInK");
    expandedInV = expandDimensions(expandedInV, SmallVector<int64_t>{inVRank - 1, inVRank - 2},
                                   SmallVector<int64_t>{inSPad, inEvPad}, inVRank, "_expandedInV");

    auto paddedAttentionMask = mlir::Value{origOp.getInputMask()};
    if (paddedAttentionMask != nullptr) {
        const auto inMaskType = mlir::cast<vpux::NDTypeInterface>(paddedAttentionMask.getType());
        const auto inMaskShape = inMaskType.getShape().toValues();
        auto inMaskRank = checked_cast<int64_t>(inMaskType.getRank());
        auto maskSDim = inMaskShape[Dim(inMaskRank - 1)];
        if (inSDim == maskSDim) {  // Is not broadcast 1d dimension for mask, so it will be aligned as inSDim
            paddedAttentionMask = expandDimensions(paddedAttentionMask, SmallVector<int64_t>{inMaskRank - 1},
                                                   SmallVector<int64_t>{inSPad}, inMaskRank, "_expandedAttentionMask");
        }
    }

    auto paddedBias = mlir::Value{origOp.getInputBias()};
    if (paddedBias != nullptr) {
        const auto inBiasType = mlir::cast<vpux::NDTypeInterface>(paddedBias.getType());
        const auto inBiasShape = inBiasType.getShape().toValues();
        auto inBiasRank = checked_cast<int64_t>(inBiasType.getRank());
        auto biasSDim = inBiasShape[Dim(inBiasRank - 1)];
        if (inSDim == biasSDim) {  // Is not broadcast 1d dimension for mask, so it will be aligned as inSDim
            paddedBias = expandDimensions(paddedBias, SmallVector<int64_t>{inBiasRank - 1},
                                          SmallVector<int64_t>{inSPad}, inBiasRank, "_expandedBias");
        }
    }

    auto paddedSink = mlir::Value{origOp.getInputSink()};
    if (paddedSink != nullptr) {
        const auto inSinkType = mlir::cast<vpux::NDTypeInterface>(paddedSink.getType());
        const auto inSinkShape = inSinkType.getShape().toValues();
        auto inSinkRank = checked_cast<int64_t>(inSinkType.getRank());
        auto sinkSDim = inSinkShape[Dim(inSinkRank - 2)];
        if (inSDim == sinkSDim) {  // Is not broadcast 1d dimension for sink, so it will be aligned as inSDim
            paddedSink = expandDimensions(paddedSink, SmallVector<int64_t>{inSinkRank - 2},
                                          SmallVector<int64_t>{inSPad}, inSinkRank, "_expandedSink");
        }
    }

    auto attention = rewriter.create<IE::AttentionOp>(
            takeOpLoc(origOp, "expanded"), expandedInQ, expandedInK, expandedInV, paddedAttentionMask,
            origOp.getInputScale(), paddedSink, paddedBias, getIntAttr(rewriter.getContext(), inSPad));

    if (inEvPad) {
        _log.trace("Slice Attention output with padding {0}", inEvPad);
        const auto outShape = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType()).getShape();
        auto offsets = SmallVector<int64_t>(outShape.size(), 0);

        auto sliceOp = rewriter.createOrFold<IE::SliceOp>(
                appendLoc(origOp.getLoc(), "sliced"), origOp->getResult(0).getType(), attention.getOutput(),
                getIntArrayAttr(rewriter, offsets), getIntArrayAttr(rewriter, outShape));
        rewriter.replaceOp(origOp, sliceOp);
    } else {
        rewriter.replaceOp(origOp, attention.getOutput());
    }

    return mlir::success();
}

mlir::LogicalResult vpux::IE::FlashSDPARewriter::matchAndRewrite(IE::FlashSDPAOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' layer at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto keyType = mlir::cast<NDTypeInterface>(origOp.getKey().getType());
    auto keyShape = keyType.getShape();
    auto valueShape = getShape(origOp.getValue());

    // Dimensions that should be aligned to satisfy DPU requirements for 2 DWConv layers in FlashSDPA kernel
    auto qkEmbedding = keyShape[Dims4D::Act::W];
    auto sourceSeqLen = keyShape[Dims4D::Act::H];
    auto vEmbedding = valueShape[Dims4D::Act::H];

    auto elemType = keyType.getElementType();
    auto alignment = vpux::VPU::NCEInvariant::getAlignment(elemType);

    auto alignedQkEmbedding = alignValUp(qkEmbedding, alignment);
    auto alignedSourceSeqLen = alignValUp(sourceSeqLen, alignment);
    auto alignedVEmbedding = alignValUp(vEmbedding, alignment);

    auto qkEmbeddingPad = alignedQkEmbedding - qkEmbedding;
    auto sourceSeqLenPad = alignedSourceSeqLen - sourceSeqLen;
    auto vEmbeddingPad = alignedVEmbedding - vEmbedding;

    auto queryPadEnd = Shape{0, 0, 0, qkEmbeddingPad};
    auto keyPadEnd = Shape{0, 0, sourceSeqLenPad, qkEmbeddingPad};
    auto valuePadEnd = Shape{0, 0, vEmbeddingPad, sourceSeqLenPad};
    auto runningOutputPadEnd = Shape{0, 0, 0, vEmbeddingPad};

    auto expand = [origOp, &rewriter](mlir::Value value, ShapeRef padsEnd, StringRef operandName) -> mlir::Value {
        // pads_end must contain at most one non-zero value.
        for (auto [index, padEnd] : enumerate(padsEnd)) {
            if (padEnd == 0) {
                continue;
            }

            auto padsBegin = SmallVector<int64_t>(padsEnd.size());

            auto padsEndOneDim = SmallVector<int64_t>(padsEnd.size());
            padsEndOneDim[index] = padEnd;

            auto loc = takeOpLoc(origOp, "expand_{0}_dim{1}", operandName, index);
            value = rewriter.createOrFold<IE::ExpandOp>(loc, value, getIntArrayAttr(rewriter, padsBegin),
                                                        getIntArrayAttr(rewriter, padsEndOneDim));
        }

        return value;
    };

    auto paddedQuery = expand(origOp.getQuery(), queryPadEnd, "query");
    auto paddedKey = expand(origOp.getKey(), keyPadEnd, "key");
    auto paddedValue = expand(origOp.getValue(), valuePadEnd, "value");
    auto paddedRunningOutput = expand(origOp.getInputRunningOutput(), runningOutputPadEnd, "running_output");

    auto paddedAttentionMask = mlir::Value{origOp.getAttentionMask()};
    if (paddedAttentionMask != nullptr) {
        auto attentionMaskPadsEnd = Shape{0, 0, 0, sourceSeqLenPad};
        paddedAttentionMask = expand(origOp.getAttentionMask(), attentionMaskPadsEnd, "attention_mask");
    }

    auto newOp = rewriter.create<IE::FlashSDPAOp>(
            takeOpLoc(origOp, "expanded"), paddedQuery, paddedKey, paddedValue, paddedRunningOutput,
            origOp.getInputRunningMax(), origOp.getInputRunningSum(), paddedAttentionMask, origOp.getIsHeadAttr(),
            origOp.getIsTailAttr(), getIntAttr(rewriter, sourceSeqLenPad));

    if (vEmbeddingPad == 0) {
        _log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp);
    } else {
        _log.trace("Extract meaningful part from extended output");

        const auto outputShape = getShape(origOp.getResultRunningOutput());
        auto offsets = SmallVector<int64_t>(outputShape.size());

        auto sliceOp = rewriter.createOrFold<IE::SliceOp>(
                appendLoc(origOp.getLoc(), "sliced"), origOp.getResultRunningOutput().getType(),
                newOp.getResultRunningOutput(), getIntArrayAttr(rewriter, offsets),
                getIntArrayAttr(rewriter, outputShape));

        rewriter.replaceOp(origOp, mlir::ValueRange{sliceOp, newOp.getResultRunningMax(), newOp.getResultRunningSum()});
    }

    return mlir::success();
};

namespace {
//
// ExpandActivationChannelsPass
//

class ExpandActivationChannelsPass final : public IE::impl::ExpandActivationChannelsBase<ExpandActivationChannelsPass> {
public:
    explicit ExpandActivationChannelsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() override;
};  // class ExpandActivationChannelsPass

void ExpandActivationChannelsPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    const auto moduleOp = getModuleOp(func);
    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getExpandActivationChannelsStrategy(
            vpux::config::hasEnableSEPtrsOperations(moduleOp), _log);

    mlir::ConversionTarget target(ctx);
    strategy->addTargets(target);

    mlir::RewritePatternSet patterns(&ctx);
    strategy->addPatterns(patterns);

    if (mlir::failed(mlir::applyFullConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createExpandActivationChannelsPass
//

namespace vpux::IE {
std::unique_ptr<mlir::Pass> createExpandActivationChannelsPass(Logger log) {
    return std::make_unique<ExpandActivationChannelsPass>(log);
}
}  // namespace vpux::IE
