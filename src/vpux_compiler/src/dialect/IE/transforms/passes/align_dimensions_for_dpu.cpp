//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_ALIGNDIMENSIONSFORDPU
#define GEN_PASS_DEF_ALIGNDIMENSIONSFORDPU
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// calcPadsEnd
//

Shape calcPadsEnd(ShapeRef origShape, ShapeRef extendedShape) {
    Shape padsEnd(origShape.size());

    for (auto i : irange(origShape.size())) {
        const auto d = Dim(i);
        padsEnd[d] = extendedShape[d] - origShape[d];
    }

    return padsEnd;
}

Shape calcOutPadsEnd(vpux::NDTypeInterface origType, int64_t channelAlignment) {
    const auto origShape = origType.getShape();

    auto extendedShape = origShape.toValues();
    extendedShape[Dims4D::Act::W] = alignValUp(origShape[Dims4D::Act::W], channelAlignment);

    return calcPadsEnd(origShape, extendedShape);
}

Shape calcInPadsEnd(vpux::NDTypeInterface inputType, vpux::NDTypeInterface outputType, const ShapeRef outputPads,
                    const int64_t kernelX, const int64_t strideX, const int64_t padLeft, const int64_t padRight) {
    const auto inputShape = inputType.getShape();
    const auto outputShape = outputType.getShape();
    const auto outputWidth = outputShape[Dims4D::Act::W] + outputPads[Dims4D::Act::W];

    auto extendedShape = inputShape.toValues();
    extendedShape[Dims4D::Act::W] = (outputWidth - 1) * strideX + kernelX - padLeft - padRight;

    return calcPadsEnd(inputShape, extendedShape);
}

mlir::Operation* opCreator(mlir::Operation* origOp, vpux::NDTypeInterface ndType, ArrayRef<mlir::Value> expandedInputs,
                           int64_t outWidthPadEnd, mlir::PatternRewriter& rewriter) {
    const Shape outPadBefore(checked_cast<size_t>(ndType.getRank()), 0);

    Shape outPadAfter(checked_cast<size_t>(ndType.getRank()), 0);
    outPadAfter[Dims4D::Act::W] = outWidthPadEnd;

    const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

    auto* newNCEOp = rewriter.clone(*origOp);
    for (size_t inIdx = 0; inIdx < expandedInputs.size(); inIdx++) {
        newNCEOp->setOperand(checked_cast<unsigned>(inIdx), expandedInputs[inIdx]);
    }
    newNCEOp->getResult(0).setType(newOutputType);
    return newNCEOp;
}

//
// PermuteQuantizeRewriter
//

class PermuteQuantizeRewriter final : public mlir::OpRewritePattern<IE::PermuteQuantizeOp> {
public:
    PermuteQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::PermuteQuantizeOp>(ctx), _log(log) {
        this->setDebugName("PermuteQuantizeRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::PermuteQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult PermuteQuantizeRewriter::matchAndRewrite(IE::PermuteQuantizeOp origOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.PermuteQuantize at '{1}'", this->getDebugName(), origOp->getLoc());

    auto* ctx = origOp->getContext();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp->getOperand(0).getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp->getResult(0).getType());
    const auto outElemType = outputType.getElementType();

    const auto alignment = VPU::NCEInvariant::getAlignment(outElemType);
    const auto outPadsEnd = calcOutPadsEnd(outputType, alignment);
    const int64_t kernelX = 1;
    const int64_t strideX = 1;
    const int64_t padLeft = 0;
    const int64_t padRight = 0;
    const auto inPadsEnd = calcInPadsEnd(inputType, outputType, outPadsEnd, kernelX, strideX, padLeft, padRight);

    _log.trace("Input padding : {0}", inPadsEnd);
    _log.trace("Output padding : {0}", outPadsEnd);

    if (inPadsEnd[Dims4D::Act::W] == 0 && outPadsEnd[Dims4D::Act::W] == 0) {
        return matchFailed(_log, rewriter, origOp, "Both input and output width are already aligned");
    }

    mlir::Value paddedInput;
    if (inPadsEnd[Dims4D::Act::W] == 0) {
        _log.trace("Input width is already aligned");
        paddedInput = origOp->getOperand(0);
    } else {
        _log.trace("Expand input tensor");
        paddedInput = rewriter.createOrFold<IE::ExpandOp>(appendLoc(origOp->getLoc(), "expand_input"),
                                                          origOp->getOperand(0), std::nullopt, ShapeRef(inPadsEnd));
    }

    SmallVector<mlir::Value> paddedInputs = {paddedInput};

    _log.trace("Create new operation with extended input and output");
    auto* newOp = opCreator(origOp, outputType, paddedInputs, outPadsEnd[Dims4D::Act::W], rewriter);

    if (outPadsEnd[Dims4D::Act::W] == 0) {
        _log.trace("Output channels are already aligned");
        rewriter.replaceOp(origOp, newOp->getResult(0));
    } else {
        _log.trace("Extract meaningful part from extended output");

        const auto outShape = outputType.getShape();
        const SmallVector<int64_t> offsets(outShape.size(), 0);

        auto newSlice =
                rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                         getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
        extendOpLoc(newSlice, "slice_output");
    }

    return mlir::success();
}

//
// LogSoftmaxRewriter
//

class LogSoftmaxRewriter final : public mlir::OpRewritePattern<IE::LogSoftmaxOp> {
public:
    LogSoftmaxRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::LogSoftmaxOp>(ctx), _log(log) {
        this->setDebugName("LogSoftmaxRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::LogSoftmaxOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult LogSoftmaxRewriter::matchAndRewrite(IE::LogSoftmaxOp origOp,
                                                        mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.LogSoftmax at '{1}'", this->getDebugName(), origOp->getLoc());

    auto* ctx = origOp->getContext();

    const auto inputType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    const auto elemType = inputType.getElementType();

    const auto axisInd = origOp.getAxisInd();
    const auto inputShape = inputType.getShape();

    // Calculate padding for the axis dimension
    const int64_t alignment = VPU::NCEInvariant::getAlignment(elemType);
    const auto axisDimSize = inputShape[Dim(axisInd)];
    const auto alignedAxisSize = alignValUp(axisDimSize, alignment);
    const auto axisPad = alignedAxisSize - axisDimSize;

    _log.trace("LogSoftmax axis {0}, dim size {1}, aligned {2}, pad {3}", axisInd, axisDimSize, alignedAxisSize,
               axisPad);

    if (axisPad == 0) {
        return matchFailed(_log, rewriter, origOp, "Axis dimension is already aligned");
    }

    Shape padsEnd(inputShape.size(), 0);
    padsEnd[Dim(axisInd)] = axisPad;

    _log.trace("Expand input tensor along axis {0} with padding {1}", axisInd, axisPad);
    auto expandedInput = rewriter.createOrFold<IE::ExpandOp>(appendLoc(origOp->getLoc(), "_expand_axis"),
                                                             origOp.getInput(), std::nullopt, ShapeRef(padsEnd));

    const Shape outPadBefore(outputType.getRank(), 0);
    Shape outPadAfter(outputType.getRank(), 0);
    outPadAfter[Dim(axisInd)] = axisPad;
    const auto newOutputType = outputType.pad(outPadBefore, outPadAfter);

    auto newOp = rewriter.create<IE::LogSoftmaxOp>(appendLoc(origOp->getLoc(), "_expanded"), newOutputType,
                                                   expandedInput, origOp.getAxisIndAttr(), getIntAttr(ctx, axisPad));

    const auto outShape = outputType.getShape();
    SmallVector<int64_t> offsets(outShape.size(), 0);

    auto sliceOp =
            rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp.getOutput().getType(), newOp.getOutput(),
                                                     getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));

    extendOpLoc(sliceOp, "_slice_axis");

    return mlir::success();
}

//
// AlignDimensionsForDPUPass
//

class AlignDimensionsForDPUPass final : public IE::impl::AlignDimensionsForDPUBase<AlignDimensionsForDPUPass> {
public:
    explicit AlignDimensionsForDPUPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void AlignDimensionsForDPUPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalPermuteQuantize = [&](IE::PermuteQuantizeOp op) {
        const auto inType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getInput().getType());
        const auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(op.getOutput().getType());
        const auto inOrder = inType.getDimsOrder();
        const auto outOrder = outType.getDimsOrder();
        // Check that such IE.PermuteQuantize can be executed on DPU.
        if (inOrder != DimsOrder::NCHW || outOrder != DimsOrder::NHWC) {
            return true;
        }
        const auto inShape = getBoundedShape(inType);
        const auto inputElemType = inType.getElementType();
        const auto inAlignment = VPU::NCEInvariant::getAlignment(inputElemType);
        if (!IE::isODUPermuteEffectiveForShape(inShape, inAlignment)) {
            return true;
        }
        const auto outShape = getBoundedShape(outType);
        const auto outputElemType = outType.getElementType();
        const auto outAlignment = VPU::NCEInvariant::getAlignment(outputElemType);
        if (!IE::isODUPermuteEffectiveForShape(outShape, outAlignment)) {
            return true;
        }

        // We are calling NCEPermuteOp::isSupported with checkChannelAlignment=false because in this pass we
        // set the alignment to be able to run on NCE. And if we are checking also the alignment the result will
        // always be false.
        const auto logCb = [&](const formatv_object_base&) {};
        if (!VPU::NCEPermuteOp::isSupported(op, logCb, /*checkLayout=*/false,
                                            /*checkChannelAlignment=*/false)) {
            return true;
        }

        const auto outputType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType());
        const auto outElemType = outputType.getElementType();
        const int64_t alignment = VPU::NCEInvariant::getAlignment(outElemType);
        const auto outPadsEnd = calcOutPadsEnd(outputType, alignment);

        return outPadsEnd[Dims4D::Act::W] == 0;
    };

    const auto isLegalLogSoftmax = [&](IE::LogSoftmaxOp op) {
        const auto inputType = mlir::cast<vpux::NDTypeInterface>(op.getInput().getType());
        const auto elemType = inputType.getElementType();
        const auto axisInd = op.getAxisInd();
        const auto inputShape = inputType.getShape();

        // Should expand only if the axis is on the innermost dim
        const auto dimsOrder = inputType.getDimsOrder();
        const auto innermostDim = dimsOrder.dimAt(dimsOrder.numDims() - 1);
        if (Dim(axisInd) != innermostDim) {
            return true;
        }

        const int64_t alignment = VPU::NCEInvariant::getAlignment(elemType);
        const auto axisDimSize = inputShape[Dim(axisInd)];
        const auto alignedAxisSize = alignValUp(axisDimSize, alignment);

        return alignedAxisSize == axisDimSize;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::PermuteQuantizeOp>(isLegalPermuteQuantize);
    target.addDynamicallyLegalOp<IE::LogSoftmaxOp>(isLegalLogSoftmax);
    target.addLegalOp<Const::DeclareOp, IE::ExpandOp, IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteQuantizeRewriter>(&ctx, _log);
    patterns.add<LogSoftmaxRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAlignDimensionsForDPUPass(Logger log) {
    return std::make_unique<AlignDimensionsForDPUPass>(log);
}
