//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Transforms/DialectConversion.h>

namespace vpux::IE {
#define GEN_PASS_DECL_EXPANDACTIVATIONWIDTH
#define GEN_PASS_DEF_EXPANDACTIVATIONWIDTH
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

    const auto inputType = origOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const auto outputType = origOp->getResult(0).getType().cast<vpux::NDTypeInterface>();
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
        paddedInput = rewriter.createOrFold<IE::ExpandOp>(origOp->getLoc(), origOp->getOperand(0), std::nullopt,
                                                          ShapeRef(inPadsEnd));
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

        rewriter.replaceOpWithNewOp<IE::SliceOp>(origOp, origOp->getResult(0).getType(), newOp->getResult(0),
                                                 getIntArrayAttr(ctx, offsets), getIntArrayAttr(ctx, outShape));
    }

    return mlir::success();
}

//
// ExpandActivationWidthPass
//

class ExpandActivationWidthPass final : public IE::impl::ExpandActivationWidthBase<ExpandActivationWidthPass> {
public:
    explicit ExpandActivationWidthPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ExpandActivationWidthPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto isLegalPermuteQuantize = [&](IE::PermuteQuantizeOp op) {
        const auto inType = op.getInput().getType().dyn_cast<vpux::NDTypeInterface>();
        const auto outType = op.getOutput().getType().dyn_cast<vpux::NDTypeInterface>();
        const auto inOrder = inType.getDimsOrder();
        const auto outOrder = outType.getDimsOrder();
        // Check that such IE.PermuteQuantize can be executed on DPU.
        if (inOrder != DimsOrder::NCHW || outOrder != DimsOrder::NHWC) {
            return true;
        }
        const ShapeRef inShape = inType.getShape();
        const auto inputElemType = inType.getElementType();
        const auto inAlignment = VPU::NCEInvariant::getAlignment(inputElemType);
        if (!IE::isODUPermuteEffectiveForShape(inShape, inAlignment)) {
            return true;
        }
        const ShapeRef outShape = outType.getShape();
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

        const auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();
        const auto outElemType = outputType.getElementType();
        const int64_t alignment = VPU::NCEInvariant::getAlignment(outElemType);
        const auto outPadsEnd = calcOutPadsEnd(outputType, alignment);

        return outPadsEnd[Dims4D::Act::W] == 0;
    };

    mlir::ConversionTarget target(ctx);
    target.addDynamicallyLegalOp<IE::PermuteQuantizeOp>(isLegalPermuteQuantize);
    target.addLegalOp<Const::DeclareOp, IE::ExpandOp, IE::SliceOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteQuantizeRewriter>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createExpandActivationWidthPass(Logger log) {
    return std::make_unique<ExpandActivationWidthPass>(log);
}
