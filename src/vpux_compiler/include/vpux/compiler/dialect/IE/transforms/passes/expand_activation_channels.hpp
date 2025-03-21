//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/expand_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/auto_padding_utils.hpp"
#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/numeric.hpp"
namespace vpux {

namespace IE {

mlir::LogicalResult generalRewrite(mlir::Operation* origOp, mlir::PatternRewriter& rewriter,
                                   FuncRef<mlir::Operation*(mlir::Value, int64_t)> opCreator,
                                   FuncRef<SmallVector<int64_t>(mlir::Operation*, ShapeRef)> calcOutputSliceOffset,
                                   FuncRef<void()> autopadAttributeModifier, Logger log);

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

//
// ConvolutionRewriter
//

class ConvolutionRewriter final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    ConvolutionRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("ConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// MatmulRewriter
//

class MatMulRewriter final : public mlir::OpRewritePattern<IE::MatMulOp> {
public:
    MatMulRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::MatMulOp>(ctx), _log(log) {
        setDebugName("MatMulRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::MatMulOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// EltwiseRewriter
//

template <class ConcreteOp>
class EltwiseRewriter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    EltwiseRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("EltwiseRewriter");
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult EltwiseRewriter<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got eltwise layer at '{1}'", this->getDebugName(), origOp->getLoc());

    const auto autopadModifier = [&]() {
        rewriter.modifyOpInPlace(origOp, [&] {
            origOp.setOutputChannels(getShape(origOp.getResult())[Dims4D::Act::C]);
        });
    };

    const auto opCreator = [&](mlir::Value expandedInput1, int64_t outChanPadEnd) -> mlir::Operation* {
        mlir::Value expandedInput2;
        if (origOp.getInput1() == origOp.getInput2()) {
            expandedInput2 = expandedInput1;
        } else if (outChanPadEnd == 0 && !(VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp))) {
            expandedInput2 = origOp.getInput2();
        } else {
            _log.trace("Expand second input tensor");

            const auto origShape = getShape(origOp.getInput2());
            const auto extendedShape = getShape(expandedInput1);
            VPUX_THROW_UNLESS(origShape.size() == extendedShape.size(), "Got non equal shapes in EltwiseRewriter");

            const auto padsEnd = IE::calcPadsEnd(origShape, extendedShape);

            auto sliceOp = origOp.getInput1().template getDefiningOp<IE::SliceOp>();
            expandedInput2 =
                    IE::expandWithOffset(rewriter, origOp, sliceOp, origOp.getInput2(), padsEnd, Dims4D::Act::C.ind());
        }

        const Shape outPadBefore(checked_cast<size_t>(origOp.getType().getRank()), 0);

        Shape outPadAfter(checked_cast<size_t>(origOp.getType().getRank()), 0);
        outPadAfter[Dims4D::Act::C] = outChanPadEnd;

        const auto ndType = origOp.getType().template cast<vpux::NDTypeInterface>();

        auto outChanBeforeAttr = origOp.getOutputChannelsAttr();
        if (VPU::hasOnlyDirectSWConsumers(origOp) && VPU::canAutopadOutput(origOp)) {
            outChanBeforeAttr = vpux::getIntAttr(origOp.getContext(), ndType.getShape()[Dims4D::Act::C]);
        }

        const auto newOutputType = ndType.pad(outPadBefore, outPadAfter);

        return rewriter.create<ConcreteOp>(origOp.getLoc(), newOutputType, expandedInput1, expandedInput2,
                                           origOp.getAutoBroadcast(), origOp.getPostOpAttr(), origOp.getClampAttr(),
                                           outChanBeforeAttr, origOp.getInputChannelsAttr());
    };

    return generalRewrite(origOp, rewriter, opCreator, IE::extractMeaningfulOutput, autopadModifier, _log.nest());
}

//
// GroupConvolutionRewriter
//

class GroupConvolutionRewriter final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    GroupConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("GroupConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// InterpolateRewriter
//

class InterpolateRewriter final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
        setDebugName("InterpolateRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// TransposedConvolutionRewriter
//

class TransposedConvolutionRewriter final : public mlir::OpRewritePattern<IE::TransposedConvolutionOp> {
public:
    TransposedConvolutionRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::TransposedConvolutionOp>(ctx), _log(log) {
        setDebugName("TransposedConvolutionRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::TransposedConvolutionOp origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// PadRewriter
//

class PadRewriter final : public mlir::OpRewritePattern<IE::PadOp> {
public:
    PadRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::PadOp>(ctx), _log(log) {
        setDebugName("PadRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::PadOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// AvgPoolRewriter
//

class AvgPoolRewriter final : public mlir::OpRewritePattern<IE::AvgPoolOp> {
public:
    AvgPoolRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::AvgPoolOp>(ctx), _log(log) {
        setDebugName("AvgPoolRewriter");
    }

    mlir::LogicalResult matchAndRewrite(IE::AvgPoolOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

}  // namespace IE
}  // namespace vpux
