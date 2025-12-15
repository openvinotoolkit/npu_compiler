//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_SPLITINTERPOLATEAXES
#define GEN_PASS_DEF_SPLITINTERPOLATEAXES
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// InterpolateOpConverter
//

class InterpolateOpConverter final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateOpConverter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
        setDebugName("InterpolateOpConverter");
    }

    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// SplitInterpolatePass
//

class SplitInterpolateAxesPass final : public IE::impl::SplitInterpolateAxesBase<SplitInterpolateAxesPass> {
public:
    explicit SplitInterpolateAxesPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult InterpolateOpConverter::matchAndRewrite(IE::InterpolateOp origOp,
                                                            mlir::PatternRewriter& rewriter) const {
    const auto loc = origOp.getLoc();
    auto inputShape = getShape(origOp.getInput());
    auto outputShape = getShape(origOp.getOutput());

    auto axesAttr = origOp.getAxesAttrAttr();
    auto axesSize = origOp.getAxesAttrAttr().size();
    if (axesSize != 3) {
        return mlir::failure();
    }

    _log.trace("Found Interpolate Operation {0} ", loc);

    SmallVector<int64_t> axes;
    for (auto axis : axesAttr.getValue()) {
        axes.push_back(mlir::cast<mlir::IntegerAttr>(axis).getInt());
    }

    SmallVector<int64_t> axes1, axes2;
    if (axes.size() > 1) {
        for (size_t i = 0; i < axes.size() - 1; ++i) {
            axes1.push_back(axes[i]);
        }
        axes2.push_back(axes.back());
    } else {
        axes2 = std::move(axes);
    }

    SmallVector<int64_t> sizes1, scale1;
    for (auto axis : axes1) {
        sizes1.push_back(outputShape[vpux::Dim(axis)]);
        scale1.push_back(outputShape[vpux::Dim(axis)] / inputShape[vpux::Dim(axis)]);
    }

    const auto sizesAttr1 = getIntArrayAttr(origOp.getContext(), sizes1);
    const auto scalesAttr1 = getFPArrayAttr(origOp.getContext(), scale1);
    const auto axesAttr1 = getIntArrayAttr(origOp.getContext(), axes1);

    SmallVector<int64_t> interpolate1Shape(inputShape.raw());
    for (size_t i = 0; i < axes1.size(); ++i) {
        interpolate1Shape[axes1[i]] *= scale1[i];
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto interpolate1OutputType = outputType.changeShape(vpux::ShapeRef(interpolate1Shape));

    auto interpolate1 = rewriter.create<IE::InterpolateOp>(
            appendLoc(loc, "_interpolate1"), interpolate1OutputType, origOp.getInput(), origOp.getSizes(),
            origOp.getScales(), origOp.getAxes(), sizesAttr1, scalesAttr1, axesAttr1, origOp.getTileOffsetAttrAttr(),
            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    SmallVector<int64_t> sizes2, scale2;
    for (auto axis : axes2) {
        auto dim = vpux::Dim(axis);
        sizes2.push_back(outputShape[dim]);
        scale2.push_back(outputShape[dim] / interpolate1Shape[axis]);
    }

    const auto sizesAttr2 = getIntArrayAttr(origOp.getContext(), sizes2);
    const auto scalesAttr2 = getFPArrayAttr(origOp.getContext(), scale2);
    const auto axesAttr2 = getIntArrayAttr(origOp.getContext(), axes2);

    SmallVector<int64_t> interpolate2Shape(interpolate1Shape.begin(), interpolate1Shape.end());
    for (size_t i = 0; i < axes2.size(); ++i) {
        interpolate2Shape[axes2[i]] *= scale2[i];
    }

    auto interpolate2OutputType = outputType.changeShape(vpux::ShapeRef(interpolate2Shape));

    auto interpolate2 = rewriter.create<IE::InterpolateOp>(
            appendLoc(loc, "_interpolate2"), interpolate2OutputType, interpolate1.getOutput(), origOp.getSizes(),
            origOp.getScales(), origOp.getAxes(), sizesAttr2, scalesAttr2, axesAttr2, origOp.getTileOffsetAttrAttr(),
            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, interpolate2);
    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitInterpolateAxesPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InterpolateOpConverter>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSplitInterpolateAxesPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitInterpolateAxesPass(Logger log) {
    return std::make_unique<SplitInterpolateAxesPass>(log);
}
