//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

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

    // Check if original scales are available for SCALES mode
    const auto calcMode = origOp.getAttr().getShapeCalcMode();
    const bool isScalesMode = calcMode != nullptr && calcMode.getValue() == IE::InterpolateCalcMode::SCALES;
    SmallVector<double> origScales;
    if (isScalesMode && origOp.getScalesAttrAttr()) {
        origScales = parseFPArrayAttr<double>(origOp.getScalesAttrAttr());
    }

    SmallVector<int64_t> sizes1;
    SmallVector<double> scale1;
    for (size_t i = 0; i < axes1.size(); ++i) {
        sizes1.push_back(outputShape[vpux::Dim(axes1[i])]);
        if (isScalesMode && i < origScales.size()) {
            scale1.push_back(origScales[i]);
        } else {
            scale1.push_back(static_cast<double>(outputShape[vpux::Dim(axes1[i])]) /
                             static_cast<double>(inputShape[vpux::Dim(axes1[i])]));
        }
    }

    const auto sizesAttr1 = getIntArrayAttr(origOp.getContext(), sizes1);
    const auto scalesAttr1 = getFPArrayAttr(origOp.getContext(), scale1);
    const auto axesAttr1 = getIntArrayAttr(origOp.getContext(), axes1);

    SmallVector<int64_t> interpolate1Shape(inputShape.raw());
    for (size_t i = 0; i < axes1.size(); ++i) {
        interpolate1Shape[axes1[i]] = sizes1[i];
    }

    const auto outputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto interpolate1OutputType = outputType.changeShape(vpux::ShapeRef(interpolate1Shape));

    auto interpolate1Result = rewriter.createOrFold<IE::InterpolateOp>(
            appendLoc(loc, "interpolate1"), interpolate1OutputType, origOp.getInput(), origOp.getSizes(),
            origOp.getScales(), origOp.getAxes(), sizesAttr1, scalesAttr1, axesAttr1, origOp.getTileOffsetAttrAttr(),
            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    SmallVector<int64_t> sizes2;
    SmallVector<double> scale2;
    for (size_t i = 0; i < axes2.size(); ++i) {
        sizes2.push_back(outputShape[vpux::Dim(axes2[i])]);
        size_t origScaleIdx = axes1.size() + i;
        if (isScalesMode && origScaleIdx < origScales.size()) {
            scale2.push_back(origScales[origScaleIdx]);
        } else {
            scale2.push_back(static_cast<double>(outputShape[vpux::Dim(axes2[i])]) /
                             static_cast<double>(interpolate1Shape[axes2[i]]));
        }
    }

    const auto sizesAttr2 = getIntArrayAttr(origOp.getContext(), sizes2);
    const auto scalesAttr2 = getFPArrayAttr(origOp.getContext(), scale2);
    const auto axesAttr2 = getIntArrayAttr(origOp.getContext(), axes2);

    SmallVector<int64_t> interpolate2Shape(interpolate1Shape.begin(), interpolate1Shape.end());
    for (size_t i = 0; i < axes2.size(); ++i) {
        interpolate2Shape[axes2[i]] = sizes2[i];
    }

    auto interpolate2OutputType = outputType.changeShape(vpux::ShapeRef(interpolate2Shape));

    auto interpolate2Result = rewriter.createOrFold<IE::InterpolateOp>(
            appendLoc(loc, "interpolate2"), interpolate2OutputType, interpolate1Result, origOp.getSizes(),
            origOp.getScales(), origOp.getAxes(), sizesAttr2, scalesAttr2, axesAttr2, origOp.getTileOffsetAttrAttr(),
            origOp.getInitialInputDimsAttrAttr(), origOp.getInitialOutputDimsAttrAttr(), origOp.getAttr(),
            origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, interpolate2Result);
    return mlir::success();
}

//
// safeRunOnFunc
//

void SplitInterpolateAxesPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InterpolateOpConverter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createSplitInterpolateAxesPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSplitInterpolateAxesPass(Logger log) {
    return std::make_unique<SplitInterpolateAxesPass>(log);
}
