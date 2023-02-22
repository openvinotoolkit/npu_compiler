//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/IE/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/IE/utils/shape_infer.hpp"
#include "vpux/compiler/dialect/VPUIP/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Pass/PassManager.h>

#include <legacy/ngraph_ops/convolution_ie.hpp>

using namespace vpux;

namespace {

//
// generalFusion
//

mlir::LogicalResult generalFusion(mlir::Operation* origOp, mlir::ArrayAttr kernelSizeAttr, mlir::ArrayAttr padBegin,
                                  mlir::ArrayAttr padEnd,
                                  FuncRef<void(mlir::Value, mlir::ArrayAttr, mlir::ArrayAttr)> opRewriter, Logger log) {
    auto origPadOp = origOp->getOperand(0).getDefiningOp<IE::PadOp>();
    if (origPadOp == nullptr) {
        return mlir::failure();
    }

    if (origPadOp.mode() != IE::PadMode::CONSTANT) {
        return mlir::failure();
    }

    auto padsBegin = vpux::IE::extractPads(origPadOp.pads_begin_attrAttr(), log);
    if (mlir::failed(padsBegin)) {
        return mlir::failure();
    }

    auto padsEnd = vpux::IE::extractPads(origPadOp.pads_end_attrAttr(), log);
    if (mlir::failed(padsEnd)) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(origPadOp.pad_value_attr().hasValue(), "IE::PadOp has pad_value_attr() == nullptr {0}",
                      origPadOp->getLoc());
    const double padsValue = origPadOp.pad_value_attr().getValue().convertToDouble();
    if (!isDoubleEqual(padsValue, 0.f)) {
        return mlir::failure();
    }

    const auto origPadBegin = parseIntArrayAttr<int64_t>(padBegin);
    const auto origPadEnd = parseIntArrayAttr<int64_t>(padEnd);

    auto newPadsBegin = getIntArrayAttr(
            origOp->getContext(),
            ngraph::CoordinateDiff{
                    origPadBegin[Dims4D::PadsBegin::Top.ind()] + padsBegin.getValue()[Dims4D::Act::H.ind()],
                    origPadBegin[Dims4D::PadsBegin::Left.ind()] + padsBegin.getValue()[Dims4D::Act::W.ind()]});
    auto newPadsEnd = getIntArrayAttr(
            origOp->getContext(),
            ngraph::CoordinateDiff{
                    origPadEnd[Dims4D::PadsEnd::Bottom.ind()] + padsEnd.getValue()[Dims4D::Act::H.ind()],
                    origPadEnd[Dims4D::PadsEnd::Right.ind()] + padsEnd.getValue()[Dims4D::Act::W.ind()]});

    if (!VPU::NCEInvariant::verifyPads(kernelSizeAttr, newPadsBegin, newPadsEnd)) {
        return mlir::failure();
    }

    log.trace("Fuse PadOp {0} into {1}", origPadOp.getLoc(), origOp->getLoc());

    opRewriter(origPadOp.input(), newPadsBegin, newPadsEnd);

    return mlir::success();
}

//
// FuseConstantPadWithConv
//

class FuseConstantPadWithConv final : public mlir::OpRewritePattern<IE::ConvolutionOp> {
public:
    FuseConstantPadWithConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvolutionOp>(ctx), _log(log) {
        setDebugName("FuseConstantPadWithConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ConvolutionOp origConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseConstantPadWithConv::matchAndRewrite(IE::ConvolutionOp origConvolutionOp,
                                                             mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got Convolution layer at '{1}'", getDebugName(), origConvolutionOp->getLoc());

    const auto kernelSize = origConvolutionOp.filter().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto kernelSizeAttr = getIntArrayAttr(getContext(), kernelSize);

    return generalFusion(
            origConvolutionOp, kernelSizeAttr, origConvolutionOp.pads_beginAttr(), origConvolutionOp.pads_endAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(origConvolutionOp, origPadInput,
                                                               origConvolutionOp.filter(), origConvolutionOp.bias(),
                                                               origConvolutionOp.stridesAttr(), newPadsBegin,
                                                               newPadsEnd, origConvolutionOp.dilationsAttr(), nullptr);
            },
            _log.nest());
}

//
// FuseConstantPadWithGroupConv
//

class FuseConstantPadWithGroupConv final : public mlir::OpRewritePattern<IE::GroupConvolutionOp> {
public:
    FuseConstantPadWithGroupConv(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::GroupConvolutionOp>(ctx), _log(log) {
        setDebugName("FuseConstantPadWithGroupConv");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::GroupConvolutionOp origGroupConvolutionOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseConstantPadWithGroupConv::matchAndRewrite(IE::GroupConvolutionOp origGroupConvolutionOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got GroupConvolution layer at '{1}'", getDebugName(), origGroupConvolutionOp->getLoc());

    const auto kernelSize = origGroupConvolutionOp.filter().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto kernelSizeAttr = getIntArrayAttr(getContext(), kernelSize);

    return generalFusion(
            origGroupConvolutionOp, kernelSizeAttr, origGroupConvolutionOp.pads_beginAttr(),
            origGroupConvolutionOp.pads_endAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
                        origGroupConvolutionOp, origPadInput, origGroupConvolutionOp.filter(),
                        origGroupConvolutionOp.bias(), origGroupConvolutionOp.stridesAttr(), newPadsBegin, newPadsEnd,
                        origGroupConvolutionOp.dilationsAttr(), origGroupConvolutionOp.groupsAttr(),
                        origGroupConvolutionOp.post_opAttr());
            },
            _log.nest());
}

//
// FuseConstantPadWithMaxpool
//

class FuseConstantPadWithMaxpool final : public mlir::OpRewritePattern<IE::MaxPoolOp> {
public:
    FuseConstantPadWithMaxpool(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MaxPoolOp>(ctx), _log(log) {
        setDebugName("FuseConstantPadWithMaxpool");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::MaxPoolOp origMaxPoolOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseConstantPadWithMaxpool::matchAndRewrite(IE::MaxPoolOp origMaxPoolOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got MaxPool layer at '{1}'", getDebugName(), origMaxPoolOp->getLoc());

    const auto kernelSizeAttr = origMaxPoolOp.kernel_size();

    return generalFusion(
            origMaxPoolOp, kernelSizeAttr, origMaxPoolOp.pads_beginAttr(), origMaxPoolOp.pads_endAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(origMaxPoolOp, origPadInput, origMaxPoolOp.kernel_sizeAttr(),
                                                           origMaxPoolOp.stridesAttr(), newPadsBegin, newPadsEnd,
                                                           origMaxPoolOp.rounding_type(), origMaxPoolOp.post_opAttr());
            },
            _log.nest());
}

//
// FusePadOpsPass
//

class FusePadOpsPass final : public IE::FusePadOpsBase<FusePadOpsPass> {
public:
    explicit FusePadOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void FusePadOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FuseConstantPadWithConv>(&ctx, _log);
    patterns.add<FuseConstantPadWithGroupConv>(&ctx, _log);
    patterns.add<FuseConstantPadWithMaxpool>(&ctx, _log);

    auto func = getFunction();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSupportFusePadOpsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFusePadOpsPass(Logger log) {
    return std::make_unique<FusePadOpsPass>(log);
}
