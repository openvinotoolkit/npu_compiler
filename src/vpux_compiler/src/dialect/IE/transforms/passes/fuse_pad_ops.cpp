//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/pad_extract.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <openvino/core/coordinate_diff.hpp>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEPADOPS
#define GEN_PASS_DEF_FUSEPADOPS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

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

    if (origPadOp.getMode() != IE::PadMode::CONSTANT) {
        return mlir::failure();
    }

    auto padsBegin = vpux::IE::extractPads(origPadOp.getPadsBeginAttrAttr(), log);
    if (mlir::failed(padsBegin) || padsBegin.value().size() != 4) {
        return mlir::failure();
    }

    auto padsEnd = vpux::IE::extractPads(origPadOp.getPadsEndAttrAttr(), log);
    if (mlir::failed(padsEnd) || padsEnd.value().size() != 4) {
        return mlir::failure();
    }

    VPUX_THROW_UNLESS(origPadOp.getPadValueAttr().has_value(), "IE::PadOp has getPadValueAttr() == nullptr {0}",
                      origPadOp->getLoc());
    const double padsValue = origPadOp.getPadValueAttr().value().convertToDouble();
    if (!isDoubleEqual(padsValue, 0.f)) {
        return mlir::failure();
    }

    const auto origPadBegin = parseIntArrayAttr<int64_t>(padBegin);
    const auto origPadEnd = parseIntArrayAttr<int64_t>(padEnd);

    auto newPadsBegin = getIntArrayAttr(
            origOp->getContext(),
            ov::CoordinateDiff{origPadBegin[Dims4D::PadsBegin::Top.ind()] + padsBegin.value()[Dims4D::Act::H.ind()],
                               origPadBegin[Dims4D::PadsBegin::Left.ind()] + padsBegin.value()[Dims4D::Act::W.ind()]});
    auto newPadsEnd = getIntArrayAttr(
            origOp->getContext(),
            ov::CoordinateDiff{origPadEnd[Dims4D::PadsEnd::Bottom.ind()] + padsEnd.value()[Dims4D::Act::H.ind()],
                               origPadEnd[Dims4D::PadsEnd::Right.ind()] + padsEnd.value()[Dims4D::Act::W.ind()]});

    if (!VPU::NCEInvariant::verifyPads(kernelSizeAttr, newPadsBegin, newPadsEnd)) {
        return mlir::failure();
    }

    log.trace("Fuse PadOp {0} into {1}", origPadOp.getLoc(), origOp->getLoc());

    opRewriter(origPadOp.getInput(), newPadsBegin, newPadsEnd);

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

    const auto kernelSize = origConvolutionOp.getFilter().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto kernelSizeAttr = getIntArrayAttr(getContext(), kernelSize);

    return generalFusion(
            origConvolutionOp, kernelSizeAttr, origConvolutionOp.getPadsBeginAttr(), origConvolutionOp.getPadsEndAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::ConvolutionOp>(
                        origConvolutionOp, origPadInput, origConvolutionOp.getFilter(), origConvolutionOp.getBias(),
                        origConvolutionOp.getStridesAttr(), newPadsBegin, newPadsEnd,
                        origConvolutionOp.getDilationsAttr(), nullptr, nullptr, origConvolutionOp.getStaticScaleAttr(),
                        origConvolutionOp.getOutputChannelsAttr(), origConvolutionOp.getInputChannelsAttr());
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

    const auto kernelSize = origGroupConvolutionOp.getFilter().getType().cast<vpux::NDTypeInterface>().getShape();
    const auto kernelSizeAttr = getIntArrayAttr(getContext(), kernelSize);

    return generalFusion(
            origGroupConvolutionOp, kernelSizeAttr, origGroupConvolutionOp.getPadsBeginAttr(),
            origGroupConvolutionOp.getPadsEndAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::GroupConvolutionOp>(
                        origGroupConvolutionOp, origPadInput, origGroupConvolutionOp.getFilter(),
                        origGroupConvolutionOp.getBias(), origGroupConvolutionOp.getStridesAttr(), newPadsBegin,
                        newPadsEnd, origGroupConvolutionOp.getDilationsAttr(), origGroupConvolutionOp.getGroupsAttr(),
                        origGroupConvolutionOp.getPostOpAttr(), origGroupConvolutionOp.getClampAttr(),
                        origGroupConvolutionOp.getOutputChannelsAttr(), origGroupConvolutionOp.getInputChannelsAttr());
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

    const auto kernelSizeAttr = origMaxPoolOp.getKernelSize();

    return generalFusion(
            origMaxPoolOp, kernelSizeAttr, origMaxPoolOp.getPadsBeginAttr(), origMaxPoolOp.getPadsEndAttr(),
            [&](mlir::Value origPadInput, mlir::ArrayAttr newPadsBegin, mlir::ArrayAttr newPadsEnd) {
                rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
                        origMaxPoolOp, origPadInput, origMaxPoolOp.getKernelSizeAttr(), origMaxPoolOp.getStridesAttr(),
                        newPadsBegin, newPadsEnd, origMaxPoolOp.getRoundingType(), origMaxPoolOp.getPostOpAttr(),
                        origMaxPoolOp.getClampAttr(), origMaxPoolOp.getOutputChannelsAttr(),
                        origMaxPoolOp.getInputChannelsAttr());
            },
            _log.nest());
}

//
// FusePadOpsPass
//

class FusePadOpsPass final : public IE::impl::FusePadOpsBase<FusePadOpsPass> {
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

    auto func = getOperation();
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
