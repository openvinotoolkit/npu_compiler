//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"

#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/utils/m2i_utils.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux {
#define GEN_PASS_DECL_CONVERTIETOVPUM2I
#define GEN_PASS_DEF_CONVERTIETOVPUM2I
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// ColorConvertToM2I
//

class ColorConvertToM2I final : public mlir::OpRewritePattern<IE::YuvToRgbOp> {
public:
    ColorConvertToM2I(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::YuvToRgbOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::YuvToRgbOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ColorConvertToM2I::matchAndRewrite(IE::YuvToRgbOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2IColorConvertOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2IColorConvertOp>(origOp->getLoc(), origOp.getType(), origOp.getInput1(),
                                                         origOp.getInFmtAttr(), origOp.getOutFmtAttr());
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// InterpolateToM2I
//

class InterpolateToM2I final : public mlir::OpRewritePattern<IE::InterpolateOp> {
public:
    InterpolateToM2I(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::InterpolateOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult InterpolateToM2I::matchAndRewrite(IE::InterpolateOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2IResizeOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    const auto interpMode = VPU::IEtoM2iInterpMode(origOp.getAttr().getMode().getValue());

    auto m2iOp = rewriter.create<VPU::M2IResizeOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(),
                                                   origOp.getSizesAttrAttr(), origOp.getAxesAttrAttr(), interpMode);
    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// BatchNormToM2I
//

class BatchNormToM2I final : public mlir::OpRewritePattern<IE::BatchNormInferenceOp> {
public:
    BatchNormToM2I(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::BatchNormInferenceOp>(ctx, vpux::benefitMid), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::BatchNormInferenceOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult BatchNormToM2I::matchAndRewrite(IE::BatchNormInferenceOp origOp,
                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto logCb = [&](const llvm::formatv_object_base& msg) {
        std::ignore = matchFailed(_log, rewriter, origOp, "[{0}] {1}", getDebugName(), msg.str());
    };

    if (!VPU::M2INormOp::isSupported(origOp, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true)) {
        return mlir::failure();
    }

    auto m2iOp = rewriter.create<VPU::M2INormOp>(
            origOp->getLoc(), origOp.getType(), origOp.getInput(), origOp.getGammaValueAttr(),
            origOp.getBetaValueAttr(), origOp.getMeanValueAttr(), origOp.getVarianceValueAttr(), origOp.getEpsAttr());

    rewriter.replaceOp(origOp, m2iOp.getOutput());

    return mlir::success();
}

//
// ConvertIEToVPUM2IPass
//

class ConvertIEToVPUM2IPass final : public impl::ConvertIEToVPUM2IBase<ConvertIEToVPUM2IPass> {
public:
    explicit ConvertIEToVPUM2IPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void ConvertIEToVPUM2IPass::safeRunOnFunc() {
    auto& ctx = getContext();

    auto module = getOperation();
    const auto arch = config::getArch(module);
    if (arch < config::ArchKind::NPU40XX) {
        _log.trace("Convert to VPU-M2I Pass enabled only for NPU40XX+ devices. Got: {0}", arch);
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ColorConvertToM2I>(&ctx, _log);
    patterns.add<InterpolateToM2I>(&ctx, _log);
    patterns.add<BatchNormToM2I>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertIEToVPUM2IPass
//

std::unique_ptr<mlir::Pass> vpux::createConvertIEToVPUM2IPass(Logger log) {
    return std::make_unique<ConvertIEToVPUM2IPass>(log);
}
