//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/QuantTypes.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPVIEWOPANDCLAMP
#define GEN_PASS_DEF_SWAPVIEWOPANDCLAMP
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// LayerRewriter
//

class LayerRewriter final : public mlir::OpRewritePattern<IE::ClampOp> {
public:
    LayerRewriter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::ClampOp>(ctx), _log(log) {
        setDebugName("LayerRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

bool isSupportedQuantizeCast(IE::QuantizeCastOp quantizeCastOp) {
    auto inType = quantizeCastOp.getInput()
                          .getType()
                          .cast<vpux::NDTypeInterface>()
                          .getElementType()
                          .dyn_cast<mlir::quant::UniformQuantizedType>();
    auto outType = quantizeCastOp.getOutput()
                           .getType()
                           .cast<vpux::NDTypeInterface>()
                           .getElementType()
                           .dyn_cast<mlir::quant::UniformQuantizedType>();
    if (inType == nullptr || outType == nullptr) {
        return false;
    }

    auto inZeroPoint = inType.getZeroPoint();
    auto outZeroPoint = outType.getZeroPoint();
    auto outScale = outType.getScale();

    if (!isDoubleEqual(inZeroPoint, outZeroPoint)) {
        return false;
    }

    if (isDoubleEqual(outScale, 0.0f)) {
        return false;
    }
    return true;
}

bool parentCanFuseClamp(mlir::Operation* parentOp, IE::ClampOp origOp, Logger log) {
    if (!parentOp->hasOneUse()) {
        return false;
    }

    if (mlir::isa<IE::SliceOp>(parentOp) || IE::isPureViewOp(parentOp)) {
        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (!isSupportedQuantizeCast(quantizeCastOp)) {
                return false;
            }
        }
        if (parentOp->getOperand(0).isa<mlir::BlockArgument>()) {
            return false;
        }
        return parentCanFuseClamp(parentOp->getOperand(0).getDefiningOp(), origOp, log);
    }

    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(parentOp);
    if (layerWithPostOp != nullptr) {
        const auto logCb = [&](const formatv_object_base& msg) {
            log.trace("{0}", msg.str());
        };
        if (!layerWithPostOp.getPostOp().has_value() && layerWithPostOp.isSupportedPostOp(origOp, logCb)) {
            return true;
        }
    }

    return false;
}

bool isBenefitToSwap(IE::ClampOp origOp, Logger log) {
    if (origOp.getInput().isa<mlir::BlockArgument>()) {
        return false;
    }

    return parentCanFuseClamp(origOp.getInput().getDefiningOp(), origOp, log);
}

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got clamp layer at '{1}' ", origOp->getName(), origOp->getLoc());

    if (!isBenefitToSwap(origOp, _log)) {
        return mlir::failure();
    }
    auto parentOp = origOp.getInput().getDefiningOp();
    if (!mlir::isa<IE::SliceOp>(parentOp) && !IE::isPureViewOp(parentOp)) {
        return mlir::failure();
    }

    auto clampMin = origOp.getMinAttr();
    auto clampMax = origOp.getMaxAttr();
    // for quantizeCast, we need to re-calculate the min max for clamp
    if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
        auto inType = quantizeCastOp.getInput()
                              .getType()
                              .cast<vpux::NDTypeInterface>()
                              .getElementType()
                              .dyn_cast<mlir::quant::UniformQuantizedType>();
        auto outType = quantizeCastOp.getOutput()
                               .getType()
                               .cast<vpux::NDTypeInterface>()
                               .getElementType()
                               .dyn_cast<mlir::quant::UniformQuantizedType>();

        auto inScale = inType.getScale();
        auto outScale = outType.getScale();

        auto newMin = clampMin.getValueAsDouble() * inScale / outScale;
        auto newMax = clampMax.getValueAsDouble() * inScale / outScale;
        clampMin = getFPAttr(rewriter, newMin);
        clampMax = getFPAttr(rewriter, newMax);
    }

    auto newClampOp = rewriter.create<IE::ClampOp>(origOp->getLoc(), parentOp->getOperand(0), clampMin, clampMax);
    mlir::IRMapping mapper;
    mapper.map(parentOp->getOperand(0), newClampOp.getOutput());
    auto newOp = rewriter.clone(*parentOp, mapper);
    vpux::inferReturnTypes(newOp, vpux::InferShapedTypeMode::ALL);
    rewriter.replaceOp(origOp, newOp->getResult(0));

    _log.trace("successfully swap clamp with {0}", newOp->getName());

    return mlir::success();
}

//
// SwapViewOpAndClampPass
//

class SwapViewOpAndClampPass final : public IE::impl::SwapViewOpAndClampBase<SwapViewOpAndClampPass> {
public:
    explicit SwapViewOpAndClampPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void SwapViewOpAndClampPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<LayerRewriter>(&ctx, _log.nest());

    auto func = getOperation();
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createSwapViewOpAndClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapViewOpAndClampPass(Logger log) {
    return std::make_unique<SwapViewOpAndClampPass>(log);
}
