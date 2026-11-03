//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"
#include "vpux/utils/core/numeric.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
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
    auto inType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getInput().getType()).getElementType());
    auto outType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
            mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getOutput().getType()).getElementType());
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

struct MatchResult {
    mlir::Operation* parentOp = nullptr;
    mlir::FloatAttr clampMin;
    mlir::FloatAttr clampMax;
};

// NOTE: If MatchResult returns parentOp as nullptr then we failed the search
mlir::FailureOr<MatchResult> findLayerWithPostOp(IE::ClampOp origOp, mlir::PatternRewriter& rewriter, Logger log) {
    MatchResult result;
    result.clampMin = origOp.getMinAttr();
    result.clampMax = origOp.getMaxAttr();

    auto clampInput = origOp.getInput();
    if (mlir::isa<mlir::BlockArgument>(clampInput)) {
        return matchFailed(log, rewriter, origOp, "Clamp at loc '{0}' has input BlockArgument", origOp->getLoc());
    }

    auto currentOp = clampInput.getDefiningOp();
    while (currentOp->hasOneUse()) {
        if (!mlir::isa<IE::SliceOp>(currentOp) && !IE::isPureViewOp(currentOp)) {
            break;
        }

        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(currentOp)) {
            if (!isSupportedQuantizeCast(quantizeCastOp)) {
                return matchFailed(log, rewriter, quantizeCastOp, "QuantizeCastOp at loc '{0}' is not supported",
                                   quantizeCastOp->getLoc());
            }

            auto inType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
                    mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getInput().getType()).getElementType());
            auto outType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(
                    mlir::cast<vpux::NDTypeInterface>(quantizeCastOp.getOutput().getType()).getElementType());

            auto inScale = inType.getScale();
            auto outScale = outType.getScale();

            auto newMin = result.clampMin.getValueAsDouble() * inScale / outScale;
            auto newMax = result.clampMax.getValueAsDouble() * inScale / outScale;

            result.clampMin = getFPAttr(rewriter, newMin);
            result.clampMax = getFPAttr(rewriter, newMax);
        }

        if (mlir::isa<mlir::BlockArgument>(currentOp->getOperand(0))) {
            return matchFailed(log, rewriter, currentOp, "Op '{0}' has input which is a BlockArgument at loc '{1}'",
                               currentOp->getName(), currentOp->getLoc());
        }

        currentOp = currentOp->getOperand(0).getDefiningOp();
    }

    if (!currentOp->hasOneUse()) {
        return matchFailed(log, rewriter, currentOp, "Op '{0}' at loc '{1}' has more than one use",
                           currentOp->getName(), currentOp->getLoc());
    }

    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(currentOp);
    if (layerWithPostOp == nullptr) {
        return matchFailed(log, rewriter, currentOp,
                           "Op '{0}' at loc '{1}' does not implement LayerWithPostOpInterface", currentOp->getName(),
                           currentOp->getLoc());
    }

    result.parentOp = currentOp;

    return result;
}

mlir::LogicalResult LayerRewriter::matchAndRewrite(IE::ClampOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got clamp layer at '{1}' ", origOp->getName(), origOp->getLoc());

    auto foundLayer = findLayerWithPostOp(origOp, rewriter, _log);
    if (mlir::failed(foundLayer)) {
        return matchFailed(_log, rewriter, origOp,
                           "Could not find LayerWithPostOpInterface producer through ViewOps for ClampOp at loc '{0}'",
                           origOp->getLoc());
    }

    auto matchResult = foundLayer.value();
    auto* parentOp = matchResult.parentOp;
    if (parentOp == origOp.getInput().getDefiningOp()) {
        return matchFailed(_log, rewriter, origOp, "ClampOp at loc '{0}' is already placed right after its producer",
                           origOp->getLoc());
    }

    rewriter.setInsertionPointAfter(parentOp);
    auto newClampOp = IE::ClampOp::create(rewriter, origOp->getLoc(), parentOp->getResult(0), matchResult.clampMin,
                                          matchResult.clampMax);
    rewriter.replaceAllUsesExcept(parentOp->getResult(0), newClampOp.getOutput(), newClampOp);
    rewriter.replaceOp(origOp, origOp.getInput());

    _log.trace("Successfully moved Clamp after Op '{0}' at loc '{1}'", parentOp->getName(), parentOp->getLoc());

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
    collectOpsAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

//
// createSwapViewOpAndClampPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createSwapViewOpAndClampPass(Logger log) {
    return std::make_unique<SwapViewOpAndClampPass>(log);
}
