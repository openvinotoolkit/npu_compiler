//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"

namespace vpux::VPU {
#define GEN_PASS_DECL_APPLYTILING
#define GEN_PASS_DEF_APPLYTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// ApplyTiling
//

class ApplyTiling final : public mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface> {
public:
    ApplyTiling(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx), _log(log) {
        this->setDebugName("ApplyTiling");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ApplyTiling::matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto op = origOp.getOperation();
    if (!op->hasAttr(tilingStrategy)) {
        _log.nest().trace("No tiling strategy or it has already been applied.");
        return mlir::failure();
    }

    auto tilingInfo = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op);
    VPUX_THROW_WHEN(tilingInfo == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface", op->getName());

    const auto outputShape = getShape(op->getResult(0));
    const auto strategy = Shape(parseIntArrayAttr<int64_t>(op->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()));

    VPUX_THROW_UNLESS(outputShape.size() == strategy.size(),
                      "Number of dimensions of output shape and tiling strategy must match");

    _log.nest().trace("Applying tiling for op {0} at {1}, tiles: {2}", op->getName(), op->getLoc(), strategy);

    const auto tiles = fillDividedTiles(origOp, strategy, outputShape);

    op->removeAttr(tilingStrategy);

    if (mlir::failed(tiles)) {
        return mlir::failure();
    }

    _log.nest().trace("Creating {0} tiles", tiles.value().size());
    return VPU::applyTileStrategy(origOp, tiles.value(), rewriter, _log);
}

//
// ApplyTilingPass
//
class ApplyTilingPass final : public VPU::impl::ApplyTilingBase<ApplyTilingPass> {
public:
    explicit ApplyTilingPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//
void ApplyTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
            return !op->hasAttr(tilingStrategy);
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ApplyTiling>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createApplyTilingPass(Logger log) {
    return std::make_unique<ApplyTilingPass>(log);
}
