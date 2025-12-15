//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "vpux/compiler/utils/attributes.hpp"

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
    ApplyTiling(mlir::MLIRContext* ctx, bool enableSCFTiling, Logger log)
            : mlir::OpInterfaceRewritePattern<VPU::TilingBuilderOpInterface>(ctx),
              _enableSCFTiling(enableSCFTiling),
              _log(log) {
        this->setDebugName("ApplyTiling");
    }
    mlir::LogicalResult matchAndRewrite(VPU::TilingBuilderOpInterface origOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    bool _enableSCFTiling = false;
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

    const auto outputShape = getShape(op->getResult(0));
    const auto strategy = Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(op->getAttr(tilingStrategy))));

    VPUX_THROW_UNLESS(outputShape.size() == strategy.size(),
                      "Number of dimensions of output shape and tiling strategy must match");

    _log.nest().trace("Applying tiling for op {0} at {1}, tiles: {2}", op->getName(), op->getLoc(), strategy);

    auto tilingContext = VPU::createTilingContext(op, _enableSCFTiling);
    auto tilingResult = tilingContext.applyTiling(rewriter, _log);

    return tilingResult;
}

//
// ApplyTilingPass
//
class ApplyTilingPass final : public VPU::impl::ApplyTilingBase<ApplyTilingPass> {
public:
    explicit ApplyTilingPass(bool enableSCFTiling, Logger log): _enableSCFTiling(enableSCFTiling) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableSCFTiling = false;
};

mlir::LogicalResult ApplyTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableSCFTiling.hasValue()) {
        _log.trace("Overloading ApplyTilingPass argument by MLIR variable");
        _enableSCFTiling = enableSCFTiling;
    }
    return mlir::success();
}

//
// safeRunOnFunc
//
void ApplyTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalOp<VPU::SliceOp, VPU::ConcatOp>();
    target.addLegalDialect<mlir::scf::SCFDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
        if (auto iface = mlir::dyn_cast<VPU::TilingInfoOpInterface>(op)) {
            return !op->hasAttr(tilingStrategy);
        }
        return true;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<ApplyTiling>(&ctx, _enableSCFTiling, _log);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createApplyTilingPass(bool enableSCFTiling, Logger log) {
    return std::make_unique<ApplyTilingPass>(enableSCFTiling, log);
}
