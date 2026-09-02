//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_context.hpp"
#include "vpux/compiler/dialect/VPU/utils/tiling_pass_config_utils.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"

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

    const auto options = VPU::TilingContextOptions(VPU::TilingContextOptions::ContextType::TILING, _enableSCFTiling);
    auto tilingContext = VPU::createTilingContext(op, options);
    auto tilingResult = tilingContext.applyTiling(rewriter, _log);

    return tilingResult;
}

//
// ApplyTilingPass
//
class ApplyTilingPass final : public VPU::impl::ApplyTilingBase<ApplyTilingPass> {
public:
    ApplyTilingPass(bool enableSCFTiling, bool enableDynamicDimAlignment, Logger log)
            : _enableSCFTiling(enableSCFTiling), _enableDynamicDimAlignment(enableDynamicDimAlignment) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
    bool _enableSCFTiling = false;
    bool _enableDynamicDimAlignment = false;
};

mlir::LogicalResult ApplyTilingPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (enableSCFTiling.hasValue()) {
        _log.trace("Overloading ApplyTilingPass argument by MLIR variable");
        _enableSCFTiling = enableSCFTiling;
    }
    if (enableDynamicDimAlignment.hasValue()) {
        _log.trace("Overloading ApplyTilingPass argument by MLIR variable");
        _enableDynamicDimAlignment = enableDynamicDimAlignment.getValue();
    }
    return mlir::success();
}

//
// safeRunOnFunc
//
void ApplyTilingPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto funcOp = getOperation();

    if (_enableDynamicDimAlignment) {
        VPU::setDynamicDimAlignment(funcOp);
    }

    funcOp->walk([tilingIndex = 0ll](VPU::TilingInfoOpInterface iface) mutable {
        if (iface->hasAttr(tilingStrategy)) {
            iface->setAttr(TILING_LOOP_INDEX_ATTR_NAME, TilingLoopIndexAttr::get(iface->getContext(), tilingIndex));
            ++tilingIndex;
        }
    });

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

    VPU::removeDynamicDimAlignment(funcOp);

    // Clean up the temporary pinnedStrategy attribute after apply tiling
    funcOp->walk([](mlir::Operation* op) {
        op->removeAttr(VPU::pinnedStrategy);
    });
}
}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createApplyTilingPass(bool enableSCFTiling, bool enableDynamicDimAlignment,
                                                             Logger log) {
    return std::make_unique<ApplyTilingPass>(enableSCFTiling, enableDynamicDimAlignment, log);
}
