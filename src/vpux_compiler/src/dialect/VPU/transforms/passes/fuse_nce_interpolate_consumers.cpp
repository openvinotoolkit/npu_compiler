//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_FUSENCEINTERPOLATECONSUMERS
#define GEN_PASS_DEF_FUSENCEINTERPOLATECONSUMERS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
namespace {

//
// FuseNCEInterpolateConsumers
//

class FuseNCEInterpolateConsumers final : public mlir::OpRewritePattern<VPU::NCEInterpolateOp> {
public:
    FuseNCEInterpolateConsumers(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEInterpolateOp>(ctx), _log(log) {
        setDebugName("FuseNCEInterpolateConsumers");
    }

    mlir::LogicalResult matchAndRewrite(VPU::NCEInterpolateOp op, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseNCEInterpolateConsumers::matchAndRewrite(VPU::NCEInterpolateOp interpOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Found NCEInterpolate: {0}", interpOp->getLoc());

    auto sparseInput = mlir::cast<vpux::VPU::SparseTensorType>(interpOp.getInput().getType());

    auto seAttr = mlir::cast<vpux::VPU::SEInterpolateAttr>(sparseInput.getSeAttr());
    auto mode = seAttr.getMode().getValue();
    if (mode != VPU::NCEInterpolateMode::NEAREST) {
        return matchFailed(rewriter, interpOp, "Only NEAREST interpolate can be fused");
    }

    // Currently the operation is fused only if there is only one consumer operation
    // Experiments should be run in order to determine if the performance increases when all users have a sparse input
    if (!interpOp.getOutput().hasOneUse()) {
        return matchFailed(rewriter, interpOp, "Operation has more than one user");
    }

    auto users = interpOp->getUsers();
    if (!mlir::isa<VPU::NCEConvolutionOp>(*users.begin())) {
        return matchFailed(rewriter, interpOp, "User is not an NCEConvolution");
    }

    interpOp.getOutput().replaceAllUsesWith(interpOp.getInput());
    rewriter.eraseOp(interpOp);

    return mlir::success();
}

//
// FuseNCEInterpolateConsumersPass
//

class FuseNCEInterpolateConsumersPass final :
        public VPU::impl::FuseNCEInterpolateConsumersBase<FuseNCEInterpolateConsumersPass> {
public:
    explicit FuseNCEInterpolateConsumersPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void FuseNCEInterpolateConsumersPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<FuseNCEInterpolateConsumers>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createFuseNCEInterpolateConsumersPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createFuseNCEInterpolateConsumersPass(Logger log) {
    return std::make_unique<FuseNCEInterpolateConsumersPass>(log);
}
