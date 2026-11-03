//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/ppe_version_config.hpp"
#include "vpux/compiler/dialect/config/utils/config_option_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_INSERTIDENTITYPOOLBETWEEND2SANDCONV
#define GEN_PASS_DEF_INSERTIDENTITYPOOLBETWEEND2SANDCONV
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

class InsertIdentityPoolBetweenD2SAndConvRewriter final : public mlir::OpRewritePattern<VPU::NCEConvolutionOp> {
public:
    InsertIdentityPoolBetweenD2SAndConvRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEConvolutionOp>(ctx), _log(log) {
        setDebugName("InsertIdentityPoolBetweenD2SAndConvRewriter");
    }

private:
    mlir::LogicalResult matchAndRewrite(VPU::NCEConvolutionOp convOp, mlir::PatternRewriter& rewriter) const final;

    Logger _log;
};

mlir::LogicalResult InsertIdentityPoolBetweenD2SAndConvRewriter::matchAndRewrite(
        VPU::NCEConvolutionOp convOp, mlir::PatternRewriter& rewriter) const {
    const auto rawFilterShape = mlir::getConstantIntValues(convOp.getMixedRawFilterShape());
    if (!rawFilterShape.has_value()) {
        return mlir::failure();
    }

    if (rawFilterShape.value()[Dims4D::Filter::KY.ind()] == 1 &&
        rawFilterShape.value()[Dims4D::Filter::KX.ind()] == 1) {
        return mlir::failure();
    }

    auto depthToSpaceOp = convOp.getInput().getDefiningOp<VPU::DepthToSpaceOp>();
    if (depthToSpaceOp == nullptr) {
        return mlir::failure();
    }

    auto* ctx = convOp.getContext();
    const auto kernelSizeAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto stridesAttr = getIntArrayAttr(ctx, SmallVector<int64_t>{1, 1});
    const auto padAttr = VPU::getPaddingAttr(ctx, PadInfo(0, 0, 0, 0));
    const auto ppeAttr = VPU::getPpeConfig(ctx).retrievePPEAttribute(depthToSpaceOp);
    const auto inputPaddingAttr = convOp.getInputPaddingAttr();

    rewriter.setInsertionPointAfter(depthToSpaceOp);
    auto avgPoolOp = VPU::NCEAveragePoolOp::create(
            rewriter, takeOpLoc(convOp, "identity_avgpool"), convOp.getInput().getType(), convOp.getInput(),
            /*weight_table_scale=*/nullptr, /*weight_table_bias=*/nullptr, kernelSizeAttr, stridesAttr, padAttr,
            ppeAttr,
            /*mpe_engine=*/nullptr, /*multiClusterStrategy=*/nullptr, inputPaddingAttr, inputPaddingAttr);

    rewriter.replaceAllUsesExcept(depthToSpaceOp.getOutput(), avgPoolOp.getOutput(), {avgPoolOp});
    _log.trace("Inserted identity NCE AveragePool between DepthToSpace and Convolution at '{0}'", convOp->getLoc());
    return mlir::success();
}

class InsertIdentityPoolBetweenD2SAndConvPass final :
        public VPU::impl::InsertIdentityPoolBetweenD2SAndConvBase<InsertIdentityPoolBetweenD2SAndConvPass> {
public:
    explicit InsertIdentityPoolBetweenD2SAndConvPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void InsertIdentityPoolBetweenD2SAndConvPass::safeRunOnFunc() {
    auto func = getOperation();
    // The identity pool only pays off for the power-oriented vertical fusion objective.
    if (!config::isPowerOptimized(func)) {
        return;
    }

    auto& ctx = getContext();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<InsertIdentityPoolBetweenD2SAndConvRewriter>(&ctx, _log);
    walkAndApplyPatterns(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPU::createInsertIdentityPoolBetweenD2SAndConvPass(Logger log) {
    return std::make_unique<InsertIdentityPoolBetweenD2SAndConvPass>(log);
}
