//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp>
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/Dialect/Quant/IR/Quant.h>
#include <mlir/Transforms/WalkPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEUNALIGNEDQDQSEQ
#define GEN_PASS_DEF_OPTIMIZEUNALIGNEDQDQSEQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool isDPUOp(mlir::Operation* op, Logger log) {
    auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op);
    if (convOp != nullptr) {
        if (!VPU::NCEConvolutionOp::verifyKernel(convOp, log).failed()) {
            return true;
        }
    }
    auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op);
    if (grConvOp != nullptr) {
        if (!VPU::NCEDepthConvolutionOp::verifyKernel(grConvOp, log).failed()) {
            return true;
        }
    }
    auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
    if (maxPoolOp != nullptr) {
        if (!VPU::NCEMaxPoolOp::verifyKernel(maxPoolOp, log).failed()) {
            return true;
        }
    }
    auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(op);
    if (subtractOp != nullptr) {
        if (!VPU::NCEEltwiseOp::verifyKernel(subtractOp, log).failed()) {
            return true;
        }
    }
    auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(op);
    if (multiplyOp != nullptr) {
        if (!VPU::NCEEltwiseOp::verifyKernel(multiplyOp, log).failed()) {
            return true;
        }
    }
    auto addOp = mlir::dyn_cast<IE::AddOp>(op);
    if (addOp != nullptr) {
        if (!VPU::NCEEltwiseOp::verifyKernel(addOp, log).failed()) {
            return true;
        }
    }
    return false;
}

bool shouldConvertFakeQuantizeOp(IE::FakeQuantizeOp fakeQuantize, Logger log) {
    if (!fakeQuantize->hasOneUse()) {
        return false;
    }
    if (!IE::isPerTensorFQ({fakeQuantize})) {
        return false;
    }
    auto affineReshape = fakeQuantize.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (affineReshape == nullptr) {
        return false;
    }
    if (!affineReshape->hasOneUse()) {
        return false;
    }
    const auto outType = mlir::dyn_cast<vpux::NDTypeInterface>(affineReshape.getType());
    if (outType.getRank() != 4) {
        return false;
    }
    if ((outType.getShape()[Dims4D::Act::C] % 16) == 0) {
        return false;
    }
    auto prevOp = affineReshape.getInput().getDefiningOp();
    if (prevOp == nullptr) {
        return false;
    }
    return isDPUOp(prevOp, log);
}

//
// UnalignedFakeQuantizeRewriter
//

class UnalignedFakeQuantizeRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    UnalignedFakeQuantizeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("UnalignedFakeQuantizeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnalignedFakeQuantizeRewriter::matchAndRewrite(IE::FakeQuantizeOp oldFakeQuantize,
                                                                   mlir::PatternRewriter& rewriter) const {
    if (!shouldConvertFakeQuantizeOp(oldFakeQuantize, _log)) {
        return mlir::failure();
    }

    auto oldAffineReshape = oldFakeQuantize.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (oldAffineReshape == nullptr) {
        return matchFailed(_log.nest(), rewriter, oldAffineReshape, "No following FakeQuantize");
    }
    auto newFakeQuantize = IE::FakeQuantizeOp::create(
            rewriter, oldFakeQuantize->getLoc(), oldAffineReshape.getInput(), oldFakeQuantize.getInputLow(),
            oldFakeQuantize.getInputHigh(), oldFakeQuantize.getOutputLow(), oldFakeQuantize.getOutputHigh(),
            oldFakeQuantize.getLevelsAttr(), oldFakeQuantize.getLowFpTypeAttr(),
            oldFakeQuantize.getAutoBroadcastAttr());
    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(oldFakeQuantize, newFakeQuantize.getOutput(),
                                                     oldAffineReshape.getDimMappingAttr(),
                                                     oldAffineReshape.getShapeValueAttr());
    rewriter.eraseOp(oldAffineReshape);
    return mlir::success();
}

class OptimizeUnalignedQDQSeqPass final : public IE::impl::OptimizeUnalignedQDQSeqBase<OptimizeUnalignedQDQSeqPass> {
public:
    explicit OptimizeUnalignedQDQSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeUnalignedQDQSeqPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnalignedFakeQuantizeRewriter>(&ctx, _log);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
}

}  // namespace

//
// createOptimizeUnalignedQDQSeq
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeUnalignedQDQSeqPass(Logger log) {
    return std::make_unique<OptimizeUnalignedQDQSeqPass>(log);
}
