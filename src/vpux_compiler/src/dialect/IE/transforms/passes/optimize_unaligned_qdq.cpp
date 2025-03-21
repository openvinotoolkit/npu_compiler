//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEUNALIGNEDQDQSEQ
#define GEN_PASS_DEF_OPTIMIZEUNALIGNEDQDQSEQ
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

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
    auto oldAffineReshape = oldFakeQuantize.getInput().getDefiningOp<IE::AffineReshapeOp>();
    if (oldAffineReshape == nullptr) {
        return matchFailed(_log.nest(), rewriter, oldAffineReshape, "No following FakeQuantize");
    }
    auto newFakeQuantize = rewriter.create<IE::FakeQuantizeOp>(
            oldFakeQuantize->getLoc(), oldAffineReshape.getInput(), oldFakeQuantize.getInputLow(),
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

    bool isDPU(mlir::Operation* op) {
        auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op);
        if (convOp != nullptr) {
            if (!VPU::NCEConvolutionOp::verifyKernel(convOp, _log).failed()) {
                return true;
            }
        }
        auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op);
        if (grConvOp != nullptr) {
            if (!VPU::NCEDepthConvolutionOp::verifyKernel(grConvOp, _log).failed()) {
                return true;
            }
        }
        auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        if (maxPoolOp != nullptr) {
            if (!VPU::NCEMaxPoolOp::verifyKernel(maxPoolOp, _log).failed()) {
                return true;
            }
        }
        auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(op);
        if (subtractOp != nullptr) {
            if (!VPU::NCEEltwiseOp::verifyKernel(subtractOp, _log).failed()) {
                return true;
            }
        }
        auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(op);
        if (multiplyOp != nullptr) {
            if (!VPU::NCEEltwiseOp::verifyKernel(multiplyOp, _log).failed()) {
                return true;
            }
        }
        auto addOp = mlir::dyn_cast<IE::AddOp>(op);
        if (addOp != nullptr) {
            if (!VPU::NCEEltwiseOp::verifyKernel(addOp, _log).failed()) {
                return true;
            }
        }
        return false;
    }
};

void OptimizeUnalignedQDQSeqPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();
    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp fakeQuantize) {
        if (!fakeQuantize->hasOneUse()) {
            return true;
        }
        if (!IE::isPerTensorFQ({fakeQuantize})) {
            return true;
        }
        auto affineReshape = fakeQuantize.getInput().getDefiningOp<IE::AffineReshapeOp>();
        if (affineReshape == nullptr) {
            return true;
        }
        if (!affineReshape->hasOneUse()) {
            return true;
        }
        const auto outType = affineReshape.getType().dyn_cast<vpux::NDTypeInterface>();
        if (outType.getRank() != 4) {
            return true;
        }
        if ((outType.getShape()[Dims4D::Act::C] % 16) == 0) {
            return true;
        }
        auto prevOp = affineReshape.getInput().getDefiningOp();
        if (prevOp == nullptr) {
            return true;
        }
        if (!isDPU(prevOp)) {
            return true;
        }
        return false;
    });
    target.addLegalOp<IE::AffineReshapeOp>();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnalignedFakeQuantizeRewriter>(&ctx, _log);
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeUnalignedQDQSeq
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeUnalignedQDQSeqPass(Logger log) {
    return std::make_unique<OptimizeUnalignedQDQSeqPass>(log);
}
