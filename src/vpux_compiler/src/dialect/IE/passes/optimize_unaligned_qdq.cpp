//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <vpux/compiler/dialect/VPUIP/nce_invariant.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IE/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

namespace {

//
// ConvertAffineReshapeRewriter
//

class ConvertAffineReshapeRewriter final : public mlir::OpRewritePattern<IE::FakeQuantizeOp> {
public:
    ConvertAffineReshapeRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FakeQuantizeOp>(ctx), _log(log) {
        setDebugName("ConvertAffineReshapeRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FakeQuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult ConvertAffineReshapeRewriter::matchAndRewrite(IE::FakeQuantizeOp oldFakeQuantize,
                                                                  mlir::PatternRewriter& rewriter) const {
    auto oldAffineReshape = oldFakeQuantize.input().getDefiningOp<IE::AffineReshapeOp>();
    if (oldAffineReshape == nullptr) {
        return matchFailed(_log.nest(), rewriter, oldAffineReshape, "No following FakeQuantize");
    }
    auto newFakeQuantize = rewriter.create<IE::FakeQuantizeOp>(
            oldFakeQuantize->getLoc(), oldAffineReshape.input(), oldFakeQuantize.input_low(),
            oldFakeQuantize.input_high(), oldFakeQuantize.output_low(), oldFakeQuantize.output_high(),
            oldFakeQuantize.levelsAttr(), oldFakeQuantize.auto_broadcastAttr());

    rewriter.replaceOpWithNewOp<IE::AffineReshapeOp>(oldFakeQuantize, newFakeQuantize.output(),
                                                     oldAffineReshape.dim_mappingAttr(),
                                                     oldAffineReshape.shape_valueAttr());
    return mlir::success();
}

class OptimizeUnalignedQDQSeqPass final : public IE::OptimizeUnalignedQDQSeqBase<OptimizeUnalignedQDQSeqPass> {
public:
    explicit OptimizeUnalignedQDQSeqPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    bool isDPU(mlir::Operation* op) {
        auto convOp = mlir::dyn_cast<IE::ConvolutionOp>(op);
        if (convOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(convOp, _log).failed()) {
                return true;
            }
        }
        auto grConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(op);
        if (grConvOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(grConvOp, _log).failed()) {
                return true;
            }
        }
        auto maxPoolOp = mlir::dyn_cast<IE::MaxPoolOp>(op);
        if (maxPoolOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(maxPoolOp, _log).failed()) {
                return true;
            }
        }
        auto andOp = mlir::dyn_cast<IE::AndOp>(op);
        if (andOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(andOp, _log).failed()) {
                return true;
            }
        }
        auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(op);
        if (subtractOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(subtractOp, _log).failed()) {
                return true;
            }
        }
        auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(op);
        if (multiplyOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(multiplyOp, _log).failed()) {
                return true;
            }
        }
        auto addOp = mlir::dyn_cast<IE::AddOp>(op);
        if (addOp != nullptr) {
            if (!VPUIP::NCEInvariant::verifyKernel(addOp, _log).failed()) {
                return true;
            }
        }
        return false;
    }
};

void OptimizeUnalignedQDQSeqPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getFunction();
    mlir::ConversionTarget target(ctx);

    target.addDynamicallyLegalOp<IE::FakeQuantizeOp>([&](IE::FakeQuantizeOp fakeQuantize) {
        if (!fakeQuantize->hasOneUse()) {
            return true;
        }
        const auto axis = IE::getQuantAxisIndex(fakeQuantize);
        if (axis.hasValue()) {
            return true;
        }
        auto affineReshape = fakeQuantize.input().getDefiningOp<IE::AffineReshapeOp>();
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
        if (!isDPU(affineReshape.input().getDefiningOp())) {
            return true;
        }
        return false;
    });
    target.addLegalOp<IE::AffineReshapeOp>();
    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<ConvertAffineReshapeRewriter>(&ctx, _log);
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
