//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_SWAPMAXPOOLWITHACTIVATION
#define GEN_PASS_DEF_SWAPMAXPOOLWITHACTIVATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// GenericConverter
//

template <class ConcreteOp>
class GenericConverter final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    GenericConverter(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log) {
        this->setDebugName("SwapMaxPoolWithActivation::GenericConverter");
    }

private:
    mlir::LogicalResult matchAndRewrite(ConcreteOp originOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <class ConcreteOp>
mlir::LogicalResult GenericConverter<ConcreteOp>::matchAndRewrite(ConcreteOp originOp,
                                                                  mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", originOp->getName(), originOp->getLoc());
    if (!originOp.getInput().hasOneUse()) {
        return matchFailed(rewriter, originOp, "Operation {0} is not the only user of its operand",
                           originOp->getName());
    }

    auto maxPool = originOp.getInput().template getDefiningOp<IE::MaxPoolOp>();
    if (maxPool == nullptr) {
        return matchFailed(rewriter, originOp, "Producer is not a MaxPool operation");
    }

    auto producerOp = maxPool.getInput().template getDefiningOp<IE::LayerWithPostOpInterface>();
    if (producerOp == nullptr) {
        return matchFailed(rewriter, originOp, "Producer of MaxPool does not support post-processing");
    }

    _log.trace("Swap MaxPool with '{0}'", originOp->getName());
    const auto activationOutType = maxPool.getInput().getType();
    auto newActivationOp = rewriter.create<ConcreteOp>(maxPool.getLoc(), activationOutType, maxPool.getInput());

    rewriter.replaceOpWithNewOp<IE::MaxPoolOp>(
            originOp, maxPool.getType(), newActivationOp.getOutput(), maxPool.getKernelSize(), maxPool.getStrides(),
            maxPool.getPadsBegin(), maxPool.getPadsEnd(), maxPool.getRoundingType(), maxPool.getPostOpAttr(),
            maxPool.getClampAttr(), maxPool.getOutputPaddingAttr(), maxPool.getInputPaddingAttr());

    return mlir::success();
}

//
// SwapMaxPoolWithActivation
//

class SwapMaxPoolWithActivation final : public IE::impl::SwapMaxPoolWithActivationBase<SwapMaxPoolWithActivation> {
public:
    explicit SwapMaxPoolWithActivation(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void SwapMaxPoolWithActivation::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GenericConverter<IE::ReLUOp>>(&ctx, _log);

    if (mlir::failed(applyPatternsGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createSwapMaxPoolWithActivation(Logger log) {
    return std::make_unique<SwapMaxPoolWithActivation>(log);
}
