//
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTSDPATOFLASHSDPA
#define GEN_PASS_DEF_CONVERTSDPATOFLASHSDPA
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

class SDPARewrite final : public mlir::OpRewritePattern<IE::SDPAOp> {
public:
    SDPARewrite(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::SDPAOp>(ctx), _log(log) {
        setDebugName("SDPARewrite");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::SDPAOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult SDPARewrite::matchAndRewrite(IE::SDPAOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    auto flashSdpa = rewriter.create<IE::FlashSDPAOp>(appendLoc(origOp->getLoc(), "FlashAttention"), origOp.getInputQ(),
                                                      origOp.getInputK(), origOp.getInputV(), origOp.getInputMask(),
                                                      origOp.getInputScale());

    rewriter.replaceOp(origOp, flashSdpa.getOutput());

    return mlir::success();
}

//
// ConvertSDPAToFlashSDPA
//

class ConvertSDPAToFlashSDPA final : public IE::impl::ConvertSDPAToFlashSDPABase<ConvertSDPAToFlashSDPA> {
public:
    explicit ConvertSDPAToFlashSDPA(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void ConvertSDPAToFlashSDPA::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);

    const auto isLegal = [](IE::SDPAOp) {
        // A more sophisticated condition should be implemented to decide
        // when this conversion is necessary or favorable
        return false;
    };

    target.addDynamicallyLegalOp<IE::SDPAOp>(isLegal);
    target.addLegalOp<IE::FlashSDPAOp>();
    target.addLegalOp<Const::DeclareOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SDPARewrite>(&ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertSDPAToFlashSDPAPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertSDPAToFlashSDPAPass(Logger log) {
    return std::make_unique<ConvertSDPAToFlashSDPA>(log);
}
