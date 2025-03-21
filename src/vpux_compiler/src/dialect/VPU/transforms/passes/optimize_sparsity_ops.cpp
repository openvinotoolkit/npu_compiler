//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_sparsity.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_OPTIMIZESPARSITYOPS
#define GEN_PASS_DEF_OPTIMIZESPARSITYOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// OptimizeSparsityOpsPass
//

class OptimizeSparsityOpsPass final : public VPU::impl::OptimizeSparsityOpsBase<OptimizeSparsityOpsPass> {
public:
    explicit OptimizeSparsityOpsPass(VPU::SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log)
            : _sparsityProfileCreateCb(std::move(sparsityProfileCreateCb)) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    VPU::ActivationSparsityProfile _sparsityProfile{VPU::ActivationSparsityProfile::S0};
    VPU::SparsityProfileCreateFunc _sparsityProfileCreateCb;
};

mlir::LogicalResult OptimizeSparsityOpsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    const auto parsedSparsityProfile = _sparsityProfileCreateCb(sparsityProfile.getValue());
    if (!parsedSparsityProfile.has_value()) {
        return mlir::failure();
    }
    _sparsityProfile = parsedSparsityProfile.value();
    return mlir::success();
}

//
// RemoveDuplicatedSparsifyOps
//

class RemoveDuplicatedSparsifyOps final : public mlir::OpRewritePattern<VPU::NCEEltwiseOp> {
public:
    RemoveDuplicatedSparsifyOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::NCEEltwiseOp>(ctx), _log(log) {
        setDebugName("RemoveDuplicatedSparsifyOps");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::NCEEltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Legend:
// === sparse path
// --- dense path
// Removing extra VPU.Sparsify operations before VPU.NCE.Eltwise
// Usefull to optimize [De]Quantize ops implemented as VPU.NCE.Eltwise
//   /-> VPU.Sparsify =>
// ->                   VPU.NCE.Eltwise
//   \-> VPU.Sparsify =>
// To
// -> VPU.Sparsify => VPU.NCE.Eltwise
mlir::LogicalResult RemoveDuplicatedSparsifyOps::matchAndRewrite(VPU::NCEEltwiseOp eltwiseOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    auto firstSparsify = eltwiseOp.getInput1().getDefiningOp<VPU::SparsifyOp>();
    if (!firstSparsify) {
        return mlir::failure();
    }
    auto secondSparsify = eltwiseOp.getInput2().getDefiningOp<VPU::SparsifyOp>();
    if (!secondSparsify) {
        return mlir::failure();
    }
    if (firstSparsify == secondSparsify) {
        return mlir::failure();
    }
    if (firstSparsify.getInput() != secondSparsify.getInput()) {
        return mlir::failure();
    }
    secondSparsify.getOutput().replaceAllUsesWith(firstSparsify.getOutput());
    rewriter.eraseOp(secondSparsify);
    _log.trace("Removed duplicated SparsifyOps for '{0}'", eltwiseOp);
    return mlir::success();
}

//
// RemoveExtraDesparsifyOp
//

class RemoveExtraDesparsifyOp final : public mlir::OpRewritePattern<VPU::DesparsifyOp> {
public:
    RemoveExtraDesparsifyOp(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::DesparsifyOp>(ctx), _log(log) {
        setDebugName("RemoveExtraDesparsifyOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::DesparsifyOp desparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

// Removing extra VPU.Desparsify where producer can be dense (has no other sparse consumers)
// SparseOp => VPU.Desparsify -> return
// To
// DenseOp -> return
mlir::LogicalResult RemoveExtraDesparsifyOp::matchAndRewrite(VPU::DesparsifyOp desparsifyOp,
                                                             mlir::PatternRewriter& rewriter) const {
    auto producingOp = desparsifyOp.getInput().getDefiningOp();
    if (!producingOp->hasOneUse()) {
        return mlir::failure();
    }
    auto sparseOp = mlir::dyn_cast<VPU::SparseOpInterface>(producingOp);
    if (!sparseOp) {
        return mlir::failure();
    }
    auto output = sparseOp->getResult(0);
    output.setType(desparsifyOp.getOutput().getType());

    desparsifyOp.getOutput().replaceAllUsesWith(output);

    _log.trace("Removed extra '{0}'", desparsifyOp);
    rewriter.eraseOp(desparsifyOp);
    return mlir::success();
}

class RemoveExtraSparsifyOp final : public mlir::OpRewritePattern<VPU::SparsifyOp> {
public:
    RemoveExtraSparsifyOp(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::SparsifyOp>(ctx), _log(log) {
        setDebugName("RemoveExtraSparsifyOp");
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::SparsifyOp sparsifyOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult RemoveExtraSparsifyOp::matchAndRewrite(VPU::SparsifyOp sparsifyOp,
                                                           mlir::PatternRewriter& rewriter) const {
    auto opOutput = sparsifyOp.getOutput();
    auto opInput = sparsifyOp.getInput();
    if (!opOutput.hasOneUse()) {
        return sparsifyOp->emitOpError("Should have only one user");
    }
    opOutput.replaceAllUsesWith(opInput);
    _log.trace("Removed '{0}'", sparsifyOp);
    rewriter.eraseOp(sparsifyOp);
    return mlir::success();
}

//
// safeRunOnFunc
//

void OptimizeSparsityOpsPass::safeRunOnFunc() {
    using namespace VPU;
    using namespace VPU::NCESparsity;

    auto func = getOperation();
    auto& ctx = getContext();

    if (_sparsityProfile != ActivationSparsityProfile::S1) {
        mlir::ConversionTarget target(ctx);
        target.addIllegalOp<VPU::SparsifyOp>();
        target.addLegalDialect<Const::ConstDialect>();
        target.addLegalDialect<VPU::VPUDialect>();
        target.addLegalDialect<mlir::linalg::LinalgDialect>();
        target.addLegalDialect<mlir::math::MathDialect>();

        target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

        mlir::RewritePatternSet legalPatterns(&ctx);
        legalPatterns.add<RemoveExtraSparsifyOp>(&ctx, _log);

        if (mlir::failed(mlir::applyFullConversion(func, target, std::move(legalPatterns)))) {
            signalPassFailure();
        }
    }

    mlir::RewritePatternSet greedyPatterns(&ctx);
    greedyPatterns.add<RemoveDuplicatedSparsifyOps>(&ctx, _log);
    greedyPatterns.add<RemoveExtraDesparsifyOp>(&ctx, _log);
    if (mlir::failed(applyPatternsAndFoldGreedily(func, std::move(greedyPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeSparsityOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeSparsityOpsPass(
        VPU::SparsityProfileCreateFunc sparsityProfileCreateCb, Logger log) {
    return std::make_unique<OptimizeSparsityOpsPass>(sparsityProfileCreateCb, log);
}
