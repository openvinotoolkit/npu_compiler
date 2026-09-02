//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_INITIALLOWPRECISIONTRANSFORMATIONSPIPELINEREWRITEREXECUTOR
#define GEN_PASS_DEF_INITIALLOWPRECISIONTRANSFORMATIONSPIPELINEREWRITEREXECUTOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// InitialLowPrecisionTransformationsPipelineRewriterExecutorPass
//

class InitialLowPrecisionTransformationsPipelineRewriterExecutorPass final :
        public impl::InitialLowPrecisionTransformationsPipelineRewriterExecutorBase<
                InitialLowPrecisionTransformationsPipelineRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base = impl::InitialLowPrecisionTransformationsPipelineRewriterExecutorBase<
            InitialLowPrecisionTransformationsPipelineRewriterExecutorPass>;

    explicit InitialLowPrecisionTransformationsPipelineRewriterExecutorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
};

mlir::LogicalResult InitialLowPrecisionTransformationsPipelineRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }
    return mlir::success();
}

void InitialLowPrecisionTransformationsPipelineRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    const auto& strategyFactory = IE::getIEStrategyFactory(&ctx);
    auto strategy = strategyFactory->getInitialLowPrecisionTransformationsPipelineStrategy();
    auto customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*customRegistry, _log);

    auto config = getDefaultGreedyRewriteConfig();
    config.setMaxIterations(mlir::GreedyRewriteConfig::kNoLimit);
    // quantization-like patterns are converted to FQ. thus, we have to run
    // forever until convergence. if this halts, there's a bug somewhere in the
    // pass.
    if (mlir::failed(this->executeRewriters(&ctx, _log, func, customRegistry.get(), config))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInitialLowPrecisionTransformationsPipelineRewriterExecutorPass(Logger log) {
    return std::make_unique<InitialLowPrecisionTransformationsPipelineRewriterExecutorPass>(log);
}
