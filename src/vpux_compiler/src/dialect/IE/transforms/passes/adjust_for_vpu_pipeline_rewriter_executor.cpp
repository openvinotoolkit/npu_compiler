//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/transforms/factories/adjust_for_vpu_pipeline_strategy_getter.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_ADJUSTFORVPUPIPELINEREWRITEREXECUTOR
#define GEN_PASS_DEF_ADJUSTFORVPUPIPELINEREWRITEREXECUTOR
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// AdjustForVPUPipelineRewriterExecutorPass
//

class AdjustForVPUPipelineRewriterExecutorPass final :
        public impl::AdjustForVPUPipelineRewriterExecutorBase<AdjustForVPUPipelineRewriterExecutorPass>,
        public RewriterExecutorInterface {
public:
    using Base = impl::AdjustForVPUPipelineRewriterExecutorBase<AdjustForVPUPipelineRewriterExecutorPass>;

    explicit AdjustForVPUPipelineRewriterExecutorPass(const IE::AdjustForVPUOptions& options, Logger log)
            : _enableFuseClamp(options.enableFuseClampOperations) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
    bool _enableFuseClamp = false;
};

mlir::LogicalResult AdjustForVPUPipelineRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    if (rewriterName.hasValue()) {
        setRewriterName(rewriterName.getValue());
    }

    if (enableFuseClamp.hasValue()) {
        _enableFuseClamp = enableFuseClamp.getValue();
    }

    return mlir::success();
}

void AdjustForVPUPipelineRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    auto strategy = IE::createAdjustForVPUPipelineStrategy(func, _enableFuseClamp);
    auto customRegistry = vpux::RegistryManager::createCustomRegistry();
    strategy->registerRewriters(*customRegistry, _log);

    if (mlir::failed(this->executeRewriters(&ctx, _log, func, customRegistry.get()))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustForVPUPipelineRewriterExecutorPass(
        const IE::AdjustForVPUOptions& options, Logger log) {
    return std::make_unique<AdjustForVPUPipelineRewriterExecutorPass>(options, log);
}

std::unique_ptr<mlir::Pass> vpux::IE::createAdjustForVPUPipelineRewriterExecutorPass(Logger log) {
    return createAdjustForVPUPipelineRewriterExecutorPass(IE::AdjustForVPUOptions{}, log);
}
