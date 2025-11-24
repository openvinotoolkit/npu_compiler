//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/compiler/dynamic_rewriter/passes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux {
#define GEN_PASS_DECL_DYNAMICREWRITEREXECUTOR
#define GEN_PASS_DEF_DYNAMICREWRITEREXECUTOR
#include "vpux/compiler/dynamic_rewriter/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// DynamicRewriterExecutorPass
//

class DynamicRewriterExecutorPass final :
        public impl::DynamicRewriterExecutorBase<DynamicRewriterExecutorPass>,
        public RewriterExecutorInterfaceBase {
public:
    explicit DynamicRewriterExecutorPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    explicit DynamicRewriterExecutorPass(StringRef rewriterName, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        this->setRewriterName(rewriterName.str());
    }

private:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;
    void safeRunOnFunc() final;
};

// Override to enforce rewriter name must be specified for DynamicRewriterExecutorPass
// Because this pass only works for lit-test purpose and always requires a specific rewriter
mlir::LogicalResult DynamicRewriterExecutorPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }
    // For unit test purpose
    // If this->getRewriterName() is specified, it is value from unit test constructor
    if (!this->getRewriterName().empty()) {
        return mlir::success();
    }
    // For lit-test purpose
    if (this->rewriterName.hasValue()) {
        this->setRewriterName(this->rewriterName.getValue());
    }
    return mlir::success();
}

void DynamicRewriterExecutorPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();
    _log.trace("Running pass: {0}", Base::getArgumentName());

    if (mlir::failed(this->executeRewriters(&ctx, _log, func))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::createDynamicRewriterExecutorPass(Logger log) {
    return std::make_unique<DynamicRewriterExecutorPass>(log);
}

// For unit test purpose only
std::unique_ptr<mlir::Pass> vpux::createDynamicRewriterExecutorPass(StringRef rewriterName, Logger log) {
    return std::make_unique<DynamicRewriterExecutorPass>(rewriterName, log);
}
