//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/range.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_WRAPINTOASYNCREGIONS
#define GEN_PASS_DEF_WRAPINTOASYNCREGIONS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

void warpIntoAsyncRegion(VPUIP::AsyncLayerOpInterface op, Logger log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    const auto executor = op.getExecutor();
    if (executor != nullptr) {
        log.trace("It will be executed on '{0}' Executor", executor);
    }

    log.trace("Create 'async.execute' Operation");

    const auto bodyBuilder = [op](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange) {
        auto* newOp = builder.clone(*op);
        builder.create<mlir::async::YieldOp>(loc, newOp->getResults());
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    auto execOp = builder.create<mlir::async::ExecuteOp>(op->getLoc(), op->getResultTypes(), std::nullopt, std::nullopt,
                                                         bodyBuilder);
    if (executor != nullptr) {
        VPUIP::VPUIPDialect::setExecutor(execOp, executor);
    }

    log.trace("Create 'async.await' Operations per each original result");

    SmallVector<mlir::Value> newResults;
    newResults.resize(op->getNumResults());
    for (auto i : irange(op->getNumResults())) {
        auto waitOp = builder.create<mlir::async::AwaitOp>(op->getLoc(), execOp.getBodyResults()[i]);
        newResults[i] = waitOp.getResult();
    }

    log.trace("Replace the operation with new 'async.await' results");

    op->replaceAllUsesWith(newResults);
    op->erase();
}

//
// WrapIntoAsyncRegionsPass
//

class WrapIntoAsyncRegionsPass final : public VPUIP::impl::WrapIntoAsyncRegionsBase<WrapIntoAsyncRegionsPass> {
public:
    explicit WrapIntoAsyncRegionsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void WrapIntoAsyncRegionsPass::safeRunOnFunc() {
    const auto callback = [&](VPUIP::AsyncLayerOpInterface op) {
        if (mlir::isa<VPUIP::AsyncLayerOpInterface>(op->getParentOp())) {
            _log.trace("Skip for operation '{0}' at '{1}' which is wrapped in other op", op->getName(), op->getLoc());
            return;
        }
        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        warpIntoAsyncRegion(op, _log.nest());
    };

    getOperation().walk(callback);
}

}  // namespace

//
// createWrapIntoAsyncRegionsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createWrapIntoAsyncRegionsPass(Logger log) {
    return std::make_unique<WrapIntoAsyncRegionsPass>(log);
}
