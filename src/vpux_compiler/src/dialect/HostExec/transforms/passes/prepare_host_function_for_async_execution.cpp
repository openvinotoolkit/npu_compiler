//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/range.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <unordered_map>

namespace vpux::HostExec {
#define GEN_PASS_DECL_PREPAREHOSTFUNCFORASYNCEXECUTION
#define GEN_PASS_DEF_PREPAREHOSTFUNCFORASYNCEXECUTION
#include "vpux/compiler/dialect/HostExec/passes.hpp.inc"
}  // namespace vpux::HostExec

using namespace vpux;

namespace {

void wrapIntoAsyncRegion(mlir::Operation* op,
                         std::unordered_map<mlir::Operation*, mlir::async::CreateGroupOp> forOpToAsyncGroupMap,
                         const Logger& log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    if (auto forOp = getTopParentOpOfType<mlir::scf::ForOp>(op);
        forOp && forOpToAsyncGroupMap.count(forOp.getOperation()) == 0) {
        mlir::OpBuilder builder(forOp);

        builder.setInsertionPoint(forOp);
        auto step = forOp.getStep();
        auto lb = forOp.getLowerBound();
        auto ub = forOp.getUpperBound();
        auto numberOfIterations = builder.create<mlir::arith::SubIOp>(forOp.getLoc(), ub, lb);
        auto numberOfIterationsDivStep = builder.create<mlir::arith::DivSIOp>(forOp.getLoc(), numberOfIterations, step);
        auto group = builder.create<mlir::async::CreateGroupOp>(forOp.getLoc(), numberOfIterationsDivStep);

        builder.setInsertionPointAfter(forOp);
        builder.create<mlir::async::AwaitAllOp>(forOp.getLoc(), group);
        forOpToAsyncGroupMap[forOp.getOperation()] = group;
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
    if (auto forOp = getTopParentOpOfType<mlir::scf::ForOp>(op); forOp != nullptr) {
        builder.create<mlir::async::AddToGroupOp>(op->getLoc(), execOp.getToken(),
                                                  forOpToAsyncGroupMap[forOp.getOperation()]);
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

void makeIndexSwitchReturnVoid(mlir::scf::IndexSwitchOp op) {
    bool hasUsers = false;
    for (auto result : op.getResults()) {
        if (!result.use_empty()) {
            hasUsers = true;
            break;
        }
    }

    if (hasUsers) {
        return;
    }

    mlir::OpBuilder builder(op);
    auto newOp = builder.create<mlir::scf::IndexSwitchOp>(op.getLoc(), mlir::TypeRange{}, op.getArg(), op.getCases(),
                                                          op.getNumCases());

    for (auto i : irange(op.getNumRegions())) {
        newOp.getRegion(i).takeBody(op.getRegion(i));
    }

    for (mlir::Region* region : newOp.getRegions()) {
        if (region->empty()) {
            continue;
        }

        auto& block = region->front();
        auto yieldOp = mlir::cast<mlir::scf::YieldOp>(block.getTerminator());

        mlir::OpBuilder yieldBuilder(yieldOp);
        yieldBuilder.create<mlir::scf::YieldOp>(yieldOp.getLoc(), mlir::ValueRange{});
        yieldOp.erase();
    }

    auto& defaultBlock = newOp.getDefaultBlock();
    for (auto& opInBlock : llvm::make_early_inc_range(defaultBlock.getOperations() | reversed)) {
        if (!mlir::isa<mlir::scf::YieldOp>(opInBlock) && !mlir::isa<mlir::cf::AssertOp>(opInBlock)) {
            opInBlock.erase();
        }
    }

    op.erase();
}

//
// PrepareHostFuncForAsyncExecutionPass
//

class PrepareHostFuncForAsyncExecutionPass final :
        public HostExec::impl::PrepareHostFuncForAsyncExecutionBase<PrepareHostFuncForAsyncExecutionPass> {
public:
    explicit PrepareHostFuncForAsyncExecutionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void PrepareHostFuncForAsyncExecutionPass::safeRunOnFunc() {
    getOperation().walk(makeIndexSwitchReturnVoid);

    std::unordered_map<mlir::Operation*, mlir::async::CreateGroupOp> forOpToAsyncGroupMap;
    const auto wrapCallOpsIntoAsyncRegion = [&](mlir::Operation* op) {
        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        if (mlir::isa<mlir::CallOpInterface>(op)) {
            wrapIntoAsyncRegion(op, forOpToAsyncGroupMap, _log.nest());
        }
    };
    getOperation().walk(wrapCallOpsIntoAsyncRegion);
}

}  // namespace

//
// createPrepareHostFuncForAsyncExecutionPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass(Logger log) {
    return std::make_unique<PrepareHostFuncForAsyncExecutionPass>(log);
}
