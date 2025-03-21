//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/utils/analysis.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_MOVEWAITRESULTTOASYNCBLOCKARGS
#define GEN_PASS_DEF_MOVEWAITRESULTTOASYNCBLOCKARGS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

void moveWaitResults(mlir::async::AwaitOp waitOp, Logger log) {
    const auto futureVal = waitOp.getOperand();

    auto producerExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(futureVal.getDefiningOp());
    VPUX_THROW_UNLESS(producerExecOp != nullptr, "'async.await' operand is produced by unsupported operation");

    const auto futureType = futureVal.getType().dyn_cast<mlir::async::ValueType>();
    VPUX_THROW_UNLESS(futureType != nullptr, "'async.await' operand has unexpected type : '{0}'", futureVal.getType());

    log.trace("Collect all uses inside 'async.execute' regions");

    using ExecutorToAsyncUsesMap = DenseMap<mlir::Operation*, SmallVector<mlir::OpOperand*>>;
    ExecutorToAsyncUsesMap allAsyncUses;

    for (auto& use : waitOp.getResult().getUses()) {
        if (auto userExecOp = use.getOwner()->getParentOfType<mlir::async::ExecuteOp>()) {
            allAsyncUses[userExecOp].push_back(&use);
        }
    }

    log.trace("Update uses inside 'async.execute' regions");

    for (const auto& p : allAsyncUses) {
        auto userExecOp = mlir::cast<mlir::async::ExecuteOp>(p.first);

        userExecOp.getBodyOperandsMutable().append(futureVal);
        const auto innerArg = userExecOp.getBody()->addArgument(futureType.getValueType(), futureVal.getLoc());

        log.nest().trace("Add '{0}' as async operand for '{1}', which is mapped to '{2}'", futureVal,
                         userExecOp->getLoc(), innerArg);

        for (auto* use : p.second) {
            log.nest().trace("Redirect '{0}' to '{1}'", *use->getOwner(), innerArg);
            use->set(innerArg);
        }
    }
}

class MoveWaitResultToAsyncBlockArgsPass final :
        public VPUIP::impl::MoveWaitResultToAsyncBlockArgsBase<MoveWaitResultToAsyncBlockArgsPass> {
public:
    explicit MoveWaitResultToAsyncBlockArgsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void MoveWaitResultToAsyncBlockArgsPass::safeRunOnFunc() {
    auto func = getOperation();

    const auto allWaitOps = to_small_vector(func.getOps<mlir::async::AwaitOp>());

    for (auto waitOp : llvm::make_early_inc_range(allWaitOps)) {
        if (waitOp.getResult() == nullptr) {
            return;
        }

        _log.trace("Process 'async.await' Operation at '{0}'", waitOp->getLoc());
        moveWaitResults(waitOp, _log.nest());

        if (waitOp.getResult().use_empty()) {
            _log.nest().trace("The operation result has no use left, remove it");
            waitOp->erase();
        } else {
            if (auto* firstUser = getFirstUser(waitOp.getResult())) {
                _log.nest().trace("Move the operation close to its first user at '{0}'", firstUser->getLoc());
                waitOp->moveBefore(firstUser);
            }
        }
    }

    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createMoveWaitResultToAsyncBlockArgsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createMoveWaitResultToAsyncBlockArgsPass(Logger log) {
    return std::make_unique<MoveWaitResultToAsyncBlockArgsPass>(log);
}
