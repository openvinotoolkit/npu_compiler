//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include "vpux/utils/core/range.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_GROUPASYNCEXECUTEOPS
#define GEN_PASS_DEF_GROUPASYNCEXECUTEOPS
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// Helpers
//

bool isOptimizableOp(mlir::async::ExecuteOp execOp) {
    const auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(execOp);
    return !IE::isNceTile(executor.getFullReference());
}

bool isSameExecutor(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    auto executor1 = vpux::VPUIP::VPUIPDialect::getExecutor(execOp1);
    auto executor2 = vpux::VPUIP::VPUIPDialect::getExecutor(execOp2);
    if (executor1 != executor2) {
        return false;
    }

    auto hasExecutorInstance1 = vpux::VPUIP::VPUIPDialect::hasExecutorInstanceMask(execOp1);
    auto hasExecutorInstance2 = vpux::VPUIP::VPUIPDialect::hasExecutorInstanceMask(execOp2);

    if (!hasExecutorInstance1 && !hasExecutorInstance2) {
        return true;
    }

    if (hasExecutorInstance1 && hasExecutorInstance2) {
        auto executorInstance1 = vpux::VPUIP::VPUIPDialect::getExecutorInstanceMask(execOp1);
        auto executorInstance2 = vpux::VPUIP::VPUIPDialect::getExecutorInstanceMask(execOp2);
        return (executorInstance1 == executorInstance2);
    }

    return false;
}

bool haveSameDependencies(mlir::async::ExecuteOp execOp1, mlir::async::ExecuteOp execOp2) {
    // check for prefetched data operations which have an injected dependency
    if (execOp1.getDependencies().size() == 1 && execOp2.getDependencies().size() == 1) {
        llvm::DenseSet<mlir::Value> dependencies;
        for (auto dep : execOp1.getDependencies()) {
            dependencies.insert(dep);
        }
        for (auto dep : execOp2.getDependencies()) {
            if (dependencies.find(dep) == dependencies.end()) {
                return false;
            }
        }
    }
    return true;
}

bool prevHasUniqueUsers(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp) {
    auto getUsers = [](mlir::async::ExecuteOp op) {
        std::set<mlir::async::ExecuteOp> users;
        for (auto res : op.getBodyResults()) {
            for (auto user : res.getUsers()) {
                users.insert(mlir::dyn_cast<mlir::async::ExecuteOp>(user));
            }
        }
        return users;
    };

    std::set<mlir::async::ExecuteOp> usersPrevOp = getUsers(prevExecOp);
    std::set<mlir::async::ExecuteOp> usersOp = getUsers(execOp);
    usersPrevOp.erase(execOp);

    SmallVector<mlir::async::ExecuteOp> uniqueUsersPrevOp;
    std::set_difference(usersPrevOp.begin(), usersPrevOp.end(), usersOp.begin(), usersOp.end(),
                        std::back_inserter(uniqueUsersPrevOp));
    if (!uniqueUsersPrevOp.empty())
        return true;

    return false;
}

//
// mergeAsyncExecuteOps
//

mlir::async::ExecuteOp mergeAsyncExecuteOps(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp,
                                            mlir::PatternRewriter& rewriter) {
    auto* prevBodyBlock = prevExecOp.getBody();
    auto* bodyBlock = execOp.getBody();

    const auto bodyBuilder = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange blockArgs) {
        mlir::IRMapping mapper;

        const auto prevBlockArgs = prevBodyBlock->getArguments();
        const auto curBlockArgs = bodyBlock->getArguments();

        for (size_t i = 0; i < blockArgs.size(); ++i) {
            if (i < prevBlockArgs.size()) {
                mapper.map(prevBlockArgs[i], blockArgs[i]);
            } else {
                mapper.map(curBlockArgs[i - prevBlockArgs.size()], blockArgs[i]);
            }
        }

        SmallVector<mlir::Value> newResults;

        const auto copyOps = [&](mlir::Block* bodyBlock) {
            for (auto& op : bodyBlock->getOperations()) {
                if (!mlir::isa<mlir::async::YieldOp>(op)) {
                    builder.clone(op, mapper);
                } else {
                    for (auto operand : op.getOperands()) {
                        newResults.push_back(mapper.lookupOrDefault(operand));
                    }
                }
            }
        };

        copyOps(prevBodyBlock);
        copyOps(bodyBlock);

        builder.create<mlir::async::YieldOp>(loc, newResults);
    };

    const auto prevResultTypes = prevBodyBlock->getTerminator()->getOperandTypes();
    const auto resultTypes = bodyBlock->getTerminator()->getOperandTypes();

    SmallVector<mlir::Type> newResultTypes(prevResultTypes);
    newResultTypes.insert(newResultTypes.end(), resultTypes.begin(), resultTypes.end());

    SmallVector<mlir::Value> newDependencies(prevExecOp.getDependencies());
    newDependencies.insert(newDependencies.end(), execOp.getDependencies().begin(), execOp.getDependencies().end());

    SmallVector<mlir::Value> newOperands(prevExecOp.getBodyOperands());
    newOperands.insert(newOperands.end(), execOp.getBodyOperands().begin(), execOp.getBodyOperands().end());

    auto newExecOp = rewriter.create<mlir::async::ExecuteOp>(prevExecOp->getLoc(), newResultTypes, newDependencies,
                                                             newOperands, bodyBuilder);

    auto executor = vpux::VPUIP::VPUIPDialect::getExecutor(execOp);
    VPUIP::VPUIPDialect::setExecutor(newExecOp, executor);

    if (vpux::VPUIP::VPUIPDialect::hasExecutorInstanceMask(execOp)) {
        auto executorInstance = vpux::VPUIP::VPUIPDialect::getExecutorInstanceMask(execOp);
        VPUIP::VPUIPDialect::setExecutorInstanceMask(newExecOp, executorInstance);
    }

    return newExecOp;
}

//
// cleanup
//

void cleanup(mlir::async::ExecuteOp prevExecOp, mlir::async::ExecuteOp execOp, mlir::async::ExecuteOp newExecOp,
             mlir::PatternRewriter& rewriter, Logger log) {
    log.trace("Redirect results of original 'async.execute' operations");

    SmallVector<mlir::Value> matchedResultsPrev;
    SmallVector<mlir::Value> matchedResultsCurr;

    // newExecOp returns one token which replaces both tokens from original ops
    matchedResultsPrev.push_back(newExecOp.getToken());
    matchedResultsCurr.push_back(newExecOp.getToken());

    for (auto p : newExecOp.getBodyResults() | indexed) {
        const auto ind = p.index();
        const auto newRes = p.value();

        if (ind < prevExecOp.getBodyResults().size()) {
            matchedResultsPrev.push_back(newRes);
        } else {
            matchedResultsCurr.push_back(newRes);
        }
    }

    rewriter.replaceOp(prevExecOp, matchedResultsPrev);
    rewriter.replaceOp(execOp, matchedResultsCurr);
}

//
// GroupAsyncExecuteOps
//

class GroupAsyncExecuteOps final : public mlir::OpRewritePattern<mlir::async::ExecuteOp> {
public:
    GroupAsyncExecuteOps(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<mlir::async::ExecuteOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(mlir::async::ExecuteOp execOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult GroupAsyncExecuteOps::matchAndRewrite(mlir::async::ExecuteOp execOp,
                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got 'async.execute' operation at '{0}'", execOp->getLoc());

    if (!isOptimizableOp(execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "The operation is not optimizible");
    }

    auto prevExecOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(execOp->getPrevNode());
    if (prevExecOp == nullptr) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous operation is not 'async.execute'");
    }

    if (!isSameExecutor(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' uses another executor");
    }

    if (!haveSameDependencies(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' has different dependencies");
    }

    /*  TODO: Remove check below when proper operands mapping is implemented.
        If current execute op depends on previous then it requires more complex operands/results mapping.
        Mapping path:
        1st execute op yield op operands -> 1st execute op result->
        -> 2nd execute op operand-> internal block argument-> operation operand */
    if (llvm::is_contained(prevExecOp->getUsers(), execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp,
                           "Current 'async.execute' depends on previous 'async.execute'");
    }

    if (prevHasUniqueUsers(prevExecOp, execOp)) {
        return matchFailed(_log.nest(), rewriter, execOp, "Previous 'async.execute' has exclusive users");
    }

    auto prevExecOpCycleBegin = getAsyncExecuteCycleBegin(prevExecOp);
    auto prevExecOpCycleEnd = getAsyncExecuteCycleEnd(prevExecOp);
    auto prevExecOpCycleCost = prevExecOpCycleEnd - prevExecOpCycleBegin;
    auto execOpCycleBegin = getAsyncExecuteCycleBegin(execOp);
    auto execOpCycleEnd = getAsyncExecuteCycleEnd(execOp);
    auto execOpCycleCost = execOpCycleEnd - execOpCycleBegin;

    auto newCycleBegin = std::min(prevExecOpCycleBegin, execOpCycleBegin);
    auto newCycleEnd = std::max(prevExecOpCycleEnd, execOpCycleEnd);

    // Allow to merge only consecutive operations according to cycle cost - without cycle gap
    // If cycle span is larger then cost of merged ops then those ops cannot be merged
    // as cost of such merged task will not correspond to acutal cost of contianed ops
    if (newCycleEnd - newCycleBegin != prevExecOpCycleCost + execOpCycleCost) {
        return matchFailed(_log.nest(), rewriter, execOp,
                           "Current and previous 'async.execute' are not consecutive in terms of cycles");
    }

    auto newExecOp = mergeAsyncExecuteOps(prevExecOp, execOp, rewriter);
    cleanup(prevExecOp, execOp, newExecOp, rewriter, _log.nest());

    newExecOp->setAttr(cycleBegin, getIntAttr(newExecOp->getContext(), newCycleBegin));
    newExecOp->setAttr(cycleEnd, getIntAttr(newExecOp->getContext(), newCycleEnd));
    newExecOp->setAttr(cycleCostAttrName, getIntAttr(newExecOp->getContext(), newCycleEnd - newCycleBegin));

    return mlir::success();
}

//
// GroupAsyncExecuteOpsPass
//

class GroupAsyncExecuteOpsPass final : public VPUIP::impl::GroupAsyncExecuteOpsBase<GroupAsyncExecuteOpsPass> {
public:
    explicit GroupAsyncExecuteOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void GroupAsyncExecuteOpsPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<GroupAsyncExecuteOps>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                                        getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }

    auto& depsInfo = getAnalysis<AsyncDepsInfo>();
    depsInfo.updateTokenDependencies();
}

}  // namespace

//
// createGroupAsyncExecuteOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createGroupAsyncExecuteOpsPass(Logger log) {
    return std::make_unique<GroupAsyncExecuteOpsPass>(log);
}
