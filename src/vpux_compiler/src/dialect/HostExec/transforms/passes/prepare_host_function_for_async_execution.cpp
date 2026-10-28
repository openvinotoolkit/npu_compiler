//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/attributes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/range.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
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
                         std::unordered_map<mlir::Operation*, mlir::async::CreateGroupOp>& forOpToAsyncGroupMap,
                         std::size_t& groupId, const Logger& log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    mlir::async::CreateGroupOp group = nullptr;
    mlir::Operation* parentForOp = nullptr;

    // Lambda to create a unique CreateGroupOp with group_id and optional attributes
    auto createCreateGroupOp = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::Value groupSize,
                                   bool noResetCmdlist = false) {
        auto groupOp = mlir::async::CreateGroupOp::create(builder, loc, groupSize);
        // Make each standalone group structurally unique so later CSE does not merge
        // unrelated calls into the same command-list group
        groupOp->setAttr("group_id", builder.getI64IntegerAttr(groupId++));

        if (noResetCmdlist) {
            groupOp->setAttr("no_reset_cmdlist", builder.getBoolAttr(true));
        }
        return groupOp;
    };

    if (auto forOp = getTopParentOpOfType<mlir::scf::ForOp>(op); forOp != nullptr) {
        parentForOp = forOp.getOperation();
        if (forOpToAsyncGroupMap.count(parentForOp) == 0) {
            mlir::OpBuilder builder(forOp);

            builder.setInsertionPoint(forOp);
            auto step = forOp.getStep();
            auto lb = forOp.getLowerBound();
            auto ub = forOp.getUpperBound();
            auto numberOfIterations = mlir::arith::SubIOp::create(builder, forOp.getLoc(), ub, lb);
            auto numberOfIterationsDivStep =
                    mlir::arith::DivSIOp::create(builder, forOp.getLoc(), numberOfIterations, step);
            group = createCreateGroupOp(builder, forOp.getLoc(), numberOfIterationsDivStep,
                                        forOp->hasAttr("no_reset_cmdlist"));
            builder.setInsertionPointAfter(forOp);

            if (forOp->hasAttr("no_await_all") == false) {
                mlir::async::AwaitAllOp::create(builder, forOp.getLoc(), group);
            }
            forOpToAsyncGroupMap[parentForOp] = group;
        } else {
            group = forOpToAsyncGroupMap[parentForOp];
        }
    } else {
        mlir::OpBuilder builder(op);
        // Standalone async ops do not share a surrounding scf.for group, so create a
        // group with a single element to keep the same add-to-group / await-all lowering shape
        auto groupSize = mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 1);
        group = createCreateGroupOp(builder, op->getLoc(), groupSize);
    }

    log.trace("Create 'async.execute' Operation");

    const bool allResultsUnused = op->use_empty();

    const auto bodyBuilder = [op, allResultsUnused](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange) {
        auto* newOp = builder.clone(*op);
        if (allResultsUnused) {
            mlir::async::YieldOp::create(builder, loc, mlir::ValueRange{});
        } else {
            mlir::async::YieldOp::create(builder, loc, newOp->getResults());
        }
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    const auto resultTypes =
            allResultsUnused ? SmallVector<mlir::Type>{} : SmallVector<mlir::Type>(op->getResultTypes());
    auto execOp = mlir::async::ExecuteOp::create(builder, op->getLoc(), mlir::TypeRange{resultTypes},
                                                 /* dependencies */ mlir::ValueRange{},
                                                 /* operands */ mlir::ValueRange{}, bodyBuilder);
    if (group != nullptr) {
        mlir::async::AddToGroupOp::create(builder, op->getLoc(), execOp.getToken(), group);
    }
    if (parentForOp == nullptr) {
        mlir::async::AwaitAllOp::create(builder, op->getLoc(), group);
    }

    if (allResultsUnused) {
        log.trace("All results unused — skipping 'async.await' generation");
        op->erase();
        return;
    }

    log.trace("Create 'async.await' Operations per each original result");

    SmallVector<mlir::Value> newResults;
    newResults.resize(op->getNumResults());
    for (auto i : irange(op->getNumResults())) {
        auto waitOp = mlir::async::AwaitOp::create(builder, op->getLoc(), execOp.getBodyResults()[i]);
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
    auto newOp = mlir::scf::IndexSwitchOp::create(builder, op.getLoc(), mlir::TypeRange{}, op.getArg(), op.getCases(),
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
        mlir::scf::YieldOp::create(yieldBuilder, yieldOp.getLoc(), mlir::ValueRange{});
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

bool shouldStripReturnValues(mlir::func::FuncOp func) {
    return HostExec::isHostCompileInferenceExecFunc(func) || vpux::config::isPureHostCompileFunc(func);
}

void removeAllHostFuncReturnValues(mlir::ModuleOp module, const Logger& log) {
    SmallVector<mlir::func::FuncOp> functionsToStrip;
    module.walk([&](mlir::func::FuncOp func) {
        if (!func.getResultTypes().empty() && shouldStripReturnValues(func)) {
            functionsToStrip.push_back(func);
        }
    });

    if (functionsToStrip.empty()) {
        return;
    }

    // Strip func.return operands
    for (auto func : functionsToStrip) {
        for (auto& block : func.getBody()) {
            if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(block.getTerminator())) {
                log.trace("Remove return operands from func.return at '{0}'", returnOp.getLoc());
                mlir::OpBuilder builder(returnOp);
                mlir::func::ReturnOp::create(builder, returnOp.getLoc());
                returnOp.erase();
            }
        }
    }

    // Update func.call sites for stripped callees
    mlir::DenseSet<mlir::func::FuncOp> stripSet(functionsToStrip.begin(), functionsToStrip.end());
    module.walk([&](mlir::func::CallOp callOp) {
        if (callOp.getNumResults() == 0) {
            return;
        }
        auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(callOp, callOp.getCalleeAttr());
        if (!callee || !stripSet.count(callee)) {
            return;
        }
        for (auto result : callOp.getResults()) {
            VPUX_THROW_WHEN(!result.use_empty(),
                            "func.call result at '{0}' still has uses after return-value stripping — "
                            "the caller must also have its return values stripped",
                            callOp.getLoc());
        }
        log.trace("Drop results from func.call to '{0}' at '{1}'", callOp.getCallee(), callOp.getLoc());
        mlir::OpBuilder builder(callOp);
        mlir::func::CallOp::create(builder, callOp.getLoc(), mlir::TypeRange{}, callOp.getCallee(),
                                   callOp.getArgOperands());
        callOp.erase();
    });

    // Update function types
    for (auto func : functionsToStrip) {
        auto newFuncType = mlir::FunctionType::get(func.getContext(), func.getArgumentTypes(), mlir::TypeRange{});
        func.setType(newFuncType);
    }
}

//
// PrepareHostFuncForAsyncExecutionPass
//

class PrepareHostFuncForAsyncExecutionPass final :
        public HostExec::impl::PrepareHostFuncForAsyncExecutionBase<PrepareHostFuncForAsyncExecutionPass> {
public:
    explicit PrepareHostFuncForAsyncExecutionPass(bool removeReturnValues, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        this->removeReturnValues = removeReturnValues;
    }

private:
    void safeRunOnModule() final;
};

void PrepareHostFuncForAsyncExecutionPass::safeRunOnModule() {
    auto module = getOperation();

    if (removeReturnValues) {
        removeAllHostFuncReturnValues(module, _log);
    }

    module.walk([&](mlir::func::FuncOp func) {
        if (!HostExec::isHostCompileInferenceExecFunc(func)) {
            _log.debug("Skip function: \"{0}\" as it isn't suitable for the processing", func.getName());
            return;
        }
        func.walk(makeIndexSwitchReturnVoid);

        std::unordered_map<mlir::Operation*, mlir::async::CreateGroupOp> forOpToAsyncGroupMap;
        std::size_t standaloneGroupId = 0;
        const auto wrapCallOpsIntoAsyncRegion = [&](mlir::Operation* op) {
            _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
            if (mlir::isa<mlir::CallOpInterface>(op)) {
                wrapIntoAsyncRegion(op, forOpToAsyncGroupMap, standaloneGroupId, _log.nest());
            }
        };
        func.walk(wrapCallOpsIntoAsyncRegion);
    });
}

}  // namespace

//
// createPrepareHostFuncForAsyncExecutionPass
//

std::unique_ptr<mlir::Pass> vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass(bool removeReturnValues,
                                                                                       Logger log) {
    return std::make_unique<PrepareHostFuncForAsyncExecutionPass>(removeReturnValues, log);
}
