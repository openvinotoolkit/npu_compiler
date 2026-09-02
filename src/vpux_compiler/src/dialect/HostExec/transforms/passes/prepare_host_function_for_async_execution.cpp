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
                         std::size_t& standaloneGroupId, const Logger& log) {
    if (op->getParentOfType<mlir::async::ExecuteOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into asynchronous region");
        return;
    }

    mlir::async::CreateGroupOp group = nullptr;
    mlir::Operation* parentForOp = nullptr;

    if (auto forOp = getTopParentOpOfType<mlir::scf::ForOp>(op); forOp != nullptr) {
        parentForOp = forOp.getOperation();
        if (forOpToAsyncGroupMap.count(parentForOp) == 0) {
            mlir::OpBuilder builder(forOp);

            builder.setInsertionPoint(forOp);
            auto step = forOp.getStep();
            auto lb = forOp.getLowerBound();
            auto ub = forOp.getUpperBound();
            auto numberOfIterations = builder.create<mlir::arith::SubIOp>(forOp.getLoc(), ub, lb);
            auto numberOfIterationsDivStep =
                    builder.create<mlir::arith::DivSIOp>(forOp.getLoc(), numberOfIterations, step);
            group = builder.create<mlir::async::CreateGroupOp>(forOp.getLoc(), numberOfIterationsDivStep);
            builder.setInsertionPointAfter(forOp);

            if (forOp->hasAttr("no_await_all") == false) {
                builder.create<mlir::async::AwaitAllOp>(forOp.getLoc(), group);
            }

            if (forOp->hasAttr("no_reset_cmdlist")) {
                // The following attributes will be referred in ConvertToLLVMUMD pass
                group->setAttr("no_reset_cmdlist", builder.getBoolAttr(true));
            }

            forOpToAsyncGroupMap[parentForOp] = group;
        } else {
            group = forOpToAsyncGroupMap[parentForOp];
        }
    } else {
        mlir::OpBuilder builder(op);
        // Standalone async ops do not share a surrounding scf.for group, so create a
        // group with a single element to keep the same add-to-group / await-all lowering shape
        auto groupSize = builder.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 1);
        group = builder.create<mlir::async::CreateGroupOp>(op->getLoc(), groupSize);
        // Make each standalone group structurally unique so later CSE does not merge
        // unrelated standalone calls into the same command-list group
        group->setAttr("standalone_group_id", builder.getI64IntegerAttr(standaloneGroupId++));
    }

    log.trace("Create 'async.execute' Operation");

    const bool allResultsUnused = op->use_empty();

    const auto bodyBuilder = [op, allResultsUnused](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange) {
        auto* newOp = builder.clone(*op);
        if (allResultsUnused) {
            builder.create<mlir::async::YieldOp>(loc, mlir::ValueRange{});
        } else {
            builder.create<mlir::async::YieldOp>(loc, newOp->getResults());
        }
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    const auto resultTypes =
            allResultsUnused ? SmallVector<mlir::Type>{} : SmallVector<mlir::Type>(op->getResultTypes());
    auto execOp = builder.create<mlir::async::ExecuteOp>(op->getLoc(), mlir::TypeRange{resultTypes},
                                                         /* dependencies */ mlir::ValueRange{},
                                                         /* operands */ mlir::ValueRange{}, bodyBuilder);
    if (group != nullptr) {
        builder.create<mlir::async::AddToGroupOp>(op->getLoc(), execOp.getToken(), group);
    }
    if (parentForOp == nullptr) {
        builder.create<mlir::async::AwaitAllOp>(op->getLoc(), group);
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
                builder.create<mlir::func::ReturnOp>(returnOp.getLoc());
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
        builder.create<mlir::func::CallOp>(callOp.getLoc(), mlir::TypeRange{}, callOp.getCallee(),
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
