//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"

#include "vpux/compiler/utils/allocate_buffers_for_net_results.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ASYNCREGIONSOUTLINING
#define GEN_PASS_DEF_ASYNCREGIONSOUTLINING
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;
using namespace VPUIP;

namespace outliner {

//
// AsyncRegionsOutliner
//

class AsyncRegionsOutliner final : public OutlinerBase {
public:
    AsyncRegionsOutliner(size_t asyncRegionOutliningMinOpsInBlock, AsyncDepsInfo& depsInfo, const Logger& log)
            : OutlinerBase(
                      std::make_unique<FunctionOutlinerAsyncRegion>(asyncRegionOutliningMinOpsInBlock, depsInfo, log),
                      log) {
    }

    void outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) override;

    static constexpr StringRef name() {
        return "async-region";
    }

private:
    void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override;

    void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                      ArrayRef<OutliningInstance> outlinedTargets) override;
    void updateMainFuncOp(mlir::ModuleOp moduleOp, ArrayRef<OutliningInstance> outlinedTargets) override;
    void addBuffersForNetResults(mlir::ModuleOp moduleOp);

private:
    SmallVector<mlir::func::FuncOp> _outlinedFunctions = {};
};

void AsyncRegionsOutliner::outline(mlir::ModuleOp moduleOp, StringRef functionSuffix) {
    OutlinerBase::outline(moduleOp, functionSuffix);
    addBuffersForNetResults(moduleOp);
}

// cloneAsyncOpsWithMapping() is a helper function to clone async.execute operations
// To clone the async.execute operation, the approach is:
//   1. update dependencies - remove the %Token if its definingOp is not included in the
//      operations list.
//   2. update body operands - remove the %bodyOperand if it is passed through function argument
//   3. clone the async.execute operation
//       a. cloneWithoutRegions() - clone the async.execute warpper itself
//       b. cloneInto() - clone the body of async.execute operation
void cloneAsyncOpsWithMapping(mlir::OpBuilder& builder, mlir::DenseMap<mlir::Value, mlir::Value>& oldToNewMap,
                              ArrayRef<mlir::Operation*> operations) {
    // If the operand is async::TokenType and its definingOp is not included in this
    // sub-functions' operations list, then update the dependencies by removing this token
    // Example:
    // Before updating dependencies: %token_0 has no definingOp in this sub-function
    //      ... async.execute[%token_0, %token_2, %token_4] (...) {
    //           %0 = VPUIP.NNDMA input
    //      }
    // After updating dependencies: delete %token_0 from dependencies
    //     ... async.execute[%token_2, %token_4] (...) {
    //          %0 = VPUIP.NNDMA input
    //     }
    auto updateDependencies = [&](mlir::async::ExecuteOp& execOp, mlir::Value operand) {
        auto depOpTokenPtr = llvm::find(execOp.getDependencies(), operand);
        if (depOpTokenPtr != execOp.getDependencies().end()) {
            auto definingOp = operand.getDefiningOp<mlir::async::ExecuteOp>();
            if (definingOp && std::find(operations.begin(), operations.end(), definingOp) == operations.end()) {
                auto depOpTokenIndex =
                        static_cast<unsigned>(std::distance(execOp.getDependencies().begin(), depOpTokenPtr));
                execOp.getDependenciesMutable().erase(depOpTokenIndex, 1);
                return true;
            }
        }
        return false;
    };

    // If the body operand of ExecuteOp is passed through function argument, use function argument directly
    // remove the operand from the body operands and propagate the function argument to the users
    // Example:
    // Before updating body operand: %bodyResults_1 is passed to function @outlining as %arg0
    //     func.func @outlining(%arg0: ...) {
    //         ... async.execute[%token_0] (%bodyResults_1: %arg1) {
    //              %0 = VPUIP.NNDMA input(%arg1)
    //         }
    //     }
    // After updating body operand: directly use %arg0 in async.execute
    //     func.func @outlining(%arg0: ...) {
    //         ... async.execute () {
    //              %0 = VPUIP.NNDMA input(%arg0)
    //         }
    //     }
    auto updateBodyOperands = [&](mlir::async::ExecuteOp& execOp, mlir::Value operand) {
        if (!mlir::isa<mlir::async::ValueType, mlir::async::TokenType>(oldToNewMap[operand].getType())) {
            auto bodyOperandPtr = llvm::find(execOp.getBodyOperands(), operand);
            if (bodyOperandPtr != execOp.getBodyOperands().end()) {
                auto bodyOperandIndex =
                        static_cast<unsigned>(std::distance(execOp.getBodyOperands().begin(), bodyOperandPtr));
                for (auto& user :
                     llvm::make_early_inc_range(execOp.getBody()->getArguments()[bodyOperandIndex].getUses())) {
                    user.set(oldToNewMap[operand]);
                    oldToNewMap[user.get()] = oldToNewMap[operand];
                }
                execOp.getBody()->eraseArgument(bodyOperandIndex);
                execOp.getBodyOperandsMutable().erase(bodyOperandIndex, 1);
            }
        }
    };

    auto updateExecOpOperands = [&](mlir::async::ExecuteOp execOp) {
        auto origOperands = to_small_vector(execOp->getOperands());
        for (auto operand : llvm::make_early_inc_range(origOperands)) {
            if (updateDependencies(execOp, operand)) {
                continue;
            }
            updateBodyOperands(execOp, operand);
        }
    };

    auto cloneAsyncExecuteOp = [&](mlir::async::ExecuteOp& execOp, mlir::IRMapping& mapper) -> mlir::Operation* {
        // Step 1: clone the async.execute operation itself
        auto clonedOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(
                builder.cloneWithoutRegions(*execOp.getOperation(), mapper));
        if (clonedOp == nullptr) {
            VPUX_THROW("Failed to clone async.execute operation");
        }
        // Step 2: clone the body of async.execute operation
        auto* bodyBlock = execOp.getBody();
        for (auto& innerOp : bodyBlock->getOperations()) {
            for (auto operand : innerOp.getOperands()) {
                if (auto blockArg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
                    if (blockArg.getOwner() != bodyBlock) {
                        mapper.map(operand, oldToNewMap[operand]);
                    }
                } else {
                    auto parentOp = operand.getDefiningOp();
                    if (parentOp == nullptr) {
                        continue;
                    }
                    if (parentOp->getParentRegion() != innerOp.getParentRegion()) {
                        mapper.map(operand, oldToNewMap[operand]);
                    }
                }
            }
        }
        for (auto pair : llvm::zip_first(execOp->getRegions(), clonedOp->getRegions())) {
            std::get<0>(pair).cloneInto(&std::get<1>(pair), mapper);
        }
        return clonedOp;
    };

    for (auto* op : operations) {
        if (auto execOp = mlir::dyn_cast<mlir::async::ExecuteOp>(op)) {
            updateExecOpOperands(execOp);
        }

        mlir::IRMapping mapper;
        for (auto operand : op->getOperands()) {
            mapper.map(operand, oldToNewMap[operand]);
        }

        mlir::Operation* clonedOp = nullptr;
        if (auto execOp = mlir::dyn_cast_or_null<mlir::async::ExecuteOp>(op)) {
            clonedOp = cloneAsyncExecuteOp(execOp, mapper);
        } else {
            clonedOp = builder.clone(*op, mapper);
        }

        if (mlir::isa<mlir::func::ReturnOp>(op)) {
            continue;
        }

        for (size_t i = 0; i < clonedOp->getResults().size(); i++) {
            oldToNewMap[op->getResult(i)] = clonedOp->getResult(i);
        }
    }
}

void AsyncRegionsOutliner::buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                        ArrayRef<OutliningInstance> outlinedTargets) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    OpBuilderLogger builderLog(getLogger().nest());
    auto builder = mlir::OpBuilder(moduleOp.getBodyRegion(), &builderLog);
    builder.setInsertionPoint(mainFuncOp);

    auto* ctx = moduleOp.getContext();
    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;
        const auto funcType = mlir::FunctionType::get(ctx, ArrayRef(funcsInfo[targetIdx][sliceIdx].inputTypes),
                                                      ArrayRef(funcsInfo[targetIdx][sliceIdx].outputTypes));
        const auto funcLoc = appendLoc(mainFuncOp.getLoc(), "_part{0}", targetIdx + 1);
        auto func = builder.create<mlir::func::FuncOp>(funcLoc, funcsInfo[targetIdx][sliceIdx].funcName, funcType);
        _outlinedFunctions.push_back(func);
        func.setPrivate();

        auto builder = mlir::OpBuilder::atBlockEnd(func.addEntryBlock(), &builderLog);

        mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
        for (size_t i = 0; i < slice.inputs.size(); i++) {
            oldToNewMap[slice.inputs[i]] = func.getArgument(i);
        }

        cloneAsyncOpsWithMapping(builder, oldToNewMap, ArrayRef(slice.operations));

        SmallVector<mlir::Value> funcOutputFromSlices;
        for (const auto output : slice.outputs) {
            if (mlir::isa<mlir::async::ValueType>(oldToNewMap[output].getType())) {
                auto waitOp = builder.create<mlir::async::AwaitOp>(output.getLoc(), oldToNewMap[output]);
                funcOutputFromSlices.push_back(waitOp.getResult());
            }
        }
        const auto returnLoc = appendLoc(mainFuncOp.getLoc(), "_part{0}_return", targetIdx + 1);
        builder.create<mlir::func::ReturnOp>(returnLoc, funcOutputFromSlices);
    }
}

void AsyncRegionsOutliner::buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                        ArrayRef<OutliningInstance> outlinedTargets) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    OpBuilderLogger builderLog(getLogger().nest());
    auto builder = mlir::OpBuilder::atBlockBegin(&mainFuncOp.getBody().front(), &builderLog);
    DenseMap<mlir::Value, mlir::Value> oldToNewArgMap;
    // Pass function arguments to new call functions
    for (const auto& arg : mainFuncOp.getArguments()) {
        oldToNewArgMap[arg] = arg;
    }

    for (const auto& [targetIdx, slices] : outlinedTargets | indexed) {
        const auto& slice = slices.front();
        const size_t sliceIdx = 0;

        SmallVector<mlir::Value> newInputs;
        for (const auto input : slice.inputs) {
            newInputs.push_back(oldToNewArgMap[input]);
        }

        const auto callLoc = appendLoc(mainFuncOp.getLoc(), "_part{0}_call", targetIdx + 1);
        auto newCall = builder.create<mlir::func::CallOp>(callLoc, funcsInfo[targetIdx][sliceIdx].funcName,
                                                          funcsInfo[targetIdx][sliceIdx].outputTypes, newInputs);
        for (auto res : newCall.getResults()) {
            size_t idx = res.getResultNumber();
            oldToNewArgMap[slice.outputs[idx]] = res;
            const_cast<mlir::Value&>(slice.outputs[idx]).replaceAllUsesWith(res);
        }
        // Move callOp to right place
        for (const auto& operand : newCall.getOperands()) {
            auto parentOp = operand.getDefiningOp();
            if (parentOp && newCall->isBeforeInBlock(parentOp)) {
                parentOp->moveBefore(newCall);
            }
        }
    }
}

// After buildCallOps and buildCallOps, the cloned operations might still exist in the main function.
// Possibly because the inputs to the async region might not necessarily be included in its operand list.
// For example:
//     func.func @outlining(%arg0: ...) {
//         ... async.execute () {               - operand list is empty
//             %0 = VPUIP.NNDMA input(%arg0)
//         }
//     }
// Therefore, the safest approach is to add another layer of insurance to ensure a valid IR mapping and a clear main
// function.
// Additionally, This function handles two await cases in main function:
// 1. After the callOp is created, it is not directly copied into the async region.
//    Instead, the createWrapIntoAsyncRegionsPass is invoked using a dramatic pass manager, which introduces a redundant
//    await operation.
// 2. If an operation is moved into a sub-function but has an await user, this await user becomes an isolated node
// (without a producer or consumer). Such isolated nodes need to be deleted.
void AsyncRegionsOutliner::updateMainFuncOp(mlir::ModuleOp moduleOp, ArrayRef<OutliningInstance> outlinedTargets) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    auto collectOpsToClone = [&]() {
        std::unordered_set<mlir::Operation*> outlinedOperations;
        for (const auto& slices : outlinedTargets) {
            const auto& slice = slices.front();
            for (auto& op : slice.operations) {
                outlinedOperations.insert(op);
            }
        }

        OpOrderedSet opsToClone;
        for (auto& op : mainFuncOp.getOps()) {
            if (outlinedOperations.count(&op) == 0) {
                opsToClone.insert(&op);
            }
        }
        return opsToClone;
    };

    OpBuilderLogger builderLog(getLogger().nest());
    auto builder = mlir::OpBuilder::atBlockEnd(moduleOp.getBody(), &builderLog);
    auto newFunc =
            builder.create<mlir::func::FuncOp>(mainFuncOp.getLoc(), mainFuncOp.getName(), mainFuncOp.getFunctionType());

    mlir::DenseMap<mlir::Value, mlir::Value> oldToNewMap;
    builder = mlir::OpBuilder::atBlockEnd(newFunc.addEntryBlock(), &builderLog);
    for (size_t i = 0; i < mainFuncOp.getArguments().size(); i++) {
        oldToNewMap[mainFuncOp.getArgument(i)] = newFunc.getArgument(i);
    }

    cloneAsyncOpsWithMapping(builder, oldToNewMap, to_small_vector(collectOpsToClone()));

    const auto allWaitOps = to_small_vector(newFunc.getOps<mlir::async::AwaitOp>());
    for (auto waitOp : llvm::make_early_inc_range(allWaitOps)) {
        if (waitOp.getResult().use_empty()) {
            // waitOp has no use left, remove it
            waitOp->erase();
        } else {
            // operand passed from func callOp, propagate operand to func callOp
            auto parentOp = waitOp.getOperand().getDefiningOp<mlir::func::CallOp>();
            if (parentOp == nullptr) {
                continue;
            }
            for (auto result : parentOp->getResults()) {
                if (result == waitOp.getOperand()) {
                    waitOp.getResult().replaceAllUsesWith(result);
                    waitOp->erase();
                }
            }
        }
    }

    mainFuncOp.erase();
}

void AsyncRegionsOutliner::addBuffersForNetResults(mlir::ModuleOp moduleOp) {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    auto logger = getLogger().nest();
    SmallVector<mlir::CallOpInterface> outlinedCallOps;
    mainFuncOp.walk([&](mlir::func::CallOp callOp) {
        if (callOp.getCallee().contains("async_region")) {
            outlinedCallOps.push_back(callOp);
        }
    });
    vpux::allocateBuffersForNetResults(outlinedCallOps, _outlinedFunctions, logger);
}

}  // namespace outliner

namespace {

//
// AsyncRegionsOutliningPass
//
class AsyncRegionsOutliningPass final : public VPUIP::impl::AsyncRegionsOutliningBase<AsyncRegionsOutliningPass> {
public:
    AsyncRegionsOutliningPass() = default;
    explicit AsyncRegionsOutliningPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    explicit AsyncRegionsOutliningPass(int64_t asyncRegionOutliningMinOpsInBlock, Logger log)
            : _asyncRegionOutliningMinOpsInBlock(static_cast<size_t>(asyncRegionOutliningMinOpsInBlock)) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

private:
    size_t _asyncRegionOutliningMinOpsInBlock = 100;
};

mlir::LogicalResult AsyncRegionsOutliningPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (asyncRegionOutliningMinOpsInBlock.hasValue()) {
        _asyncRegionOutliningMinOpsInBlock = asyncRegionOutliningMinOpsInBlock.getValue();
    }
    return mlir::success();
}

void AsyncRegionsOutliningPass::safeRunOnModule() {
    net::NetworkInfoOp netInfo;
    mlir::func::FuncOp mainFuncOp;
    auto moduleOp = getOperation();
    net::NetworkInfoOp::getFromModule(moduleOp, netInfo, mainFuncOp);

    auto& depsInfo = getChildAnalysis<AsyncDepsInfo>(mainFuncOp);
    outliner::AsyncRegionsOutliner outliner(_asyncRegionOutliningMinOpsInBlock, depsInfo, _log);
    outliner.outline(moduleOp, "async_region");

    {
        mlir::OpPassManager dynamicPM("builtin.module");
        dynamicPM.addNestedPass<mlir::func::FuncOp>(VPUIP::createConvertTransferOpsToDMAsPass(_log));
        dynamicPM.addNestedPass<mlir::func::FuncOp>(VPUIP::createWrapIntoAsyncRegionsPass(_log));
        dynamicPM.addNestedPass<mlir::func::FuncOp>(VPUIP::createMoveWaitResultToAsyncBlockArgsPass(_log));
        dynamicPM.addNestedPass<mlir::func::FuncOp>(Core::createMoveDeclarationsToTopPass(_log));
        if (mlir::failed(runPipeline(dynamicPM, moduleOp))) {
            signalPassFailure();
        }
    }
}

}  // namespace

//
// createAsyncRegionsOutliningPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAsyncRegionsOutliningPass(Logger log) {
    return std::make_unique<AsyncRegionsOutliningPass>(log);
}

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAsyncRegionsOutliningPass(size_t asyncRegionOutliningMinOpsInBlock,
                                                                         Logger log) {
    return std::make_unique<AsyncRegionsOutliningPass>(asyncRegionOutliningMinOpsInBlock, log);
}
