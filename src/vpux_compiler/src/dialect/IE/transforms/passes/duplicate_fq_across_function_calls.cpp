//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/range.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/RegionUtils.h>

namespace vpux::IE {
#define GEN_PASS_DECL_DUPLICATEFQACROSSFUNCTIONCALLS
#define GEN_PASS_DEF_DUPLICATEFQACROSSFUNCTIONCALLS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

bool doesFQHaveConstRanges(IE::FakeQuantizeOp fqOp) {
    return llvm::all_of(fqOp.getOperands().drop_front(), [](mlir::Value operand) {
        return mlir::isa_and_nonnull<Const::DeclareOp>(operand.getDefiningOp());
    });
}

bool isIntermediateOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::ReshapeOp, IE::AffineReshapeOp>(op);
}

IE::FakeQuantizeOp getFakeQuantizeParent(mlir::Operation* op, SmallVector<mlir::Operation*>& intermediateOps) {
    while (isIntermediateOp(op)) {
        intermediateOps.push_back(op);
        op = op->getOperand(0).getDefiningOp();
    }
    return mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op);
}

IE::FakeQuantizeOp getFakeQuantizeUser(mlir::Operation* op, SmallVector<mlir::Operation*>& intermediateOps) {
    while (isIntermediateOp(op)) {
        intermediateOps.push_back(op);
        if (!op->getResult(0).hasOneUse()) {
            break;
        }
        op = *op->getResult(0).getUsers().begin();
    }
    return mlir::dyn_cast_or_null<IE::FakeQuantizeOp>(op);
}

IE::FakeQuantizeOp getFakeQuantizeParent(mlir::Operation* op) {
    SmallVector<mlir::Operation*> intermediateOps;
    return getFakeQuantizeParent(op, intermediateOps);
}

IE::FakeQuantizeOp getFakeQuantizeUser(mlir::Operation* op) {
    SmallVector<mlir::Operation*> intermediateOps;
    return getFakeQuantizeUser(op, intermediateOps);
}

}  // namespace

namespace {

class DuplicateFQAcrossFunctionCallsPass final :
        public IE::impl::DuplicateFQAcrossFunctionCallsBase<DuplicateFQAcrossFunctionCallsPass> {
private:
    using FunctionCalls = DenseMap<mlir::func::FuncOp, SmallVector<mlir::func::CallOp>>;
    using CallFunction = DenseMap<mlir::func::CallOp, mlir::func::FuncOp>;
    FunctionCalls _functionCalls;
    CallFunction _callFunction;
    // Used to update the location for each clone of an operation, to ensure there are no duplicated locations
    size_t _cloneInstanceIdx = 0;

public:
    explicit DuplicateFQAcrossFunctionCallsPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    void safeRunOnModule() final {
        collectFunctionCalls();
        if (_functionCalls.empty()) {
            _log.trace("No functions to optimize");
        }

        auto moduleOp = getOperation();
        IE::CNNNetworkOp netInfo;
        mlir::func::FuncOp netFunc;
        IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

        duplicateFQOpsOutside(moduleOp, netFunc);
        duplicateFQOpsInside(moduleOp, netFunc);
        eraseUnusedFQOps(netFunc);
    }

private:
    // Collect information on what CallOps each function has, as well as what FuncOp each CallOp refers to
    void collectFunctionCalls() {
        auto moduleOp = getOperation();
        moduleOp.walk([&](mlir::func::FuncOp funcOp) {
            funcOp.walk([&](mlir::func::CallOp callOp) {
                auto calledFuncOp = vpux::getCalledFunction(callOp);
                _functionCalls[calledFuncOp].push_back(callOp);
                _callFunction[callOp] = calledFuncOp;
            });
        });
    }

    // Find the Return operation of the given function
    mlir::func::ReturnOp findReturnOp(mlir::func::FuncOp funcOp) {
        mlir::func::ReturnOp returnOp;
        funcOp.walk([&](mlir::func::ReturnOp op) {
            returnOp = op;
        });
        return returnOp;
    }

    // Find FakeQuantize operations that are found at the boundaries of functions (i.e. user of argument or producer of
    // result) and duplicate them outside the function, if there is no associated FakeQuantize operation outside
    void duplicateFQOpsOutside(mlir::ModuleOp moduleOp, mlir::func::FuncOp netFunc) {
        _log.trace("Duplicating FakeQuantize ops outside functions");
        for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
            if (funcOp == netFunc) {
                continue;
            }
            if (_functionCalls[funcOp].empty()) {
                continue;
            }

            _log.trace("Handling function {0}", funcOp.getSymName());

            // Duplicate FQ ops used by function arguments
            for (auto arg : funcOp.getArguments()) {
                _log.nest().trace("Checking argument {0}", arg.getArgNumber());
                if (!arg.hasOneUse()) {
                    _log.nest(2).trace("Argument has multiple users");
                    continue;
                }
                const auto userOp = *arg.getUsers().begin();
                SmallVector<mlir::Operation*> intermediateOps;
                auto fqOp = getFakeQuantizeUser(userOp, intermediateOps);
                if (fqOp == nullptr) {
                    _log.nest(2).trace("Argument has no FakeQuantize user");
                    continue;
                }
                if (!doesFQHaveConstRanges(fqOp)) {
                    _log.nest(2).trace("FakeQuantize user has ranges as non-constants");
                    continue;
                }

                _log.nest(2).trace("Found user FakeQuantize operation '{0}'", fqOp->getLoc());

                for (auto callOp : _functionCalls[funcOp]) {
                    _log.nest(2).trace("Handling call operation '{0}'", callOp.getLoc());
                    const auto outerOperand = callOp->getOperand(arg.getArgNumber());
                    if (getFakeQuantizeParent(outerOperand.getDefiningOp()) != nullptr) {
                        _log.nest(3).trace("Call's outer operand already has a FakeQuantize operation");
                        continue;
                    }
                    _log.nest(3).trace("Duplicating FakeQuantize operation");
                    mlir::OpBuilder builder(callOp);
                    const auto newOperand = cloneIntermediateAndFQOps(builder, outerOperand, intermediateOps, fqOp);
                    callOp.setOperand(arg.getArgNumber(), newOperand);
                }
            }

            // Duplicate FQ ops which produce function results
            auto returnOp = findReturnOp(funcOp);
            if (returnOp == nullptr) {
                _log.nest().trace("Found no return operation for function");
                continue;
            }
            for (auto& operand : returnOp->getOpOperands()) {
                const auto resultNum = operand.getOperandNumber();
                _log.nest().trace("Checking result {0}", resultNum);

                SmallVector<mlir::Operation*> intermediateOps;
                auto fqOp = getFakeQuantizeParent(operand.get().getDefiningOp(), intermediateOps);
                if (fqOp == nullptr) {
                    _log.nest(2).trace("Result has no FakeQuantize producer");
                    continue;
                }
                if (!doesFQHaveConstRanges(fqOp)) {
                    _log.nest(2).trace("FakeQuantize operation has ranges as non-constants");
                    continue;
                }

                _log.nest(2).trace("Found producer FakeQuantize operation '{0}'", fqOp->getLoc());

                for (auto callOp : _functionCalls[funcOp]) {
                    _log.nest(2).trace("Handling call operation '{0}'", callOp.getLoc());
                    auto outerResult = callOp.getResult(resultNum);
                    SmallVector<std::pair<mlir::Operation*, size_t>> targetUsers;
                    for (auto& use : outerResult.getUses()) {
                        if (getFakeQuantizeUser(use.getOwner()) == nullptr) {
                            targetUsers.push_back({use.getOwner(), use.getOperandNumber()});
                        }
                    }
                    if (targetUsers.empty()) {
                        _log.nest(3).trace("Call's outer result already has FakeQuantize users");
                        continue;
                    }

                    _log.nest(3).trace("Duplicating FakeQuantize operation");

                    mlir::OpBuilder builder(&getContext());
                    builder.setInsertionPointAfter(callOp);
                    const auto [newValue, newUserOp] =
                            cloneFQAndIntermediateOps(builder, outerResult, intermediateOps, fqOp);
                    outerResult.replaceAllUsesExcept(newValue, newUserOp);
                }
            }
        }
    }

    // Find FakeQuantize operations that are producers / users of call operations and duplicate them inside the
    // function, if there is no associated FakeQuantize operation inside
    void duplicateFQOpsInside(mlir::ModuleOp moduleOp, mlir::func::FuncOp netFunc) {
        _log.trace("Duplicating FakeQuantize ops inside functions");
        for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
            if (funcOp == netFunc) {
                continue;
            }
            if (_functionCalls[funcOp].empty()) {
                continue;
            }

            _log.trace("Handling function {0}", funcOp.getSymName());

            // Duplicate FQ ops associated with function arguments
            for (auto arg : funcOp.getArguments()) {
                _log.nest().trace("Checking argument {0}", arg.getArgNumber());
                const auto anyUserHasFQ = llvm::any_of(arg.getUsers(), [&](mlir::Operation* userOp) {
                    return getFakeQuantizeUser(userOp) != nullptr;
                });
                if (anyUserHasFQ) {
                    _log.nest(2).trace("Argument already has FakeQuantize users");
                    continue;
                }

                const auto callOps = ArrayRef(_functionCalls[funcOp]);
                const auto firstCallOp = callOps.front();
                const auto outerOperand = firstCallOp->getOperand(arg.getArgNumber());
                SmallVector<mlir::Operation*> intermediateOps;
                auto fqOp = getFakeQuantizeParent(outerOperand.getDefiningOp(), intermediateOps);
                if (fqOp == nullptr) {
                    _log.nest(2).trace("There is no outer FakeQuantize operation to duplicate");
                    continue;
                }
                if (!doesFQHaveConstRanges(fqOp)) {
                    _log.nest(2).trace("Outer FakeQuantize operation has ranges as non-constants");
                    continue;
                }

                const auto allCallOpsQuantized = llvm::all_of(callOps.drop_front(), [&](mlir::func::CallOp callOp) {
                    return getFakeQuantizeParent(callOp->getOperand(arg.getArgNumber()).getDefiningOp()) != nullptr;
                });
                if (!allCallOpsQuantized) {
                    _log.nest(2).trace("Not all call operations have the outer operand quantized");
                    continue;
                }

                _log.nest(2).trace("Duplicating outer FakeQuantize operation '{0}'", fqOp->getLoc());

                auto builder = mlir::OpBuilder::atBlockBegin(arg.getOwner());
                const auto [newValue, newUserOp] = cloneFQAndIntermediateOps(builder, arg, intermediateOps, fqOp);
                arg.replaceAllUsesExcept(newValue, newUserOp);
            }

            // Duplicate FQ ops associated with function result
            auto returnOp = findReturnOp(funcOp);
            if (returnOp == nullptr) {
                _log.nest().trace("Found no return operation for function");
                continue;
            }
            for (auto& operand : returnOp->getOpOperands()) {
                const auto resultNum = operand.getOperandNumber();
                _log.nest().trace("Checking result {0}", resultNum);

                if (getFakeQuantizeParent(operand.get().getDefiningOp()) != nullptr) {
                    _log.nest(2).trace("Result already has a FakeQuantize producer");
                    continue;
                }

                const auto callOps = ArrayRef(_functionCalls[funcOp]);
                const auto firstCallOp = callOps.front();
                const auto outerResult = firstCallOp->getResult(resultNum);
                if (!outerResult.hasOneUse()) {
                    _log.nest(2).trace("Outer result has multiple users");
                    continue;
                }
                SmallVector<mlir::Operation*> intermediateOps;
                const auto fqOp = getFakeQuantizeUser(*outerResult.getUsers().begin(), intermediateOps);
                if (fqOp == nullptr) {
                    _log.nest(2).trace("There is no outer FakeQuantize operation to duplicate");
                    continue;
                }
                if (!doesFQHaveConstRanges(fqOp)) {
                    _log.nest(2).trace("Outer FakeQuantize operation has ranges as non-constants");
                    continue;
                }

                const auto allCallOpsQuantized = llvm::all_of(callOps.drop_front(), [&](mlir::func::CallOp callOp) {
                    const auto result = callOp->getResult(resultNum);
                    if (!result.hasOneUse()) {
                        return false;
                    }
                    return getFakeQuantizeUser(*result.getUsers().begin()) != nullptr;
                });
                if (!allCallOpsQuantized) {
                    _log.nest(2).trace("Not all call operations have the outer result quantized");
                    continue;
                }

                _log.nest(2).trace("Duplicating outer FakeQuantize operation '{0}'", fqOp->getLoc());

                mlir::OpBuilder builder(returnOp);
                const auto newOperand =
                        cloneIntermediateAndFQOps(builder, returnOp->getOperand(resultNum), intermediateOps, fqOp);
                returnOp->setOperand(resultNum, newOperand);
            }
        }
    }

    // Clone chain of intermediate operations (if it exists), followed by a FakeQuantize operation
    // For example: Reshape -> Reshape -> FakeQuantize
    mlir::Value cloneIntermediateAndFQOps(mlir::OpBuilder& builder, mlir::Value input,
                                          ArrayRef<mlir::Operation*> intermediateOps, IE::FakeQuantizeOp fqOp) {
        auto inputValue = input;
        for (auto op : intermediateOps) {
            mlir::IRMapping mapper;
            mapper.map(op->getOperand(0), inputValue);
            auto newOp = builder.clone(*op, mapper);
            newOp->setLoc(appendLoc(newOp->getLoc(), "_duplicate{0}", _cloneInstanceIdx));
            inputValue = newOp->getResult(0);
        }
        mlir::IRMapping mapper;
        mapper.map(fqOp->getOperand(0), inputValue);
        for (auto& operand : fqOp->getOpOperands().drop_front()) {
            const auto newParentOp = builder.clone(*operand.get().getDefiningOp());
            mapper.map(operand.get(), newParentOp->getResult(0));
        }
        auto newFqOp = builder.clone(*fqOp, mapper);
        newFqOp->setLoc(appendLoc(newFqOp->getLoc(), "_duplicate{0}", _cloneInstanceIdx));

        const auto shape = mlir::cast<NDTypeInterface>(input.getType()).getShape();
        auto newOperand = builder.createOrFold<IE::ReshapeOp>(
                appendLoc(fqOp->getLoc(), "_reshape"), newFqOp->getResult(0), /*shape=*/nullptr,
                /*specialZero=*/nullptr, getIntArrayAttr(builder, shape.raw()));
        ++_cloneInstanceIdx;
        return newOperand;
    }

    // Clone FakeQuantize operation, followed by a chain of intermediate operations (if it exists)
    // For example: FakeQuantize -> Reshape -> Reshape
    std::pair<mlir::Value, mlir::Operation*> cloneFQAndIntermediateOps(mlir::OpBuilder& builder, mlir::Value input,
                                                                       ArrayRef<mlir::Operation*> intermediateOps,
                                                                       IE::FakeQuantizeOp fqOp) {
        mlir::Value inputValue = input;
        mlir::Operation* newInputUserOp = nullptr;
        const auto shape = mlir::cast<NDTypeInterface>(fqOp->getOperand(0).getType()).getShape();
        if (shape != mlir::cast<NDTypeInterface>(input.getType()).getShape()) {
            auto reshapeOp =
                    builder.create<IE::ReshapeOp>(appendLoc(fqOp->getLoc(), "_reshape"), input, /*shape=*/nullptr,
                                                  /*specialZero=*/nullptr, getIntArrayAttr(builder, shape.raw()));
            reshapeOp->setLoc(appendLoc(reshapeOp->getLoc(), "_duplicate{0}", _cloneInstanceIdx));
            inputValue = reshapeOp->getResult(0);
            newInputUserOp = reshapeOp;
        }

        mlir::IRMapping mapper;
        mapper.map(fqOp->getOperand(0), inputValue);
        for (auto& operand : fqOp->getOpOperands().drop_front()) {
            const auto newParentOp = builder.clone(*operand.get().getDefiningOp());
            mapper.map(operand.get(), newParentOp->getResult(0));
        }
        auto newFqOp = builder.clone(*fqOp, mapper);
        newFqOp->setLoc(appendLoc(newFqOp->getLoc(), "_duplicate{0}", _cloneInstanceIdx));
        inputValue = newFqOp->getResult(0);
        if (newInputUserOp == nullptr) {
            newInputUserOp = newFqOp;
        }

        for (auto op : intermediateOps) {
            mlir::IRMapping mapper;
            mapper.map(op->getOperand(0), inputValue);
            auto newOp = builder.clone(*op, mapper);
            newOp->setLoc(appendLoc(newOp->getLoc(), "_duplicate{0}", _cloneInstanceIdx));
            inputValue = newOp->getResult(0);
        }

        ++_cloneInstanceIdx;

        return {inputValue, newInputUserOp};
    }

    // Erase FakeQuantize operations that have no operations to use them. This can happen if:
    // - the operation has no parent or only a call operation as parent
    // - the operation has no users or only call / return operation as users
    // For example:
    //   func @main(%arg) {
    //     %fq1 = FakeQuantize(%arg)      // can be erased
    //     %call1 = call @function(%fq1)
    //     %fq2 = FakeQuantize(%call1)    // can be erased
    //     %call2 = call @function(%fq2)
    //     %fq3 = FakeQuantize(%call2)    // can be erased
    //     return %fq3
    //   }
    void eraseUnusedFQOps(mlir::func::FuncOp netFunc) {
        _log.trace("Erasing unused FakeQuantize ops");
        const auto fqOps = to_std_vector(netFunc.getOps<IE::FakeQuantizeOp>());
        for (auto fqOp : fqOps) {
            auto parentOp = fqOp->getOperand(0).getDefiningOp();
            while (isIntermediateOp(parentOp)) {
                parentOp = parentOp->getOperand(0).getDefiningOp();
            }
            const auto parentNeedsFQ = parentOp != nullptr && !mlir::isa<mlir::func::CallOp>(parentOp);
            if (parentNeedsFQ) {
                continue;
            }

            const auto userNeedsFQ = llvm::any_of(fqOp->getResult(0).getUsers(), [&](mlir::Operation* userOp) {
                while (isIntermediateOp(userOp)) {
                    if (!userOp->getResult(0).hasOneUse()) {
                        break;
                    }
                    userOp = *userOp->getResult(0).getUsers().begin();
                }
                return userOp != nullptr && !mlir::isa<mlir::func::CallOp, mlir::func::ReturnOp>(userOp);
            });
            if (userNeedsFQ) {
                continue;
            }

            _log.nest().trace("Erasing '{0}'", fqOp.getLoc());
            fqOp->getResult(0).replaceAllUsesWith(fqOp->getOperand(0));
            fqOp->erase();
        }
    }
};

}  // namespace

//
// createDuplicateFQAcrossFunctionCallsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createDuplicateFQAcrossFunctionCallsPass(const Logger& log) {
    return std::make_unique<DuplicateFQAcrossFunctionCallsPass>(log);
}
