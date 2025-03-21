//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/func_dialect.hpp"
#include "vpux/compiler/utils/logging.hpp"
#include "vpux/utils/core/format.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_OPTIMIZEREORDERSACROSSFUNCTIONCALLS
#define GEN_PASS_DEF_OPTIMIZEREORDERSACROSSFUNCTIONCALLS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

using FunctionCalls = std::map<mlir::func::FuncOp, std::vector<mlir::func::CallOp>>;
using CallFunction = std::map<mlir::func::CallOp, mlir::func::FuncOp>;

struct Usage {
    mlir::Operation* userOp;
    size_t operandIdx;
};

struct ArgOperations {
    SmallVector<IE::ReorderOp> producerOps;
    SmallVector<Usage> userOps;
};

//
// OptimizeReordersAcrossFunctionCallsPass
//

class OptimizeReordersAcrossFunctionCallsPass final :
        public IE::impl::OptimizeReordersAcrossFunctionCallsBase<OptimizeReordersAcrossFunctionCallsPass> {
public:
    explicit OptimizeReordersAcrossFunctionCallsPass(const bool seOpsEnabled, const bool seExperimentalOpsEnabled,
                                                     Logger log)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnModule() final;

    void collectFunctionCalls();
    SmallVector<IE::ReorderOp> getOuterReorderProducerOps(mlir::func::FuncOp funcOp, mlir::BlockArgument arg);
    SmallVector<Usage> getNonReorderUsers(mlir::Value value);
    SmallVector<Usage> getCompatibleUsers(ArrayRef<Usage> userOps, DimsOrder inputOrder);
    std::map<size_t, ArgOperations> getOptimizableArguments(mlir::func::FuncOp funcOp);
    mlir::FailureOr<DimsOrder> getProducersInputOrder(ArrayRef<IE::ReorderOp> producerOps);
    std::map<size_t, size_t> updateFunctionArguments(mlir::func::FuncOp funcOp,
                                                     const std::map<size_t, ArgOperations>& argOperations,
                                                     SmallVector<size_t>& erasedArguments);
    void connectProducerInputsToNewArg(mlir::func::FuncOp funcOp, const ArgOperations& argOperations,
                                       size_t newArgNumber);
    void eraseConnectionsToOriginalArg(mlir::func::FuncOp funcOp, size_t origArgNumber,
                                       const ArgOperations& argOperations);
    void updateCallOperations(mlir::func::FuncOp funcOp, const std::map<size_t, ArgOperations>& argOperations,
                              const std::map<size_t, size_t>& oldToNewArgMapping, SmallVector<size_t> erasedArguments);

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;

    FunctionCalls _functionCalls;
    CallFunction _callFunction;
};

mlir::LogicalResult OptimizeReordersAcrossFunctionCallsPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    if (seOpsEnabled.hasValue()) {
        _seOpsEnabled = seOpsEnabled.getValue();
    }
    if (seExperimentalOpsEnabled.hasValue()) {
        _seExperimentalOpsEnabled = seExperimentalOpsEnabled.getValue();
    }

    return mlir::success();
}

/**
 * Collect information on what CallOps each function has, as well as what FuncOp each CallOp refers to
 */
void OptimizeReordersAcrossFunctionCallsPass::collectFunctionCalls() {
    auto moduleOp = getOperation();
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        funcOp.walk([&](mlir::func::CallOp callOp) {
            auto calledFuncOp = vpux::getCalledFunction(callOp);
            _functionCalls[calledFuncOp].push_back(callOp);
            _callFunction[callOp] = calledFuncOp;
        });
    });
}

/**
 * Find the Reorder operation(s) which produce the block argument of the given function
 * The Reorders can either be direct parents to the function's call operation, or they
 * can be found inside another function when the parent op is a call operation
 * Multiple producers can be returned in case the function is called multiple times
 */
SmallVector<IE::ReorderOp> OptimizeReordersAcrossFunctionCallsPass::getOuterReorderProducerOps(
        mlir::func::FuncOp funcOp, mlir::BlockArgument arg) {
    SmallVector<IE::ReorderOp> producerOps;
    for (auto callOp : _functionCalls[funcOp]) {
        auto operand = callOp.getOperand(arg.getArgNumber());
        auto producerOp = operand.getDefiningOp();
        if (producerOp == nullptr) {
            continue;
        }

        if (auto producerCallOp = mlir::dyn_cast_or_null<mlir::func::CallOp>(producerOp)) {
            size_t resultNumber = 0;
            for (auto result : producerCallOp.getResults()) {
                if (result == operand) {
                    resultNumber = result.getResultNumber();
                    break;
                }
            }
            _callFunction[producerCallOp].walk([&](mlir::func::ReturnOp returnOp) {
                auto operand = returnOp.getOperand(resultNumber);
                auto producerOp = operand.getDefiningOp();
                if (producerOp == nullptr) {
                    return;
                }
                if (auto reorderOp = mlir::dyn_cast<IE::ReorderOp>(producerOp)) {
                    producerOps.push_back(reorderOp);
                }
            });
            continue;
        }

        if (auto reorderOp = mlir::dyn_cast<IE::ReorderOp>(producerOp)) {
            producerOps.push_back(reorderOp);
        }
    }
    return producerOps;
}

/**
 * Find users of a value, ignoring the direct Reorder operations if they exist. The operand number of each user is also
 * returned. For example:
 *    [value] -> Reorder -> User1
 *            \> User2
 *    would identify User1 and User2
 */
SmallVector<Usage> OptimizeReordersAcrossFunctionCallsPass::getNonReorderUsers(mlir::Value value) {
    SmallVector<Usage> userOps;
    for (auto& use : value.getUses()) {
        auto user = use.getOwner();
        // Nested calls and block arguments that are directly returned are not supported
        if (mlir::isa<mlir::func::CallOp, mlir::func::ReturnOp>(user)) {
            continue;
        }
        if (auto reorderOp = mlir::dyn_cast<IE::ReorderOp>(user)) {
            for (auto& reorderUse : reorderOp.getOutput().getUses()) {
                if (mlir::isa<mlir::func::CallOp, mlir::func::ReturnOp>(reorderUse.getOwner())) {
                    continue;
                }
                userOps.push_back({reorderUse.getOwner(), reorderUse.getOperandNumber()});
            }
            continue;
        }
        userOps.push_back({user, use.getOperandNumber()});
    }
    return userOps;
}

/**
 * Find the input order of the given Reorder operations. In case not all operations have the same input order, a failure
 * is returned
 */
mlir::FailureOr<DimsOrder> OptimizeReordersAcrossFunctionCallsPass::getProducersInputOrder(
        ArrayRef<IE::ReorderOp> producerOps) {
    DimsOrder inputOrder;
    for (auto producerOp : producerOps) {
        auto operandType = producerOp->getOperand(0).getType().cast<vpux::NDTypeInterface>();
        if (inputOrder.empty()) {
            inputOrder = operandType.getDimsOrder();
            continue;
        }
        if (inputOrder != operandType.getDimsOrder()) {
            return mlir::failure();
        }
    }
    return inputOrder;
}

/**
 * Filter the user operations so that only those compatible with the input order are returned
 * The compatibility is determined based on the layout interface attached to the operations, so that when the operation
 * receives the operand with the new order, all other operands and return values must remain unchanged
 */
SmallVector<Usage> OptimizeReordersAcrossFunctionCallsPass::getCompatibleUsers(ArrayRef<Usage> userOps,
                                                                               DimsOrder inputOrder) {
    SmallVector<Usage> compatibleUserOps;
    for (auto& userPair : userOps) {
        auto userOp = userPair.userOp;
        auto layoutIf = mlir::dyn_cast<IE::LayoutInfoOpInterface>(userOp);
        if (layoutIf == nullptr) {
            continue;
        }
        auto orderInfo = layoutIf.getLayoutInfo();
        const auto operandNumber = userPair.operandIdx;
        orderInfo.setInput(operandNumber, inputOrder);
        layoutIf.inferLayoutInfo(orderInfo, _seOpsEnabled, _seExperimentalOpsEnabled);

        bool inputsCompatible = true;
        for (size_t inputIdx = 0; inputIdx < orderInfo.getNumInputs(); ++inputIdx) {
            if (inputIdx == operandNumber) {
                inputsCompatible &= orderInfo.getInput(operandNumber) == inputOrder;
                continue;
            }
            const auto actualInputOrder =
                    userOp->getOperand(inputIdx).getType().cast<vpux::NDTypeInterface>().getDimsOrder();
            inputsCompatible &= orderInfo.getInput(inputIdx) == actualInputOrder;
        }
        if (!inputsCompatible) {
            continue;
        }

        const auto outputsCompatible = llvm::all_of(userOp->getResults(), [&](mlir::OpResult result) {
            const auto actualResultOrder = result.getType().cast<vpux::NDTypeInterface>().getDimsOrder();
            const auto supportedResultOrder = orderInfo.getOutput(result.getResultNumber());
            return actualResultOrder == supportedResultOrder;
        });
        if (!outputsCompatible) {
            continue;
        }

        compatibleUserOps.push_back({userOp, operandNumber});
    }
    return compatibleUserOps;
}

/**
 * Find the list of function arguments which can be optimized in terms of layout, by reducing the Reorder operations
 * between the argument's user and producer
 */
std::map<size_t, ArgOperations> OptimizeReordersAcrossFunctionCallsPass::getOptimizableArguments(
        mlir::func::FuncOp funcOp) {
    auto log = _log.nest();

    std::map<size_t, ArgOperations> argOperations;
    for (auto arg : funcOp.getArguments()) {
        const auto userOps = getNonReorderUsers(arg);
        if (userOps.empty()) {
            continue;
        }
        auto producerOps = getOuterReorderProducerOps(funcOp, arg);
        if (producerOps.empty()) {
            continue;
        }

        log.trace("Argument {0} is a candidate with {1} Reorder producer(s) and {2} user(s)", arg.getArgNumber(),
                  producerOps.size(), userOps.size());

        const auto inputOrder = getProducersInputOrder(producerOps);
        if (mlir::failed(inputOrder)) {
            log.nest().trace("Not all producers have the same input order");
            continue;
        }

        auto compatibleUsers = getCompatibleUsers(userOps, inputOrder.value());
        if (compatibleUsers.empty()) {
            log.nest().trace("No users are compatible with the producers' input order {0}", inputOrder.value());
            continue;
        }

        log.trace("{0} user(s) are compatible with the producers' input order {1}", compatibleUsers.size(),
                  inputOrder.value());

        argOperations[arg.getArgNumber()] = ArgOperations{std::move(producerOps), std::move(compatibleUsers)};
    }

    return argOperations;
}

/**
 * Update the arguments of the function by adding new arguments corresponding to the original arguments whose users can
 * be optimized. The new arguments are then connected to the users that are compatible with the layout. The original
 * arguments that are left without any user and erased
 */
std::map<size_t, size_t> OptimizeReordersAcrossFunctionCallsPass::updateFunctionArguments(
        mlir::func::FuncOp funcOp, const std::map<size_t, ArgOperations>& argOperations,
        SmallVector<size_t>& erasedArguments) {
    // Add new arguments at the end of the list
    std::map<size_t, size_t> oldToNewArgMapping;
    for (auto& argInfo : argOperations) {
        const auto origArgNumber = argInfo.first;
        const auto arg = funcOp.getArgument(origArgNumber);
        const auto newArgType = argInfo.second.producerOps.front()->getOperand(0).getType();
        const auto newArgNumber = funcOp.getNumArguments();
        funcOp.insertArgument(newArgNumber, newArgType, nullptr, arg.getLoc());
        oldToNewArgMapping[origArgNumber] = newArgNumber;
        _log.nest().trace("Introduced new argument of type {0} at position {1}, for the original argument {2}",
                          newArgType, newArgNumber, origArgNumber);

        auto newArg = funcOp.getArgument(newArgNumber);
        for (auto& [userOp, operandNumber] : argInfo.second.userOps) {
            userOp->setOperand(operandNumber, newArg);
        }
    }

    // Erase the original arguments if they have no users
    for (auto& argInfo : argOperations | reversed) {
        const auto arg = funcOp.getArgument(argInfo.first);
        const auto currentOrigArgUsers = getNonReorderUsers(arg);
        if (!currentOrigArgUsers.empty()) {
            continue;
        }

        for (auto origArgUser : llvm::make_early_inc_range(arg.getUsers())) {
            if (mlir::isa<IE::ReorderOp>(origArgUser)) {
                origArgUser->erase();
            }
        }

        const auto origArgNumber = arg.getArgNumber();
        erasedArguments.push_back(origArgNumber);
        funcOp.eraseArgument(origArgNumber);
    }

    return oldToNewArgMapping;
}

/**
 * Connect the producer Reorder operations to the new arguments that were added to the function (i.e. the associated
 * operands of the function's CallOps)
 * In case the producers are found in other functions, these functions and their call operations are adjusted
 */
void OptimizeReordersAcrossFunctionCallsPass::connectProducerInputsToNewArg(mlir::func::FuncOp funcOp,
                                                                            const ArgOperations& argOperations,
                                                                            size_t newArgNumber) {
    const auto addNewResultToFunction = [&](mlir::func::FuncOp producerFuncOp, IE::ReorderOp producerOp,
                                            mlir::func::ReturnOp returnOp) {
        const auto newReturnValue = producerOp->getOperand(0);
        returnOp.getOperandsMutable().append({newReturnValue});

        SmallVector<mlir::Type> resultTypes(producerFuncOp.getResultTypes());
        resultTypes.push_back(newReturnValue.getType());
        producerFuncOp.setType(
                mlir::FunctionType::get(producerFuncOp->getContext(), producerFuncOp.getArgumentTypes(), resultTypes));
    };

    const auto updateCallOps = [&](mlir::func::FuncOp producerFuncOp) {
        for (auto producerCallOpIdx : irange(_functionCalls[producerFuncOp].size())) {
            // Create a new CallOp which has the correct number of return values and erase the original CallOp
            auto producerCallOp = _functionCalls[producerFuncOp][producerCallOpIdx];
            mlir::OpBuilder builder(producerCallOp);
            auto newProducerCallOp =
                    builder.create<mlir::func::CallOp>(producerCallOp.getLoc(), producerCallOp.getCalleeAttr(),
                                                       producerFuncOp.getResultTypes(), producerCallOp.getOperands());
            for (auto result : producerCallOp->getResults() | indexed) {
                result.value().replaceAllUsesWith(newProducerCallOp->getResult(result.index()));
            }
            _functionCalls[producerFuncOp][producerCallOpIdx] = newProducerCallOp;
            _callFunction[newProducerCallOp] = _callFunction[producerCallOp];
            _callFunction.erase(producerCallOp);
            producerCallOp.erase();

            // Find the CallOp users of the new CallOp which correspond to the original function whose arguments were
            // increased
            const auto newResultIdx = newProducerCallOp->getNumResults() - 1;
            std::set<mlir::Operation*> producerCallOpUsers(newProducerCallOp->getUsers().begin(),
                                                           newProducerCallOp->getUsers().end());
            for (auto producerCallOpUser : producerCallOpUsers) {
                auto callOp = mlir::dyn_cast<mlir::func::CallOp>(producerCallOpUser);
                if (callOp == nullptr) {
                    continue;
                }
                if (llvm::find(_functionCalls[funcOp], callOp) == _functionCalls[funcOp].end()) {
                    continue;
                }
                callOp.getOperandsMutable().append(newProducerCallOp->getResult(newResultIdx));
                const auto argPos = callOp.getNumOperands() - 1;
                VPUX_THROW_WHEN(argPos != newArgNumber, "Invalid position for new argument {0}, expected position {1}",
                                argPos, newArgNumber);
            }
        }
    };

    const auto producerOps = argOperations.producerOps;
    for (auto producerOp : producerOps) {
        for (auto user : producerOp->getUsers()) {
            if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(user)) {
                // Skip other call users that point to a different function
                if (llvm::find(_functionCalls[funcOp], callOp) == _functionCalls[funcOp].end()) {
                    continue;
                }
                callOp.getOperandsMutable().append(producerOp->getOperand(0));
                const auto argPos = callOp.getNumOperands() - 1;
                VPUX_THROW_WHEN(argPos != newArgNumber, "Invalid position for new argument {0}, expected position {1}",
                                argPos, newArgNumber);
            } else if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(user)) {
                auto producerFuncOp = returnOp->getParentOfType<mlir::func::FuncOp>();
                addNewResultToFunction(producerFuncOp, producerOp, returnOp);
                updateCallOps(producerFuncOp);
            }
        }
    }
}

/**
 * Erase the connections to the original argument, for all producers which have no other uses
 * In case the unused producers are found in other functions, these functions and their call operations are also updated
 */
void OptimizeReordersAcrossFunctionCallsPass::eraseConnectionsToOriginalArg(mlir::func::FuncOp funcOp,
                                                                            size_t origArgNumber,
                                                                            const ArgOperations& argOperations) {
    const auto eraseResultFromFunction = [&](mlir::func::FuncOp producerFuncOp, size_t resultNumber) {
        SmallVector<mlir::Type> resultTypes(producerFuncOp.getResultTypes());
        resultTypes.erase(resultTypes.begin() + resultNumber);
        producerFuncOp.setType(
                mlir::FunctionType::get(producerFuncOp->getContext(), producerFuncOp.getArgumentTypes(), resultTypes));
    };

    const auto updateCallOps = [&](mlir::func::FuncOp producerFuncOp) {
        for (auto producerCallOp : _functionCalls[producerFuncOp]) {
            // Create a new CallOp which has the correct number of return values and erase the original CallOp
            mlir::OpBuilder builder(producerCallOp);
            auto newProducerCallOp =
                    builder.create<mlir::func::CallOp>(producerCallOp.getLoc(), producerCallOp.getCalleeAttr(),
                                                       producerFuncOp.getResultTypes(), producerCallOp.getOperands());
            size_t resultIdx = 0;
            for (auto result : producerCallOp->getResults()) {
                const auto numUses = std::distance(result.getUses().begin(), result.getUses().end());
                if (numUses == 0) {
                    continue;
                }
                result.replaceAllUsesWith(newProducerCallOp->getResult(resultIdx++));
            }
            for (size_t i = 0; i < _functionCalls[producerFuncOp].size(); ++i) {
                if (_functionCalls[producerFuncOp][i] == producerCallOp) {
                    _functionCalls[producerFuncOp][i] = newProducerCallOp;
                }
            }
            _callFunction[newProducerCallOp] = _callFunction[producerCallOp];
            _callFunction.erase(producerCallOp);
            producerCallOp.erase();
        }
    };

    for (auto callOp : _functionCalls[funcOp]) {
        auto operand = callOp.getOperand(origArgNumber);

        auto parentOp = operand.getDefiningOp();
        size_t resultNumber = 0;
        for (auto result : parentOp->getResults()) {
            if (result == operand) {
                resultNumber = result.getResultNumber();
                break;
            }
        }

        callOp.getOperandsMutable().erase(origArgNumber, 1);
        const auto numUses = std::distance(operand.getUses().begin(), operand.getUses().end());
        if (numUses > 0) {
            continue;
        }

        // The producer operation might be part of the same function as the CallOp, in which case it can be directly
        // removed since there are no other users
        const auto& producerOps = argOperations.producerOps;
        if (llvm::find(producerOps, parentOp) != producerOps.end()) {
            parentOp->erase();
            continue;
        }

        // Remove the Reorder when it is found inside another function
        if (auto producerCallOp = mlir::dyn_cast<mlir::func::CallOp>(parentOp)) {
            auto producerFuncOp = _callFunction[producerCallOp];
            producerFuncOp.walk([&](mlir::func::ReturnOp returnOp) {
                auto operand = returnOp.getOperand(resultNumber);
                VPUX_THROW_WHEN(parentOp == nullptr,
                                "Expected result {0} of function {1} to be an operation, but it is a block argument",
                                resultNumber, producerFuncOp.getSymName());
                auto parentOp = operand.getDefiningOp();
                VPUX_THROW_WHEN(llvm::find(producerOps, parentOp) == producerOps.end(),
                                "Expected producer Reorder to be returned, but got operation {0} at {1}",
                                parentOp->getName(), parentOp->getLoc());
                returnOp.getOperandsMutable().erase(resultNumber, 1);
                const auto numUses =
                        std::distance(parentOp->getResult(0).getUses().begin(), parentOp->getResult(0).getUses().end());
                if (numUses > 0) {
                    return;
                }
                parentOp->erase();
            });
            eraseResultFromFunction(producerFuncOp, resultNumber);
            updateCallOps(producerFuncOp);
            continue;
        }

        VPUX_THROW("Could not find producer Reorder operation. Parent operation is {0}", parentOp);
    }
}

/**
 * Update the call operations of the function so that connections are made from the producers' inputs to the new
 * arguments and the connections from unused producers are removed
 */
void OptimizeReordersAcrossFunctionCallsPass::updateCallOperations(mlir::func::FuncOp funcOp,
                                                                   const std::map<size_t, ArgOperations>& argOperations,
                                                                   const std::map<size_t, size_t>& oldToNewArgMapping,
                                                                   SmallVector<size_t> erasedArguments) {
    const auto log = _log.nest(2);

    for (auto& argInfo : argOperations) {
        const auto origArgNumber = argInfo.first;
        const auto newArgNumber = oldToNewArgMapping.at(origArgNumber);
        log.trace("Updating connections for original argument {0} to new argument {1}", origArgNumber, newArgNumber);
        connectProducerInputsToNewArg(funcOp, argInfo.second, newArgNumber);
    }

    llvm::sort(erasedArguments, [](size_t lhs, size_t rhs) {
        return lhs > rhs;
    });
    for (auto origArgNumber : erasedArguments) {
        const auto& argOps = argOperations.at(origArgNumber);
        log.trace("Erasing connections to the original argument {0}", origArgNumber);
        eraseConnectionsToOriginalArg(funcOp, origArgNumber, argOps);
    }
}

//
// safeRunOnModule
//

void OptimizeReordersAcrossFunctionCallsPass::safeRunOnModule() {
    collectFunctionCalls();

    auto moduleOp = getOperation();
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
        _log.trace("Handling function {0}", funcOp.getSymName());

        if (_functionCalls[funcOp].empty()) {
            _log.nest().trace("Function has no calls");
            return;
        }

        if (_functionCalls[funcOp].size() > 1) {
            _log.nest().trace("Function has {0} calls. Currently supporting cases with a single call",
                              _functionCalls[funcOp].size());
            return;
        }

        const auto argOperations = getOptimizableArguments(funcOp);
        if (argOperations.empty()) {
            return;
        }

        SmallVector<size_t> erasedArguments;
        const auto oldToNewArgMapping = updateFunctionArguments(funcOp, argOperations, erasedArguments);
        updateCallOperations(funcOp, argOperations, oldToNewArgMapping, erasedArguments);
    });
}

}  // namespace

//
// createOptimizeReordersAcrossFunctionCallsPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createOptimizeReordersAcrossFunctionCallsPass(const bool seOpsEnabled,
                                                                                    const bool seExperimentalOpsEnabled,
                                                                                    Logger log) {
    return std::make_unique<OptimizeReordersAcrossFunctionCallsPass>(seOpsEnabled, seExperimentalOpsEnabled, log);
}
