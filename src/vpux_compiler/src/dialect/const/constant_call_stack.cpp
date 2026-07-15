//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringExtras.h>
#include <mlir/Debug/ExecutionContext.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/DialectResourceBlobManager.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <memory>

namespace vpux::Const {

const mlir::tracing::ActionActiveStack*& getCurrentActionStack() {
    thread_local const mlir::tracing::ActionActiveStack* currentStack = nullptr;
    return currentStack;
}

CallStackCache::CallStackTy& CallStackCache::getCallStack() {
    return _callStack;
}

std::mutex& CallStackCache::callStackCacheMutex() {
    return _callStackMutex;
}

std::string gatherTrace() {
    SmallVector<std::string> trace;
    auto currentStack = getCurrentActionStack();

    while (currentStack != nullptr) {
        if (const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&currentStack->getAction())) {
            trace.push_back(passAction->getPass().getName().str());
        } else {
            std::string actionName;
            llvm::raw_string_ostream actionOS(actionName);
            currentStack->getAction().print(actionOS);
            trace.push_back(std::move(actionName));
        }
        currentStack = currentStack->getParent();
    }

    return llvm::join(llvm::reverse(trace), " -> ");
}

std::string CallStackCache::getSpecificCallStack(mlir::ElementsAttr baseContent,
                                                 Const::TransformAttrInterface transformation) {
    std::lock_guard<std::mutex> lock(_callStackMutex);
    const auto& trs = _callStack[baseContent];
    auto it = std::find_if(trs.begin(), trs.end(), [transformation](const auto& t) {
        return std::get<0>(t) == transformation;
    });
    if (it != trs.end()) {
        return std::get<1>(*it);
    }
    return "UNKNOWN_CALL_STACK";
}

void setAction(const mlir::tracing::ActionActiveStack* action) {
    getCurrentActionStack() = action;
}

void CallStackObserver::beforeExecute(const mlir::tracing::ActionActiveStack* action,
                                      mlir::tracing::Breakpoint* /*breakpoint*/, bool /*willExecute*/) {
    setAction(action);
}

void CallStackObserver::afterExecute(const mlir::tracing::ActionActiveStack* action) {
    setAction(action != nullptr ? action->getParent() : nullptr);
}

}  // namespace vpux::Const
