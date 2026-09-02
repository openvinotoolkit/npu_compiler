//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/const/constant_call_stack.hpp"
#include "vpux/compiler/core/interfaces/dialect_cache.hpp"
#include "vpux/compiler/dialect/const/attr_interfaces.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/npu_actions.hpp"
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

std::string gatherTrace() {
    SmallVector<std::string> trace;

    for (auto currentStack = getCurrentActionStack(); currentStack != nullptr;
         currentStack = currentStack->getParent()) {
        const auto* action = &currentStack->getAction();
        auto actionName = getPrettyName(action);
        if (actionName == "mlir::detail::OpToOpPassAdaptor") {
            // ignore OpToOpPassAdaptor as it is "useless": provides no
            // information, just a wrapper around pass execution
            continue;
        }

        trace.push_back(std::move(actionName));
    }

    return llvm::join(llvm::reverse(trace), " -> ");
}

std::string CallStackCache::getSpecificCallStack(mlir::ElementsAttr baseContent,
                                                 Const::TransformAttrInterface transformation) {
    auto handle = _callStack.lock();
    auto& callstack = *handle;

    auto it = callstack.find(baseContent);
    if (it == callstack.end()) {
        return "NO_TRACE_FOR_PARSED_BASE_CONTENT";
    }

    const auto& trs = it->second;
    auto transformIt = std::find_if(trs.begin(), trs.end(), [transformation](const auto& t) {
        return std::get<0>(t) == transformation;
    });
    if (transformIt != trs.end()) {
        return std::get<1>(*transformIt);
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
