//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/pass_usage_observer.hpp"

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

using namespace vpux;
using namespace mlir::tracing;

PassUsageObserver::PassUsageObserver(Logger log): _log(log) {
    _log.setName("pass-usage-observer");
}

void PassUsageObserver::beforeExecute(const ActionActiveStack* actionStack, Breakpoint*, bool willExecute) {
    if (!willExecute) {
        return;
    }

    const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction == nullptr) {
        return;
    }

    auto handle = _activePasses.lock();
    handle->emplace(actionStack, mlir::OperationFingerPrint(passAction->getOp()));
}

void PassUsageObserver::afterExecute(const ActionActiveStack* actionStack) {
    const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction == nullptr) {
        return;
    }

    const auto newHash = mlir::OperationFingerPrint(passAction->getOp());
    const auto passName = passAction->getPass().getName().str();

    auto handle = _activePasses.lock();
    auto& activePasses = *handle;
    auto it = activePasses.find(actionStack);
    if (it == activePasses.end()) {
        return;
    }

    const auto oldHash = it->second;
    activePasses.erase(it);

    _log.info("{0}\t{1}", passName, oldHash != newHash ? "CHANGED" : "SAME");
}
