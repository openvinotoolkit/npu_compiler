//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/pass_usage_observer.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/OperationSupport.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassInstrumentation.h>

#include <fstream>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

using namespace vpux;
using namespace mlir::tracing;

struct PassUsageObserver::State {
    std::mutex mutex;
    std::ofstream outputFile;
    std::unordered_map<const ActionActiveStack*, mlir::OperationFingerPrint> activePasses;
};

PassUsageObserver::PassUsageObserver(std::string outputFile): _state(std::make_shared<State>()) {
    std::lock_guard lock(_state->mutex);

    _state->outputFile.open(outputFile, std::ios::out);
    VPUX_THROW_UNLESS(_state->outputFile.is_open(), "PassUsageObserver failed to open file '{0}'", outputFile);
}

void PassUsageObserver::beforeExecute(const ActionActiveStack* actionStack, Breakpoint*, bool willExecute) {
    if (!willExecute) {
        return;
    }

    const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction == nullptr) {
        return;
    }

    std::lock_guard lock(_state->mutex);
    _state->activePasses.emplace(actionStack, mlir::OperationFingerPrint(passAction->getOp()));
}

void PassUsageObserver::afterExecute(const ActionActiveStack* actionStack) {
    const auto passAction = llvm::dyn_cast<mlir::PassExecutionAction>(&actionStack->getAction());
    if (passAction == nullptr) {
        return;
    }

    const auto newHash = mlir::OperationFingerPrint(passAction->getOp());
    const auto passName = passAction->getPass().getName().str();

    std::lock_guard lock(_state->mutex);
    auto it = _state->activePasses.find(actionStack);
    if (it == _state->activePasses.end()) {
        return;
    }
    const auto oldHash = it->second;
    _state->activePasses.erase(it);

    _state->outputFile << passName << "\t" << (oldHash != newHash ? "CHANGED" : "SAME") << "\n";
}
