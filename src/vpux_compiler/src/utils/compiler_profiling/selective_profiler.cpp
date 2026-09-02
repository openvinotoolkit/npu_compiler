//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/selective_profiler.hpp"
#include "vpux/compiler/utils/npu_actions.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Pass/Pass.h>

namespace {
bool shouldSelectAction(const mlir::tracing::Action* action, const llvm::Regex& regex) {
    // Note: there are primarily two types of actions that we are interested in:
    // PassExecutionAction and NPU action(s).
    return llvm::TypeSwitch<const mlir::tracing::Action*, bool>(action)
            .Case([&](const mlir::PassExecutionAction* passAction) {
                const auto& pass = passAction->getPass();
                // anything goes: "argument" (pass-one-two) or name (PassOneTwo)
                return regex.match(pass.getArgument()) || regex.match(pass.getName());
            })
            .Case([&](const vpux::NpuCompilerAction* npuAction) {
                const auto& actionName = npuAction->getName();
                return regex.match(actionName);
            })
            .Default([&](const mlir::tracing::Action*) {
                return false;
            });
}

std::pair<bool, size_t> findSuitableActionInTheStackWithCurrentDepth(const mlir::tracing::ActionActiveStack* action,
                                                                     const llvm::Regex& regex) {
    size_t depth = 0;
    for (; action != nullptr; action = action->getParent(), ++depth) {
        if (shouldSelectAction(&action->getAction(), regex)) {
            return {true, depth};
        }
    }
    return {false, depth};
}
}  // namespace

namespace vpux::compiler_profiling {
SelectiveProfiler::SelectiveProfiler(llvm::StringRef selection) {
    if (selection.empty()) {
        return;
    }

    _selection = llvm::Regex(selection);
    std::string error;
    VPUX_THROW_UNLESS(_selection->isValid(error), "Invalid regex pattern '{0}': {1}", selection, error);
}

void SelectiveProfiler::beforeExecute(const mlir::tracing::ActionActiveStack* action, mlir::tracing::Breakpoint*,
                                      bool willExecute) {
    if (!willExecute || !_selection.has_value()) {
        return;
    }

    const auto [isAnythingSelected, depth] = findSuitableActionInTheStackWithCurrentDepth(action, *_selection);
    if (!isAnythingSelected) {
        return;
    }
    profileBeforeExecute(&action->getAction(), depth);
}

void SelectiveProfiler::afterExecute(const mlir::tracing::ActionActiveStack* action) {
    if (!_selection.has_value()) {
        return;
    }

    const auto [isAnythingSelected, depth] = findSuitableActionInTheStackWithCurrentDepth(action, *_selection);
    if (!isAnythingSelected) {
        return;
    }
    profileAfterExecute(&action->getAction(), depth);
}
}  // namespace vpux::compiler_profiling
