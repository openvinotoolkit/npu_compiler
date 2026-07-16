//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/Debug/ExecutionContext.h>
#include <mlir/IR/Action.h>

#include <memory>
#include <string>

namespace vpux {

class PassUsageObserver final : public mlir::tracing::ExecutionContext::Observer {
private:
    struct State;
    std::shared_ptr<State> _state;

public:
    explicit PassUsageObserver(std::string outputFile);
    void beforeExecute(const mlir::tracing::ActionActiveStack* actionStack, mlir::tracing::Breakpoint* breakpoint,
                       bool willExecute) final;
    void afterExecute(const mlir::tracing::ActionActiveStack* actionStack) final;
};

}  // namespace vpux
