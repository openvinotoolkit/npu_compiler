//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/thread_safe_accessors.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <mlir/Debug/ExecutionContext.h>
#include <mlir/IR/Action.h>
#include <mlir/IR/OperationSupport.h>

#include <unordered_map>

namespace vpux {

class PassUsageObserver final : public mlir::tracing::ExecutionContext::Observer {
private:
    using ActionStackToHashMap =
            std::unordered_map<const mlir::tracing::ActionActiveStack*, mlir::OperationFingerPrint>;

public:
    explicit PassUsageObserver(Logger log);
    void beforeExecute(const mlir::tracing::ActionActiveStack* actionStack, mlir::tracing::Breakpoint* breakpoint,
                       bool willExecute) final;
    void afterExecute(const mlir::tracing::ActionActiveStack* actionStack) final;

private:
    SimpleThreadSafeAccessor<ActionStackToHashMap> _activePasses;
    Logger _log;
};

}  // namespace vpux
