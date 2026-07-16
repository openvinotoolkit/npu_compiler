//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/small_vector.hpp"

#include <mlir/Debug/ExecutionContext.h>
#include <mlir/IR/MLIRContext.h>
#include <memory>

namespace vpux {

/// This class is a wrapper around `mlir::tracing::ExecutionContext` that should
/// be used as the action handler for the MLIRContext in the NPU plugin. It adds
/// lifetime/ownership management to the MLIR's ExecutionContext and otherwise
/// provides the same functionality.
class NpuActionHandler {
public:
    using Control = mlir::tracing::ExecutionContext::Control;
    using Observer = mlir::tracing::ExecutionContext::Observer;

    NpuActionHandler() = default;
    ~NpuActionHandler();
    NpuActionHandler(const NpuActionHandler& other);
    NpuActionHandler& operator=(const NpuActionHandler& other);
    NpuActionHandler(NpuActionHandler&& other) noexcept;
    NpuActionHandler& operator=(NpuActionHandler&& other) noexcept;

    /// Set the callback that is used to control the execution.
    void setCallback(std::function<Control(const mlir::tracing::ActionActiveStack*)> callback);

    /// Register a new `Observer` on this context. It'll be notified before and
    /// after executing an action. Note that this method is not thread-safe: it
    /// isn't supported to add a new observer while actions may be executed.
    void registerObserver(std::unique_ptr<Observer> observer);

    /// Register a new `BreakpointManager` on this context. It'll have a chance to
    /// match an action before it gets executed. Note that this method is not
    /// thread-safe: it isn't supported to add a new manager while actions may be
    /// executed.
    void addBreakpointManager(std::unique_ptr<mlir::tracing::BreakpointManager> manager);

    /// Process the given action. This is the operator called by MLIRContext on
    /// `executeAction()`.
    void operator()(mlir::function_ref<void()> transform, const mlir::tracing::Action& action);

private:
    mlir::tracing::ExecutionContext _executionContext;

    // Lifetime-managed storage for the callback, observers and breakpoint
    // managers used by the mlir::tracing::ExecutionContext
    std::function<Control(const mlir::tracing::ActionActiveStack*)> _callback;
    SmallVector<std::shared_ptr<Observer>> _observers;
    SmallVector<std::shared_ptr<mlir::tracing::BreakpointManager>> _breakpoints;
};

NpuActionHandler& getActionHandler(mlir::MLIRContext& ctx);

}  // namespace vpux
