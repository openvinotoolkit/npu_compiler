//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/npu_action_handler.hpp"

#include <utility>

using namespace vpux;

namespace {
void setExecutionContextCallback(
        mlir::tracing::ExecutionContext& executionContext,
        const std::function<NpuActionHandler::Control(const mlir::tracing::ActionActiveStack*)>& callback) {
    if (callback) {
        executionContext.setCallback(callback);
    } else {
        executionContext.setCallback(nullptr);
    }
}
}  // namespace

NpuActionHandler::~NpuActionHandler() {
}

NpuActionHandler::NpuActionHandler(const NpuActionHandler& other)
        : _executionContext(other._executionContext),
          _callback(other._callback),
          _observers(other._observers),
          _breakpoints(other._breakpoints) {
    setExecutionContextCallback(_executionContext, _callback);
}

NpuActionHandler& NpuActionHandler::operator=(const NpuActionHandler& other) {
    if (this == &other) {
        return *this;
    }

    _executionContext = other._executionContext;
    _callback = other._callback;
    _observers = other._observers;
    _breakpoints = other._breakpoints;
    setExecutionContextCallback(_executionContext, _callback);
    return *this;
}

NpuActionHandler::NpuActionHandler(NpuActionHandler&& other) noexcept
        : _executionContext(std::move(other._executionContext)),
          _callback(std::move(other._callback)),
          _observers(std::move(other._observers)),
          _breakpoints(std::move(other._breakpoints)) {
    setExecutionContextCallback(_executionContext, _callback);
}

NpuActionHandler& NpuActionHandler::operator=(NpuActionHandler&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    _executionContext = std::move(other._executionContext);
    _callback = std::move(other._callback);
    _observers = std::move(other._observers);
    _breakpoints = std::move(other._breakpoints);
    setExecutionContextCallback(_executionContext, _callback);
    return *this;
}

void NpuActionHandler::setCallback(std::function<Control(const mlir::tracing::ActionActiveStack*)> callback) {
    _callback = std::move(callback);
    setExecutionContextCallback(_executionContext, _callback);
}

void NpuActionHandler::registerObserver(std::unique_ptr<Observer> observer) {
    _observers.push_back(std::move(observer));
    _executionContext.registerObserver(_observers.back().get());
}

void NpuActionHandler::addBreakpointManager(std::unique_ptr<mlir::tracing::BreakpointManager> manager) {
    _breakpoints.push_back(std::move(manager));
    _executionContext.addBreakpointManager(_breakpoints.back().get());
}

void NpuActionHandler::operator()(mlir::function_ref<void()> transform, const mlir::tracing::Action& action) {
    _executionContext(std::move(transform), action);
}

NpuActionHandler& vpux::getActionHandler(mlir::MLIRContext& ctx) {
    auto actionHandler = ctx.getActionHandler().target<NpuActionHandler>();
    assert(actionHandler != nullptr && "NpuActionHandler is not registered in the MLIRContext");
    return *const_cast<NpuActionHandler*>(actionHandler);
}
