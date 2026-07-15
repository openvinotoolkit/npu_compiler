//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/Support/Regex.h>
#include <mlir/Debug/ExecutionContext.h>
#include <memory>

namespace vpux {

class PassDisablingCallback {
private:
    std::shared_ptr<llvm::Regex> _disabledPasses;

public:
    explicit PassDisablingCallback(llvm::StringRef disabledPasses);
    mlir::tracing::ExecutionContext::Control operator()(const mlir::tracing::ActionActiveStack* actionStack) const;
    static std::unique_ptr<mlir::tracing::BreakpointManager> createBreakpointManager();
};
}  // namespace vpux
