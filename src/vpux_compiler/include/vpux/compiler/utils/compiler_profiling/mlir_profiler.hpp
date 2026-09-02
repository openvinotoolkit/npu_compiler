//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Debug/Observers/ActionProfiler.h>

namespace vpux::compiler_profiling {
// A simple wrapper around MLIR's ActionProfiler that owns the file stream
// (which is a file) to which the profiling data is written.
class MlirProfiler : public mlir::tracing::ExecutionContext::Observer {
    std::error_code _error;
    llvm::raw_fd_ostream _os;
    mlir::tracing::ActionProfiler _impl;

public:
    MlirProfiler(llvm::StringRef filePath);

    void beforeExecute(const mlir::tracing::ActionActiveStack* action, mlir::tracing::Breakpoint* breakpoint,
                       bool willExecute) override;
    void afterExecute(const mlir::tracing::ActionActiveStack* action) override;
};
}  // namespace vpux::compiler_profiling
