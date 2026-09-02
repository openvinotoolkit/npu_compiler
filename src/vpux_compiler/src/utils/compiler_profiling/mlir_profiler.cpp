//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/mlir_profiler.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::compiler_profiling {
MlirProfiler::MlirProfiler(llvm::StringRef filePath): _os(filePath, _error), _impl(_os) {
    VPUX_THROW_WHEN(_os.has_error(), "Failed to open file '{0}' for write : {1}", filePath, _error.message());
}

void MlirProfiler::beforeExecute(const mlir::tracing::ActionActiveStack* action, mlir::tracing::Breakpoint* breakpoint,
                                 bool willExecute) {
    _impl.beforeExecute(action, breakpoint, willExecute);
}

void MlirProfiler::afterExecute(const mlir::tracing::ActionActiveStack* action) {
    _impl.afterExecute(action);
}
}  // namespace vpux::compiler_profiling
