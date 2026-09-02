//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/compiler_profiling/selective_profiler.hpp"

#include <memory>

namespace vpux::compiler_profiling {
namespace details {
struct CallgrindProfilerImpl;
}

class CallgrindProfiler : public SelectiveProfiler {
    std::unique_ptr<details::CallgrindProfilerImpl> _impl;

public:
    CallgrindProfiler(llvm::StringRef regex, bool separateDumps);
    ~CallgrindProfiler();

    void profileBeforeExecute(const mlir::tracing::Action* action, size_t depth) override;
    void profileAfterExecute(const mlir::tracing::Action* action, size_t depth) override;
};
}  // namespace vpux::compiler_profiling
