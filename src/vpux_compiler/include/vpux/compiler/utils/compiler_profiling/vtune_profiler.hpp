//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/compiler_profiling/selective_profiler.hpp"

namespace vpux::compiler_profiling {

class VTuneProfiler : public SelectiveProfiler {
public:
    VTuneProfiler(llvm::StringRef selection);

    void profileBeforeExecute(const mlir::tracing::Action* action, size_t depth) override;
    void profileAfterExecute(const mlir::tracing::Action* action, size_t depth) override;
};
}  // namespace vpux::compiler_profiling
