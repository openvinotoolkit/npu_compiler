//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/callgrind_profiler.hpp"

#include "vpux/compiler/utils/npu_actions.hpp"
#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"

#ifdef NPU_VALGRIND_FOUND

// some misconfiguration
#if !__has_include(<valgrind/callgrind.h>)
#error "CMake detected Valgrind but the necessary callgrind header is not found."
#endif

#include <valgrind/callgrind.h>
#define ERROR_OUT_WHEN_VALGRIND_IS_NOT_PRESENT

#else

#define ERROR_OUT_WHEN_VALGRIND_IS_NOT_PRESENT \
    VPUX_THROW("Callgrind profiler is requested but valgrind is not found in the system.")

// define callgrind's macros used below to simplify the C++ code:
#define CALLGRIND_START_INSTRUMENTATION
#define CALLGRIND_STOP_INSTRUMENTATION
#define CALLGRIND_TOGGLE_COLLECT
#define CALLGRIND_DUMP_STATS_AT(...)

#endif

namespace vpux::compiler_profiling {
namespace details {
struct CallgrindProfilerImpl {
    bool separateDumps = false;
    DenseMap<const mlir::tracing::Action*, size_t> actionRepetitions;

    // Callgrind seems to be a simple profiler where it can only start and stop
    // instrumentation once (there is no notion of nested regions as in ITT,
    // etc.). This should be manually controlled.
    std::pair<const mlir::tracing::Action*, bool> instrumentationCache{nullptr, false};

    CallgrindProfilerImpl(bool separateDumps): separateDumps(separateDumps) {
    }

    void runBefore(const mlir::tracing::Action* action, size_t depth) {
        // note: depth > 0 means we're already "inside" an instrumented section
        if (depth == 0 && !instrumentationCache.second) {
            instrumentationCache = {action, true};

            CALLGRIND_START_INSTRUMENTATION;
            CALLGRIND_TOGGLE_COLLECT;
        }
    }

    void runAfter(const mlir::tracing::Action* action, size_t depth) {
        if (depth == 0 && instrumentationCache.first == action) {
            CALLGRIND_TOGGLE_COLLECT;
            CALLGRIND_STOP_INSTRUMENTATION;

            instrumentationCache = {nullptr, false};

            if (separateDumps) {
                const auto uniqueDumpName = getPrettyName(action) + "_" + std::to_string(++actionRepetitions[action]);
                CALLGRIND_DUMP_STATS_AT(uniqueDumpName.c_str());
            }
        }
    }
};
}  // namespace details

CallgrindProfiler::CallgrindProfiler(llvm::StringRef regex, bool separateDumps)
        : SelectiveProfiler(regex), _impl(std::make_unique<details::CallgrindProfilerImpl>(separateDumps)) {
    ERROR_OUT_WHEN_VALGRIND_IS_NOT_PRESENT;

    if (regex.empty()) {
        // note: for callgrind, it makes little sense to collect everything
        // through this primitive simply because one can use valgrind directly -
        // there's no fancy instrumentation available anyway.
        constexpr llvm::StringLiteral message = R"(Callgrind profiler is selected but its regex is empty.
If you intend to profile everything,
    a) just use valgrind directly without this compilation option (preferred);
    b) specify a universal regular expression with callgrind profiler)";

        Logger::global().nest("compiler-profiling").warning(message);
    }
}
CallgrindProfiler::~CallgrindProfiler() = default;

void CallgrindProfiler::profileBeforeExecute(const mlir::tracing::Action* action, size_t depth) {
    _impl->runBefore(action, depth);
}

void CallgrindProfiler::profileAfterExecute(const mlir::tracing::Action* action, size_t depth) {
    _impl->runAfter(action, depth);
}

}  // namespace vpux::compiler_profiling
