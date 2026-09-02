//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/compiler_profiling/vtune_profiler.hpp"
#include "vpux/compiler/utils/npu_actions.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/itt.hpp"

namespace vpux::compiler_profiling {

VTuneProfiler::VTuneProfiler(llvm::StringRef selection): SelectiveProfiler(selection.empty() ? ".*" : selection) {
    if (selection.empty()) {
        Logger::global()
                .nest("compiler-profiling")
                .warning("VTuneProfiler is initialized with an empty selection. All actions will be profiled.");
    }
}

void VTuneProfiler::profileBeforeExecute(const mlir::tracing::Action* action, size_t /*depth*/) {
    const std::string name = getPrettyName(action);
    auto taskHandle = openvino::itt::handle(name.c_str());
    openvino::itt::internal::taskBegin(vpux::itt::domains::VPUXPlugin(), taskHandle);
}

void VTuneProfiler::profileAfterExecute(const mlir::tracing::Action* /*action*/, size_t /*depth*/) {
    openvino::itt::internal::taskEnd(vpux::itt::domains::VPUXPlugin());
}
}  // namespace vpux::compiler_profiling
