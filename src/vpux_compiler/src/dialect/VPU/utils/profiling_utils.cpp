//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/profiling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

using namespace vpux;
using namespace vpux::VPU;

bool vpux::VPU::isProfilingEnabled(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_PROFILING).value_or(false);
}
