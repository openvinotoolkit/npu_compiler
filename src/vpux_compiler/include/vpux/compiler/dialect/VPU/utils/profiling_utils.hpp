//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/utils/core/string_ref.hpp"

namespace vpux::VPU {

constexpr StringRef ENABLE_PROFILING = "VPU.EnableProfiling";

bool isProfilingEnabled(mlir::ModuleOp module);

}  // namespace vpux::VPU
