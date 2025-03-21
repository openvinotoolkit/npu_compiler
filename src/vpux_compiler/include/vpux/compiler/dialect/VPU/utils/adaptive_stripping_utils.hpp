//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <vpux/compiler/utils/passes.hpp>
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/setup_pipeline_options_utils.hpp"

namespace vpux {
namespace VPU {

constexpr StringRef ENABLE_ADAPTIVE_STRIPPING = "VPU.EnableAdaptiveStripping";

bool hasEnableAdaptiveStripping(mlir::ModuleOp module);

}  // namespace VPU
}  // namespace vpux
