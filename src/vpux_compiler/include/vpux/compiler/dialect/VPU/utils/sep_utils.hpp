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

constexpr StringRef ENABLE_SE_PTRS_OPERATIONS = "VPU.EnableSEPtrsOperations";
constexpr StringRef ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS = "VPU.EnableExperimentalSEPtrsOperations";

bool hasEnableSEPtrsOperations(mlir::ModuleOp module);
bool hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module);

}  // namespace VPU
}  // namespace vpux
