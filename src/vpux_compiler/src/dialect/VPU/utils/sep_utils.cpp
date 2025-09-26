//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/sep_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool VPU::hasEnableSEPtrsOperations(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_SE_PTRS_OPERATIONS).value_or(false);
}

bool VPU::hasEnableExperimentalSEPtrsOperations(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_EXPERIMENTAL_SE_PTRS_OPERATIONS).value_or(false);
}
