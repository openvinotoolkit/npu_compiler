//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/adaptive_stripping_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool VPU::hasEnableAdaptiveStripping(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_ADAPTIVE_STRIPPING).value_or(false);
}
