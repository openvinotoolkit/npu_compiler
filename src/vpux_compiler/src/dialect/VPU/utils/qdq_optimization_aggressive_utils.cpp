//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/qdq_optimization_aggressive_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool VPU::hasEnableQDQOptimizationAggressive(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_QDQ_OPTIMIZATION_AGGRESSIVE).value_or(false);
}
