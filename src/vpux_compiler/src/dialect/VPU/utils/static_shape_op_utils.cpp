//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/static_shape_op_utils.hpp"

#include <llvm/ADT/TypeSwitch.h>
#include <algorithm>

using namespace vpux;

bool VPU::hasEnableExtraStaticShapeOps(mlir::ModuleOp module) {
    return VPU::tryGetBoolPassOption(module, ENABLE_EXTRA_STATIC_SHAPE_OPS).value_or(false);
}
