//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux::IE {
class InterpolateOp;
class YuvToRgbOp;
class BatchNormInferenceOp;
}  // namespace vpux::IE

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/m2i.hpp.inc>
