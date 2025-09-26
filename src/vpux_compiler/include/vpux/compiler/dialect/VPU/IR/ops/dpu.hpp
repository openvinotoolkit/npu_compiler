//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux::VPU {
class InterpolateOp;
}

namespace vpux::IE {
class InterpolateOp;
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/dpu.hpp.inc>
