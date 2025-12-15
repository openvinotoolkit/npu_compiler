//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"

namespace vpux::VPU {
class VerticalFusionOp;
}

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/control_flow.hpp.inc>
