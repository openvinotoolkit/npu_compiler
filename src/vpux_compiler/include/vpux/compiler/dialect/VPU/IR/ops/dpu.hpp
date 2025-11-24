//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/asm.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

namespace vpux::VPU {
class InterpolateOp;
}

namespace vpux::IE {
class AvgPoolOp;
class AddOp;
class ConvolutionOp;
class GroupConvolutionOp;
class InterpolateOp;
class MatMulOp;
class MaxPoolOp;
class MultiplyOp;
class PermuteQuantizeOp;
class SubtractOp;
class TransposedConvolutionOp;
}  // namespace vpux::IE

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPU/ops/dpu.hpp.inc>
