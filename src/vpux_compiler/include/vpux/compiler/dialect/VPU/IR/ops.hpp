//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/types.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

// E#173010: remove dependency on IE operations for VPU operations
namespace vpux::IE {
class AvgPoolOp;
class AddOp;
class BatchNormInferenceOp;
class ConvolutionOp;
class GroupConvolutionOp;
class InterpolateOp;
class LSTMCellOp;
class LSTMSequenceOp;
class MatMulOp;
class MaxPoolOp;
class MultiplyOp;
class PermuteQuantizeOp;
class SubtractOp;
class TransposedConvolutionOp;
class YuvToRgbOp;
}  // namespace vpux::IE

//
// Generated
//

#include "vpux/compiler/dialect/VPU/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/bitwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/comparison.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/control_flow.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/logical.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/m2i.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
