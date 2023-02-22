//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/core/ops_interfaces.hpp"
#include "vpux/compiler/dialect/EMU/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/dialect.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

#include <mlir/Dialect/Quant/QuantOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

//
// Generated
//

#include <vpux/compiler/dialect/EMU/generated/dialect.hpp.inc>

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/EMU/generated/ops.hpp.inc>

//
// Operation verifiers
//

namespace vpux {
namespace EMU {

constexpr Bit FP16_SIZE = 16_Bit;

mlir::LogicalResult verifyOp(NCEClusterTaskOp op);
mlir::LogicalResult verifyPostOp(mlir::Operation* op);

}  // namespace EMU
}  // namespace vpux
