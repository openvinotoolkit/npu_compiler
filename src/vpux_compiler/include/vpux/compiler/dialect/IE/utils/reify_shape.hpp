//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

namespace vpux {

IE::ConcatOp buildConcat(mlir::Location loc, mlir::OpBuilder& builder, ShapeRef producerShape,
                         mlir::ValueRange dynamicOperands);

mlir::Value repackDynamicTensor(mlir::Location loc, mlir::OpBuilder& builder, mlir::Operation* producer,
                                NDTypeInterface operandType, IE::ConcatOp newShapeValue);

}  // namespace vpux
