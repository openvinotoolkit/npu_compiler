//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/IE/IR/ops.hpp"

namespace vpux {

IE::ConcatOp buildConcat(mlir::Location loc, mlir::OpBuilder& builder, ShapeRef producerShape,
                         mlir::ValueRange dynamicOperands);

mlir::Value repackDynamicTensor(mlir::OpBuilder& builder, mlir::Operation* producer, NDTypeInterface operandType,
                                IE::ConcatOp newShapeValue);

}  // namespace vpux
