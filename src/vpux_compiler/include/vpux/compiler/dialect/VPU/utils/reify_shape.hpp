//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

namespace vpux::VPU {

// VPU-dialect counterparts of vpux::buildConcat/vpux::repackDynamicTensor (see
// vpux/compiler/dialect/IE/utils/reify_shape.hpp). VPU::DynamicReshapeOp requires its `shape`
// operand to be a si32 tensor (unlike IE::DynamicReshapeOp, which accepts si64), so the concat
// built here packs each dimension as a 1-element si32 tensor instead of si64.

VPU::ConcatOp buildConcat(mlir::Location loc, mlir::OpBuilder& builder, ShapeRef producerShape,
                          mlir::ValueRange dynamicOperands);

mlir::Value repackDynamicTensor(mlir::OpBuilder& builder, mlir::Operation* producer, NDTypeInterface operandType,
                                VPU::ConcatOp newShapeValue);

}  // namespace vpux::VPU
