//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

namespace vpux {
namespace VPU {

mlir::Value createAuxiliaryBuffer(mlir::OpBuilder& builder, mlir::Location opLoc, mlir::Type type);
mlir::LogicalResult compareTypes(mlir::Location loc, mlir::Type actual, mlir::Type expected);

}  // namespace VPU
}  // namespace vpux
