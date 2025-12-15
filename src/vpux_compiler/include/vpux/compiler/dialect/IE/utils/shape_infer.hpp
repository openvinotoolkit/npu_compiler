//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/OpDefinition.h>

namespace vpux::IE {
enum class AutoBroadcastType : uint64_t;
}

namespace vpux {
namespace IE {

vpux::DimsOrder inferOrder(vpux::NDTypeInterface lhsType, vpux::NDTypeInterface rhsType);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> constInputToData(mlir::Location loc, mlir::Value value);

mlir::FailureOr<Shape> getShapeCastExpandedShape(mlir::Operation* operation, ShapeRef expandedShape,
                                                 ShapeRef unExpandedShape, Logger log);
mlir::FailureOr<Shape> getShapeCastExpandedShapeInDimC(mlir::Operation* operation, ShapeRef originShape, Logger log);
mlir::FailureOr<Shape> getShapeCastExpandedShapeKeepDimC(mlir::Operation* operation, ShapeRef originShape, Logger log);

mlir::FailureOr<Shape> getShapeCastExpandedShapeCanNotAlign(mlir::Operation* operation, ShapeRef inputShape,
                                                            Logger log);

mlir::FailureOr<Shape> getShapeCastExpandedShapeWithMinimalDimChange(mlir::Operation* operation, ShapeRef inputShape,
                                                                     Logger log);

bool isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment);
bool isODUPermuteEffectiveForShape(const ShapeRef shape, const int64_t alignment);

SmallVector<int64_t> dispatchBounds(const mlir::Value operand);
SmallVector<int64_t> inferOutputBounds(const mlir::Value lhs, const mlir::Value rhs, const ShapeRef outputShape,
                                       const IE::AutoBroadcastType autoBroadcast);
}  // namespace IE
}  // namespace vpux
