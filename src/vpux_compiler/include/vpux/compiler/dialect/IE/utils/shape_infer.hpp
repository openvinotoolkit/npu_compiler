//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/OpDefinition.h>

namespace vpux {
namespace IE {

bool isBroadcastable(int64_t d0, int64_t d1);

mlir::OpFoldResult reifyDim(mlir::OpBuilder builder, mlir::Value value, mlir::RankedTensorType type, size_t idx,
                            std::optional<mlir::Location> loc = std::nullopt);
mlir::OpFoldResult reifyDim(mlir::OpBuilder builder, mlir::Value value, size_t idx,
                            std::optional<mlir::Location> loc = std::nullopt);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> broadcastEltwiseShape(ArrayRef<ArrayRef<int64_t>> shapes,
                                                            AutoBroadcastType broadcastType, mlir::Location loc);

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyEltwiseTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                     mlir::Value input2,
                                                                     IE::AutoBroadcastType broadcastType,
                                                                     mlir::Location loc);

mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyMatMulTensors(mlir::OpBuilder& builder, mlir::Value input1,
                                                                    mlir::Value input2, bool transposeA,
                                                                    bool transposeB, mlir::Location loc);

/**
 * @brief Reify tensors for convolution or pooling operations. Currently, it supports only convolution with dilation
 * equal to 1 and pooling.
 *
 * @param builder - builder to create new operations
 * @param input - input tensor
 * @param output - output tensor
 * @param kernelSize - kernel size
 * @param strides - strides
 * @param padBegin - padding begin
 * @param padEnd - padding end
 *
 * @return reified shapes for output tensor
 */
mlir::FailureOr<SmallVector<mlir::OpFoldResult>> reifyConvPoolTensors(mlir::OpBuilder& builder, mlir::Value input,
                                                                      mlir::Value output, ArrayRef<int64_t> kernelSize,
                                                                      ArrayRef<int64_t> strides,
                                                                      ArrayRef<int64_t> padBegin,
                                                                      ArrayRef<int64_t> padEnd, mlir::Location loc);

mlir::FailureOr<SmallVector<int64_t>> constInputToData(mlir::Location loc, const mlir::Value& value);

mlir::FailureOr<Shape> getShapeCastExpandedShape(mlir::Operation* operation, ShapeRef expandedShape,
                                                 ShapeRef unExpandedShape, Logger log);
mlir::FailureOr<Shape> getShapeCastExpandedShapeInDimC(mlir::Operation* operation, ShapeRef originShape, Logger log);
mlir::FailureOr<Shape> getShapeCastExpandedShapeKeepDimC(mlir::Operation* operation, ShapeRef originShape, Logger log);

mlir::FailureOr<Shape> getShapeCastExpandedShapeCanNotAlign(mlir::Operation* operation, ShapeRef inputShape,
                                                            Logger log);

bool isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment);
bool isODUPermuteEffectiveForShape(const ShapeRef shape, const int64_t alignment);
}  // namespace IE
}  // namespace vpux
