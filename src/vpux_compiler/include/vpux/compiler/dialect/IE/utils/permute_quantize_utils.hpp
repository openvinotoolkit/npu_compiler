//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

namespace vpux {
namespace IE {

bool isPurePermuteCompatiblePrecision(mlir::Type inElemType, mlir::Type outElemType);
bool isLegalReorderAddPattern(IE::ReorderOp origOp);
bool isLegalReorderAvgPoolPattern(IE::ReorderOp origOp);
bool isBeneficialConvertToPermuteQuantize(ShapeRef shape);
bool isLegalReorderLikeToPermuteQuantize(vpux::NDTypeInterface inType, vpux::NDTypeInterface outType, Logger log);
std::optional<SmallVector<int64_t>> getAdjustHW(int64_t alignment, int64_t width, int64_t height);
bool isODUPermuteEffectiveForShape(const ShapeRef shape, const int64_t alignment);
bool isShapeCompatibleWithODUPermute(const ShapeRef shape, const int64_t alignment);
bool canConvertToNCHWInOrderWithPermuteCast(vpux::NDTypeInterface inType, mlir::AffineMap memPerm);
bool checkNCEPermuteShapeCompatibility(ShapeRef inShape, ShapeRef outShape, int64_t alignment);

}  // namespace IE
}  // namespace vpux
