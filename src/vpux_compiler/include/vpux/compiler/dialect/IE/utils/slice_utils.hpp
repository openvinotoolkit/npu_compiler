//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"

namespace vpux {
namespace IE {

DimArr getDiffInOutSizeDims(ShapeRef inShape, ShapeRef outShape);
std::optional<vpux::Dim> getSingleDiffAxis(ShapeRef inShape, ShapeRef outShape);
SmallVector<uint64_t> getSliceAxes(IE::SliceOp sliceOp);
std::optional<bool> slicesHaveOverlap(ArrayRef<IE::SliceOp> slicesOp, const Dim& dim);

}  // namespace IE
}  // namespace vpux
