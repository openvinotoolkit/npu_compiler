//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "vpux/compiler/core/attributes/dims_order.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {

namespace IE {

mlir::LogicalResult inferReduceReturnTypeComponents(mlir::Location loc, mlir::Value input, bool keepDims,
                                                    SmallVector<int64_t>& axes,
                                                    SmallVectorImpl<mlir::ShapedTypeComponents>& inferredReturnShapes,
                                                    mlir::ArrayAttr inputPadding = nullptr,
                                                    mlir::ArrayAttr outputPadding = nullptr);
DimsOrder calculateReducedOutputLayout(const DimsOrder& inputDimOrder, const SmallVector<int64_t>& axes);

// Checks that axes represent a single channel-axis reduction and the parent op's
// activation input and output have the same layout (no ODU permute active).
// Used as a sub-check in both IE and VPU reduce optimization passes.
bool isChannelAxisReductionWithMatchingLayout(vpux::NDTypeInterface parentInputType,
                                              vpux::NDTypeInterface parentOutputType, ArrayRef<int64_t> axes,
                                              Logger log);

bool isChannelAxisReductionWithDPUParent(mlir::Operation* op, ArrayRef<int64_t> axes, Logger log);

// Extracts a statically-known axes vector from a constant value. Ensures axes
// are always normalized (non-negative and sorted).
mlir::FailureOr<SmallVector<int64_t>> getReduceAxes(mlir::Location loc, mlir::Value axesInput, size_t inputRank);

}  // namespace IE
}  // namespace vpux
