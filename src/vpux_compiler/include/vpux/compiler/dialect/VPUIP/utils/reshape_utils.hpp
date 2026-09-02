//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/dim.hpp"
#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/strides_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <optional>
#include <utility>

namespace vpux {
namespace VPUIP {

std::optional<MemDimArr> deduceLegalOutputMemDims(MemShapeRef inMemShape, MemShapeRef outMemShape, MemDim inMemDim);
mlir::FailureOr<vpux::NDTypeInterface> updateStridesForReshape(const vpux::NDTypeInterface& inType,
                                                               const vpux::NDTypeInterface& outType);
bool isInAndOutStridesCompatible(const vpux::NDTypeInterface& inType, const vpux::NDTypeInterface& outType);

//
// Back type inference for reshape/shape-change view ops
//
// Helpers shared by VPUIP view-like op implementations to reconstruct the input
// or output type across a shape change. Pass code should use BackInferUtils or
// BackInferViewTypeOpInterface rather than calling these directly.
//

// Infers the element type on the opposite side of a shape-only view.
//
// The "known" side is the side whose type is already rewritten by the caller.
// Forward example: use rewritten input as known side to infer output.
// Reverse example: use desired output as known side to infer input.
std::optional<mlir::Type> inferReshapeElementType(mlir::Type newKnownElemType, mlir::Type origKnownElemType,
                                                  mlir::Type origOtherElemType, ShapeRef otherShape);
// Infers the full output type of a reshape for a new input type, including
// distributed axis remapping when the input is a DistributedBufferType.
std::optional<mlir::Type> inferReshapeOutputType(vpux::NDTypeInterface newInputNDType,
                                                 vpux::NDTypeInterface origInputNDType,
                                                 vpux::NDTypeInterface origOutputNDType, ShapeRef outShape);
// Maps the distributed tiling axis of the source type onto the target type across a reshape.
std::optional<std::pair<int64_t, int64_t>> inferReshapeDistributedAxesMapping(vpux::NDTypeInterface sourceType,
                                                                              vpux::NDTypeInterface targetType,
                                                                              VPU::DistributionInfoAttr distribution);
// Infers the input type of a reshape needed to produce the desired output type.
std::optional<mlir::Type> inferReshapeInputType(mlir::Operation* op, vpux::NDTypeInterface desiredOutputNDType,
                                                vpux::NDTypeInterface origInputNDType,
                                                vpux::NDTypeInterface origOutputNDType);

}  // namespace VPUIP
}  // namespace vpux
