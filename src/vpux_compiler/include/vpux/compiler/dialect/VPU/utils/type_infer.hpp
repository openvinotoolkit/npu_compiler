//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/core/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>

namespace vpux::IE {
enum class AutoBroadcastType : uint64_t;
}

namespace vpux {
namespace VPU {

mlir::LogicalResult inferReduceReturnTypes(mlir::Location loc, mlir::Value input, bool keepDims,
                                           SmallVector<int64_t>& axes,
                                           mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes,
                                           mlir::ArrayAttr inputPadding = nullptr,
                                           mlir::ArrayAttr outputPadding = nullptr);
void inferPermuteReturnTypes(mlir::Value input, mlir::AffineMap mem_perm, mlir::AffineMap dst_order,
                             SmallVectorImpl<mlir::Type>& inferredReturnTypes);
mlir::LogicalResult inferEltwiseReturnTypes(SmallVectorImpl<mlir::Type>& inferredReturnTypes, mlir::Location loc,
                                            mlir::Value input1, mlir::Value input2, IE::AutoBroadcastType broadcast,
                                            std::optional<mlir::Type> outElemType = std::nullopt);

TensorAttr createTensorAttrFromType(NDTypeInterface inType);
mlir::FailureOr<TensorAttr> createOutTensorAttrFromType(NDTypeInterface inType, size_t outRank);

}  // namespace VPU
}  // namespace vpux
