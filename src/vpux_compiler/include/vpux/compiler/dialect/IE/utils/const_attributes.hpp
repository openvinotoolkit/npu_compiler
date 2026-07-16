//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/const/ops.hpp"

namespace vpux {
namespace IE {

mlir::ArrayAttr getIntArrayAttrValue(mlir::Value operand);
mlir::ArrayAttr getFloatArrayAttrValue(mlir::Value operand);
mlir::IntegerAttr getIntAttrValue(mlir::Value operand, mlir::PatternRewriter& rewriter);
mlir::FailureOr<Const::DeclareOp> getConstParentOp(mlir::Value input);
mlir::FailureOr<int64_t> getBaseContentNumElements(Const::DeclareOp constOp);

// Get base content number elements that excludes the influence of attribution (broadcast, expand...)
// Example 1: "dense<[[[[1.0]]], [[[2.0]]], [[[3.0]]]]> : tensor<3x1x1x1xf16>"
//  - return: 3x1x1x1 = 3
// Example 2: "dense<1.0> : tensor<1x1x1x1xf16>"
//  - return: 1x1x1x1 = 1
// Example 3: "dense<1.0> : tensor<3x1x1x1xf16>"
//  - return: 3x1x1x1 = 3
mlir::FailureOr<int64_t> getBaseContentNumElements(Const::DeclareOp constOp);

// Check the Constant only has a single value
// Example 1: "dense<[[[[1.0]]], [[[2.0]]], [[[3.0]]]]> : tensor<3x1x1x1xf16>"
//  - return: false
// Example 2: "dense<1.0> : tensor<1x1x1x1xf16>"
//  - return: true
// Example 3: "dense<1.0> : tensor<3x1x1x1xf16>"
//  - return: true
bool isBaseContentSplat(Const::DeclareOp constOp);

SmallVector<double> readConstAsDoubles(Const::DeclareOp constOp);
SmallVector<int64_t> readConstAsInt64(Const::DeclareOp constOp);

}  // namespace IE
}  // namespace vpux
