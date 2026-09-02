//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <string>
#include <tuple>

namespace vpux::IE::BatchNormInference {

using BatchNormAttrs = std::tuple<mlir::ArrayAttr, mlir::ArrayAttr, mlir::ArrayAttr, mlir::ArrayAttr>;

/** @brief Returns the attributes of an IE::BatchNormInferenceOp.
 */
llvm::FailureOr<BatchNormAttrs> parseAttributes(mlir::MLIRContext* ctx, llvm::ArrayRef<mlir::Value> inputs);

}  // namespace vpux::IE::BatchNormInference
