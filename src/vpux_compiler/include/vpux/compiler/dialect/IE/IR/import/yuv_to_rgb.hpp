//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

namespace vpux::IE::YuvToRgb {

/** @brief Canonicalizes the input for an IE::YuvToRgb operation by applying necessary transpositions.
 */
mlir::Value canonicalizeYuvToRgbInput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value input,
                                      llvm::StringRef locSuffix);
mlir::Value restoreYuvToRgbOutput(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value output);

}  // namespace vpux::IE::YuvToRgb
