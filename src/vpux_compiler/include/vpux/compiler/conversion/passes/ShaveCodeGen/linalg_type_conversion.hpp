//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <optional>

namespace mlir {
class PatternRewriter;
class MLIRContext;
}  // namespace mlir

namespace vpux {
namespace ShaveCodeGen {

/// @brief Remove padding (if any) from the type-converted operand
/// @param operand - the original operand
/// @param typeConvertedOperand - the original operand converted to identity layout
///        and with normalized element types.
/// @param rewriter - the current PatternRewriter
/// @param padding - the per-dimension padding amounts, in the original operand's order.
/// @return If no padding is present returns the type-converted operand, otherwise
///        emits operations to remove the padding from the type-converted operand
///        and returns the result
mlir::Value removePadding(mlir::Value operand, mlir::Value typeConvertedOperand, mlir::PatternRewriter& rewriter,
                          llvm::ArrayRef<int64_t> padding);

/// @brief Add the zero-filled padding (if any) back onto the unpadded equivalent of the result
/// @param result - the original (padded) op result. Provides the padded shape and the
///        dims order the padding is expressed in.
/// @param unpaddedResult - the computed result, in identity layout and without padding.
/// @param rewriter - the current PatternRewriter
/// @param padding - the per-dimension padding amounts, in the original result's order.
/// @return If no padding is present returns unpaddedResult, otherwise emits a tensor.pad
///        filling the padded region with zeros and returns its result.
mlir::Value addPadding(mlir::Value result, mlir::Value unpaddedResult, mlir::PatternRewriter& rewriter,
                       llvm::ArrayRef<int64_t> padding);

/// @brief Get the element type of a linalg-compatible tensor for the input type.
/// @param ty - the input tensor type
/// @param ctx
/// @return the element type
mlir::Type getLinalgElementType(mlir::Type ty, mlir::MLIRContext* ctx);

/// @brief Get the unpadded tensor type from a padded tensor type and the padding specification.
/// If the padded tensor type has an order specified the result will maintain this order.
/// @param type - the padded tensor type
/// @param loc - the location information
/// @param padding - an optional padding specification. If no padding is used the original
///        type is returned
/// @return the unpadded tensor type
mlir::RankedTensorType getUnpaddedTensorType(mlir::RankedTensorType type, mlir::Location loc,
                                             std::optional<mlir::ArrayAttr> padding);

/// @brief Get the identity order equivalent of the input tensor type such that the
/// in-memory representation is the same. If the element type is a non-signless integer
/// then the element type is changed to be signless.
/// @param type - a pottentially ordered ranked tensor type
/// @return - the equivalent memory-ordered type
mlir::RankedTensorType normalizeType(mlir::RankedTensorType type);

}  // namespace ShaveCodeGen
}  // namespace vpux
