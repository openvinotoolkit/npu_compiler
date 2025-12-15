//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>

#include <optional>
#include <tuple>

namespace mlir {
class PatternRewriter;
class MLIRContext;
}  // namespace mlir

namespace vpux {
namespace ShaveCodeGen {

/// @brief Converts a ranked tensor to a linalg compatible tensor. The output tensor
/// will have an identity layout. If the element type is a non-signless integer then
/// the element type is changed to be signless. The output tensor has the same
/// memory layout as the input tensor.
/// @param operand - the tensor to convert
/// @param rewriter
/// @return the converted tensor
mlir::Value convertToLinalgValue(mlir::Value operand, mlir::PatternRewriter& rewriter,
                                 std::optional<mlir::ArrayAttr> padding = std::nullopt);

/// @brief Converts from a ranked linalg-compatible tensor (identity layout, no non-signless
/// integer element types) to one with the specified output (same memory layout). The specified
/// output type must have the same memory layout as the input tensor.
/// @param operand - the tensor to convert
/// @param outputTy - the output tensor type
/// @return the converted tensor
mlir::Value convertFromLinalgValue(mlir::Value operand, mlir::Type outputTy, mlir::PatternRewriter& rewriter);

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

/// @brief Creates a tensor of the specified shape as a slice extracted from the start of a larger tensor filled with
/// zeros. The post-bufferization effect is that the extracted slice is zero-padded to large type. All tensor types are
/// expected to have an identity layout.
/// @param loc - the location information
/// @param sliceShape - the slice shape
/// @param allocType - the type of the larger zero-filled tensor
/// @param rewriter - the rewriter used to emit the operations
/// @return a tuple of the extracted slice and the zero-filled tensor
std::tuple<mlir::Value, mlir::Value> emitTensorSlice(mlir::Location loc, llvm::SmallVectorImpl<int64_t>& sliceShape,
                                                     mlir::RankedTensorType allocType, mlir::PatternRewriter& rewriter);
}  // namespace ShaveCodeGen
}  // namespace vpux
