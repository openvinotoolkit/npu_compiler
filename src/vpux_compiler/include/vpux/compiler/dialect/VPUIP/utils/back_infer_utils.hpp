//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>

#include <llvm/ADT/ArrayRef.h>

#include <optional>

namespace vpux {
namespace VPUIP {

/// Type propagation for view-like ops used when a pass moves or recreates a
/// view around a rewritten buffer.
///
/// Example: if a DDR->CMX copy is moved across a ShapeCast, the pass can ask
/// ShapeCast what output type would be produced from the new CMX input. For the
/// reverse direction, it can ask what input type is required to obtain an
/// already chosen output type.
class BackInferUtils {
public:
    /// Given a view-like op and a new input type, compute the output type that
    /// the op would produce.
    /// Returns std::nullopt if type inference fails.
    static std::optional<mlir::Type> inferOutputType(mlir::Operation* viewOp, mlir::Type newInputType);

    /// Given a view-like op and a desired output type, compute the input type
    /// needed for the op to produce that output. The implementation validates
    /// the result with a forward round-trip before returning it.
    /// Returns std::nullopt if reverse type inference fails.
    static std::optional<mlir::Type> reverseInferInputType(mlir::Operation* viewOp, mlir::Type desiredOutputType);
};

/// Infer the type produced by a planned contiguous SubView tile when no
/// VPUIP::SubViewOp exists yet. This overload uses unit strides. Example: a pass
/// can test whether offsets [0, 0, 0, 0] and sizes [1, 16, 32, 32] are legal for
/// a candidate distributed buffer before creating the real SubViewOp. Callers
/// with a real SubViewOp should use BackInferUtils through the generic view-like
/// interface instead.
std::optional<mlir::Type> inferSubViewOutputTypeFromTile(mlir::Type inputType, llvm::ArrayRef<int64_t> staticOffsets,
                                                         llvm::ArrayRef<int64_t> staticSizes);

/// Infer the type produced by a planned strided SubView tile. The strides array
/// must be explicit and have the same rank as offsets/sizes/input type.
std::optional<mlir::Type> inferSubViewOutputTypeFromTile(mlir::Type inputType, llvm::ArrayRef<int64_t> staticOffsets,
                                                         llvm::ArrayRef<int64_t> staticSizes,
                                                         llvm::ArrayRef<int64_t> staticStrides);

}  // namespace VPUIP
}  // namespace vpux
