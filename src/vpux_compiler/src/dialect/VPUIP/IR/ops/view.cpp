//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

//
// ViewLikeOpInterface
//

mlir::Value vpux::VPUIP::ViewOp::getViewSource() {
    return getSource();
}

namespace {

bool areViewTypesCompatible(mlir::Type sourceType, mlir::Type resultType) {
    if (!vpux::isBufferType(sourceType) || !vpux::isBufferType(resultType) ||
        sourceType.getTypeID() != resultType.getTypeID() || !mlir::isa<vpux::NDTypeInterface>(sourceType) ||
        !mlir::isa<vpux::NDTypeInterface>(resultType)) {
        return false;
    }
    auto sourceBufferType = vpux::getBufferType(sourceType);
    auto resultBufferType = vpux::getBufferType(resultType);
    // ViewOp is a pure buffer reinterpretation. It may expose a different
    // shape/layout view over the same allocation.
    // It must not model memory movement, so memory space have to stay unchanged.
    // The result buffer type must not exceed the source buffer type in terms of total allocation size.
    return sourceBufferType.getMemSpace() == resultBufferType.getMemSpace() &&
           resultBufferType.getTotalAllocSize().count() <= sourceBufferType.getTotalAllocSize().count();
}

}  // namespace

//
// BackInferViewTypeOpInterface
//

std::optional<mlir::Type> VPUIP::ViewOp::inferOutputTypeFromInput(mlir::Type newInputType) {
    if (!areViewTypesCompatible(newInputType, getResult().getType())) {
        return std::nullopt;
    }
    return getResult().getType();
}

std::optional<mlir::Type> VPUIP::ViewOp::inferInputTypeFromOutput(mlir::Type desiredOutputType) {
    if (!areViewTypesCompatible(getSource().getType(), desiredOutputType)) {
        return std::nullopt;
    }
    return getSource().getType();
}
