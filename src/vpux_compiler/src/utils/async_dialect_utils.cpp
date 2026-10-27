//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/async_dialect_utils.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

using namespace vpux;

mlir::Type vpux::getAsyncValueType(mlir::Value value) {
    auto type = value.getType();
    if (const auto asyncType = mlir::dyn_cast<mlir::async::ValueType>(type)) {
        type = asyncType.getValueType();
    }
    return type;
}

// Allocate a new buffer for re-read operation with the same type and attributes as the original buffer
mlir::Value vpux::allocateSpillReadBuffer(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value bufferToSpill) {
    mlir::Operation* newBufferOp = nullptr;

    if (auto distAllocOp = bufferToSpill.getDefiningOp<VPURT::AllocDistributed>()) {
        newBufferOp =
                VPURT::AllocDistributed::create(builder, loc, bufferToSpill.getType(), distAllocOp.getAlignmentAttr(),
                                                distAllocOp.getSwizzlingKeyAttr());
    } else if (auto allocOp = bufferToSpill.getDefiningOp<VPURT::Alloc>()) {
        newBufferOp = VPURT::Alloc::create(builder, loc, bufferToSpill.getType(), allocOp.getAlignmentAttr(),
                                           allocOp.getSwizzlingKeyAttr());
    } else {
        newBufferOp =
                mlir::memref::AllocOp::create(builder, loc, mlir::cast<mlir::MemRefType>(bufferToSpill.getType()));
    }

    return newBufferOp->getResult(0);
}
