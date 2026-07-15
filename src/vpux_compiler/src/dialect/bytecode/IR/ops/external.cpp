//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/bytecode/IR/ops/external.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vpux/utils/core/small_vector.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>

#include <cstdint>

using namespace vpux;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/bytecode/ops/external.cpp.inc>

mlir::LogicalResult bytecode::ExtBufferViewOp::verify() {
    if (getShape().size() != getStrides().size()) {
        return emitOpError("expected shape and strides to have the same number of registers, got ")
               << getShape().size() << " and " << getStrides().size();
    }
    const int64_t rank = static_cast<int64_t>(getShape().size());
    if (rank > intel_npu::vm::BufferType::MAX_RANK) {
        return emitOpError("expected rank to fit in buffer rank type (max ")
               << intel_npu::vm::BufferType::MAX_RANK << "), got " << rank;
    }
    return mlir::success();
}

mlir::LogicalResult bytecode::ExtBufferCreateOp::verify() {
    auto memrefType = getBufferType();
    const int64_t rank = memrefType.getRank();
    if (rank < 0 || rank > intel_npu::vm::BufferType::MAX_RANK) {
        return emitOpError("expected memref rank to fit in buffer rank type (max ")
               << intel_npu::vm::BufferType::MAX_RANK << "), got " << rank;
    }

    // Each dynamic dimension takes its runtime extent from one dynamic_sizes register; static allocations carry none.
    const auto dynamicDims = static_cast<size_t>(memrefType.getNumDynamicDims());
    if (dynamicDims != getDynamicSizes().size()) {
        return emitOpError("expected ") << dynamicDims << " dynamic-size operand(s) for the dynamic dimensions of "
                                        << memrefType << ", got " << getDynamicSizes().size();
    }

    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (mlir::failed(memrefType.getStridesAndOffset(strides, offset))) {
        return emitOpError("memref layout must be representable as a strided layout, got ") << memrefType;
    }
    for (const auto stride : strides) {
        if (mlir::ShapedType::isDynamic(stride)) {
            return emitOpError("memref strides must be fully static, got ") << memrefType;
        }
    }

    if (mlir::ShapedType::isDynamic(offset)) {
        return emitOpError("memref offset must be static zero, got dynamic offset");
    }
    if (offset != 0) {
        return emitOpError("memref offset must be static zero, got ") << offset;
    }

    return mlir::success();
}
