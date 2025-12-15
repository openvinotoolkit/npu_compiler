//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/auxiliary_buffers.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

using namespace vpux;

mlir::Value VPU::createAuxiliaryBuffer(mlir::OpBuilder& builder, mlir::Location opLoc, mlir::Type type) {
    const auto loc = appendLoc(opLoc, "_aux");
    auto ndType = mlir::cast<NDTypeInterface>(type);
    if (mlir::isa<mlir::Float16Type>(ndType.getElementType())) {
        std::vector<type::float16> vals(ndType.getShape().totalSize(), 0.0f);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else if (mlir::isa<mlir::Float32Type>(ndType.getElementType())) {
        std::vector<float> vals(ndType.getShape().totalSize(), 0.0f);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else if (ndType.getElementType() == getUInt32Type(builder.getContext())) {
        std::vector<uint32_t> vals(ndType.getShape().totalSize(), 0);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else if (ndType.getElementType() == getSInt32Type(builder.getContext())) {
        std::vector<int32_t> vals(ndType.getShape().totalSize(), 0);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else {
        std::vector<uint8_t> vals(ndType.getTotalAllocSize().count(), 0);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    }
}

mlir::LogicalResult VPU::compareTypes(mlir::Location loc, mlir::Type actual, mlir::Type expected) {
    auto expectedType = mlir::cast<NDTypeInterface>(expected);
    auto auxBufferType = mlir::cast<NDTypeInterface>(actual);
    if (auxBufferType.getShape() != expectedType.getShape()) {
        return errorAt(loc, "Expected auxiliary buffer shape {0}, but got {1}", expectedType.getShape(),
                       auxBufferType.getShape());
    }
    if (auxBufferType.getElementType() != expectedType.getElementType()) {
        return errorAt(loc, "Expected auxiliary buffer element type {0}, but got {1}", expectedType.getElementType(),
                       auxBufferType.getElementType());
    }
    return mlir::success();
};
