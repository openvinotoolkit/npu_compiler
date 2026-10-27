//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/types.hpp"
#include "vpux/compiler/dialect/core/IR/dynamic_attrs.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace vpux;

namespace {
mlir::LogicalResult verifyTypeIsEitherBoundedBufferOrDynamicMemref(mlir::Operation* op, mlir::Type type) {
    return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(type)
            .Case([](VPUIP::BoundedBufferType) {
                return mlir::success();
            })
            .Case([op](mlir::MemRefType memref) {
                if (memref.hasStaticShape()) {
                    return errorAt(op, "memref must have dynamic shape, but got static: {0}", memref);
                }
                const auto bounds = getBounds(memref);
                if (bounds.empty()) {
                    return errorAt(op, "dynamic memref must have bounds, but got none: {0}", memref);
                }
                return mlir::success();
            })
            .Default([op](mlir::Type unknown) {
                return errorAt(op, "type must be either VPUIP.BoundedBuffer or dynamic memref with bounds, got: {0}",
                               unknown);
            });
}
}  // namespace

void VPUIP::GroupBoundedBufferOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::Value data,
                                        mlir::Value shape) {
    state.addOperands({data, shape});
    const auto dataType = data.getType();
    const auto shapeType = shape.getType();
    state.addTypes(VPUIP::BoundedBufferType::get(dataType, shapeType));
}

mlir::ValueRange VPUIP::GroupBoundedBufferOp::getViewSources() {
    return getOperands();
}

mlir::LogicalResult VPUIP::GroupBoundedBufferOp::verify() {
    return verifyTypeIsEitherBoundedBufferOrDynamicMemref(getOperation(), getOutput().getType());
}

void VPUIP::UngroupBoundedBufferOp::build(mlir::OpBuilder&, mlir::OperationState& state, mlir::Value input) {
    state.addOperands({input});
    const auto boundedBufferType = mlir::cast<VPUIP::BoundedBufferType>(input.getType());
    state.addTypes({boundedBufferType.getData(), boundedBufferType.getDynamicShape()});
}

mlir::Value VPUIP::UngroupBoundedBufferOp::getViewSource(ptrdiff_t idx) {
    VPUX_THROW_UNLESS(idx == 0 || idx == 1,
                      "UngroupBoundedBufferOp should have one view source with two aliases, got {0} offset", idx);
    return getOperand();
}

mlir::LogicalResult VPUIP::UngroupBoundedBufferOp::verify() {
    return verifyTypeIsEitherBoundedBufferOrDynamicMemref(getOperation(), getInput().getType());
}
