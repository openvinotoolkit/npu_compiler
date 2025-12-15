//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/dialect.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/ops_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/traits_utils.hpp"

#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>

#include <functional>

using namespace vpux::VPUIPDPU;
using namespace mlir;

//
// Generated
//

#define GET_OP_CLASSES
#include <vpux/compiler/dialect/VPUIPDPU/ops.cpp.inc>

//
// Custom
//

mlir::LogicalResult vpux::VPUIPDPU::DPUInvariantOp::verify() {
    if (!hasMandatorySingleInstanceChildren<DPUInvariantOp, IDUCfgOp, ODUCfgOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: missing mandatory child ops", getOperationName());
    }
    if (!hasOptionalSingleInstanceChildren<DPUInvariantOp, MPECfgOp>(*this)) {
        return errorAt(getLoc(), "Operation {0}: too many optional child ops", getOperationName());
    }

    return ::mlir::success();
}

mlir::LogicalResult vpux::VPUIPDPU::MPECfgOp::verify() {
    auto op = this->getOperation();
    if (!mlir::isa<VPUIPDPU::MPECfgOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not MPECfg", op->getName());
    }

    auto iface = mlir::cast<VPUIPDPU::MPECfgOpInterface>(op);
    return iface.verifyInnerOps();
}

mlir::LogicalResult vpux::VPUIPDPU::ODUCfgOp::verify() {
    auto op = this->getOperation();
    if (!mlir::isa<VPUIPDPU::ODUCfgOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not ODUCfg", op->getName());
    }

    auto iface = mlir::cast<VPUIPDPU::ODUCfgOpInterface>(op);
    return iface.verifyInnerOps();
}

mlir::LogicalResult vpux::VPUIPDPU::DPUVariantOp::verify() {
    auto op = this->getOperation();
    if (!mlir::isa<VPUIPDPU::DPUVariantOpInterface>(op)) {
        return errorAt(op, "Operation '{0}' is not DPUVariant", op->getName());
    }

    auto iface = mlir::cast<VPUIPDPU::DPUVariantOpInterface>(op);
    return iface.verifyInnerOps();
}
