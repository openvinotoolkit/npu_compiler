//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUMI37XX/ops_interfaces.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult vpux::VPUMI37XX::verifySingleOutputAsIndexOp(mlir::Operation* op) {
    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation '{0}' does not have a single index type result", op->getName());
    }
    if (!op->getResult(0).getType().isa<VPURegMapped::IndexType>()) {
        return errorAt(op, "Operation '{0}' result type is not VPURegMapped::IndexType", op->getName());
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/dialect/VPUMI37XX/ops_interfaces.cpp.inc>
