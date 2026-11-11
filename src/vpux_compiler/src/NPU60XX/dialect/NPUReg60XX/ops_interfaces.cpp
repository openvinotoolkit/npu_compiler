//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops_interfaces.hpp"
#include "vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

//
// SingleOutputAsIndexOp
//

mlir::LogicalResult vpux::NPUReg60XX::verifySingleOutputAsIndexOp(mlir::Operation* op) {
    if (op->getNumResults() != 1) {
        return errorAt(op, "Operation '{0}' does not have a single index type result", op->getName());
    }
    if (!mlir::isa<vpux::VPURegMapped::IndexType>(op->getResult(0).getType())) {
        return errorAt(op, "Operation '{0}' result type is not VPURegMapped::IndexType", op->getName());
    }

    return mlir::success();
}

//
// Generated
//

#include <vpux/compiler/NPU60XX/dialect/NPUReg60XX/ops_interfaces.cpp.inc>
