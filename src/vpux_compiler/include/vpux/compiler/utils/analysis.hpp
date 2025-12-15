//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux::VPUIP {
class LayerOpInterface;
}

namespace vpux {

//
// getFirstUser
//

mlir::Operation* getFirstUser(mlir::Value output);

//
// hasOneUniqueUser
//

bool hasOneUniqueUser(mlir::Operation* op);

//
// isBufAllocOp
//

bool isBufAllocOp(mlir::Operation* op);

//
// getInputsSanitized
//

mlir::SmallVector<mlir::Value> getInputsSanitized(VPUIP::LayerOpInterface layerOp);

//
// getModuleOp
//

mlir::ModuleOp getModuleOp(mlir::Operation* op);
mlir::ModuleOp getModuleOp(mlir::OpBuilder& builder);

/// @brief This function returns the top parent operation of type OpT which contains this op. If the op itself is of
/// type OpT, then op is returned. If it doesn't exist, nullptr is returned.
template <typename OpT>
OpT getTopParentOpOfType(mlir::Operation* op) {
    OpT parent;
    while ((parent = op->getParentOfType<OpT>()) != nullptr) {
        op = parent;
    }

    return mlir::dyn_cast_or_null<OpT>(op);
}

}  // namespace vpux
