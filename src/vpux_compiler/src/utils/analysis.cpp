//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/types.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <algorithm>

using namespace vpux;

//
// getFirstUser
//

mlir::Operation* vpux::getFirstUser(mlir::Value output) {
    VPUX_THROW_WHEN(output == nullptr, "Got NULL pointer in getFirstUser");

    const auto users = output.getUsers();
    const auto firstUser = std::min_element(users.begin(), users.end(), [](mlir::Operation* lhs, mlir::Operation* rhs) {
        return lhs->getBlock() == rhs->getBlock() && lhs->isBeforeInBlock(rhs);
    });

    return firstUser == users.end() ? nullptr : *firstUser;
}

//
// hasOneUniqueUser
//

bool vpux::hasOneUniqueUser(mlir::Operation* op) {
    auto users = op->getUsers();
    if (users.empty()) {
        return false;
    }

    auto firstUser = *users.begin();
    return std::all_of(std::next(users.begin()), users.end(), [&](mlir::Operation* userOp) {
        return firstUser == userOp;
    });
}

//
// isBufAllocOp
//

bool vpux::isBufAllocOp(mlir::Operation* op) {
    if (!op) {
        return false;
    }

    if (op->getNumOperands() != 0 || op->getNumResults() != 1) {
        return false;
    }

    if (!isBufferType(op->getResult(0).getType())) {
        return false;
    }

    if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        return iface.onlyHasEffect<mlir::MemoryEffects::Allocate>();
    }

    return false;
}

//
// getModuleOp
//

mlir::ModuleOp vpux::getModuleOp(mlir::Operation* op) {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) {
        return module;
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_WHEN(module == nullptr, "Can't get parent Module from Operation '{0}' at '{1}'", op->getName(),
                    op->getLoc());
    return module;
}

mlir::ModuleOp vpux::getModuleOp(mlir::OpBuilder& builder) {
    if (auto* block = builder.getInsertionBlock()) {
        if (auto* parentOp = block->getParentOp()) {
            return parentOp->getParentOfType<mlir::ModuleOp>();
        }
    }

    VPUX_THROW("No insertion point is set on a builder. Can't get a ModuleOp.");
    return nullptr;
}

//
// findReturnOp
//
mlir::func::ReturnOp vpux::findReturnOp(mlir::func::FuncOp funcOp) {
    return mlir::cast<mlir::func::ReturnOp>(funcOp.getBody().front().getTerminator());
}
