//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <algorithm>

using namespace vpux;

//
// getFirstUser
//

mlir::Operation* vpux::getFirstUser(mlir::Value output) {
    VPUX_THROW_UNLESS(output != nullptr, "Got NULL pointer in getFirstUser");

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

mlir::SmallVector<mlir::Value> vpux::getInputsSanitized(VPUIP::LayerOpInterface layerOp) {
    auto inputs = vpux::to_small_vector(layerOp.getInputs());

    // handle dynamic input shapes which are not part of getInputs()
    if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(layerOp.getOperation())) {
        auto dynamicInputs = swOp.getDynamicInputShapes();
        std::move(dynamicInputs.begin(), dynamicInputs.end(), std::back_inserter(inputs));
    }

    // handle parent input / output buffer duplication
    if (auto nceTaskOp = mlir::dyn_cast<VPUIP::NCEClusterTaskOp>(layerOp.getOperation())) {
        // in case of NCEClusterTaskOp we need to remove parent outputs from inputs
        // in order to make dependency calculation work correctly
        auto parentOutput = nceTaskOp.getParentOutput();
        auto parentOutputSparsityMap = nceTaskOp.getParentOutputSparsityMap();
        auto input = nceTaskOp.getInput();
        auto inputSparsityMap = nceTaskOp.getInputSparsityMap();
        auto weights = nceTaskOp.getWeights();
        auto weightsSparsityMap = nceTaskOp.getWeightsSparsityMap();
        llvm::SmallVector<mlir::Value> inputsToSanitize{};
        inputsToSanitize.swap(inputs);
        std::copy_if(inputsToSanitize.begin(), inputsToSanitize.end(), std::back_inserter(inputs),
                     [&](mlir::Value value) {
                         // For in-place eltwise op it might happen that parentOutput == input.
                         // Check those first to make sure they don't get removed.
                         if (value == input || value == inputSparsityMap || value == weights ||
                             value == weightsSparsityMap) {
                             return true;
                         }
                         return (value != parentOutput) && (value != parentOutputSparsityMap);
                     });
    }

    return inputs;
}

//
// getModuleOp
//

mlir::ModuleOp vpux::getModuleOp(mlir::Operation* op) {
    if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op)) {
        return module;
    }

    auto module = op->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(module != nullptr, "Can't get parent Module from Operation '{0}' at '{1}'", op->getName(),
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
