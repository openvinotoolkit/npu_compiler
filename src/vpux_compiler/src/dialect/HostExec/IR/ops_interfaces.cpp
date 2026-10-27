//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/HostExec/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/HostExec/params.hpp"
#include "vpux/compiler/utils/analysis.hpp"

using namespace vpux;

void HostExec::getOpEffects(mlir::Operation* op, mlir::SmallVectorImpl<HostExec::MemoryEffect>& effects) {
    if (op == nullptr) {
        return;
    }

    if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op); callOp != nullptr) {
        // Note: the effects are queried by function-scoped passes which may run in parallel on sibling functions,
        // so all the information local to the call operation must be checked before the callee is resolved through
        // the parent-module symbol table, which reads the operations owned by the other functions.
        auto numOperands = callOp.getOperands().size();
        if (numOperands < HostMainFuncArgs::HOST_MAIN_FUNC_ARGS_COUNT) {
            return;
        }

        auto module = vpux::getModuleOp(callOp);
        VPUX_THROW_WHEN(module == nullptr, "Call operation: {0} must have a parent-module", callOp->getName());
        auto firstFuncOp = module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
        VPUX_THROW_WHEN(firstFuncOp == nullptr, "Call operation: {0} must have an associated parent-function",
                        callOp->getName());
        if (isHostCompileInferenceExecFunc(firstFuncOp)) {
            effects.emplace_back(mlir::MemoryEffects::Write::get(),
                                 callOp.getOperand(GET_ARG_INDEX_COMMAND_FENCE(numOperands)));
            effects.emplace_back(mlir::MemoryEffects::Write::get(),
                                 callOp.getOperand(GET_ARG_INDEX_COMMAND_EVENT(numOperands)));
            effects.emplace_back(mlir::MemoryEffects::Write::get(),
                                 callOp.getOperand(GET_ARG_INDEX_COMMAND_QUEUE(numOperands)));
            effects.emplace_back(mlir::MemoryEffects::Write::get(),
                                 callOp.getOperand(GET_ARG_INDEX_COMMAND_LIST(numOperands)));
            effects.emplace_back(mlir::MemoryEffects::Write::get(),
                                 callOp.getOperand(GET_ARG_INDEX_COMMAND_EXECUTION_CONTEXT(numOperands)));
        }
    }
}
