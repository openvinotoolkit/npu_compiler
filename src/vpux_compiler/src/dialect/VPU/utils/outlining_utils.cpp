//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/outlining_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Visitors.h>

using namespace vpux;

bool VPU::isConstOperandOp(mlir::Operation* op) {
    if (mlir::isa<VPU::StorageElementTableOp, VPU::DataPointerTableOp, VPU::ZeroPointTableOp, Const::DeclareOp>(op)) {
        return true;
    }

    if (mlir::isa<VPU::GroupedViewLikeOpInterface>(op)) {
        return llvm::all_of(op->getOperands(), [&](mlir::Value v) {
            if (mlir::isa<mlir::BlockArgument>(v)) {
                return true;
            }
            auto parentOp = v.getDefiningOp();
            return isConstOperandOp(parentOp);
        });
    }

    return false;
}

// Helper function to check if an operation should be treated as constant-like.
bool VPU::isConstantLikeOp(mlir::Operation* op) {
    return op->hasTrait<mlir::OpTrait::ConstantLike>() || VPU::isConstOperandOp(op);
}

void VPU::removeUnusedConstantOutputs(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                                      ArrayRef<OutliningInstance> outliningInstances, const Logger& log) {
    auto mainFuncOp = net::getMainFunc(moduleOp);

    // For each outlined function, find which outputs are from constants and check if they're used
    for (const auto& [targetIdx, slices] : outliningInstances | indexed) {
        for (const auto& [sliceIdx, slice] : slices | indexed) {
            auto funcName = funcsInfo[targetIdx][sliceIdx].funcName;
            auto funcOp = moduleOp.lookupSymbol<mlir::func::FuncOp>(funcName);
            if (!funcOp) {
                continue;
            }

            // Find the call op for this function
            mlir::func::CallOp callOp = nullptr;
            mainFuncOp.walk([&](mlir::func::CallOp op) {
                if (op.getCallee() == funcName) {
                    callOp = op;
                    return mlir::WalkResult::interrupt();
                }
                return mlir::WalkResult::advance();
            });

            if (!callOp) {
                continue;
            }

            // Check which outputs are from ConstantLikeOps and are unused
            SmallVector<size_t> unusedOutputIndices;
            for (const auto& [outputIdx, output] : slice.outputs | indexed) {
                auto defOp = output.getDefiningOp();
                if (!defOp || !isConstantLikeOp(defOp)) {
                    continue;
                }

                // Check if the corresponding call result is used
                auto callResult = callOp.getResult(outputIdx);
                if (callResult.use_empty()) {
                    unusedOutputIndices.push_back(outputIdx);
                }
            }

            if (unusedOutputIndices.empty()) {
                continue;
            }

            log.nest().trace("Removing {0} unused constant outputs from function {1}", unusedOutputIndices.size(),
                             funcName);

            // Create new function with reduced outputs
            SmallVector<mlir::Type> newOutputTypes;
            SmallVector<size_t> oldToNewOutputIdx;
            size_t newIdx = 0;
            for (size_t i = 0; i < slice.outputs.size(); ++i) {
                if (llvm::find(unusedOutputIndices, i) == unusedOutputIndices.end()) {
                    newOutputTypes.push_back(slice.outputs[i].getType());
                    oldToNewOutputIdx.push_back(newIdx++);
                } else {
                    oldToNewOutputIdx.push_back(SIZE_MAX);  // Mark as removed
                }
            }

            // Update function signature
            auto newFuncType = mlir::FunctionType::get(funcOp.getContext(), funcOp.getArgumentTypes(), newOutputTypes);
            funcOp.setFunctionType(newFuncType);

            // Update return operation
            funcOp.walk([&](mlir::func::ReturnOp returnOp) {
                SmallVector<mlir::Value> newOperands;
                for (size_t i = 0; i < returnOp.getNumOperands(); ++i) {
                    if (oldToNewOutputIdx[i] != SIZE_MAX) {
                        newOperands.push_back(returnOp.getOperand(i));
                    }
                }
                returnOp->setOperands(newOperands);
            });

            // Update call operation
            mlir::OpBuilder builder(callOp->getContext());
            builder.setInsertionPoint(callOp);
            auto newCall = mlir::func::CallOp::create(builder, callOp.getLoc(), funcName, newOutputTypes,
                                                      callOp.getOperands());

            // Replace old call results with new ones (only for non-removed outputs)
            for (size_t i = 0; i < callOp.getNumResults(); ++i) {
                if (oldToNewOutputIdx[i] != SIZE_MAX) {
                    callOp.getResult(i).replaceAllUsesWith(newCall.getResult(oldToNewOutputIdx[i]));
                }
            }
            callOp.erase();
        }
    }
}
