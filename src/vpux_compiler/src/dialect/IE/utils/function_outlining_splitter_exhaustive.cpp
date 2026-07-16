//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/utils/function_outlining_splitter.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"

using namespace vpux;

namespace {
bool isOutlineableOperation(mlir::Operation* op) {
    return !mlir::isa<Const::DeclareOp, mlir::func::ReturnOp, mlir::func::CallOp, mlir::func::FuncOp>(op);
}

bool isConstTransformation(mlir::Operation* op) {
    return mlir::isa<IE::SubtractOp, IE::MultiplyOp, IE::ConvertOp, IE::FakeQuantizeOp, IE::ReshapeOp,
                     IE::AffineReshapeOp>(op);
}

mlir::FailureOr<SmallVector<mlir::Operation*>> getConstantParents(mlir::Operation* op) {
    if (op == nullptr) {
        return mlir::failure();
    }
    if (mlir::isa<Const::DeclareOp>(op)) {
        return SmallVector<mlir::Operation*>{op};
    }

    // Quantized weights could be represented as subgraphs, such as:
    //              Cst         Cst
    //               |           |
    // Weights -> Subtract -> Multiply -> [user]
    // These subgraphs are included into the outlined function, so that the low-precision pipeline can correctly
    // quantize the user operation
    if (isConstTransformation(op)) {
        SmallVector<mlir::Operation*> parentConstOps;
        for (auto operand : op->getOperands()) {
            const auto ops = getConstantParents(operand.getDefiningOp());
            if (mlir::failed(ops)) {
                return mlir::failure();
            }
            parentConstOps.append(ops->begin(), ops->end());
        }
        parentConstOps.push_back(op);
        return parentConstOps;
    }
    return mlir::failure();
}

mlir::LogicalResult duplicateNeededParentOps(IRSlice& slice, mlir::DenseSet<mlir::Operation*>& opsInSlice,
                                             mlir::Operation* parentOp) {
    // In case the parent operation is a constant (or constant subgraph), it should be placed in the current instance
    // regardless of where it was placed initially
    const auto constParents = getConstantParents(parentOp);
    if (mlir::failed(constParents)) {
        return mlir::failure();
    }

    for (auto constParent : *constParents) {
        if (opsInSlice.insert(constParent).second) {
            slice.operations.push_back(constParent);
        }
    }
    return mlir::success();
}

IRSlice createSlice(ArrayRef<mlir::Operation*> operations) {
    IRSlice slice;
    mlir::DenseSet<mlir::Operation*> opsInSlice;

    for (auto op : operations) {
        for (auto operand : op->getOperands()) {
            auto parentOp = operand.getDefiningOp();
            if (parentOp != nullptr) {
                if (opsInSlice.contains(parentOp) ||
                    mlir::succeeded(duplicateNeededParentOps(slice, opsInSlice, parentOp))) {
                    continue;
                }
            }
            if (llvm::find(slice.inputs, operand) == slice.inputs.end()) {
                slice.inputs.push_back(operand);
            }
        }

        if (!opsInSlice.contains(op)) {
            slice.operations.push_back(op);
            opsInSlice.insert(op);
        }
    }

    mlir::DenseSet<mlir::Operation*> sliceOps(slice.operations.begin(), slice.operations.end());
    for (auto op : operations) {
        for (auto result : op->getResults()) {
            if (llvm::find(slice.outputs, result) != slice.outputs.end()) {
                continue;
            }
            const auto userOutsideSlice = llvm::any_of(result.getUsers(), [&](mlir::Operation* userOp) {
                return !sliceOps.contains(userOp);
            });
            if (userOutsideSlice) {
                slice.outputs.push_back(result);
            }
        }
    }

    return slice;
}

bool isOutputNeeded(mlir::Value output, const mlir::DenseSet<mlir::Value>& sliceInputs) {
    const auto escapesOutlining = llvm::any_of(output.getUsers(), [](mlir::Operation* userOp) {
        return !isOutlineableOperation(userOp);
    });
    return escapesOutlining || sliceInputs.contains(output);
}

mlir::DenseSet<mlir::Operation*> collectLiveOps(IRSlice& slice) {
    const mlir::DenseSet<mlir::Operation*> sliceOps(slice.operations.begin(), slice.operations.end());

    mlir::DenseSet<mlir::Operation*> liveOps;
    SmallVector<mlir::Operation*> worklist;
    const auto markLive = [&](mlir::Operation* op) {
        if (op != nullptr && sliceOps.contains(op) && liveOps.insert(op).second) {
            worklist.push_back(op);
        }
    };

    for (auto output : slice.outputs) {
        markLive(output.getDefiningOp());
    }
    while (!worklist.empty()) {
        auto* op = worklist.pop_back_val();
        for (auto operand : op->getOperands()) {
            markLive(operand.getDefiningOp());
        }
    }
    return liveOps;
}

std::vector<mlir::Operation*> filterOps(std::vector<mlir::Operation*> ops,
                                        const mlir::DenseSet<mlir::Operation*>& filter) {
    std::vector<mlir::Operation*> filteredOps;
    for (auto op : ops) {
        if (filter.contains(op)) {
            filteredOps.push_back(op);
        }
    }
    return filteredOps;
}

// The splitter may duplicate a constant subgraph into a slice (see duplicateNeededParentOps).
// Then values produced by an earlier slice can become unused, because the later slice
// recomputes them locally. This function removes such redundant outputs.
void pruneConstantSubgraphs(SmallVector<IRSlice>& slices) {
    mlir::DenseSet<mlir::Value> sliceInputs;
    for (const auto& slice : slices) {
        sliceInputs.insert(slice.inputs.begin(), slice.inputs.end());
    }

    for (auto& slice : slices) {
        SmallVector<mlir::Value> neededOutputs;
        for (auto output : slice.outputs) {
            if (isOutputNeeded(output, sliceInputs)) {
                neededOutputs.push_back(output);
            }
        }
        if (neededOutputs.size() != slice.outputs.size()) {
            slice.outputs = std::move(neededOutputs);
            if (!slice.outputs.empty()) {
                const auto liveOps = collectLiveOps(slice);
                slice.operations = filterOps(std::move(slice.operations), liveOps);
            }
        }
    }

    llvm::erase_if(slices, [](const IRSlice& slice) {
        return slice.outputs.empty();
    });
}

}  // namespace

namespace vpux {
namespace IE {

FunctionOutlinerExhaustive::FunctionOutlinerExhaustive(Logger log): _log(log) {
    _log.setName("function-outliner-exhaustive");
}

SmallVector<OutliningInstance> FunctionOutlinerExhaustive::getOutliningTargets(mlir::func::FuncOp mainFunction) {
    SmallVector<SmallVector<mlir::Operation*>> operationGroups(1);
    for (auto& op : mainFunction.getBody().front().without_terminator()) {
        if (isOutlineableOperation(&op)) {
            operationGroups.back().push_back(&op);
        } else if (!mlir::isa<Const::DeclareOp>(op) && !operationGroups.back().empty()) {
            operationGroups.emplace_back();
        }
    }

    SmallVector<IRSlice> slices;
    for (auto& operations : operationGroups) {
        auto slice = createSlice(operations);
        if (!slice.operations.empty()) {
            slices.push_back(std::move(slice));
        }
    }

    pruneConstantSubgraphs(slices);

    SmallVector<OutliningInstance> outliningInstances;
    for (auto& slice : slices) {
        OutliningInstance outliningInstance;
        outliningInstance.push_back(std::move(slice));
        outliningInstances.push_back(std::move(outliningInstance));
    }
    return outliningInstances;
}

}  // namespace IE
}  // namespace vpux
