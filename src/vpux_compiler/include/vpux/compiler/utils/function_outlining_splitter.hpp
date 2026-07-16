//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/array_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir {
class Operation;
class Value;
}  // namespace mlir

namespace vpux {

// A subset of the IR intended to be extracted into a function. It contains a list of operations in topological order
struct IRSlice {
    SmallVector<mlir::Value> inputs;
    SmallVector<mlir::Value> outputs;
    std::vector<mlir::Operation*> operations;
    SmallVector<std::pair<mlir::Operation*, size_t>> inputUserMapping;
};

// A vector of IR slices which should be outlined with the same function. This means all of these instances should be
// identical in terms of operations, attributes and types - only the data may be different (activations and constants,
// if allowed). Can have only one element if the block is not repeating
using OutliningInstance = SmallVector<IRSlice>;
void printOutliningInstances(ArrayRef<OutliningInstance> outliningInstances, Logger& log);

//
// IFunctionOutliner
//

class IFunctionOutliner {
public:
    virtual ~IFunctionOutliner() = default;

    virtual SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp /*mainFunction*/) {
        return {};
    }
};

//
// Outliner Helper Structures
//

struct FuncInfo {
    SmallVector<mlir::Type> inputTypes;
    SmallVector<mlir::Type> outputTypes;
    std::string funcName;
};

//
// OutlinerBase
//

class OutlinerBase {
public:
    virtual ~OutlinerBase() = default;
    virtual bool outline(mlir::ModuleOp moduleOp, StringRef functionSuffix);

protected:
    OutlinerBase(std::unique_ptr<IFunctionOutliner> splitter, const Logger& log)
            : _splitter(std::move(splitter)), _log(log) {
    }

    SmallVector<OutliningInstance> getOutliningTargets(mlir::func::FuncOp funcOp) {
        return _splitter->getOutliningTargets(funcOp);
    }

    Logger getLogger() const {
        return _log;
    }

private:
    virtual void buildFuncOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;
    virtual void buildCallOps(mlir::ModuleOp moduleOp, ArrayRef<SmallVector<FuncInfo>> funcsInfo,
                              ArrayRef<OutliningInstance> outlinedTargets) = 0;
    virtual void updateMainFuncOp(mlir::ModuleOp /*moduleOp*/, ArrayRef<OutliningInstance> /*outlinedTargets*/) {
    }

private:
    std::unique_ptr<IFunctionOutliner> _splitter;
    Logger _log;
};

}  // namespace vpux
