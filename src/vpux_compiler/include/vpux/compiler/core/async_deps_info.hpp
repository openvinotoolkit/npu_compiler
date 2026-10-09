//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/core/func_ref.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/DenseSet.h>

#include <optional>

namespace vpux {

class AsyncDepsInfo final {
public:
    explicit AsyncDepsInfo(mlir::func::FuncOp func);

public:
    void addDependency(mlir::async::ExecuteOp from, mlir::async::ExecuteOp to);
    void addDependency(size_t fromOpIdx, size_t toOpIdx);
    void buildConsMap();
    void optimizeDepsMap();
    void updateTokenDependencies();
    size_t insertNewExecOpToDepsMap(mlir::async::ExecuteOp execOp);
    void preAllocateForNewOps(size_t numOfNewOps);
    mlir::async::ExecuteOp getExecuteOpAtIndex(size_t opIdx) const;
    const llvm::SmallVector<size_t> getOpDeps(size_t opIdx) const;
    const llvm::SmallVector<size_t> getConsumerOps(size_t opIdx) const;
    std::unordered_map<size_t, size_t> calculateOpInDegreeTable() const;
    std::unordered_map<size_t, size_t> calculateOpOutDegreeTable() const;
    uint32_t getIndex(mlir::async::ExecuteOp execOp) const;
    size_t getExecOpCount() const;
    void verifyAcyclic() const;

private:
    using DepsMap = SmallVector<llvm::DenseSet<size_t>>;
    using DepsVecCache = SmallVector<std::optional<SmallVector<size_t>>>;

    void setIndex(mlir::async::ExecuteOp execOp, uint64_t index);
    SmallVector<size_t> getDepsVec(const llvm::DenseSet<size_t>& deps) const;
    const SmallVector<size_t>& getCachedDepsVec(size_t opIdx, const DepsMap& depsMap, DepsVecCache& depsVecCache) const;
    static void invalidateDepsVecCacheEntry(DepsVecCache& depsVecCache, size_t opIdx);
    bool hasCycle() const;

private:
    void buildDepsMap(mlir::func::FuncOp func);
    void addExecOp(mlir::async::ExecuteOp execOp);

private:
    Logger _log;

    mlir::StringAttr _indexAttrName;

    SmallVector<mlir::async::ExecuteOp> _allExecOps;

    // indexOf(mlir::async::ExecuteOp) 'depends on' [ indexOf(mlir::async::ExecuteOp)... ].
    DepsMap _depsMap;
    DepsMap _consumerMap;

    // getOpDeps() and getConsumerOps() are called repeatedly by the schedulers. Keep the
    // deterministically sorted representation and invalidate only entries whose set changes.
    mutable DepsVecCache _depsVecCache;
    mutable DepsVecCache _consumerVecCache;

    size_t _execOpCount = 0;
};

}  // namespace vpux
