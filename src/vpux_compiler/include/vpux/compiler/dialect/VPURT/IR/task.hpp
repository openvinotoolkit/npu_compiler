//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/IR/Builders.h>

namespace vpux {
namespace VPURT {

template <typename OpTy, typename... Args>
OpTy wrapIntoTaskOp(mlir::OpBuilder& builder, mlir::ValueRange waitBarriers, mlir::ValueRange updateBarriers,
                    mlir::Location loc, Args&&... args) {
    auto taskOp = builder.create<vpux::VPURT::TaskOp>(loc, waitBarriers, updateBarriers);
    auto& block = taskOp.getBody().emplaceBlock();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&block);

    return builder.create<OpTy>(loc, std::forward<Args>(args)...);
}

template <typename OpTy, typename... Args>
OpTy createOp(mlir::PatternRewriter& rewriter, mlir::Operation* insertionPoint, Args&&... args) {
    VPUX_THROW_WHEN(insertionPoint == nullptr, "Insertion point is empty");
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(insertionPoint);
    return rewriter.create<OpTy>(std::forward<Args>(args)...);
}

template <typename OpTy, typename... Args>
OpTy createOp(mlir::OpBuilder& builder, mlir::Operation* insertionPoint, Args&&... args) {
    VPUX_THROW_WHEN(insertionPoint == nullptr, "Insertion point is empty");
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(insertionPoint);
    return builder.create<OpTy>(std::forward<Args>(args)...);
}

struct TaskQueueType {
    VPU::ExecutorKind type;
    int64_t id;
    bool operator<(const TaskQueueType& other) const {
        if (type == other.type) {
            return id < other.id;
        }
        return type < other.type;
    }
    bool operator==(const TaskQueueType& other) const {
        return type == other.type && id == other.id;
    }
    bool operator!=(const TaskQueueType& other) const {
        return !(*this == other);
    }
};

SmallVector<int64_t> getDMATaskPorts(TaskOp task);

std::optional<SmallVector<TaskQueueType>> getDMATaskQueueType(TaskOp task);

TaskQueueType getTaskQueueType(TaskOp task, bool ignoreIndexForNce = true);

std::map<TaskQueueType, std::pair<TaskOp, TaskOp>> getTaskQueuesFirstAndLastOp(mlir::func::FuncOp funcOp);
}  // namespace VPURT

// Tasks execution groups are used to validate schedule for correct task descriptor fetch operations
using ExecutionGroup = llvm::SmallVector<size_t>;
using ExecutionGroupList = llvm::SmallVector<ExecutionGroup>;
using ExecutionGroupListMap = llvm::DenseMap<VPURT::TaskQueueType, ExecutionGroupList>;

}  // namespace vpux

using namespace vpux;

namespace llvm {
template <>
struct DenseMapInfo<VPURT::TaskQueueType> {
    static VPURT::TaskQueueType getEmptyKey() {
        return VPURT::TaskQueueType{DenseMapInfo<VPU::ExecutorKind>::getEmptyKey(), 0};
    }

    static VPURT::TaskQueueType getTombstoneKey() {
        return VPURT::TaskQueueType{DenseMapInfo<VPU::ExecutorKind>::getTombstoneKey(), -1};
    }

    static unsigned getHashValue(VPURT::TaskQueueType val) {
        auto h1 = hash_value(val.type);
        auto h2 = hash_value(val.id);

        return static_cast<unsigned>(hash_combine(h1, h2));
    }

    static bool isEqual(VPURT::TaskQueueType lhs, VPURT::TaskQueueType rhs) {
        return (lhs.type == rhs.type) && (lhs.id == rhs.id);
    }
};
}  // namespace llvm
