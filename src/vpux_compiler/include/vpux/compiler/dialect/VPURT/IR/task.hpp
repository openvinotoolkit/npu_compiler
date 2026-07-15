//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

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

/// The hardware has different executors that can run in parallel. In the abstract compute model that we use for
/// scheduling and memory allocation we assume that there is only a single executor of each type. The only
/// exception to that rule is DMA_NN for now. We assume a DMA_NN has multiple channels. To differentiate between the
/// different DMA_NN channels we use the id field in addition to the executor kind. id is unused for all other executor
/// kinds.
/// We use the queue/FIFO terminology here because these executors are controlled via FIFOs in the hardware.
struct TaskQueueType {
    vpux::config::ExecutorKind type;
    // Unused for all executor kinds except for DMA_NN.
    int64_t id;

    TaskQueueType() = default;
    TaskQueueType(vpux::config::ExecutorKind type, int64_t id = 0): type(type), id(id) {
    }
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

    // Objects of this type are hashable for std::hash-utilizing containers.
};

TaskQueueType getTaskQueueType(TaskOp task, bool ignoreIndexForNce = true);

std::map<TaskQueueType, std::pair<TaskOp, TaskOp>> getTaskQueuesFirstAndLastOp(mlir::func::FuncOp funcOp);

// Get tile and list index for given queue type as expected by backend representation
std::pair<size_t, size_t> getTileAndListIndex(VPURT::TaskQueueType queueType, int64_t numTiles,
                                              vpux::config::ArchKind arch);

TaskQueueType getQueueTypeFromTileAndListIndex(vpux::config::ExecutorKind executorKind, size_t tileIndex,
                                               size_t listIndex, int64_t numTiles);

size_t getTileIndexForDpuOrShv(VPURT::TaskOp taskOp, VPURT::TaskQueueType queueType);
size_t getListIndexForDpuOrShv(VPURT::TaskOp taskOp);

// Get number of workloads under single processed task
size_t getNumberOfWorkloads(TaskOp taskOp);

}  // namespace VPURT

// Tasks execution groups are used to validate schedule for correct task descriptor fetch operations
using ExecutionGroup = llvm::SmallVector<size_t>;
using ExecutionGroupList = llvm::SmallVector<ExecutionGroup>;
using ExecutionGroupListMap = llvm::DenseMap<VPURT::TaskQueueType, ExecutionGroupList>;

}  // namespace vpux

namespace llvm {
template <>
struct DenseMapInfo<vpux::VPURT::TaskQueueType> {
    static vpux::VPURT::TaskQueueType getEmptyKey() {
        return vpux::VPURT::TaskQueueType{DenseMapInfo<vpux::config::ExecutorKind>::getEmptyKey(), 0};
    }

    static vpux::VPURT::TaskQueueType getTombstoneKey() {
        return vpux::VPURT::TaskQueueType{DenseMapInfo<vpux::config::ExecutorKind>::getTombstoneKey(), -1};
    }

    static unsigned getHashValue(vpux::VPURT::TaskQueueType val) {
        auto h1 = hash_value(val.type);
        auto h2 = hash_value(val.id);

        return static_cast<unsigned>(hash_combine(h1, h2));
    }

    static bool isEqual(vpux::VPURT::TaskQueueType lhs, vpux::VPURT::TaskQueueType rhs) {
        return lhs == rhs;
    }
};

template <>
struct format_provider<vpux::VPURT::TaskQueueType> final {
    static void format(const vpux::VPURT::TaskQueueType& taskQueueType, raw_ostream& os, StringRef) {
        os << formatv("{0}.{1}", vpux::config::stringifyExecutorKind(taskQueueType.type),
                      static_cast<uint32_t>(taskQueueType.id));
    }
};

}  // namespace llvm

namespace std {
template <>
struct hash<vpux::VPURT::TaskQueueType> final {
    std::size_t operator()(const vpux::VPURT::TaskQueueType& qt) const noexcept {
        return llvm::DenseMapInfo<vpux::VPURT::TaskQueueType>::getHashValue(qt);
    }
};
}  // namespace std
