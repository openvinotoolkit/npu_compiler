//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/dense_map.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

constexpr char SPILL_READ_OP_NAME_SUFFIX[] = "spill_read_";
constexpr char SPILL_WRITE_OP_NAME_SUFFIX[] = "spill_write_";

//
// FeasibleMemorySchedulerSpilling class is a support class for FeasibleMemoryScheduler
// to handle spilling and insert correct spill-write and spill-read DmaOps within
// async dialect and perform required connections between new ops to achieve correct
// execution order for models that need spilling
//
class FeasibleMemorySchedulerSpilling final {
public:
    explicit FeasibleMemorySchedulerSpilling(VPU::MemoryKind memKind, VPU::MemoryKind secondLvlMemKind,
                                             AsyncDepsInfo& depsInfo, AliasesInfo& aliasInfo, Logger log,
                                             LinearScan<mlir::Value, LinearScanHandler>& scan);

    void removeComputeOpRelocationSpills(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void optimizeDataOpsSpills(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void removeRedundantSpillWrites(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);
    void insertSpillDmaOps(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);

private:
    void createSpillWrite(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps, size_t schedOpIndex);
    void createSpillRead(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps, size_t schedOpIndex);
    mlir::async::ExecuteOp insertSpillWriteDmaOp(mlir::async::ExecuteOp opThatWasSpilled,
                                                 mlir::async::ExecuteOp insertAfterExecOp, mlir::Value bufferToSpill,
                                                 size_t allocatedAddress, int spillId);
    mlir::async::ExecuteOp insertSpillReadDmaOp(mlir::async::ExecuteOp opThatWasSpilled, mlir::Value bufferToSpill,
                                                mlir::async::ExecuteOp spillWriteExecOp,
                                                mlir::async::ExecuteOp insertAfterExecOp, size_t allocatedAddress,
                                                int spillId);
    void updateSpillWriteReadUsers(mlir::Value bufferToSpill, mlir::async::ExecuteOp spillWriteExecOp,
                                   mlir::async::ExecuteOp spillReadExecOp);
    SmallVector<mlir::Value> getAsyncResultsForBuffer(mlir::async::ExecuteOp opThatWasSpilled, mlir::Value buffer);
    mlir::Value getBufferFromAsyncResult(mlir::Value asyncResult);
    llvm::DenseSet<size_t> identifyDataOpsWithInputOverwrite(FeasibleMemoryScheduler::ScheduledOpInfoVec& scheduledOps);

    // Below nested class is intended to handle data dependency updates
    // for users of spilled buffers
    class SpillUsersUpdate {
    public:
        explicit SpillUsersUpdate(FeasibleMemorySchedulerSpilling& spillingClass,
                                  mlir::async::ExecuteOp opThatWasSpilled, mlir::async::ExecuteOp spillReadExecOp,
                                  mlir::Value bufferToSpill)
                : _spillingParentClass(spillingClass),
                  _opThatWasSpilled(opThatWasSpilled),
                  _spillReadExecOp(spillReadExecOp),
                  _bufferToSpill(bufferToSpill) {
        }
        void resolveSpillBufferUsage();

    private:
        SmallVector<mlir::Operation*> getViewOpsForMasterBuffer(mlir::Value asyncResult);
        SmallVector<mlir::async::ExecuteOp> getUsersOfSpilledOpThatNeedUpdate(mlir::Value opThatWasSpilledResult);
        unsigned int getOperandIndexForSpillResultUser(mlir::async::ExecuteOp spillResultUser,
                                                       mlir::Value spilledAsyncResult);
        void updateSpillResultUsers(mlir::Value oldResult, mlir::Value newResult);
        void updateSpillBufferUsers(mlir::Value oldBuffer, mlir::Value newBuffer);

        FeasibleMemorySchedulerSpilling& _spillingParentClass;
        mlir::async::ExecuteOp _opThatWasSpilled;
        mlir::async::ExecuteOp _spillReadExecOp;
        mlir::Value _bufferToSpill;
    };

    struct SpillWriteInfo {
        mlir::Value spilledBuffer;
        mlir::async::ExecuteOp execOp;
        int spillId;
    };

private:
    Logger _log;
    // op insertion point for allocation related operations
    mlir::Operation* _allocOpInsertionPoint;
    // first level mem space
    VPU::MemoryKind _memKind;
    // second level mem space which is used for spilling
    VPU::MemoryKind _secondLvlMemKind;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
    // Vector storing information about already inserted spill-writes. Used when spill-read is inerted
    // to properly connect both operations
    SmallVector<SpillWriteInfo> _spillWriteInfoVec;
    // Map storing new buffers replacing spilled buffers: key - original spilled buffer, value - new allocated buffer
    // after spill-read
    DenseMap<mlir::Value, mlir::Value> _bufferReplacementAfterSpillRead;
    // Index identifying spill-write and corresponding spill-reads (can be more than 1)
    int _spillId{-1};
};

}  // namespace vpux
