//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/utils/partitioner.hpp"

namespace vpux {

// Vector of stalls: pair {cycleStart, cycleEnd}
using StallCycles = SmallVector<std::pair<size_t, size_t>>;
// Map: Key: pair {executorKind, executorInstance}, Value: StallCycles
using ExecutorStallCycles = std::map<std::pair<FeasibleMemoryScheduler::QueueType, size_t>, StallCycles>;
using ScheduledOpInfo = FeasibleMemoryScheduler::ScheduledOpInfo;
using ScheduledOpInfoVec = FeasibleMemoryScheduler::ScheduledOpInfoVec;

struct SpillStats {
    size_t numOfSpillWrites;
    size_t numOfSpillWritesDueToFrag;
    size_t numOfSpillWritesOfDataOps;
    size_t numOfSpillRead;
};

ExecutorStallCycles getExecutorStallRegions(ScheduledOpInfoVec& scheduledOps);
StallCycles getStallsOnAllExecutorPipelines(ScheduledOpInfoVec& scheduledOps);
void verifyDependenciesPreservedInCycles(AsyncDepsInfo& depsInfo, ScheduledOpInfoVec& scheduledOps);
StringRef getTaskType(const ScheduledOpInfo& op);
void printScheduleStatistics(mlir::func::FuncOp& netFunc, AsyncDepsInfo& depsInfo, Logger log,
                             llvm::ArrayRef<ScheduledOpInfo> scheduledOps);
SpillStats getDynamicSpillingStats(llvm::ArrayRef<ScheduledOpInfo> scheduledOps);
void printSpillingStatistics(Logger log, SpillStats& beforePrefetching, SpillStats& afterPrefetching,
                             SpillStats& afterOptimizations);
void createTracingJSON(mlir::func::FuncOp& netFunc, MemLiveRangeInfo& liveRangeInfo,
                       LinearScan<mlir::Value, LinearScanHandler>& scan, vpux::AddressType totalMemory,
                       StringRef fileName = "mallocTraceRaw.json");

}  // namespace vpux
