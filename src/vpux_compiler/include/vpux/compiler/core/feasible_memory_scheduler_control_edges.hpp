//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace vpux {

//
// FeasibleMemorySchedulerControlEdges class is a support class for FeasibleMemoryScheduler
// to handle insertion of required dependencies resulting from decisions made by scheduler
//
class FeasibleMemorySchedulerControlEdges final {
public:
    explicit FeasibleMemorySchedulerControlEdges(VPU::MemoryKind memKind, AsyncDepsInfo& depsInfo,
                                                 AliasesInfo& aliasInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan);

    // Based on scheduler output insert dependencies from all tasks in time t
    // to all tasks in time t+1
    // NOTE: This is old method. It is safe in execution but doesn't allow
    // good parallelization
    void insertDependenciesBasic(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

    // Insert control flow for overlapping memory regions
    void insertMemoryControlEdges(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

    // Insert dependencies for a given executor type so that resulting execution
    // is aligned with order generated by list-scheduler
    void insertScheduleOrderDepsForExecutor(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps,
                                            VPU::ExecutorKind executorKind, uint32_t executorInstance = 0);

    // After all new dependencies have been prepared call this function to make actual changes in IR
    void updateDependenciesInIR();

private:
    Logger _log;
    // first level mem space
    VPU::MemoryKind _memKind;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
};

// Apply dependencies from controlEdges set into depsInfo.
void updateControlEdgesInDepsInfo(AsyncDepsInfo& depsInfo, ControlEdgeSet& controlEdges, Logger& log);

}  // namespace vpux
