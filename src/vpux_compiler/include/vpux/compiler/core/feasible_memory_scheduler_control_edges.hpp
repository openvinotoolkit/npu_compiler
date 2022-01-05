//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include "vpux/compiler/core/async_deps_info.hpp"
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
    explicit FeasibleMemorySchedulerControlEdges(mlir::Attribute memSpace, AsyncDepsInfo& depsInfo,
                                                 AliasesInfo& aliasInfo, Logger log,
                                                 LinearScan<mlir::Value, LinearScanHandler>& scan);

    // Based on scheduler output insert dependencies from all tasks in time t
    // to all tasks in time t+1
    // NOTE: This is old method. It is safe in execution but doesn't allow
    // good parallization
    void insertDependenciesBasic(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

    // mateusz
    // Same as insertDependenciesBasic but focuses only on resources not withing main _memSpace
    void insertDependenciesBasicForNonCmxResources(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

    // Insert control flow for overlapping memory regions
    void insertMemoryControlEdges(ArrayRef<FeasibleMemoryScheduler::ScheduledOpInfo> scheduledOps);

private:
    Logger _log;
    // first level mem space
    mlir::Attribute _memSpace;
    // dependencies of ops
    AsyncDepsInfo& _depsInfo;
    // aliases information for buffers
    AliasesInfo& _aliasInfo;
    // allocator class
    LinearScan<mlir::Value, LinearScanHandler>& _scan;
};

}  // namespace vpux
