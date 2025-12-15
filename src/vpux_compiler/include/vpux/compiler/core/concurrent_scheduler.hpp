//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "vpux/compiler/core/feasible_memory_scheduler.hpp"

namespace vpux {

enum class SchedulerType { Default, AggressivePrefetch };

inline const char* stringifySchedulerType(SchedulerType type);

/**
 * @brief Helper class for concurrently running multiple scheduling tasks and retrieving the best result
 *
 * This class is designed to execute multiple scheduling tasks in parallel, collect their results,
 * and determine the best schedule based on the results.
 *
 */
class ConcurrentMemorySchedulerRunner {
public:
    using ScheduledOpsType = FeasibleMemoryScheduler::ScheduledOpInfoVec;
    using TaskFunc = std::function<ScheduledOpsType(AsyncDepsInfo&, LinearScan<mlir::Value, LinearScanHandler>&,
                                                    MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>&)>;

    explicit ConcurrentMemorySchedulerRunner(mlir::MLIRContext* ctx, Logger log): _ctx(ctx), _log(log) {
    }

    // The result of a successful schedule task
    // depsInfo and scan are also returned to preserve any state changes
    struct SchedulerResult {
        ScheduledOpsType scheduledOps;
        AsyncDepsInfo depsInfo;
        LinearScan<mlir::Value, LinearScanHandler> scan;
        SchedulerType friendlyName = SchedulerType::Default;
    };

    // The result of a scheduling task, including success or failure
    struct TaskResult {
        mlir::FailureOr<SchedulerResult> result;
        std::string errorMsg;
    };

    // A scheduling task encapsulating the function to execute, and the current schedule status
    // including the ops dependencies, and the linear scan state
    struct SchedulerTask {
        TaskFunc taskFunc;
        AsyncDepsInfo depsInfo;
        LinearScan<mlir::Value, LinearScanHandler> scan;
        MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN> liveRange;
        SchedulerType friendlyName;

        SchedulerTask(TaskFunc func, const AsyncDepsInfo& deps, const LinearScan<mlir::Value, LinearScanHandler>& sc,
                      const MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>& lr,
                      SchedulerType name = SchedulerType::Default)
                : taskFunc(std::move(func)), depsInfo(deps), scan(sc), liveRange(lr), friendlyName(name) {
        }
    };

    // Execute tasks concurrently
    // The number of threads is determined by the thread pool in context
    // store the costs of schedulers for statistic information
    void runTasks(std::vector<SchedulerTask> tasks);
    // Retrieve the best schedule result among all executed tasks
    // The best schedule is determined by the first successful result with the minimal cycle count
    const TaskResult& getBestSchedule() const;
    // For statistic information, print the cost of schedulers
    void printSchedulersCosts() const;

private:
    std::vector<TaskResult> _results;
    DenseMap<SchedulerType, int64_t> _schedulersCosts;
    mlir::MLIRContext* _ctx;
    Logger _log;
};

}  // namespace vpux
