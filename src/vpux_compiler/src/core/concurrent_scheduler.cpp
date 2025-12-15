//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/concurrent_scheduler.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler.hpp"
#include "vpux/compiler/utils/loop.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

inline const char* stringifySchedulerType(SchedulerType type) {
    switch (type) {
    case SchedulerType::Default:
        return "DefaultScheduler";
    case SchedulerType::AggressivePrefetch:
        return "AggressivePrefetchScheduler";
    default:
        return "UnknownScheduler";
    }
}

void ConcurrentMemorySchedulerRunner::runTasks(std::vector<SchedulerTask> tasks) {
    _results.clear();
    _results.resize(tasks.size());
    _schedulersCosts.clear();
    _schedulersCosts.reserve(tasks.size());
    auto executeTask = [&](size_t i) {
        try {
            auto scheduleResult = tasks[i].taskFunc(tasks[i].depsInfo, tasks[i].scan, tasks[i].liveRange);
            if (_schedulersCosts.contains(tasks[i].friendlyName)) {
                // Schedulers should be named properly. Repeating name makes statistic log confusing.
                // It's not a severe blocker for scheduling result but better to avoid it.
                _log.warning("Repeating name of {0}", stringifySchedulerType(tasks[i].friendlyName));
            } else {
                _schedulersCosts[tasks[i].friendlyName] = scheduleResult.back().cycleEnd_;
            }
            _results[i].result = SchedulerResult{std::move(scheduleResult), std::move(tasks[i].depsInfo),
                                                 std::move(tasks[i].scan), std::move(tasks[i].friendlyName)};
        } catch (const std::exception& e) {
            _results[i].errorMsg = e.what();
        } catch (...) {
            _results[i].errorMsg = "Scheduling task failed with unknown exception";
        }
    };

    vpux::loop_1d(LoopExecPolicy::Parallel, _ctx, tasks.size(), executeTask);
}

// Simple heuristic: first successful result with minimal cycle count.
const ConcurrentMemorySchedulerRunner::TaskResult& ConcurrentMemorySchedulerRunner::getBestSchedule() const {
    VPUX_THROW_WHEN(_results.empty(), "No results available");

    auto comparator = [](const TaskResult& a, const TaskResult& b) {
        const bool aSuccess = mlir::succeeded(a.result);
        const bool bSuccess = mlir::succeeded(b.result);
        if (aSuccess != bSuccess) {
            return aSuccess > bSuccess;
        }
        if (!aSuccess && !bSuccess) {
            return false;  // Both failed, considered equal
        }
        // Both succeeded, compare by schedule length
        int64_t aLength = a.result.value().scheduledOps.back().cycleEnd_;
        int64_t bLength = b.result.value().scheduledOps.back().cycleEnd_;
        return aLength < bLength;
    };

    return *std::min_element(_results.begin(), _results.end(), comparator);
}

void ConcurrentMemorySchedulerRunner::printSchedulersCosts() const {
    _log.info("[FeasibleAllocation statistics] Schedulers' Costs:");
    for (const auto& [name, cost] : _schedulersCosts) {
        _log.info("\t{0}: {1} cycles", stringifySchedulerType(name), cost);
    }

    if (_schedulersCosts.size() < 2) {
        return;
    }
    VPUX_THROW_UNLESS(_schedulersCosts.contains(SchedulerType::Default),
                      "Default scheduler result not found for cost comparison");
    auto defaultCost = _schedulersCosts.at(SchedulerType::Default);
    auto bestCost = this->getBestSchedule().result.value().scheduledOps.back().cycleEnd_;
    if (bestCost == defaultCost) {
        _log.info("Default scheduler result is the best one.");
        return;
    }
    _log.info("Default cost {0} vs best cost {1}, {2}% improved", defaultCost, bestCost,
              (double(int(defaultCost - bestCost)) / double(defaultCost) * 100));
}
}  // namespace vpux
