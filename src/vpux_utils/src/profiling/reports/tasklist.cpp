//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/tasklist.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/location.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include <algorithm>
#include <numeric>
#include <set>
#include <string>

using vpux::checked_cast;
namespace vpux::profiling {

template <TaskInfo::ExecType Value>
bool isTaskType(const TaskInfo& task) {
    return task.exec_type == Value;
}

template <typename InT>
enable_t<InT, std::is_integral<InT>, not_<std::is_signed<InT>>> checked_add(InT a, InT b) {
    VPUX_THROW_UNLESS(b <= std::numeric_limits<InT>::max() - a,
                      "Can not safely add two values {0} and {1} of type {2} ", a, b, llvm::getTypeName<InT>());
    return a + b;
}

TaskList::TaskList() {
}

TaskList::TaskList(const std::vector<TaskInfo>& tasks): std::vector<TaskInfo>(tasks) {
}

template <TaskInfo::ExecType T>
TaskList TaskList::selectTasksOfType() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isTaskType<T>);
    return selectedTasks;
}

TaskList TaskList::selectDPUtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DPU>();
}

TaskList TaskList::selectDMAtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::DMA>();
}

TaskList TaskList::selectSWtasks() const {
    return selectTasksOfType<TaskInfo::ExecType::SW>();
}

TaskList TaskList::selectM2Itasks() const {
    return selectTasksOfType<TaskInfo::ExecType::M2I>();
}

TaskList TaskList::getSortedByStartTime() const {
    TaskList sorted(*this);
    sorted.sortByStartTime();
    return sorted;
}

TaskList TaskList::selectClusterLevelTasks() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), isClusterLevelProfilingTask);
    return selectedTasks;
}

TaskList TaskList::selectTasksFromCluster(unsigned clusterId) const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), [=](const TaskInfo& task) {
        return task.clusterId == clusterId;
    });
    return selectedTasks;
}

TaskList TaskList::selectSubtasks() const {
    TaskList selectedTasks;
    std::copy_if(begin(), end(), std::back_inserter(selectedTasks), [](const TaskInfo& task) {
        return task.isSubtask;
    });
    return selectedTasks;
}

void TaskList::sortByStartTime() {
    std::sort(begin(), end(), profilingTaskStartTimeComparator<TaskInfo>);
}

unsigned TaskList::getClusterCount() const {
    return 1 + std::accumulate(begin(), end(), 0u, [&](unsigned n, auto task) {
               return std::max(n, task.clusterId);
           });
}

uint64_t TaskList::getSumOfDurations() const {
    return std::accumulate(begin(), end(), static_cast<uint64_t>(0),
                           [](const uint64_t& totalTime, const TaskInfo& task) {
                               return checked_add(totalTime, task.duration_ns);
                           });
}

uint64_t TaskList::getStartTime() const {
    VPUX_THROW_WHEN(empty(), "Minimal time in empty TaskList is not defined.");

    auto minElementIt = min_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return a.start_time_ns < b.start_time_ns;
    });
    return minElementIt->start_time_ns;
}

uint64_t TaskList::getEndTime() const {
    VPUX_THROW_WHEN(empty(), "Maximal time in empty TaskList is not defined.");

    auto maxElementIt = max_element(begin(), end(), [](const TaskInfo& a, const TaskInfo& b) {
        return checked_add(a.start_time_ns, a.duration_ns) < checked_add(b.start_time_ns, b.duration_ns);
    });
    return maxElementIt->start_time_ns + maxElementIt->duration_ns;
}

uint64_t TaskList::getTotalDuration() const {
    if (empty()) {
        return 0;
    }
    return getEndTime() - getStartTime();
}

TaskList& TaskList::append(const TaskList& tasks) {
    insert(end(), tasks.begin(), tasks.end());
    return *this;
}

bool isVariantLevelProfilingTask(const TaskInfo& task) {
    return task.exec_type == TaskInfo::ExecType::DPU && task.isSubtask;
}

bool isClusterLevelProfilingTask(const TaskInfo& task) {
    switch (task.exec_type) {
    case TaskInfo::ExecType::SW:
        return true;
    case TaskInfo::ExecType::DPU:
        return !task.isSubtask;  // Invariants only
    default:
        return false;
    }
}

}  // namespace vpux::profiling
