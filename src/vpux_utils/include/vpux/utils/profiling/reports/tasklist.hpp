//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/profiling/taskinfo.hpp"

#include <vector>

namespace vpux::profiling {

/**
 * @brief Profiling tasks selection and management utility
 */
class TaskList : public std::vector<TaskInfo> {
public:
    TaskList();
    TaskList(const std::vector<TaskInfo>& tasks);

    TaskList& append(const TaskList&);
    TaskList selectClusterLevelTasks() const;
    TaskList selectTasksFromCluster(unsigned clusterId) const;
    TaskList selectSubtasks() const;
    TaskList selectDMAtasks() const;
    TaskList selectDPUtasks() const;
    TaskList selectSWtasks() const;
    TaskList selectM2Itasks() const;
    TaskList getSortedByStartTime() const;
    /**
     * @brief Infer the Cluster Count from tasks names
     *
     * @return number of different clusters the tasks are assigned to.
     *
     * The returned value may be 0 if tasks do not contain CLUSTER_LEVEL_PROFILING_SUFFIX
     * in their name.
     */
    unsigned getClusterCount() const;

    /**
     * @brief Calculate sum of all tasks durations
     *
     * @return sum of all tasks durations
     */
    uint64_t getSumOfDurations() const;

    /**
     * @brief Walltime duration of all tasks in units defined by TaskInfo
     * @return time elapsed from the first chronologically task start time
     * to the last chronologically task end time
     *
     * Note: Tasks do not need to be ordered chronologically.
     */
    uint64_t getTotalDuration() const;

private:
    /**
     * @brief Get first chronologically task start time
     *
     * @return minimal start time among all tasks
     */
    uint64_t getStartTime() const;

    /**
     * @brief Get last chronologically task end time
     *
     * @return maximal end time among all tasks
     */
    uint64_t getEndTime() const;

    void sortByStartTime();

    template <TaskInfo::ExecType T>
    TaskList selectTasksOfType() const;
};

bool isClusterLevelProfilingTask(const TaskInfo& task);
bool isVariantLevelProfilingTask(const TaskInfo& task);

}  // namespace vpux::profiling
