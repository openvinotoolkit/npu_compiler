//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace vpux::profiling {

class TaskList;

struct TaskStatistics {
    uint64_t totalDuration;  // wall time duration of inference [ns]
    uint64_t idleDuration;   // sum all no-operation intervals [ns]
    uint64_t allTasksUnion;  // self-consistency: should match totalDuration - idleDuration [ns]

    uint64_t dmaDuration;  // sum of interval durations of a union of DMA tasks [ns].
                           // Union of tasks (coalesced tasks @see TaskTrack::coalesce() and @see
                           // TaskTrack::calculateOverlap(const TaskTrack& refTrack)) disregard possible overlaps of
                           // individual contributing tasks
    uint64_t dpuDuration;  // sum of interval durations of a union of DPU tasks [ns]
    uint64_t swDuration;   // sum of interval durations of a union of SW tasks [ns]

    uint64_t dmaDpuOverlapDuration;  // sum of interval durations of intersection of a union of DMA and a union of DPU
                                     // tasks [ns]
    uint64_t dmaSwOverlapDuration;   // sum of interval durations of intersection of a union of DMA and a union of SW
                                     // tasks [ns]
    uint64_t swDpuOverlapDuration;   // sum of interval durations of intersection of a union of SW and a union of DPU
                                     // tasks [ns]

    uint64_t dmaDpuIdleDuration;  // sum of idle (no operation) interval durations outside of a union of DMA and DPU
                                  // tasks [ns]
    uint64_t dmaSwIdleDuration;   // sum of idle interval durations outside of a union of DMA and SW tasks [ns]
    uint64_t swDpuIdleDuration;   // sum of idle interval durations outside of a union of SW and DPU tasks [ns]

    uint64_t sumOfDmaTaskDurations;  // sum of all DMA tasks durations [ns].
    uint64_t sumOfDpuTaskDurations;  // sum of all DPU tasks durations [ns]
    uint64_t sumOfSwTaskDurations;   // sum of all SW tasks durations [ns]

    /**
     * @brief calculate tasks timing statistics
     *
     * @param tasks - vector of tasks used for calculation
     * @return task timing statistics
     */
    TaskStatistics(const TaskList& tasks);

    /**
     * @brief calculate joint duration of SW tasks union that does not intersect with union of DPU tasks
     *
     * @return joint intervals duration in ns
     */
    int64_t getSwDurationWithoutDpuOverlap() const {
        return swDuration - swDpuOverlapDuration;
    }

    /**
     * @brief calculate joint duration of DMA tasks union that does not intersect with union of all other tasks
     *
     * @return joint intervals duration in ns
     */
    int64_t getDmaDurationWithoutOverlap() const {
        return dmaDuration - dmaSwOverlapDuration - dmaDpuOverlapDuration + swDpuOverlapDuration;
    }

    void printAsJson(std::ostream& out);

    void log(vpux::Logger& log);
};

namespace details {
/**
 * @brief TaskTrack is an NPU tasks container that stores tasks start and end times as single time events called
 * TrackEvent It encapsulates coalescing overlapping events
 *
 */
class TaskTrack {
    /**
     * @brief TrackEvent is used to store information about count of parallel tasks at any given time during execution
     * flow.
     */
    struct TrackEvent {
        uint64_t time;      // track event time
        int taskCount = 0;  // number of tasks at given time
        bool isStart;       // indicates whether the event marks the start time of a task
        uint64_t duration;  // duration of task the event is associated with
    };

public:
    TaskTrack& insert(const TaskList& tasks);

private:
    TaskTrack& insert(uint64_t trackTime, uint64_t evtDuration, bool isEvtStart);
    TaskTrack& append(const TaskTrack& taskTrack);

public:
    /**
     * @brief calculate union of all events in the track.
     * The coalesced version of the sequence of events has overlapping events merged together and output
     * as a single longer event. Neighboring events are not merged. Zero-duration events are reduced out.
     */
    TaskTrack& coalesce();

    /**
     * @return uint64_t - sum of durations of all events in the track
     */
    uint64_t getSumOfDurations() const;

    /**
     * @brief calculate tasks mutual overlap and idle times
     *
     * @param refTrack - track to calculate overlap with
     * @return pair of overlapTime and idleTime
     *
     * The overlapTime and idleTime are defined as follows:
     *
     *      test track (20 time units long): xxxx......xxxxx.....
     * reference track (20 time units long): ...yyy.......yyyyyyy
     *                             analysis:    o  iiii   oo
     *
     * o - overlap time
     * i - idle time
     *
     * totalDuration = 20
     * workload = N(x) + N(y) = 9 + 10 = 19
     * idleTime = N(i) = 4
     * overlapTime = N(o) = 3
     *
     * Note:
     *
     * The impact of zero-duration tasks is that they affect the "total duration"
     * time (i.e. inference time) but do not contribute to workload,
     * hence if duration-zero task is located at the beginning or at the end
     * of the provided task list, this may impact the statistics
     * by increasing the total execution time (totalDuration). This situation
     * is not shown in the example above.
     *
     * Default unit is ns.
     *
     */
    std::pair<uint64_t, uint64_t> calculateOverlap(const TaskTrack& refTrack) const;

private:
    TaskTrack& sortByTime();

    /**
     * @brief calculate number of stacked tasks as a function of event time
     *
     * @return std::map<int, int> containing time and task count
     */
    std::map<int, int> getTrackProfile();

    std::vector<TrackEvent> _trackEvents;
};

}  // namespace details

}  // namespace vpux::profiling
