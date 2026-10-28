//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/reports/stats.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"

#include <map>
#include <numeric>
#include <ostream>
#include <vector>

namespace vpux::profiling {

TaskStatistics::TaskStatistics(const TaskList& tasks) {
    using namespace details;

    auto dmaStatsTasks = tasks.selectDMAtasks();
    auto dpuStatsTasks = tasks.selectDPUtasks().selectClusterLevelTasks();  // for task statistics use invariants only
    auto swTasks = tasks.selectSWtasks();

    TaskList swStatsTasks;
    if (!swTasks.empty()) {
        // we use all SW tasks considered as low-level tasks being directly profiled
        // and assuming no SW task grouping is performed
        swStatsTasks.append(swTasks);
    }

    totalDuration = tasks.getTotalDuration();

    // DMA stats
    TaskTrack dmaTrack;
    dmaDuration = dmaTrack.insert(dmaStatsTasks).coalesce().getSumOfDurations();
    sumOfDmaTaskDurations = dmaStatsTasks.getSumOfDurations();

    // DPU stats
    TaskTrack dpuTrack;
    dpuDuration = dpuTrack.insert(dpuStatsTasks).coalesce().getSumOfDurations();
    sumOfDpuTaskDurations = dpuStatsTasks.getSumOfDurations();

    // SW (Shave) stats
    TaskTrack swTrack;
    swDuration = swTrack.insert(swStatsTasks).coalesce().getSumOfDurations();
    sumOfSwTaskDurations = swStatsTasks.getSumOfDurations();

    // DMA vs DPU overlap statistics
    auto overlapIdleDurations = dmaTrack.calculateOverlap(dpuTrack);
    dmaDpuOverlapDuration = overlapIdleDurations.first;
    dmaDpuIdleDuration = overlapIdleDurations.second;

    // DMA vs SW overlap statistics
    overlapIdleDurations = dmaTrack.calculateOverlap(swTrack);
    dmaSwOverlapDuration = overlapIdleDurations.first;
    dmaSwIdleDuration = overlapIdleDurations.second;

    // SW vs DPU overlap statistics
    overlapIdleDurations = swTrack.calculateOverlap(dpuTrack);
    swDpuOverlapDuration = overlapIdleDurations.first;
    swDpuIdleDuration = overlapIdleDurations.second;

    // calculate idle time and tasks union
    TaskTrack statsTasks;
    statsTasks.insert(dmaStatsTasks);
    statsTasks.insert(dpuStatsTasks);
    statsTasks.insert(swStatsTasks);
    statsTasks.coalesce();

    overlapIdleDurations = statsTasks.calculateOverlap(statsTasks);
    allTasksUnion = overlapIdleDurations.first;  // set intersection with self is union
    idleDuration = totalDuration - allTasksUnion;
}

void TaskStatistics::printAsJson(std::ostream& out) {
    auto oldFlags = out.flags();
    out << std::fixed << "\"taskStatistics\": {\n"
        << "\"total duration\":" << totalDuration * 1e-3 << ",\n"
        << "\"DMA duration\":" << dmaDuration * 1e-3 << ",\n"
        << "\"DPU duration\":" << dpuDuration * 1e-3 << ",\n"
        << "\"SW duration\":" << swDuration * 1e-3 << ",\n"
        << "\"DMA-DPU overlap\":" << dmaDpuOverlapDuration * 1e-3 << ",\n"
        << "\"DMA-SW overlap\":" << dmaSwOverlapDuration * 1e-3 << ",\n"
        << "\"SW-DPU overlap\":" << swDpuOverlapDuration * 1e-3 << ",\n"
        << "\"all tasks union\":" << allTasksUnion * 1e-3 << ",\n"
        << "\"total idle\":" << idleDuration * 1e-3 << ",\n"
        << "\"SW duration without DPU overlap\":" << getSwDurationWithoutDpuOverlap() * 1e-3 << ",\n"
        << "\"DMA duration without overlaps\":" << getDmaDurationWithoutOverlap() * 1e-3 << ",\n"
        << "\"Sum of DMA task durations\":" << sumOfDmaTaskDurations * 1e-3 << ",\n"
        << "\"Sum of DPU task durations\":" << sumOfDpuTaskDurations * 1e-3 << ",\n"
        << "\"Sum of SW task durations\":" << sumOfSwTaskDurations * 1e-3 << "\n"
        << "}," << std::endl;
    out.flags(oldFlags);
}

void TaskStatistics::log(vpux::Logger& logger) {
    logger.info("Tasks statistics:");
    auto log = logger.nest();

    log.info("- total duration [ns]: {0}", totalDuration);
    log.info("- DMA duration [ns]: {0} ({1} %)", dmaDuration, double(dmaDuration) / totalDuration * 100);
    log.info("- DPU duration [ns]: {0} ({1} %)", dpuDuration, double(dpuDuration) / totalDuration * 100);
    log.info("- SW duration [ns]: {0} ({1} %)", swDuration, double(swDuration) / totalDuration * 100);

    // tasks overlap statistics
    log.info("- DMA-DPU overlap [ns]: {0} ({1} %)", dmaDpuOverlapDuration,
             double(dmaDpuOverlapDuration) / totalDuration * 100);
    log.info("- DMA-SW overlap [ns]: {0} ({1} %)", dmaSwOverlapDuration,
             double(dmaSwOverlapDuration) / totalDuration * 100);
    log.info("- SW-DPU overlap [ns]: {0} ({1} %)", swDpuOverlapDuration,
             double(swDpuOverlapDuration) / totalDuration * 100);
    log.info("- all tasks union [ns]: {0} ({1} %)", allTasksUnion, double(allTasksUnion) / totalDuration * 100);

    // tasks idle statistics
    log.info("- total idle [ns]: {0} ({1} %)", idleDuration, double(idleDuration) / totalDuration * 100);

    // SW duration that does not overlap with DPU
    auto SWdurWithoutDPUoverlap = getSwDurationWithoutDpuOverlap();
    log.info("- SW duration without DPU overlap [ns]: {0} ({1} %)", SWdurWithoutDPUoverlap,
             double(SWdurWithoutDPUoverlap) / totalDuration * 100);

    // DMA duration that does not overlap with SW and DPU
    auto DMAdurWithoutOverlap = getDmaDurationWithoutOverlap();
    log.info("- DMA duration without overlaps [ns]: {0} ({1} %)", DMAdurWithoutOverlap,
             double(DMAdurWithoutOverlap) / totalDuration * 100);

    // tiling and scheduling performance parameters
    log.info("- Sum of DMA task durations [ns]: {0} ({1} %)", sumOfDmaTaskDurations,
             double(sumOfDmaTaskDurations) / totalDuration * 100);
    log.info("- Sum of DPU task durations [ns]: {0} ({1} %)", sumOfDpuTaskDurations,
             double(sumOfDpuTaskDurations) / totalDuration * 100);
    log.info("- Sum of SW task durations [ns]: {0} ({1} %)", sumOfSwTaskDurations,
             double(sumOfSwTaskDurations) / totalDuration * 100);
}

namespace details {

TaskTrack& TaskTrack::insert(const TaskList& tasks) {
    for (const auto& task : tasks) {
        TrackEvent eventStart = {task.start_time_ns, 0, true, task.duration_ns};
        TrackEvent eventEnd = {task.start_time_ns + task.duration_ns, 0, false, task.duration_ns};
        _trackEvents.push_back(eventStart);
        _trackEvents.push_back(eventEnd);
    }
    return *this;
}

TaskTrack& TaskTrack::insert(uint64_t trackTime, uint64_t evtDuration, bool isEvtStart) {
    TrackEvent evt = {trackTime, 0, isEvtStart, evtDuration};
    _trackEvents.push_back(evt);
    return *this;
}

TaskTrack& TaskTrack::append(const TaskTrack& taskTrack) {
    const auto& events = taskTrack._trackEvents;
    _trackEvents.insert(_trackEvents.end(), events.begin(), events.end());
    return *this;
}

TaskTrack& TaskTrack::coalesce() {
    // sort events by increasing time
    sortByTime();

    const int noTaskValue = 0;  // value indicating that no tasks are present at given time
    int startTime = 0;
    bool findNewTaskStart = true;
    auto trackProfile = getTrackProfile();
    std::vector<TrackEvent> coalescedEvents;
    for (auto it = trackProfile.begin(); it != trackProfile.end(); ++it) {
        int currentTime = it->first;
        int stackedTasksCount = it->second;
        TrackEvent evt;

        evt.isStart = stackedTasksCount != noTaskValue;
        if (stackedTasksCount > noTaskValue && findNewTaskStart) {
            findNewTaskStart = false;
            startTime = currentTime;
        }

        if (stackedTasksCount == noTaskValue) {
            evt.duration = currentTime - startTime;
            evt.time = startTime;
            evt.taskCount = stackedTasksCount;
            coalescedEvents.push_back(evt);
            findNewTaskStart = true;
        }
    }

    _trackEvents = std::move(coalescedEvents);
    return *this;
}

uint64_t TaskTrack::getSumOfDurations() const {
    return std::accumulate(_trackEvents.begin(), _trackEvents.end(), 0,
                           [](const int& totalTime, const TrackEvent& trackEvent) {
                               return totalTime + trackEvent.duration;
                           });
}

std::pair<uint64_t, uint64_t> TaskTrack::calculateOverlap(const TaskTrack& refTrack) const {
    TaskTrack combined;
    combined.append(*this).append(refTrack);

    // create a list of relevant times of event overlaps
    TaskTrack trackEvents;
    for (auto& x : combined._trackEvents) {
        if (x.duration > 0) {  // ignore zero-duration events
            trackEvents.insert(x.time, x.duration, true);
            trackEvents.insert(x.time + x.duration, x.duration, false);
        }
    }

    // calculate overlap
    trackEvents.sortByTime();
    int counter = 0;
    for (auto& trackEvent : trackEvents._trackEvents) {
        trackEvent.isStart ? counter++ : counter--;
        trackEvent.taskCount = counter;
    }

    // calculate concurrent tasks and idle durations
    uint64_t overlapTime = 0;
    uint64_t idleTime = 0;
    std::vector<TrackEvent>::iterator it;
    const int concurrentTasksThres = 2;  // at least two tasks are present at given time
    const int noTaskValue = 0;           // value indicating that no tasks are present at given time
    for (it = trackEvents._trackEvents.begin(); it != trackEvents._trackEvents.end(); ++it) {
        auto currentTimestamp = it->time;
        auto stackedTasksCount = it->taskCount;
        auto nextTimestamp = next(it, 1) == trackEvents._trackEvents.end() ? it->time : std::next(it, 1)->time;
        if (stackedTasksCount >= concurrentTasksThres) {  // at least two tasks are executed in parallel
                                                          // starting from the current event
            overlapTime += nextTimestamp - currentTimestamp;
        } else if (stackedTasksCount == noTaskValue) {  // no tasks are executed starting from current event
            idleTime += nextTimestamp - currentTimestamp;
        }
    }

    return std::make_pair(overlapTime, idleTime);
}

TaskTrack& TaskTrack::sortByTime() {
    std::sort(_trackEvents.begin(), _trackEvents.end(), [](const TrackEvent& a, const TrackEvent& b) {
        return std::make_tuple(a.time, b.duration) < std::make_tuple(b.time, a.duration);
    });
    return *this;
}

std::map<int, int> TaskTrack::getTrackProfile() {
    std::map<int, int> profile;

    int concurrencyCounter = 0;
    for (auto& evt : _trackEvents) {
        if (evt.duration == 0) {  // zero-duration events do not change track profile so do not store their times
            continue;
        }

        evt.isStart ? concurrencyCounter++ : concurrencyCounter--;
        profile[evt.time] = concurrencyCounter;
    }
    return profile;
}

}  // namespace details

}  // namespace vpux::profiling
