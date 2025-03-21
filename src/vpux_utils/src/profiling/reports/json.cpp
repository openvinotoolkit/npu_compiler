//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/reports/stats.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"
#include "vpux/utils/profiling/reports/ted.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include <exception>
#include <iomanip>
#include <ostream>
#include <vector>

namespace vpux::profiling {

namespace {

/**
 * @brief Helper class to calculate placement of profiling tasks in
 * an optimal number of Perfetto UI threads
 *
 * Stores tasks end times for each thread.
 */
class TraceEventTimeOrderedDistribution {
public:
    /**
     * @brief Get the event thread Id assuring a non-overlapping placement among other existing tasks on the same
     * thread.
     *
     * Calls to this function assume that tasks were sorted in ascending order by taskStartTime
     * This function updates the state of object with the taskEndTime in the thread the task was assigned to.
     *
     * @return int - calculate thread id unique to the current process
     */
    int getThreadId(double taskStartTime, double duration);

    /// Return current number of threads
    unsigned size() const {
        return _lastTimestamps.size();
    }

private:
    std::vector<double> _lastTimestamps;
};

int TraceEventTimeOrderedDistribution::getThreadId(double taskStartTime, double duration) {
    double taskEndTime = taskStartTime + duration;
    for (size_t i = 0; i < _lastTimestamps.size(); ++i) {
        if (_lastTimestamps[i] <= taskStartTime) {
            _lastTimestamps[i] = taskEndTime;
            return i;
        }
    }
    _lastTimestamps.push_back(taskEndTime);
    return _lastTimestamps.size() - 1;
}

class TraceEventExporter {
public:
    TraceEventExporter(std::ostream& outStream, Logger& log);
    ~TraceEventExporter() noexcept(false);

    /**
     * @brief flush queued trace events to output stream.
     */
    void flushAsTraceEvents();

    void processTasks(const std::vector<TaskInfo>& tasks);
    void processLayers(const std::vector<LayerInfo>& layers);

private:
    TraceEventExporter(const TraceEventExporter&) = delete;
    TraceEventExporter& operator=(const TraceEventExporter&) = delete;

    /**
     * @brief helper function to ease exporting profiled tasks to JSON format
     *
     * @param tasks list of tasks to be exported
     * @param threadName trace event thread name
     *
     * The function schedules tasks for output to out stream and generates meta type header trace events.
     * It internally manages trace events' thread IDs
     */
    void processTraceEvents(const TaskList& tasks, const std::string& threadName);

    void createProcess(const std::string& processName);

    /**
     * @brief set tracing event process name for given process id.
     *
     * @param processName process name
     * @param processId trace event process identifier
     */
    void setTraceEventProcessName(const std::string& processName, int processId);

    void setTraceEventThreadName(const std::string& threadName, int threadId, int processId);

    /**
     * @brief Set the Tracing Event Process Sort Index
     *
     * @param processId trace event process identifier
     * @param sortIndex index defining the process ordering in the output report. (Some UIs do not respect this value)
     */
    void setTraceEventProcessSortIndex(int processId, unsigned sortIndex);

    /**
     * @brief Perform basic sanity checks on a task
     *
     * @param task task to check
     *
     * Warning is issued if task duration is not a positive integer.
     */
    void validateTask(const TaskInfo& task) const;

    std::vector<TraceEventDesc> _events;
    std::ostream& _outStream;
    Logger _log;
    int _processId = -1;
    int _threadId = -1;
};

std::string getTaskCategory(TaskInfo::ExecType type) {
    switch (type) {
    case TaskInfo::ExecType::DPU:
        return "DPU";
    case TaskInfo::ExecType::SW:
        return "Shave";
    case TaskInfo::ExecType::DMA:
        return "DMA";
    case TaskInfo::ExecType::M2I:
        return "M2I";
    default:
        VPUX_THROW("Unexpected task type");
    }
}

std::string formatDuration(uint64_t timeNs) {
    uint64_t ms = timeNs / 1000000;
    uint64_t us = (timeNs / 1000) % 1000;
    uint64_t ns = timeNs % 1000;
    std::string timeString;
    if (ms != 0) {
        timeString += printToString("{0}ms ", ms);
    }
    if (us != 0) {
        timeString += printToString("{0}us ", us);
    }
    timeString += printToString("{0}ns", ns);
    return timeString;
}

TraceEventDesc makeTaskTraceEvent(const TaskInfo& task, int pid, int tid) {
    TraceEventDesc ted;
    ted.name = task.name;
    ted.category = getTaskCategory(task.exec_type);
    ted.pid = pid;
    ted.tid = tid;
    // use ns-resolution integers to avoid round-off errors during fixed precision output to JSON
    ted.timestamp = task.start_time_ns / 1000.;
    ted.duration = task.duration_ns / 1000.;

    if (task.total_cycles != 0) {
        ted.customArgs.push_back({"Total cycles:", std::to_string(task.total_cycles)});
    }
    if (task.stall_cycles != 0) {
        ted.customArgs.push_back({"Stall cycles:", std::to_string(task.stall_cycles)});
    }
    if (task.stall_counters.lsu0_stalls != 0) {
        ted.customArgs.push_back({"LSU0 stall cycles:", std::to_string(task.stall_counters.lsu0_stalls)});
    }
    if (task.stall_counters.lsu1_stalls != 0) {
        ted.customArgs.push_back({"LSU1 stall cycles:", std::to_string(task.stall_counters.lsu1_stalls)});
    }
    if (task.stall_counters.instruction_stalls != 0) {
        ted.customArgs.push_back({"Instruction stall cycles:", std::to_string(task.stall_counters.instruction_stalls)});
    }
    return ted;
}

TraceEventDesc makeLayerTraceEvent(const LayerInfo& layer, int pid, int tid) {
    TraceEventDesc ted;
    ted.name = layer.name;
    ted.category = "Layer";
    ted.pid = pid;
    ted.tid = tid;
    // use ns-resolution integers to avoid round-off errors during fixed precision output to JSON
    ted.timestamp = layer.start_time_ns / 1000.;
    ted.duration = layer.duration_ns / 1000.;
    ted.customArgs.push_back({"Layer type", layer.layer_type});

    if (layer.dpu_ns != 0) {
        ted.customArgs.push_back({"DPU time:", formatDuration(layer.dpu_ns)});
    }
    if (layer.sw_ns != 0) {
        ted.customArgs.push_back({"Shave time:", formatDuration(layer.sw_ns)});
    }
    if (layer.dma_ns != 0) {
        ted.customArgs.push_back({"DMA time:", formatDuration(layer.dma_ns)});
    }
    return ted;
}

void TraceEventExporter::processTasks(const std::vector<TaskInfo>& tasks) {
    for (auto& task : tasks) {
        validateTask(task);
    }

    //
    // Export DMA tasks
    //
    auto dmaTasks = TaskList(tasks).selectDMAtasks();
    if (!dmaTasks.empty()) {
        createProcess("DMA");
        processTraceEvents(dmaTasks, "DMA");
    }

    //
    // Export cluster tasks (DPU and SW)
    //
    unsigned clusterCount = TaskList(tasks).getClusterCount();

    TaskList dpuTasks = TaskList(tasks).selectDPUtasks();
    TaskList swTasks = TaskList(tasks).selectSWtasks();

    for (unsigned clusterId = 0; clusterId < clusterCount; clusterId++) {
        createProcess("Cluster (" + std::to_string(clusterId) + ")");
        auto clusterDpuTasks = dpuTasks.selectTasksFromCluster(clusterId);
        auto dpuInvariants = clusterDpuTasks.selectClusterLevelTasks();
        auto dpuVariants = clusterDpuTasks.selectSubtasks();
        processTraceEvents(dpuInvariants, "DPU");
        processTraceEvents(dpuVariants, "DPU Variants");
        auto clusterSwTasks = swTasks.selectTasksFromCluster(clusterId);
        processTraceEvents(clusterSwTasks, "Shave");
    }

    TaskList m2iTasks = TaskList(tasks).selectM2Itasks();
    if (!m2iTasks.empty()) {
        createProcess("M2I");
        processTraceEvents(m2iTasks, "M2I");
    }
}

void TraceEventExporter::processLayers(const std::vector<LayerInfo>& layers) {
    if (layers.empty()) {
        return;
    }

    createProcess("Layers");
    ++_threadId;

    TraceEventTimeOrderedDistribution layersDistr;
    for (auto& layer : layers) {
        auto tid = _threadId + layersDistr.getThreadId(layer.start_time_ns, layer.duration_ns);
        _events.push_back(makeLayerTraceEvent(layer, _processId, tid));
    }

    for (unsigned n = 0; n < layersDistr.size(); ++n) {
        setTraceEventThreadName("Layers", _threadId++, _processId);
    }
}

void TraceEventExporter::validateTask(const TaskInfo& task) const {
    // check task duration
    if (task.duration_ns <= 0) {
        _log.warning("Task {0} has duration {1} ns.", task.name, task.duration_ns);
    }
}

void TraceEventExporter::processTraceEvents(const TaskList& tasks, const std::string& threadName) {
    if (tasks.empty()) {
        return;
    }

    int lastThreadId = _threadId++;

    auto sortedTasks = tasks.getSortedByStartTime();
    TraceEventTimeOrderedDistribution threadDistr;

    for (const auto& task : sortedTasks) {
        auto tid = _threadId + threadDistr.getThreadId(task.start_time_ns, task.duration_ns);
        if (tid > lastThreadId) {
            setTraceEventThreadName(threadName, tid, _processId);
            lastThreadId = tid;
        }

        _events.push_back(makeTaskTraceEvent(task, _processId, tid));
    }

    _threadId = lastThreadId;
}

void TraceEventExporter::createProcess(const std::string& name) {
    auto pid = ++_processId;
    setTraceEventProcessName(name, pid);
    setTraceEventProcessSortIndex(pid, pid);
}

TraceEventExporter::TraceEventExporter(std::ostream& outStream, Logger& log): _outStream(outStream), _log(log) {
    // Trace Events timestamps are in microseconds, set precision to preserve nanosecond resolution
    _outStream << std::setprecision(3) << "{\"traceEvents\":[" << std::endl;
}

TraceEventExporter::~TraceEventExporter() noexcept(false) {
    if (std::uncaught_exceptions() > 0) {
        // Got bigger probelm than closing the report so ignore the errors from here
        _outStream.exceptions(std::ios::goodbit);
    }
    // Hint for a classic Perfetto UI to use nanoseconds for display
    // JSON timestamps are expected to be in microseconds regardless
    _outStream << "\"displayTimeUnit\": \"ns\"\n"
               << "}\n";
    _outStream.flush();
}

void TraceEventExporter::flushAsTraceEvents() {
    if (!_events.empty()) {
        for (auto tedIt = _events.begin(); tedIt != std::prev(_events.end()); ++tedIt) {
            _outStream << *tedIt << ",\n";
        }
        _outStream << _events.back() << std::endl;
        _events.clear();
    }
    // close traceEvents block
    _outStream << "],\n";
    _outStream.flush();
}

void TraceEventExporter::setTraceEventProcessName(const std::string& processName, int processId) {
    _outStream << std::string(R"({"name": "process_name", "ph": "M", "pid":)") << processId
               << R"(, "args": {"name" : ")" << processName << R"("}},)" << std::endl;
}

void TraceEventExporter::setTraceEventThreadName(const std::string& threadName, int threadId, int processId) {
    _outStream << std::string(R"({"name": "thread_name", "ph": "M", "pid":)") << processId << R"(, "tid":)" << threadId
               << R"(, "args": {"name" : ")" << threadName << R"("}},)" << std::endl;
}

void TraceEventExporter::setTraceEventProcessSortIndex(int processId, unsigned sortIndex) {
    _outStream << std::string(R"({"name": "process_sort_index", "ph": "M", "pid":)") << processId
               << R"(, "args": {"sort_index" : ")" << sortIndex << R"("}},)" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const FreqInfo& freqInfo) {
    if (freqInfo.freqMHz != UNINITIALIZED_FREQUENCY_VALUE) {
        return printTo(os, "\"workpoint\": { \"freq\": {0:1}, \"status\": \"{1}\" },\n", freqInfo.freqMHz,
                       to_string(freqInfo.freqStatus));
    }
    return os;
}

}  // namespace

void printProfilingAsTraceEvent(const std::vector<TaskInfo>& tasks, const std::vector<LayerInfo>& layers,
                                FreqInfo dpuFreq, std::ostream& output, Logger& log) {
    TaskStatistics stats(tasks);

    {
        TraceEventExporter events(output, log);
        events.processTasks(tasks);
        events.processLayers(layers);
        events.flushAsTraceEvents();
        stats.printAsJson(output);
        output << dpuFreq;
    }

    stats.log(log);
}

std::ostream& operator<<(std::ostream& os, const TraceEventDesc& event) {
    std::ios::fmtflags origFlags(os.flags());
    os << std::fixed << "{\"name\":\"" << event.name << "\", \"cat\":\"" << event.category << "\", \"ph\":\"X\", "
       << "\"ts\":" << event.timestamp << ", \"dur\":" << event.duration << ", \"pid\":" << event.pid
       << ", \"tid\":" << event.tid;
    if (!event.customArgs.empty()) {
        os << ", \"args\":{";
        bool isFirst = true;
        for (const auto& arg : event.customArgs) {
            os << (isFirst ? "" : ", ") << "\"" << arg.first << "\": \"" << arg.second << "\"";
            isFirst = false;
        }
        os << "}";
    }
    os << "}";
    os.flags(origFlags);
    return os;
}

}  // namespace vpux::profiling
