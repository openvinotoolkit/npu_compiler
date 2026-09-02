//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/reports/api.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/reports/stats.hpp"
#include "vpux/utils/profiling/reports/tasklist.hpp"
#include "vpux/utils/profiling/reports/ted.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"
#include "vpux/utils/profiling/utils.hpp"

#include <exception>
#include <iomanip>
#include <ostream>
#include <sstream>
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
     * @param taskFilter filter which selects which tasks to consider during processing
     *
     * The function schedules tasks for output to out stream and generates meta type header trace events.
     * It internally manages trace events' thread IDs
     */
    template <typename FilterFunction>
    void processTraceEvents(const TaskList& tasks, FilterFunction&& taskFilter);

    void processTraceEvents(const TaskList& tasks);

    void processDMATraceEvents(const TaskList& tasks);

    void processSWTraceEvents(const TaskList& tasks);

    std::string getThreadLabel(const TaskInfo& taskInfo);

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

TraceEventDesc makeTaskTraceEvent(const TaskInfo& task, int pid, int tid) {
    TraceEventDesc ted;
    ted.name = task.name;
    ted.category = getTaskCategory(task.exec_type);
    ted.pid = pid;
    ted.tid = tid;
    // use ns-resolution integers to avoid round-off errors during fixed precision output to JSON
    ted.timestamp = task.start_time_ns / 1000.;
    ted.duration = task.duration_ns / 1000.;

    ted.customArgs = task.customArgs;
    if (task.exec_type == TaskInfo::ExecType::SW && task.fifo_id.has_value()) {
        ted.customArgs.push_back({"FIFO ID", std::to_string(*task.fifo_id)});
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
        ted.customArgs.push_back({"DPU time", formatDuration(layer.dpu_ns)});
    }
    if (layer.sw_ns != 0) {
        ted.customArgs.push_back({"Shave time", formatDuration(layer.sw_ns)});
    }
    if (layer.dma_ns != 0) {
        ted.customArgs.push_back({"DMA time", formatDuration(layer.dma_ns)});
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
        processDMATraceEvents(dmaTasks);
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
        processTraceEvents(dpuInvariants);
        processTraceEvents(dpuVariants);
        auto clusterSwTasks = swTasks.selectTasksFromCluster(clusterId);
        processSWTraceEvents(clusterSwTasks);
    }

    TaskList m2iTasks = TaskList(tasks).selectM2Itasks();
    if (!m2iTasks.empty()) {
        createProcess("M2I");
        processTraceEvents(m2iTasks);
    }
}

void TraceEventExporter::processLayers(const std::vector<LayerInfo>& layers) {
    if (layers.empty()) {
        return;
    }

    createProcess("Layers");
    ++_threadId;
    const std::string layersThreadName = "Layers";

    TraceEventTimeOrderedDistribution layersDistr;
    for (auto& layer : layers) {
        auto tid = _threadId + layersDistr.getThreadId(layer.start_time_ns, layer.duration_ns);
        _events.push_back(makeLayerTraceEvent(layer, _processId, tid));
    }

    for (unsigned n = 0; n < layersDistr.size(); ++n) {
        setTraceEventThreadName(layersThreadName, _threadId++, _processId);
    }
}

void TraceEventExporter::validateTask(const TaskInfo& task) const {
    // check task duration
    if (task.duration_ns <= 0) {
        _log.warning("Task {0} has duration {1} ns.", task.name, task.duration_ns);
    }
}

std::string TraceEventExporter::getThreadLabel(const TaskInfo& taskInfo) {
    std::stringstream label;
    switch (taskInfo.exec_type) {
    case TaskInfo::ExecType::DMA:
        label << "DMA";
        // DMA channel type and port ID were added together starting
        // from profiling schema v2.1. For previous versions channel is
        // set as unknown and in such case don't print port number or channel type.
        if (taskInfo.channel_type) {
            if (taskInfo.port_id) {
                label << ' ' << static_cast<int>(taskInfo.port_id.value());
            }
            switch (*taskInfo.channel_type) {
            case DMAChannelType::DDR:
                label << " DDR";
                break;
            case DMAChannelType::CMX:
                label << " CMX";
                break;
            default:
                _log.warning("Unknown channel type");
            }
        }
        break;
    case TaskInfo::ExecType::DPU:
        label << "DPU";
        if (taskInfo.isSubtask) {
            label << " Variants";
        }
        break;
    case TaskInfo::ExecType::M2I:
        label << "M2I";
        break;
    case TaskInfo::ExecType::SW:
        if (taskInfo.fifo_id.has_value()) {
            label << "Shave" << static_cast<unsigned>(taskInfo.fifo_id.value());
        } else {
            label << "Shave";
        }
        break;
    case TaskInfo::ExecType::NONE:
    default:
        VPUX_THROW("Unexpected exec type");
        break;
    }

    return label.str();
}

template <typename FilterFunction>
void TraceEventExporter::processTraceEvents(const TaskList& tasks, FilterFunction&& taskFilter) {
    if (tasks.empty()) {
        return;
    }

    auto sortedTasks = tasks.getSortedByStartTime();
    int lastThreadId = _threadId++;
    TraceEventTimeOrderedDistribution threadDistr;
    for (const auto& task : llvm::make_filter_range(tasks, taskFilter)) {
        // Note that Perfetto requires that tasks within single track(thread) are not overlapping.
        // Since our HW supports pipelining some of the tasks on the same engine will overlap.
        // Below logic is responsible for splitting single engine track into several tracks where tasks
        // won't overlap. Failure to do so will result in incorrect track display.
        auto tid = _threadId + threadDistr.getThreadId(task.start_time_ns, task.duration_ns);
        if (tid > lastThreadId) {
            auto threadLabel = getThreadLabel(task);
            setTraceEventThreadName(threadLabel, tid, _processId);
            lastThreadId = tid;
        }
        _events.push_back(makeTaskTraceEvent(task, _processId, tid));
    }

    _threadId = lastThreadId;
}

void TraceEventExporter::processTraceEvents(const TaskList& tasks) {
    auto acceptAllTasks = [](const TaskInfo&) {
        return true;
    };
    return processTraceEvents(tasks, std::move(acceptAllTasks));
}

void TraceEventExporter::processDMATraceEvents(const TaskList& tasks) {
    // Group by channel then port
    std::set<std::pair<std::optional<DMAChannelType>, std::optional<unsigned short>>> uniqueChannels;
    for (const auto& task : tasks) {
        uniqueChannels.insert(std::make_pair(task.channel_type, task.port_id));
    }

    for (auto dmaChannel : uniqueChannels) {
        processTraceEvents(tasks, [&](const TaskInfo& task) {
            return task.port_id == dmaChannel.second && task.channel_type == dmaChannel.first;
        });
    }
}

void TraceEventExporter::processSWTraceEvents(const TaskList& tasks) {
    if (tasks.empty()) {
        return;
    }

    const auto sortedTasks = tasks.getSortedByStartTime();

    // Group by fifo_id to ensure consistent thread labels per FIFO
    std::set<std::optional<uint8_t>> uniqueFifoIds;
    for (const auto& task : sortedTasks) {
        uniqueFifoIds.insert(task.fifo_id);
    }

    for (auto fifoId : uniqueFifoIds) {
        processTraceEvents(sortedTasks, [&](const TaskInfo& task) {
            return task.fifo_id == fifoId;
        });
    }
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
