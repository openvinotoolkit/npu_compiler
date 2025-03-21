//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/utils/profiling/parser/parser.hpp"
#include "vpux/utils/profiling/parser/records.hpp"

#include "vpux/utils/profiling/reports/api.hpp"  // getLayerInfo

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/metadata.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include "schema/profiling_generated.h"

#include <string>
#include <vector>

namespace vpux::profiling {

namespace {

template <typename... Args>
void warnOrFail(bool failOnError, vpux::Logger& log, bool condition, llvm::StringLiteral format, Args&&... params) {
    if (condition) {
        return;
    }
    if (failOnError) {
        VPUX_THROW(format, std::forward<Args>(params)...);
    } else {
        log.warning(format, std::forward<Args>(params)...);
    }
}

void fillTaskInfoWithParsedRawRecords(std::vector<TaskInfo>& vec, const RawProfilingRecords& rawTasks,
                                      FrequenciesSetup frequenciesSetup) {
    for (const auto& task : rawTasks) {
        vec.push_back(task->getTaskInfo(frequenciesSetup));
    }
}

bool minStartTimeTaskComparator(const TaskInfo& a, const TaskInfo& b) {
    return a.start_time_ns < b.start_time_ns;
};

std::optional<uint64_t> getEarliestTaskBegin(const std::vector<TaskInfo>& tasks) {
    if (tasks.empty()) {
        return std::nullopt;
    }
    return std::min_element(tasks.cbegin(), tasks.cend(), minStartTimeTaskComparator)->start_time_ns;
}

// Lets find the minimal offset between timers as we know for sure that DPU/SW task are started after the end of
// DMA task because they are connected via the same barrier
int64_t getTimersOffset(const std::optional<int64_t> maybeTaskTimerDiff,
                        const std::optional<uint64_t> maybeEarliestDmaNs, uint64_t earliestTaskNs) {
    if (maybeTaskTimerDiff.has_value()) {
        return maybeTaskTimerDiff.value();
    }

    // Could not calculate offset between timers(Most likely DMA profiling is disabled)
    // -> set offset based on begin time
    if (maybeEarliestDmaNs.has_value()) {
        return maybeEarliestDmaNs.value() - earliestTaskNs;
    } else {
        // FIXME: we do not need to mix unsigned and singed here
        // currenntly, we apply workaround to enable a strict check
        // for such kind of errors
        // E#65384
        return -(int64_t)earliestTaskNs;
    }
};

// Adjust all tasks to zero point (earliest DMA task)
void adjustZeroPoint(std::vector<TaskInfo>& taskInfo, int64_t timerDiff, const std::optional<uint64_t> maybeFirstDma) {
    const auto firstDma = static_cast<int64_t>(maybeFirstDma.value_or(0));
    for (auto& task : taskInfo) {
        int64_t startTimeNs = task.start_time_ns - firstDma + timerDiff;
        task.start_time_ns = std::max(startTimeNs, (int64_t)0);
    }
};

size_t findEarliestTask(size_t currentEngineEarliestTaskNs, const std::vector<std::optional<size_t>>& otherEngines) {
    for (const auto& maybeTimestamp : otherEngines) {
        if (maybeTimestamp.has_value()) {
            currentEngineEarliestTaskNs = std::min(currentEngineEarliestTaskNs, maybeTimestamp.value());
        }
    }
    return currentEngineEarliestTaskNs;
}

RawProfilingRecords parseDmaHwTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DMATask>>* dmaTaskList, const void* output,
        size_t outputLen) {
    if (dmaTaskList == nullptr) {
        return {};
    }
    VPUX_THROW_UNLESS(outputLen % sizeof(HwpDma40Data_t) == 0, "Invalid profiling data");

    const size_t totalDmaTasks = (outputLen / sizeof(HwpDma40Data_t));

    RawProfilingRecords rawRecords;
    for (const ProfilingFB::DMATask* task : *dmaTaskList) {
        VPUX_THROW_WHEN(task->isProfBegin(), "DMA HWP do not use profBegin tasks");

        unsigned recordNumber = task->dataIndex();
        VPUX_THROW_UNLESS(recordNumber < totalDmaTasks, "Can't process DMA profiling data: {0} out of {1}",
                          recordNumber, totalDmaTasks);
        VPUX_THROW_UNLESS(recordNumber * sizeof(HwpDma40Data_t) < outputLen, "Invalid profiling data");

        auto outputBin = reinterpret_cast<const HwpDma40Data_t*>(output);
        const auto data = outputBin[recordNumber];
        const auto record = std::make_shared<RawProfilingDMA40Record>(data, task, recordNumber);
        record->checkDataOrDie();
        rawRecords.push_back(record);
    }
    return rawRecords;
}

RawProfilingRecords parseDmaSwTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DMATask>>* dmaTaskList, const void* output,
        size_t outputLen, TargetDevice) {
    if (dmaTaskList == nullptr) {
        return {};
    }

    size_t totalDmaTasks = 0;
    VPUX_THROW_WHEN(outputLen % sizeof(DMA27Data_t) != 0, "Invalid section size");
    totalDmaTasks = outputLen / sizeof(DMA27Data_t);

    RawProfilingRecord::BarriersSet lastProfilingRecordWaitBarriers;
    uint32_t foundDmaTasks = 0;
    RawProfilingRecords rawRecords;
    for (const ProfilingFB::DMATask* taskMetadata : *dmaTaskList) {
        if (!taskMetadata->isProfBegin()) {
            foundDmaTasks++;

            unsigned recordNumber = taskMetadata->dataIndex();
            const auto updateBarriers = RawProfilingRecord::getUpdateBarriersFromTask(taskMetadata);

            VPUX_THROW_UNLESS(recordNumber < totalDmaTasks, "Can't process DMA profiling data.");

            VPUX_THROW_WHEN(recordNumber * sizeof(DMA27Data_t) >= outputLen, "Invalid profiling data");

            auto outputBin = reinterpret_cast<const DMA27Data_t*>(output);
            const auto record = outputBin[recordNumber];
            rawRecords.push_back(std::make_shared<RawProfilingDMA27Record>(
                    record, taskMetadata, lastProfilingRecordWaitBarriers, updateBarriers, recordNumber));
        } else {
            lastProfilingRecordWaitBarriers = RawProfilingRecord::getWaitBarriersFromTask(taskMetadata);
        }
    }
    VPUX_THROW_UNLESS(totalDmaTasks == foundDmaTasks, "Unexpected number of DMA tasks in profiling data: {0} != {1}",
                      totalDmaTasks, foundDmaTasks);
    return rawRecords;
}

bool hasExtendedActShaveRecord(TargetDevice device) {
    if (device >= TargetDevice::TargetDevice_VPUX40XX) {
        return true;
    }
    if (device == TargetDevice::TargetDevice_VPUX37XX) {
        return false;
    }
    VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
}

RawProfilingRecords parseActShaveTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::SWTask>>* shaveTaskList, const void* output,
        size_t outputLen, TargetDevice device) {
    if (shaveTaskList == nullptr) {
        return {};
    }

    const auto recordSize = sizeof(ActShaveData_t);
    VPUX_THROW_UNLESS(outputLen % recordSize == 0, "Invalid profiling data");
    const size_t numOfActShaveTasks = outputLen / recordSize;
    size_t foundActShaveTasks = 0;

    RawProfilingRecords rawRecords;
    for (const ProfilingFB::SWTask* taskMeta : *shaveTaskList) {
        size_t currentPos =
                taskMeta->bufferOffset() + taskMeta->clusterSize() * taskMeta->clusterId() + taskMeta->dataIndex();

        VPUX_THROW_UNLESS(currentPos < numOfActShaveTasks, "Unexpected end of blob in ACT section.");
        foundActShaveTasks++;

        std::shared_ptr<RawProfilingRecord> record;
        if (hasExtendedActShaveRecord(device)) {
            const ActShaveDataEx_t outputShave = reinterpret_cast<const ActShaveDataEx_t*>(output)[currentPos];
            record = std::make_shared<RawProfilingACTExRecord>(outputShave, taskMeta, currentPos);
        } else {
            const ActShaveData_t outputShave = reinterpret_cast<const ActShaveData_t*>(output)[currentPos];
            record = std::make_shared<RawProfilingACTRecord>(outputShave, taskMeta, currentPos);
        }
        record->checkDataOrDie();
        rawRecords.push_back(std::move(record));
    }
    VPUX_THROW_UNLESS(foundActShaveTasks == shaveTaskList->size(), "All ActShave tasks should be profiled");
    return rawRecords;
}

RawProfilingRecords parseM2ITaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::M2ITask>>* m2iTaskList, const void* output,
        size_t outputLen) {
    if (m2iTaskList == nullptr) {
        return {};
    }

    const M2IData_t* outputM2I = reinterpret_cast<const M2IData_t*>(output);
    VPUX_THROW_UNLESS(outputLen % sizeof(M2IData_t) == 0, "Invalid profiling data");
    const size_t numOfM2ITasks = outputLen / sizeof(M2IData_t);
    VPUX_THROW_UNLESS(numOfM2ITasks == m2iTaskList->size(), "All M2I tasks should be profiled");

    RawProfilingRecords rawRecords;
    for (size_t taskIndex = 0; taskIndex < m2iTaskList->size(); taskIndex++) {
        const auto record =
                std::make_shared<RawProfilingM2IRecord>(outputM2I[taskIndex], m2iTaskList->Get(taskIndex), taskIndex);
        record->checkDataOrDie();
        rawRecords.push_back(record);
    }
    return rawRecords;
}

size_t getDpuRecordSize(TargetDevice device) {
    if (device >= TargetDevice::TargetDevice_VPUX40XX) {
        return sizeof(HwpDpuIduOduData_t);
    }
    if (device == TargetDevice::TargetDevice_VPUX37XX) {
        return sizeof(HwpDpu27Mode0Data_t);
    }
    VPUX_THROW("TargetDevice {0} is not supported ", EnumNameTargetDevice(device));
}

struct DpuMetaComparator {
    bool operator()(const ProfilingFB::DPUTask* a, const ProfilingFB::DPUTask* b) const {
        return std::make_tuple(a->bufferId(), a->clusterId(), a->taskId()) <
               std::make_tuple(b->bufferId(), b->clusterId(), b->taskId());
    }
};

RawProfilingRecords parseDPUTaskProfiling(
        const flatbuffers::Vector<flatbuffers::Offset<ProfilingFB::DPUTask>>* dpuTaskList, const void* output,
        size_t outputLen, TargetDevice device, vpux::Logger& log, bool ignoreSanitizationErrors) {
    if (dpuTaskList == nullptr) {
        return {};
    }
    const size_t recordSize = getDpuRecordSize(device);
    VPUX_THROW_UNLESS(outputLen % recordSize == 0, "Invalid profiling data");

    unsigned currentPos = 0;
    std::set<const ProfilingFB::DPUTask*, DpuMetaComparator> profInfoAggregator(dpuTaskList->begin(),
                                                                                dpuTaskList->end());
    RawProfilingRecords rawRecords;
    size_t clusterBeginning = 0;
    using TaskLocationDescriptor = std::tuple<size_t, size_t>;
    TaskLocationDescriptor bufferAndClusterDescriptor(0, 0);
    for (const ProfilingFB::DPUTask* taskMeta : profInfoAggregator) {
        VPUX_THROW_UNLESS(taskMeta->taskId() < dpuTaskList->size() + 1, "Invalid profiling data");

        const TaskLocationDescriptor newDescriptor{taskMeta->bufferId(), taskMeta->clusterId()};
        if (newDescriptor != bufferAndClusterDescriptor) {
            clusterBeginning = currentPos;
            bufferAndClusterDescriptor = newDescriptor;
        }
        for (uint32_t variantId = 0; variantId < taskMeta->maxVariants(); variantId++) {
            std::shared_ptr<RawProfilingDPURecord> record;
            if (variantId < taskMeta->numVariants()) {
                const auto inClusterIndex = currentPos - clusterBeginning;
                VPUX_THROW_WHEN(currentPos >= outputLen / recordSize, "Profiling index is out of range");
                if (device >= TargetDevice::TargetDevice_VPUX40XX) {
                    const HwpDpuIduOduData_t dpuTimings =
                            reinterpret_cast<const HwpDpuIduOduData_t*>(output)[currentPos];
                    const auto taskWloadId = taskMeta->workloadIds()->Get(variantId);
                    bool isValidWorkloadIdConfiguration =
                            (dpuTimings.idu_wl_id == dpuTimings.odu_wl_id) && (dpuTimings.idu_wl_id == taskWloadId);
                    warnOrFail(!ignoreSanitizationErrors, log, isValidWorkloadIdConfiguration,
                               "Wrong workload ID. Please report! Expected: {0}, but got IDU {1}, ODU {2}", taskWloadId,
                               dpuTimings.idu_wl_id, dpuTimings.odu_wl_id);

                    if (device == TargetDevice::TargetDevice_VPUX40XX) {
                        record = std::make_shared<RawProfilingDPUHW40Record>(dpuTimings, taskMeta, variantId,
                                                                             currentPos, inClusterIndex);
                    }
                } else if (device == TargetDevice::TargetDevice_VPUX37XX) {
                    const HwpDpu27Mode0Data_t dpuTimings =
                            reinterpret_cast<const HwpDpu27Mode0Data_t*>(output)[currentPos];
                    record = std::make_shared<RawProfilingDPUHW27Record>(dpuTimings, taskMeta, variantId, currentPos,
                                                                         inClusterIndex);
                }
                record->checkDataOrDie();
                rawRecords.push_back(record);
            }
            // continue increment of currentPos to walk over non-used data
            ++currentPos;
        }
    }
    return rawRecords;
}

std::vector<std::pair<WorkpointConfiguration_t, size_t>> getWorkpointData(const void* output, size_t outputLen,
                                                                          size_t offset) {
    const auto NUM_WORKPOINTS = 2;
    VPUX_THROW_UNLESS(outputLen == WORKPOINT_BUFFER_SIZE, "Unexpected workpoint size: {0}", outputLen);
    const auto* workpointsPtr = reinterpret_cast<const WorkpointConfiguration_t*>(output);

    std::vector<std::pair<WorkpointConfiguration_t, size_t>> workpoints;
    for (size_t i = 0; i < NUM_WORKPOINTS; ++i) {
        workpoints.emplace_back(workpointsPtr[i], offset);
        offset += sizeof(WorkpointConfiguration_t);
    }

    return workpoints;
}

RawProfilingData parseProfilingTaskLists(const RawDataLayout& sections, TargetDevice device, const uint8_t* profData,
                                         const ProfilingFB::ProfilingMeta* profilingSchema, vpux::Logger& log,
                                         bool ignoreSanitizationErrors) {
    RawProfilingData rawProfData;

    for (const auto& section : sections) {
        const auto offset = section.second.first;
        const auto length = section.second.second;

        switch (section.first) {
        case ExecutorType::DMA_SW: {
            rawProfData.dmaTasks =
                    parseDmaSwTaskProfiling(profilingSchema->dmaTasks(), profData + offset, length, device);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_SW, offset);
            break;
        }
        case ExecutorType::DMA_HW: {
            rawProfData.dmaTasks = parseDmaHwTaskProfiling(profilingSchema->dmaTasks(), profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::DMA_HW, offset);
            break;
        }
        case ExecutorType::ACTSHAVE: {
            rawProfData.swTasks =
                    parseActShaveTaskProfiling(profilingSchema->swTasks(), profData + offset, length, device);
            rawProfData.parseOrder.emplace_back(ExecutorType::ACTSHAVE, offset);
            break;
        }
        case ExecutorType::DPU: {
            rawProfData.dpuTasks = parseDPUTaskProfiling(profilingSchema->dpuTasks(), profData + offset, length, device,
                                                         log, ignoreSanitizationErrors);
            rawProfData.parseOrder.emplace_back(ExecutorType::DPU, offset);
            break;
        }
        case ExecutorType::WORKPOINT: {
            const auto isWorkpointAccessible =
                    device == TargetDevice::TargetDevice_VPUX37XX || device >= TargetDevice::TargetDevice_VPUX40XX;
            if (isWorkpointAccessible && length != 0) {
                rawProfData.workpoints = getWorkpointData(profData + offset, length, offset);
            }
            break;
        }
        case ExecutorType::M2I: {
            rawProfData.m2iTasks = parseM2ITaskProfiling(profilingSchema->m2iTasks(), profData + offset, length);
            rawProfData.parseOrder.emplace_back(ExecutorType::M2I, offset);
            break;
        }
        default:
            VPUX_THROW("Invalid profiling executor.");
        }
    }
    return rawProfData;
}

RawDataLayout getRawDataLayoutFB(const ProfilingFB::ProfilingBuffer* profBuffer, size_t actualBufferSize) {
    VPUX_THROW_UNLESS(profBuffer != nullptr, "Profiling buffer data must be not empty");

    const uint32_t profSize = profBuffer->size();
    VPUX_THROW_WHEN(uint32_t(actualBufferSize) != profSize,
                    "The profiling data size does not match the expected size. Expected {0}, but got {1}", profSize,
                    actualBufferSize);

    uint32_t prevSectionEnd = 0;
    RawDataLayout sections;
    for (const auto& section : *profBuffer->sections()) {
        const auto sectionBegin = section->offset();
        const auto sectionSize = section->size();
        const auto sectionEnd = sectionBegin + sectionSize;

        VPUX_THROW_UNLESS((sectionBegin < profSize && sectionEnd <= profSize),
                          "Section [{0};{1}] exceeds profiling buffer size({2}b)", sectionBegin, sectionEnd, profSize);
        VPUX_THROW_WHEN(sectionBegin < prevSectionEnd, "Section(type {0}) overlaps with previous section",
                        section->type());

        const auto execType = static_cast<ExecutorType>(section->type());
        sections[execType] = {sectionBegin, sectionSize};
        prevSectionEnd = sectionEnd;
    }
    return sections;
}

RawProfilingRecords makeFakeDpuInvariants(const RawProfilingRecords& variants) {
    RawProfilingRecords invariants;

    // Grouping of variants into one invariant
    std::multimap<std::pair<uint32_t, size_t>, RawProfilingRecordPtr> groupedClustersInfo;
    for (const auto& task : variants) {
        const auto clusteredTask = std::dynamic_pointer_cast<RawProfilingDPURecord>(task);
        VPUX_THROW_WHEN(clusteredTask == nullptr, "Expected cluster task");
        const auto clusterId = clusteredTask->getClusterId();
        const auto key = std::make_pair(clusteredTask->getTaskId(), clusterId);
        groupedClustersInfo.insert(std::make_pair(key, task));
    }

    auto it = groupedClustersInfo.cbegin();
    while (it != groupedClustersInfo.cend()) {
        RawProfilingRecords variants;
        const auto groupingKey = it->first;
        std::string name = it->second->getOriginalName();

        while (it != groupedClustersInfo.cend() && it->first == groupingKey) {
            variants.push_back(it->second);
            ++it;
        }
        invariants.push_back(std::make_shared<ArrayRecord>(name, variants));
    }

    return invariants;
}

// At parse time we don't know frequency for some platforms, so data is collected in cycles format. We need
// to determine frequency to convert from cycles to nanoseconds
std::vector<TaskInfo> convertRawTasksToTaskInfo(const RawProfilingData& rawTasks,
                                                const FrequenciesSetup& frequenciesSetup, VerbosityLevel verbosity,
                                                vpux::Logger& log) {
    for (const auto& taskList : {rawTasks.dmaTasks, rawTasks.dpuTasks, rawTasks.swTasks, rawTasks.m2iTasks}) {
        for (const auto& task : taskList) {
            task->sanitize(log, frequenciesSetup);
        }
    }

    std::vector<TaskInfo> dmaTaskInfo;
    std::vector<TaskInfo> swTaskInfo;
    std::vector<TaskInfo> dpuTaskInfo;
    std::vector<TaskInfo> m2iTaskInfo;

    fillTaskInfoWithParsedRawRecords(dmaTaskInfo, rawTasks.dmaTasks, frequenciesSetup);

    fillTaskInfoWithParsedRawRecords(swTaskInfo, rawTasks.swTasks, frequenciesSetup);

    RawProfilingRecords dpuInvariantTasks = makeFakeDpuInvariants(rawTasks.dpuTasks);
    fillTaskInfoWithParsedRawRecords(dpuTaskInfo, dpuInvariantTasks, frequenciesSetup);
    if (verbosity >= VerbosityLevel::MEDIUM) {
        fillTaskInfoWithParsedRawRecords(dpuTaskInfo, rawTasks.dpuTasks, frequenciesSetup);
    }

    fillTaskInfoWithParsedRawRecords(m2iTaskInfo, rawTasks.m2iTasks, frequenciesSetup);

    const auto earliestDpuNs = getEarliestTaskBegin(dpuTaskInfo);
    const auto earliestDmaNs = getEarliestTaskBegin(dmaTaskInfo);
    const auto earliestSwNs = getEarliestTaskBegin(swTaskInfo);
    const auto earliestM2iNs = getEarliestTaskBegin(m2iTaskInfo);

    log.trace("Earliest DMA: {0}", earliestDmaNs);
    log.trace("Earliest DPU: {0}", earliestDpuNs);
    log.trace("Earliest SW : {0}", earliestSwNs);
    log.trace("Earliest M2I : {0}", earliestM2iNs);

    adjustZeroPoint(dmaTaskInfo, 0, earliestDmaNs);

    if (!dpuTaskInfo.empty()) {
        int64_t dma2dpuOffset = 0;
        if (!frequenciesSetup.hasSharedDmaDpuCounter) {
            const auto timersShift = getDMA2OtherTimersShift(rawTasks.dmaTasks, rawTasks.dpuTasks, frequenciesSetup,
                                                             SynchronizationPointKind::STRICT_DMA_TO_DPU, log);
            log.trace("Timers DMA2DPU difference: {0}", timersShift);
            dma2dpuOffset = getTimersOffset(timersShift, earliestDmaNs, earliestDpuNs.value());
        } else {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice. Timer also can be shared with SW tasks
            // List of other engines, which share timer with DMA
            std::vector<std::optional<size_t>> otherEngineStarts = {earliestM2iNs};
            if (frequenciesSetup.hasSharedDmaSwCounter) {
                otherEngineStarts.push_back(earliestSwNs);
            }
            size_t earliestTaskNs = findEarliestTask(earliestDpuNs.value(), otherEngineStarts);
            dma2dpuOffset = earliestDmaNs.has_value() ? 0 : -static_cast<int64_t>(earliestTaskNs);
        }
        adjustZeroPoint(dpuTaskInfo, dma2dpuOffset, earliestDmaNs);
    }

    if (!swTaskInfo.empty()) {
        int64_t dma2SwOffset = 0;
        if (frequenciesSetup.hasSharedDmaSwCounter) {
            // If DMA profiling enabled difference is 0, otherwise setting to earliest task without call to
            // getTimersOffset to avoid counting twice. Timer also can be shared with SW tasks
            // List of other engines, which share timer with DMA
            std::vector<std::optional<size_t>> otherEngineStarts = {earliestM2iNs};
            if (frequenciesSetup.hasSharedDmaDpuCounter) {
                otherEngineStarts.push_back(earliestDpuNs);
            }
            size_t earliestTaskNs = findEarliestTask(earliestSwNs.value(), otherEngineStarts);
            dma2SwOffset = earliestDmaNs.has_value() ? 0 : -static_cast<int64_t>(earliestTaskNs);
        }
        adjustZeroPoint(swTaskInfo, dma2SwOffset, earliestDmaNs);
    }

    adjustZeroPoint(m2iTaskInfo, 0, earliestDmaNs);

    std::vector<TaskInfo> allTaskInfo;
    allTaskInfo.reserve(dpuTaskInfo.size() + dmaTaskInfo.size() + swTaskInfo.size() + m2iTaskInfo.size());
    allTaskInfo.insert(allTaskInfo.end(), dpuTaskInfo.begin(), dpuTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), dmaTaskInfo.begin(), dmaTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), swTaskInfo.begin(), swTaskInfo.end());
    allTaskInfo.insert(allTaskInfo.end(), m2iTaskInfo.begin(), m2iTaskInfo.end());

    std::sort(allTaskInfo.begin(), allTaskInfo.end(), profilingTaskStartTimeComparator<TaskInfo>);

    return allTaskInfo;
}

}  // namespace

RawData getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                             bool ignoreSanitizationErrors) {
    if ((nullptr == blobData) || (nullptr == profData)) {
        VPUX_THROW("Empty input data");
    }

    auto log = vpux::Logger::global();
    const auto profilingDataSchema = getProfilingSectionMeta(blobData, blobSize);
    auto device = (TargetDevice)profilingDataSchema->platform()->device();
    VPUX_THROW_WHEN(device == TargetDevice::TargetDevice_NONE, "Unknown device");
    log.trace("Using target device {0}", EnumNameTargetDevice(device));

    const auto profilingBufferMeta = profilingDataSchema->profilingBuffer();
    auto sections = getRawDataLayoutFB(profilingBufferMeta, profSize);

    RawProfilingData rawProfData =
            parseProfilingTaskLists(sections, device, profData, profilingDataSchema, log, ignoreSanitizationErrors);

    return {std::move(sections), std::move(rawProfData), device};
}

ProfInfo getProfInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                     VerbosityLevel verbosity, bool fpga, bool highFreqPerfClk) try {
    const auto rawData = getRawProfilingTasks(blobData, blobSize, profData, profSize);

    auto log = vpux::Logger::global();
    FrequenciesSetup frequenciesSetup =
            getFrequencySetup(rawData.device, rawData.rawRecords.workpoints, highFreqPerfClk, fpga, log);
    ProfInfo profInfo;
    profInfo.tasks = convertRawTasksToTaskInfo(rawData.rawRecords, frequenciesSetup, verbosity, log);
    profInfo.layers = getLayerInfo(profInfo.tasks);
    profInfo.dpuFreq.freqMHz = frequenciesSetup.dpuClk;
    profInfo.dpuFreq.freqStatus = frequenciesSetup.clockStatus;
    return profInfo;
} catch (const std::exception& ex) {
    VPUX_THROW("Profiling post-processing failed. {0}", ex.what());
}

std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  VerbosityLevel verbosity, bool fpga, bool highFreqPerfClk) try {
    const auto rawData = getRawProfilingTasks(blobData, blobSize, profData, profSize);

    auto log = vpux::Logger::global();
    FrequenciesSetup frequenciesSetup =
            getFrequencySetup(rawData.device, rawData.rawRecords.workpoints, highFreqPerfClk, fpga, log);
    return convertRawTasksToTaskInfo(rawData.rawRecords, frequenciesSetup, verbosity, log);
} catch (const std::exception& ex) {
    VPUX_THROW("Profiling post-processing failed. {0}", ex.what());
}

}  // namespace vpux::profiling
