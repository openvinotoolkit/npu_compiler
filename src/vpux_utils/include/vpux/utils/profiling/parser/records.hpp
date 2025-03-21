//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "parser.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include "schema/graphfile_generated.h"
#include "schema/profiling_generated.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

namespace vpux::profiling {

constexpr int COL_WIDTH_32 = 11;
constexpr int COL_WIDTH_64 = 19;

class DebugFormattableRecordMixin {
public:
    using ColDesc = std::vector<std::pair<std::string, int>>;

protected:
    DebugFormattableRecordMixin(size_t inMemoryOffset, size_t inClusterIndex = 0)
            : _inMemoryOffset(inMemoryOffset), _inClusterIndex(inClusterIndex) {
    }

    virtual ColDesc getColDesc() const = 0;

public:
    void printDebugHeader(std::ostream& os) {
        const auto columns = this->getColDesc();
        for (const std::pair<std::string, int>& p : columns) {
            os << std::setw(p.second) << p.first;
        }
    }

    size_t getInMemoryOffset() const {
        return _inMemoryOffset;
    }

    size_t getInClusterIndex() const {
        return _inClusterIndex;
    }

    virtual size_t getDebugDataSize() const = 0;

    virtual void printDebugInfo(std::ostream& outStream) const = 0;

private:
    size_t _inMemoryOffset;
    size_t _inClusterIndex;
};

class RawProfilingRecord {
public:
    using BarrierIdType = uint32_t;
    using TimeType = double;
    using BarriersSet = std::set<BarrierIdType>;

    template <typename T, typename std::enable_if_t<std::is_integral<T>::value, bool> = true>
    static TimeType convertTicksToNs(T cycles, double frequency) {
        VPUX_THROW_WHEN(frequency == UNINITIALIZED_FREQUENCY_VALUE, "Invalid frequency {0}", frequency);
        return static_cast<TimeType>(cycles * 1000. / frequency);
    }

    template <typename RawMetadata>
    static BarriersSet getWaitBarriersFromTask(const RawMetadata* task) {
        if (task == nullptr) {
            return {};
        }
        const auto barrierList = task->waitBarriers();
        return BarriersSet(barrierList->cbegin(), barrierList->cend());
    }

    template <typename RawMetadata>
    static BarriersSet getUpdateBarriersFromTask(const RawMetadata* task) {
        if (task == nullptr) {
            return {};
        }
        const auto barrierList = task->updateBarriers();
        return BarriersSet(barrierList->cbegin(), barrierList->cend());
    }

    static auto getBarriersIntersection(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection;
        std::set_intersection(set1.cbegin(), set1.cend(), set2.cbegin(), set2.cend(),
                              std::back_inserter(barriersIntersection));
        return barriersIntersection;
    }

private:
    static bool isSetIntersectionEmpty(const BarriersSet& set1, const BarriersSet& set2) {
        std::vector<BarrierIdType> barriersIntersection = getBarriersIntersection(set1, set2);
        VPUX_THROW_UNLESS(barriersIntersection.size() < 2, "Tasks should have at most 1 common barrier, but got {0}",
                          barriersIntersection.size());
        return barriersIntersection.empty();
    }

    static TaskInfo::ExecType convertToTaskExec(ExecutorType exec) {
        switch (exec) {
        case ExecutorType::DMA_SW:
        case ExecutorType::DMA_HW:
            return TaskInfo::ExecType::DMA;
        case ExecutorType::DPU:
            return TaskInfo::ExecType::DPU;
        case ExecutorType::ACTSHAVE:
            return TaskInfo::ExecType::SW;
        case ExecutorType::M2I:
            return TaskInfo::ExecType::M2I;
        default:
            VPUX_THROW("Unknown ExecutorType value");
        }
    }

protected:
    template <typename RawMetadata>
    RawProfilingRecord(const RawMetadata* metadata, const BarriersSet& wBarriers, const BarriersSet& uBarriers)
            : _name(metadata->name()->str()),
              _layerType(deserializeTaskName(metadata->name()->str()).layerType),
              _waitBarriers(wBarriers),
              _updateBarriers(uBarriers) {
    }

    template <typename RawMetadata>
    RawProfilingRecord(const RawMetadata* metadata)
            : RawProfilingRecord(metadata, getWaitBarriersFromTask(metadata), getUpdateBarriersFromTask(metadata)) {
    }

    RawProfilingRecord(const std::string& name, const std::string& layerType, const BarriersSet& wBarriers = {},
                       const BarriersSet& uBarriers = {})
            : _name(name), _layerType(layerType), _waitBarriers(wBarriers), _updateBarriers(uBarriers) {
    }

private:
    RawProfilingRecord(const std::string& cleanName, const std::string& layerType, const MVCNN::Task* task)
            : _name(cleanName), _layerType(layerType) {
        VPUX_THROW_WHEN(task == nullptr, "Invalid task");
        VPUX_THROW_WHEN(task->name() == nullptr, "Invalid task name");
        VPUX_THROW_WHEN(task->associated_barriers() == nullptr, "Task should have associated barriers");

        auto barriers = task->associated_barriers();
        if (auto wBarriers = barriers->wait_barriers()) {
            _waitBarriers = BarriersSet(wBarriers->cbegin(), wBarriers->cend());
        }
        if (auto uBarriers = barriers->update_barriers()) {
            _updateBarriers = BarriersSet(uBarriers->cbegin(), uBarriers->cend());
        }
    }

protected:
    virtual ~RawProfilingRecord() = default;

public:
    bool isDirectPredecessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_updateBarriers, other._waitBarriers);
    }

    bool isDirectSuccessor(const RawProfilingRecord& other) const {
        return !isSetIntersectionEmpty(_waitBarriers, other._updateBarriers);
    }

    virtual ExecutorType getExecutorType() const = 0;

    const BarriersSet& getWaitBarriers() const {
        return _waitBarriers;
    }

    const BarriersSet& getUpdateBarriers() const {
        return _updateBarriers;
    }

    std::string getOriginalName() const {
        return _name;
    }

    virtual std::string getTaskName() const {
        return _name;
    }

    std::string getLayerType() const {
        return _layerType;
    }

    virtual TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const {
        TaskInfo taskInfo{};
        taskInfo.name = getTaskName();
        taskInfo.layer_type = getLayerType();
        taskInfo.exec_type = convertToTaskExec(getExecutorType());
        taskInfo.start_time_ns = static_cast<uint64_t>(getStartTime(frequenciesSetup));
        taskInfo.duration_ns = static_cast<uint64_t>(getDuration(frequenciesSetup));
        return taskInfo;
    }

    virtual void checkDataOrDie() const {
        VPUX_THROW("checkDataOrDie not implemented");
    }

    virtual void sanitize(vpux::Logger&, FrequenciesSetup) const {
        // do nothing in base
    }

    virtual TimeType getStartTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const = 0;

    virtual TimeType getDuration(FrequenciesSetup frequenciesSetup) const {
        return getFinishTime(frequenciesSetup) - getStartTime(frequenciesSetup);
    }

private:
    const std::string _name;
    const std::string _layerType;

    BarriersSet _waitBarriers;
    BarriersSet _updateBarriers;
};

using RawProfilingRecordPtr = std::shared_ptr<RawProfilingRecord>;
using RawProfilingRecords = std::vector<RawProfilingRecordPtr>;

template <class RecordDataType>
class RawProfilingDMARecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    using ExtendedTimestampType = uint64_t;

protected:
    RawProfilingDMARecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata,
                          const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingRecord(metadata, wBarriers, uBarriers),
              DebugFormattableRecordMixin(inMemoryOffset),
              _record(record) {
    }

    RawProfilingDMARecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata), DebugFormattableRecordMixin(inMemoryOffset), _record(record) {
    }

public:
    void sanitize(vpux::Logger& log, FrequenciesSetup frequenciesSetup) const override {
        const auto dmaDurationNs = getDuration(frequenciesSetup);
        // Maximum 4MB  transfer
        const uint64_t maxTransferSize = 1024LL * 1024LL * 4LL;
        // guard band (DMA transfers seem to have significant variance in duration due to
        // variable DDR latency)
        const uint64_t guardBand = 10;
        // clock cycles upper limit (for 32 bytes/cycle) extended by guardBand margin
        const uint64_t maxTicks = static_cast<ExtendedTimestampType>(guardBand * maxTransferSize / 32);
        if (dmaDurationNs > convertTicksToNs(maxTicks, FrequenciesSetup::MIN_FREQ_MHZ)) {
            log.warning("Too long execution time of DMA task");
        }
    }

    size_t getDebugDataSize() const override {
        return sizeof(RecordDataType);
    }

protected:
    const RecordDataType _record;
};

template <class RecordDataType>
class RawProfilingSwDmaRecord : public RawProfilingDMARecord<RecordDataType> {
public:
    using RawProfilingDMARecord<RecordDataType>::RawProfilingDMARecord;
    using ColDesc = DebugFormattableRecordMixin::ColDesc;
    using BarriersSet = RawProfilingRecord::BarriersSet;

protected:
    RawProfilingSwDmaRecord(const RecordDataType& record, const ProfilingFB::DMATask* metadata,
                            const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingDMARecord<RecordDataType>(record, metadata, wBarriers, uBarriers, inMemoryOffset) {
    }

    ColDesc getColDesc() const override {
        return {{"Begin tstamp", COL_WIDTH_64}, {"End tstamp", COL_WIDTH_64}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto cols = getColDesc();
        outStream << std::setw(cols[0].second) << this->_record.startCycle << std::setw(cols[1].second)
                  << this->_record.endCycle;
    }
};

class RawProfilingDMA27Record : public RawProfilingSwDmaRecord<DMA27Data_t> {
public:
    explicit RawProfilingDMA27Record(const DMA27Data_t& record, const ProfilingFB::DMATask* metadata,
                                     const BarriersSet& wBarriers, const BarriersSet& uBarriers, size_t inMemoryOffset)
            : RawProfilingSwDmaRecord<DMA27Data_t>(record, metadata, wBarriers, uBarriers, inMemoryOffset) {
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DMA_SW;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.startCycle, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.endCycle, frequenciesSetup.profClk);
    }
};

class RawProfilingDMA40Record : public RawProfilingDMARecord<HwpDma40Data_t> {
public:
    explicit RawProfilingDMA40Record(const HwpDma40Data_t& record, const ProfilingFB::DMATask* metadata,
                                     size_t inMemoryOffset)
            : RawProfilingDMARecord<HwpDma40Data_t>(record, metadata, inMemoryOffset) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_UNLESS(_record.rsvd == 0, "Reserved value must contain 0.");
        VPUX_THROW_WHEN(_record.desc_addr == 0, "Invalid DMA descriptor address.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DMA_HW;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.start_time, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_record.finish_time, frequenciesSetup.profClk);
    }

protected:
    ColDesc getColDesc() const override {
        return {
                {"JDESC_ADDR", COL_WIDTH_64},
                {"JFETCH_TIME", COL_WIDTH_64},
                {"JREADY_TIME", COL_WIDTH_64},
                {"JSTART_TIME", COL_WIDTH_64},
                {"JWDONE_TIME", COL_WIDTH_64},
                {"JFINISH_TIME", COL_WIDTH_64},
                {"JLA_ID", 7},
                {"JCH_ID", 7},
                {"RSVD", 7},
                {"JRSTALL_CNT", 13},
                {"JWSTALL_CNT", 13},
                {"JTWBYTES_CNT", 14},
                {"JCHCYCLE_CNT", 14},
        };
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto cols = getColDesc();
        // std::ostream recognize uint8_t as char and print character instead of value, so explicitly cast for printing
        // purpose
        const auto to_int = [](uint8_t val) {
            return static_cast<uint16_t>(val);
        };
        outStream << std::setw(cols[0].second) << _record.desc_addr << std::setw(cols[1].second) << _record.fetch_time
                  << std::setw(cols[2].second) << _record.ready_time << std::setw(cols[3].second) << _record.start_time
                  << std::setw(cols[4].second) << _record.wdone_time << std::setw(cols[5].second) << _record.finish_time
                  << std::setw(cols[6].second) << to_int(_record.la_id) << std::setw(cols[7].second)
                  << to_int(_record.ch_id) << std::setw(cols[8].second) << _record.rsvd << std::setw(cols[9].second)
                  << _record.rstall_cnt << std::setw(cols[10].second) << _record.wstall_cnt
                  << std::setw(cols[11].second) << _record.twbytes_cnt << std::setw(cols[12].second)
                  << _record.chcycle_cnt;
    }
};

class RawProfilingDPURecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
protected:
    RawProfilingDPURecord(const ProfilingFB::DPUTask* metadata, uint32_t variantId, size_t inMemoryOffset,
                          uint32_t inClusterIndex)
            : RawProfilingRecord(metadata),
              DebugFormattableRecordMixin(inMemoryOffset, inClusterIndex),
              _bufferId(metadata->bufferId()),
              _clusterId(metadata->clusterId()),
              _taskId(metadata->taskId()),
              _variantId(variantId) {
    }

    virtual double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const = 0;

public:
    std::string getTaskName() const override {
        // adding variant suffix as it is not stored in meta data
        return getOriginalName() + "/" + VARIANT_LEVEL_PROFILING_SUFFIX + "_" + std::to_string(_variantId);
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.clusterId = _clusterId;
        profInfoItem.isSubtask = true;
        return profInfoItem;
    }

    void sanitize(vpux::Logger& log, FrequenciesSetup frequenciesSetup) const override {
        const auto dpuExecutionTime = this->getDuration(frequenciesSetup);
        const uint64_t maxKernel = 11 * 11;
        const uint64_t maxElem = 2ll * 1024ll * 1024ll;
        const uint64_t maxChannels = 8192;
        const uint64_t maxCycles = maxKernel * maxElem * maxChannels / 256;
        const auto frequency = this->getTaskDurationClock(frequenciesSetup);
        const auto maxNs = convertTicksToNs(maxCycles, frequency);
        if (maxNs < dpuExecutionTime) {
            log.warning("Too long execution time of DPU task");
        }
    }

    size_t getClusterId() {
        return _clusterId;
    }

    uint32_t getTaskId() {
        return _taskId;
    }

protected:
    const uint32_t _bufferId;
    const uint32_t _clusterId;
    const uint32_t _taskId;
    const uint32_t _variantId;
};

class RawProfilingDPUHW27Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW27Record(HwpDpu27Mode0Data_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterIndex)
            : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterIndex), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
        VPUX_THROW_UNLESS(_timestamps.reserved3 == 0 && _timestamps.reserved8 == 0, "Reserved values must contain 0.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DPU;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        const auto max28BitTime = convertTicksToNs(0x0FFFFFFFull, frequenciesSetup.vpuClk);
        const auto noOverflowSubtract = [](TimeType first, TimeType second, TimeType max) -> TimeType {
            return first - second + ((first < second) ? max : 0);
        };
        return noOverflowSubtract(convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.vpuClk),
                                  convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk), max28BitTime);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.vpuClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(HwpDpu27Mode0Data_t);
    }

protected:
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},
                {"Cluster ID", COL_WIDTH_64},
                {"Buffer offset", COL_WIDTH_64},
                {"IDU dur", COL_WIDTH_32},
                {"IDU tstamp", COL_WIDTH_32},
                {"SWE ID", 7},
                {"Rvd", 4},
                {"ODU dur", COL_WIDTH_32},
                {"ODU tstamp", COL_WIDTH_32},
                {"Rvd", 7}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();
        const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

        outStream << std::setw(hwpDpuCol[0].second) << _bufferId << std::setw(hwpDpuCol[1].second) << _clusterId
                  << std::setw(hwpDpuCol[2].second) << bufferOffsetBytes << std::setw(hwpDpuCol[3].second)
                  << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[4].second) << _timestamps.idu_tstamp
                  << std::setw(hwpDpuCol[5].second) << _timestamps.sve_id << std::setw(hwpDpuCol[6].second)
                  << _timestamps.reserved3 << std::setw(hwpDpuCol[7].second) << _timestamps.odu_wl_duration
                  << std::setw(hwpDpuCol[8].second) << _timestamps.odu_tstamp << std::setw(hwpDpuCol[9].second)
                  << _timestamps.reserved8;
    }

private:
    const HwpDpu27Mode0Data_t _timestamps;
};

class RawProfilingDPUHW40Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW40Record(HwpDpuIduOduData_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterIndex)
            : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterIndex), _timestamps(timestamps) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
                        "Invalid DPU task duration");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::DPU;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
               convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(HwpDpuIduOduData_t);
    }

protected:
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.dpuClk;
    }

    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},
                {"Cluster ID", COL_WIDTH_64},
                {"Buffer offset", COL_WIDTH_64},
                {"IDU dur", COL_WIDTH_32},
                {"IDU tstamp", COL_WIDTH_64},
                {"IDU WL ID", 11},
                {"IDU DPU ID", 12},
                {"ODU dur", COL_WIDTH_32},
                {"ODU tstamp", COL_WIDTH_64},
                {"ODU WL ID", 11},
                {"ODU DPU ID", 12}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto hwpDpuCol = getColDesc();
        const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

        outStream << std::setw(hwpDpuCol[0].second) << _bufferId << std::setw(hwpDpuCol[1].second) << _clusterId
                  << std::setw(hwpDpuCol[2].second) << bufferOffsetBytes << std::setw(hwpDpuCol[3].second)
                  << _timestamps.idu_wl_duration << std::setw(hwpDpuCol[4].second) << _timestamps.idu_tstamp
                  << std::setw(hwpDpuCol[5].second) << _timestamps.idu_wl_id << std::setw(hwpDpuCol[6].second)
                  << _timestamps.idu_dpu_id << std::setw(hwpDpuCol[5].second) << _timestamps.odu_wl_duration
                  << std::setw(hwpDpuCol[7].second) << _timestamps.odu_tstamp << std::setw(hwpDpuCol[8].second)
                  << _timestamps.odu_wl_id << std::setw(hwpDpuCol[9].second) << _timestamps.odu_dpu_id;
    }

protected:
    const HwpDpuIduOduData_t _timestamps;
};

class RawProfilingDPUHW50Record : public RawProfilingDPUHW40Record {
public:
    using RawProfilingDPUHW40Record::RawProfilingDPUHW40Record;

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
               convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.profClk);
    }

protected:
    double getTaskDurationClock(FrequenciesSetup frequenciesSetup) const override {
        return frequenciesSetup.profClk;
    }
};

class RawProfilingACTRecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTRecord(ActShaveData_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata),
              DebugFormattableRecordMixin(inMemoryOffset, metadata->dataIndex()),
              _data(data),
              _bufferId(metadata->bufferId()),
              _clusterId(metadata->clusterId()) {
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.clusterId = _clusterId;
        profInfoItem.total_cycles = _data.clockCycles;
        profInfoItem.stall_cycles = _data.stallCycles;
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::ACTSHAVE;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return getStartTime(frequenciesSetup) + getDuration(frequenciesSetup);
    }

    TimeType getDuration(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.duration, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(ActShaveData_t);
    }

protected:
    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32}, {"Cluster ID", COL_WIDTH_64}, {"Buffer offset", COL_WIDTH_64},
                {"Begin", COL_WIDTH_64},     {"Duration", COL_WIDTH_32},   {"Stall", COL_WIDTH_32},
                {"Executed", COL_WIDTH_32},  {"Clock", COL_WIDTH_32},      {"Branch", COL_WIDTH_32}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto actShaveCol = getColDesc();
        const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

        outStream << std::setw(actShaveCol[0].second) << _bufferId << std::setw(actShaveCol[1].second) << _clusterId
                  << std::setw(actShaveCol[2].second) << bufferOffsetBytes << std::setw(actShaveCol[3].second)
                  << _data.begin << std::setw(actShaveCol[4].second) << _data.duration
                  << std::setw(actShaveCol[5].second) << _data.stallCycles << std::setw(actShaveCol[6].second)
                  << _data.executedInstructions << std::setw(actShaveCol[7].second) << _data.clockCycles
                  << std::setw(actShaveCol[8].second) << _data.branchTaken;
    }

private:
    const ActShaveData_t _data;
    const uint32_t _bufferId;
    const uint32_t _clusterId;
};

class RawProfilingACTExRecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTExRecord(ActShaveDataEx_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata),
              DebugFormattableRecordMixin(inMemoryOffset, metadata->dataIndex()),
              _data(data),
              _bufferId(metadata->bufferId()),
              _clusterId(metadata->clusterId()) {
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
        profInfoItem.clusterId = _clusterId;
        profInfoItem.total_cycles = _data.clockCycles;
        profInfoItem.stall_cycles = _data.lsu0Stalls + _data.lsu1Stalls + _data.instStalls;
        profInfoItem.stall_counters = {_data.lsu0Stalls, _data.lsu1Stalls, _data.instStalls};
        return profInfoItem;
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::ACTSHAVE;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return getStartTime(frequenciesSetup) + getDuration(frequenciesSetup);
    }

    TimeType getDuration(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.duration, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(ActShaveDataEx_t);
    }

protected:
    ColDesc getColDesc() const override {
        return {{"Buffer ID", COL_WIDTH_32},   {"Cluster ID", COL_WIDTH_64},  {"Buffer offset", COL_WIDTH_64},
                {"Begin", COL_WIDTH_64},       {"Duration", COL_WIDTH_32},    {"Executed", COL_WIDTH_32},
                {"Clock", COL_WIDTH_32},       {"LSU0 Stalls", COL_WIDTH_64}, {"LSU1 Stalls", COL_WIDTH_64},
                {"Instr Stalls", COL_WIDTH_64}};
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto actShaveCol = getColDesc();
        const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

        outStream << std::setw(actShaveCol[0].second) << _bufferId << std::setw(actShaveCol[1].second) << _clusterId
                  << std::setw(actShaveCol[2].second) << bufferOffsetBytes << std::setw(actShaveCol[3].second)
                  << _data.begin << std::setw(actShaveCol[4].second) << _data.duration
                  << std::setw(actShaveCol[5].second) << _data.executedInstructions << std::setw(actShaveCol[6].second)
                  << _data.clockCycles << std::setw(actShaveCol[7].second) << _data.lsu0Stalls
                  << std::setw(actShaveCol[8].second) << _data.lsu1Stalls << std::setw(actShaveCol[9].second)
                  << _data.instStalls;
    }

private:
    const ActShaveDataEx_t _data;
    const uint32_t _bufferId;
    const uint32_t _clusterId;
};

class RawProfilingM2IRecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingM2IRecord(M2IData_t data, const ProfilingFB::M2ITask* metadata, size_t inMemoryOffset)
            : RawProfilingRecord(metadata), DebugFormattableRecordMixin(inMemoryOffset), _data(data) {
    }

    void checkDataOrDie() const override {
        VPUX_THROW_WHEN(_data.startTime == 0 && _data.finishTime == 0, "Can't process M2I profiling data.");
    }

    ExecutorType getExecutorType() const override {
        return ExecutorType::M2I;
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.startTime, frequenciesSetup.profClk);
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return convertTicksToNs(_data.finishTime, frequenciesSetup.profClk);
    }

    size_t getDebugDataSize() const override {
        return sizeof(M2IData_t);
    }

protected:
    ColDesc getColDesc() const override {
        return {
                {"Fetch tstamp", COL_WIDTH_64}, {"Ready tstamp", COL_WIDTH_64},  {"Start tstamp", COL_WIDTH_64},
                {"Done tstamp", COL_WIDTH_64},  {"Finish tstamp", COL_WIDTH_64}, {"LA", COL_WIDTH_32},
                {"Parent id", COL_WIDTH_32},    {"RStall cnt", COL_WIDTH_32},    {"WStall cnt", COL_WIDTH_32},
                {"WRCycle cnt", COL_WIDTH_32},  {"RDCycle cnt", COL_WIDTH_32},
        };
    }

    void printDebugInfo(std::ostream& outStream) const override {
        const auto cols = getColDesc();

        outStream << std::setw(cols[0].second) << _data.fetchTime << std::setw(cols[1].second) << _data.readyTime
                  << std::setw(cols[2].second) << _data.startTime << std::setw(cols[3].second) << _data.doneTime
                  << std::setw(cols[4].second) << _data.finishTime << std::setw(cols[5].second) << _data.linkAgentID
                  << std::setw(cols[6].second) << _data.parentID << std::setw(cols[7].second) << _data.RStallCount
                  << std::setw(cols[8].second) << _data.WStallCount << std::setw(cols[9].second) << _data.WRCycleCount
                  << std::setw(cols[10].second) << _data.RDCycleCount;
    }

private:
    const M2IData_t _data;
};

class ArrayRecord : public RawProfilingRecord {
public:
    ArrayRecord(const std::string name, const RawProfilingRecords& records)
            : RawProfilingRecord(name, records.front()->getLayerType(), records.front()->getWaitBarriers(),
                                 records.front()->getUpdateBarriers()),
              _records(records) {
    }

    TimeType getStartTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_records.cbegin(), _records.cend(), std::numeric_limits<TimeType>::max(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::min(a, variant->getStartTime(frequenciesSetup));
                               });
    }

    TimeType getFinishTime(FrequenciesSetup frequenciesSetup) const override {
        return std::accumulate(_records.cbegin(), _records.cend(), std::numeric_limits<TimeType>::min(),
                               [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                   return std::max(a, variant->getFinishTime(frequenciesSetup));
                               });
    }

    ExecutorType getExecutorType() const override {
        VPUX_THROW_WHEN(_records.size() == 0, "Empty ArrayRecord");
        return _records.front()->getExecutorType();
    }

    TaskInfo getTaskInfo(FrequenciesSetup frequenciesSetup) const override {
        VPUX_THROW_WHEN(_records.size() == 0, "Empty ArrayRecord");
        auto taskInfo = _records.front()->getTaskInfo(frequenciesSetup);
        taskInfo.name = getTaskName();
        taskInfo.isSubtask = false;
        taskInfo.start_time_ns = getStartTime(frequenciesSetup);
        taskInfo.duration_ns = getDuration(frequenciesSetup);
        //  Ignore cycle information. Not used for DPU
        return taskInfo;
    }

protected:
    const RawProfilingRecords _records;
};

}  // namespace vpux::profiling
