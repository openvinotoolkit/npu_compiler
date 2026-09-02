//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/utils/profiling/parser/records.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/utils.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>

using namespace vpux::profiling;

namespace {

constexpr int COL_WIDTH_32 = 11;
constexpr int COL_WIDTH_64 = 19;

std::string memoryKindToString(ProfilingFB::MemoryKind memoryKind) {
    return ProfilingFB::EnumNameMemoryKind(memoryKind);
}

TaskInfo::ExecType convertToTaskExec(ExecutorType exec) {
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

TimeType convertTicksToNs(uint64_t cycles, double frequency) {
    VPUX_THROW_WHEN(frequency == UNINITIALIZED_FREQUENCY_VALUE, "Invalid frequency {0}", frequency);
    return static_cast<TimeType>(cycles * 1000. / frequency);
}

}  // namespace

namespace vpux::profiling {

//
// DebugFormattableRecordMixin
//

void DebugFormattableRecordMixin::printDebugHeader(std::ostream& os) {
    const auto columns = this->getColDesc();
    for (const std::pair<std::string, int>& p : columns) {
        os << std::setw(p.second) << p.first;
    }
}

//
// RawProfilingRecord
//

RawProfilingRecord::RawProfilingRecord(const std::string& name, const std::string& layerType,
                                       const BarriersSet& wBarriers, const BarriersSet& uBarriers)
        : _name(name), _layerType(layerType), _waitBarriers(wBarriers), _updateBarriers(uBarriers) {
}

TaskInfo RawProfilingRecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    TaskInfo taskInfo{};
    taskInfo.name = getTaskName();
    taskInfo.layer_type = getLayerType();
    taskInfo.exec_type = convertToTaskExec(getExecutorType());
    taskInfo.start_time_ns = static_cast<uint64_t>(getStartTime(frequenciesSetup));
    taskInfo.duration_ns = static_cast<uint64_t>(getDuration(frequenciesSetup));
    taskInfo.customArgs = getCustomArgs(frequenciesSetup);
    return taskInfo;
}

void RawProfilingRecord::checkData(bool failOnError, Logger& log) const {
    VPUX_UNUSED(failOnError);
    VPUX_UNUSED(log);
    VPUX_THROW("checkData not implemented");
}

void RawProfilingRecord::sanitize(Logger&, const FrequenciesSetup&) const {
    // do nothing in base
}

TimeType RawProfilingRecord::getDuration(const FrequenciesSetup& frequenciesSetup) const {
    return getFinishTime(frequenciesSetup) - getStartTime(frequenciesSetup);
}

CustomArgsVector RawProfilingRecord::getCustomArgs(const FrequenciesSetup&) const {
    return {};
}

//
// RawProfilingDMARecord
//

RawProfilingDMARecord::RawProfilingDMARecord(const ProfilingFB::DMATask* metadata, size_t inMemoryOffset,
                                             const BarriersSet& wBarriers, const BarriersSet& uBarriers)
        : RawProfilingRecord(metadata, wBarriers, uBarriers),
          DebugFormattableRecordMixin(inMemoryOffset),
          _portId(metadata->portId()),
          _channelType(metadata->channelType()),
          _sourceMemoryKind(metadata->sourceMemoryKind()),
          _destinationMemoryKind(metadata->destinationMemoryKind()),
          _tensorShapeInfo(metadata->tensorShapeInfo() ? metadata->tensorShapeInfo()->UnPack() : nullptr),
          _tensorStrideInfo(metadata->tensorStrideInfo() ? metadata->tensorStrideInfo()->UnPack() : nullptr),
          _gatherIndices(metadata->gatherIndices()),
          _dynamicStridesInput(metadata->dynamicStridesInput()),
          _dynamicStridesOutput(metadata->dynamicStridesOutput()) {
}

TaskInfo RawProfilingDMARecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
    profInfoItem.port_id = _portId;
    profInfoItem.channel_type = _channelType;
    return profInfoItem;
}

void RawProfilingDMARecord::sanitize(Logger& log, const FrequenciesSetup& frequenciesSetup) const {
    const auto dmaDurationNs = getDuration(frequenciesSetup);
    // Maximum 4MB  transfer
    const uint64_t maxTransferSize = 1024LL * 1024LL * 4LL;
    // guard band (DMA transfers seem to have significant variance in duration due to
    // variable DDR latency)
    const uint64_t guardBand = 10;
    // clock cycles upper limit (for 32 bytes/cycle) extended by guardBand margin
    const uint64_t maxTicks = guardBand * maxTransferSize / 32;
    if (dmaDurationNs > convertTicksToNs(maxTicks, FrequenciesSetup::MIN_FREQ_MHZ)) {
        log.warning("Too long execution time of DMA task");
    }
}

CustomArgsVector RawProfilingDMARecord::getCustomArgs(const FrequenciesSetup&) const {
    CustomArgsVector customArgs;
    if (_sourceMemoryKind) {
        customArgs.push_back({"Source memory:", memoryKindToString(_sourceMemoryKind.value())});
    }
    if (_destinationMemoryKind) {
        customArgs.push_back({"Destination memory:", memoryKindToString(_destinationMemoryKind.value())});
    }

    if (_tensorShapeInfo) {
        auto const& [inputs, output] = *_tensorShapeInfo;
        customArgs.push_back({"Input tensor shape", to_string(inputs, _gatherIndices, _dynamicStridesInput)});
        customArgs.push_back({"Output tensor shape", to_string(output, 0, _dynamicStridesOutput)});
    }

    if (_tensorStrideInfo) {
        auto const& [inputs, outputs] = *_tensorStrideInfo;
        customArgs.push_back({"Input tensor strides", to_string(inputs, 0, _dynamicStridesInput)});
        customArgs.push_back({"Output tensor strides", to_string(outputs, 0, _dynamicStridesOutput)});
    }

    return customArgs;
}

//
// RawProfilingDMA27Record
//

RawProfilingDMA27Record::RawProfilingDMA27Record(const DMA27Data_t& record, const ProfilingFB::DMATask* metadata,
                                                 size_t inMemoryOffset, const BarriersSet& wBarriers,
                                                 const BarriersSet& uBarriers)
        : RawProfilingDMARecord(metadata, inMemoryOffset, wBarriers, uBarriers), _record(record) {
}

ExecutorType RawProfilingDMA27Record::getExecutorType() const {
    return ExecutorType::DMA_SW;
}

TimeType RawProfilingDMA27Record::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_record.startCycle, frequenciesSetup.profClk);
}

TimeType RawProfilingDMA27Record::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_record.endCycle, frequenciesSetup.profClk);
}

size_t RawProfilingDMA27Record::getDebugDataSize() const {
    return sizeof(DMA27Data_t);
}

void RawProfilingDMA27Record::printDebugInfo(std::ostream& outStream) const {
    const auto cols = getColDesc();
    outStream << std::setw(cols[0].second) << this->_record.startCycle << std::setw(cols[1].second)
              << this->_record.endCycle;
}

DebugFormattableRecordMixin::ColDesc RawProfilingDMA27Record::getColDesc() const {
    return {{"Begin tstamp", COL_WIDTH_64}, {"End tstamp", COL_WIDTH_64}};
}

//
// RawProfilingDMA40Record
//

RawProfilingDMA40Record::RawProfilingDMA40Record(const HwpDma40Data_t& record, const ProfilingFB::DMATask* metadata,
                                                 size_t inMemoryOffset)
        : RawProfilingDMARecord(metadata, inMemoryOffset), _record(record) {
}

void RawProfilingDMA40Record::checkData(bool failOnError, Logger& log) const {
    warnOrFail(failOnError, log, _record.rsvd != 0, "Reserved value must contain 0.");
    warnOrFail(failOnError, log, _record.desc_addr == 0, "Invalid DMA descriptor address.");
}

ExecutorType RawProfilingDMA40Record::getExecutorType() const {
    return ExecutorType::DMA_HW;
}

TimeType RawProfilingDMA40Record::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_record.start_time, frequenciesSetup.profClk);
}

TimeType RawProfilingDMA40Record::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_record.wdone_time, frequenciesSetup.profClk);
}

size_t RawProfilingDMA40Record::getDebugDataSize() const {
    return sizeof(HwpDma40Data_t);
}

CustomArgsVector RawProfilingDMA40Record::getCustomArgs(const FrequenciesSetup& frequenciesSetup) const {
    CustomArgsVector customArgs = RawProfilingDMARecord::getCustomArgs(frequenciesSetup);
    customArgs.push_back({"Address", llvm::formatv("{0:x}", _record.desc_addr)});
    customArgs.push_back({"Time to ready", formatDuration(convertTicksToNs(_record.ready_time - _record.fetch_time,
                                                                           frequenciesSetup.profClk))});
    customArgs.push_back({"Time to start", formatDuration(convertTicksToNs(_record.start_time - _record.ready_time,
                                                                           frequenciesSetup.profClk))});
    customArgs.push_back({"Transfer time", formatDuration(convertTicksToNs(_record.wdone_time - _record.start_time,
                                                                           frequenciesSetup.profClk))});
    customArgs.push_back({"Time to finish", formatDuration(convertTicksToNs(_record.finish_time - _record.wdone_time,
                                                                            frequenciesSetup.profClk))});
    customArgs.push_back({"Link agent id", std::to_string(_record.la_id)});
    customArgs.push_back({"Channel id", std::to_string(_record.ch_id)});
    customArgs.push_back({"Read stall cycles", std::to_string(_record.rstall_cnt)});
    customArgs.push_back({"Write stall cycles", std::to_string(_record.wstall_cnt)});
    customArgs.push_back({"Total bytes", std::to_string(_record.twbytes_cnt)});
    customArgs.push_back({"Total cycles", std::to_string(_record.chcycle_cnt)});
    return customArgs;
}

void RawProfilingDMA40Record::printDebugInfo(std::ostream& outStream) const {
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
              << _record.rstall_cnt << std::setw(cols[10].second) << _record.wstall_cnt << std::setw(cols[11].second)
              << _record.twbytes_cnt << std::setw(cols[12].second) << _record.chcycle_cnt;
}

DebugFormattableRecordMixin::ColDesc RawProfilingDMA40Record::getColDesc() const {
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

//
// RawProfilingDPURecord
//

RawProfilingDPURecord::RawProfilingDPURecord(const ProfilingFB::DPUTask* metadata, uint32_t variantId,
                                             size_t inMemoryOffset, uint32_t inClusterIndex,
                                             std::shared_ptr<const TensorInfo> tensorInfo,
                                             std::unique_ptr<const DPUVariantInfo> variantInfo)
        : RawProfilingRecord(metadata),
          DebugFormattableRecordMixin(inMemoryOffset, inClusterIndex),
          _bufferId(metadata->bufferId()),
          _clusterId(metadata->clusterId()),
          _taskId(metadata->taskId()),
          _variantId(variantId),
          _variantInfo(std::move(variantInfo)),
          _tensorInfo(std::move(tensorInfo)) {
}

std::string RawProfilingDPURecord::getTaskName() const {
    // adding variant suffix as it is not stored in meta data
    return getOriginalName() + "/" + VARIANT_LEVEL_PROFILING_SUFFIX + "_" + std::to_string(_variantId);
}

TaskInfo RawProfilingDPURecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
    profInfoItem.clusterId = _clusterId;
    profInfoItem.isSubtask = true;
    profInfoItem.variant_id = _variantId;
    return profInfoItem;
}

void RawProfilingDPURecord::sanitize(Logger& log, const FrequenciesSetup& frequenciesSetup) const {
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

CustomArgsVector RawProfilingDPURecord::getCustomArgs(const FrequenciesSetup&) const {
    if (_variantInfo == nullptr) {
        return {};
    }
    return to_custom_args(*_variantInfo);
}

const std::shared_ptr<const TensorInfo>& RawProfilingDPURecord::getTensorInfo() const {
    return _tensorInfo;
}

//
// RawProfilingDPUHW27Record
//

RawProfilingDPUHW27Record::RawProfilingDPUHW27Record(HwpDpu27Mode0Data_t timestamps,
                                                     const ProfilingFB::DPUTask* metadata, uint32_t variantId,
                                                     size_t inMemoryOffset, uint32_t inClusterIndex,
                                                     std::shared_ptr<const TensorInfo> tensorInfo,
                                                     std::unique_ptr<const DPUVariantInfo> variantInfo)
        : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterIndex, std::move(tensorInfo),
                                std::move(variantInfo)),
          _timestamps(timestamps) {
}

void RawProfilingDPUHW27Record::checkData(bool failOnError, Logger& log) const {
    warnOrFail(failOnError, log, _timestamps.idu_wl_duration == 0 && _timestamps.odu_wl_duration == 0,
               "Invalid DPU task duration");
    warnOrFail(failOnError, log, _timestamps.reserved3 != 0 || _timestamps.reserved8 != 0,
               "Reserved values must contain 0");
}

ExecutorType RawProfilingDPUHW27Record::getExecutorType() const {
    return ExecutorType::DPU;
}

TimeType RawProfilingDPUHW27Record::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    const auto max28BitTime = convertTicksToNs(0x0FFFFFFFull, frequenciesSetup.vpuClk);
    const auto noOverflowSubtract = [](TimeType first, TimeType second, TimeType max) -> TimeType {
        return first - second + ((first < second) ? max : 0);
    };
    return noOverflowSubtract(convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.vpuClk),
                              convertTicksToNs(_timestamps.idu_wl_duration, frequenciesSetup.dpuClk), max28BitTime);
}

TimeType RawProfilingDPUHW27Record::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.vpuClk);
}

double RawProfilingDPUHW27Record::getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const {
    return frequenciesSetup.dpuClk;
}

size_t RawProfilingDPUHW27Record::getDebugDataSize() const {
    return sizeof(HwpDpu27Mode0Data_t);
}

void RawProfilingDPUHW27Record::printDebugInfo(std::ostream& outStream) const {
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

DebugFormattableRecordMixin::ColDesc RawProfilingDPUHW27Record::getColDesc() const {
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

//
// RawProfilingDPUHW40Record
//

RawProfilingDPUHW40Record::RawProfilingDPUHW40Record(HwpDpuIduOduData_t timestamps,
                                                     const ProfilingFB::DPUTask* metadata, uint32_t variantId,
                                                     size_t inMemoryOffset, uint32_t inClusterIndex,
                                                     std::shared_ptr<const TensorInfo> tensorInfo,
                                                     std::unique_ptr<const DPUVariantInfo> variantInfo)
        : RawProfilingDPURecord(metadata, variantId, inMemoryOffset, inClusterIndex, std::move(tensorInfo),
                                std::move(variantInfo)),
          _timestamps(timestamps) {
}

ExecutorType RawProfilingDPUHW40Record::getExecutorType() const {
    return ExecutorType::DPU;
}

TimeType RawProfilingDPUHW40Record::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_timestamps.idu_tstamp, frequenciesSetup.profClk) -
           convertTicksToNs(_timestamps.idu_wl_duration, getTaskDurationClock(frequenciesSetup));
}

TimeType RawProfilingDPUHW40Record::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_timestamps.odu_tstamp, frequenciesSetup.profClk);
}

double RawProfilingDPUHW40Record::getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const {
    return frequenciesSetup.dpuClk;
}

size_t RawProfilingDPUHW40Record::getDebugDataSize() const {
    return sizeof(HwpDpuIduOduData_t);
}

void RawProfilingDPUHW40Record::printDebugInfo(std::ostream& outStream) const {
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

DebugFormattableRecordMixin::ColDesc RawProfilingDPUHW40Record::getColDesc() const {
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

//
// RawProfilingDPUHW50Record
//

double RawProfilingDPUHW50Record::getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const {
    return frequenciesSetup.profClk;
}

//
// RawProfilingACTRecord
//

RawProfilingACTRecord::RawProfilingACTRecord(ActShaveData_t data, const ProfilingFB::SWTask* metadata,
                                             size_t inMemoryOffset)
        : RawProfilingRecord(metadata),
          DebugFormattableRecordMixin(inMemoryOffset, metadata->dataIndex()),
          _data(data),
          _bufferId(metadata->bufferId()),
          _clusterId(metadata->clusterId()),
          _fifoId(metadata->fifoId()),
          _tensorInfo(metadata->tensorInfo() ? metadata->tensorInfo()->UnPack() : nullptr) {
}

void RawProfilingACTRecord::checkData(bool failOnError, Logger& log) const {
    warnOrFail(failOnError, log, _data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
}

TaskInfo RawProfilingACTRecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
    profInfoItem.clusterId = _clusterId;
    profInfoItem.fifo_id = _fifoId;
    profInfoItem.active_cycles = _data.executedInstructions;
    profInfoItem.stall_cycles = _data.clockCycles - _data.executedInstructions;
    return profInfoItem;
}

ExecutorType RawProfilingACTRecord::getExecutorType() const {
    return ExecutorType::ACTSHAVE;
}

TimeType RawProfilingACTRecord::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
}

TimeType RawProfilingACTRecord::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return getStartTime(frequenciesSetup) + getDuration(frequenciesSetup);
}

TimeType RawProfilingACTRecord::getDuration(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.duration, frequenciesSetup.profClk);
}

CustomArgsVector RawProfilingACTRecord::getCustomArgs(const FrequenciesSetup&) const {
    CustomArgsVector customArgs;
    customArgs.push_back({"Total cycles", std::to_string(_data.clockCycles)});
    customArgs.push_back({"Active cycles", std::to_string(_data.executedInstructions)});
    customArgs.push_back({"Stall cycles", std::to_string(_data.clockCycles - _data.executedInstructions)});

    if (_tensorInfo) {  // This check ensures backward compatibility
        auto tensorInfoArgs = to_custom_args(*_tensorInfo);
        customArgs.insert(customArgs.end(), tensorInfoArgs.begin(), tensorInfoArgs.end());
    }
    return customArgs;
}

size_t RawProfilingACTRecord::getDebugDataSize() const {
    return sizeof(ActShaveData_t);
}

DebugFormattableRecordMixin::ColDesc RawProfilingACTRecord::getColDesc() const {
    return {{"Buffer ID", COL_WIDTH_32}, {"Cluster ID", COL_WIDTH_64}, {"Buffer offset", COL_WIDTH_64},
            {"Begin", COL_WIDTH_64},     {"Duration", COL_WIDTH_32},   {"Stall", COL_WIDTH_32},
            {"Executed", COL_WIDTH_32},  {"Clock", COL_WIDTH_32},      {"Branch", COL_WIDTH_32}};
}

void RawProfilingACTRecord::printDebugInfo(std::ostream& outStream) const {
    const auto actShaveCol = getColDesc();
    const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

    outStream << std::setw(actShaveCol[0].second) << _bufferId << std::setw(actShaveCol[1].second) << _clusterId
              << std::setw(actShaveCol[2].second) << bufferOffsetBytes << std::setw(actShaveCol[3].second)
              << _data.begin << std::setw(actShaveCol[4].second) << _data.duration << std::setw(actShaveCol[5].second)
              << _data.stallCycles << std::setw(actShaveCol[6].second) << _data.executedInstructions
              << std::setw(actShaveCol[7].second) << _data.clockCycles << std::setw(actShaveCol[8].second)
              << _data.branchTaken;
}

//
// RawProfilingACTExRecord
//
RawProfilingACTExRecord::RawProfilingACTExRecord(ActShaveDataEx_t data, const ProfilingFB::SWTask* metadata,
                                                 size_t inMemoryOffset)
        : RawProfilingRecord(metadata),
          DebugFormattableRecordMixin(inMemoryOffset, metadata->dataIndex()),
          _data(data),
          _bufferId(metadata->bufferId()),
          _clusterId(metadata->clusterId()),
          _fifoId(metadata->fifoId()),
          _tensorInfo(metadata->tensorInfo() ? metadata->tensorInfo()->UnPack() : nullptr) {
}

void RawProfilingACTExRecord::checkData(bool failOnError, Logger& log) const {
    warnOrFail(failOnError, log, _data.begin == 0 && _data.duration == 0, "Can't process ACT profiling data.");
}

TaskInfo RawProfilingACTExRecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    auto profInfoItem = RawProfilingRecord::getTaskInfo(frequenciesSetup);
    profInfoItem.clusterId = _clusterId;
    profInfoItem.fifo_id = _fifoId;
    profInfoItem.active_cycles = _data.executedInstructions;
    profInfoItem.stall_cycles = _data.clockCycles - _data.executedInstructions;
    return profInfoItem;
}

ExecutorType RawProfilingACTExRecord::getExecutorType() const {
    return ExecutorType::ACTSHAVE;
}

TimeType RawProfilingACTExRecord::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.begin, frequenciesSetup.profClk);
}

TimeType RawProfilingACTExRecord::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return getStartTime(frequenciesSetup) + getDuration(frequenciesSetup);
}

TimeType RawProfilingACTExRecord::getDuration(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.duration, frequenciesSetup.profClk);
}

CustomArgsVector RawProfilingACTExRecord::getCustomArgs(const FrequenciesSetup&) const {
    CustomArgsVector customArgs;
    customArgs.push_back({"Total cycles", std::to_string(_data.clockCycles)});
    customArgs.push_back({"Active cycles", std::to_string(_data.executedInstructions)});
    customArgs.push_back({"Stall cycles", std::to_string(_data.clockCycles - _data.executedInstructions)});
    customArgs.push_back({"LSU0 stalls", std::to_string(_data.lsu0Stalls)});
    customArgs.push_back({"LSU1 stalls", std::to_string(_data.lsu1Stalls)});
    customArgs.push_back({"Instruction stalls", std::to_string(_data.instStalls)});

    if (_tensorInfo) {  // This check ensures backward compatibility
        auto tensorInfoArgs = to_custom_args(*_tensorInfo);
        customArgs.insert(customArgs.end(), tensorInfoArgs.begin(), tensorInfoArgs.end());
    }
    return customArgs;
}

size_t RawProfilingACTExRecord::getDebugDataSize() const {
    return sizeof(ActShaveDataEx_t);
}

void RawProfilingACTExRecord::printDebugInfo(std::ostream& outStream) const {
    const auto actShaveCol = getColDesc();
    const auto bufferOffsetBytes = getInClusterIndex() * getDebugDataSize();

    outStream << std::setw(actShaveCol[0].second) << _bufferId << std::setw(actShaveCol[1].second) << _clusterId
              << std::setw(actShaveCol[2].second) << bufferOffsetBytes << std::setw(actShaveCol[3].second)
              << _data.begin << std::setw(actShaveCol[4].second) << _data.duration << std::setw(actShaveCol[5].second)
              << _data.executedInstructions << std::setw(actShaveCol[6].second) << _data.clockCycles
              << std::setw(actShaveCol[7].second) << _data.lsu0Stalls << std::setw(actShaveCol[8].second)
              << _data.lsu1Stalls << std::setw(actShaveCol[9].second) << _data.instStalls;
}

DebugFormattableRecordMixin::ColDesc RawProfilingACTExRecord::getColDesc() const {
    return {{"Buffer ID", COL_WIDTH_32},   {"Cluster ID", COL_WIDTH_64},  {"Buffer offset", COL_WIDTH_64},
            {"Begin", COL_WIDTH_64},       {"Duration", COL_WIDTH_32},    {"Executed", COL_WIDTH_32},
            {"Clock", COL_WIDTH_32},       {"LSU0 Stalls", COL_WIDTH_64}, {"LSU1 Stalls", COL_WIDTH_64},
            {"Instr Stalls", COL_WIDTH_64}};
}

//
// RawProfilingM2IRecord
//
RawProfilingM2IRecord::RawProfilingM2IRecord(M2IData_t data, const ProfilingFB::M2ITask* metadata,
                                             size_t inMemoryOffset)
        : RawProfilingRecord(metadata), DebugFormattableRecordMixin(inMemoryOffset), _data(data) {
}

void RawProfilingM2IRecord::checkData(bool failOnError, Logger& log) const {
    warnOrFail(failOnError, log, _data.startTime == 0 && _data.finishTime == 0, "Can't process M2I profiling data.");
}

ExecutorType RawProfilingM2IRecord::getExecutorType() const {
    return ExecutorType::M2I;
}

TimeType RawProfilingM2IRecord::getStartTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.startTime, frequenciesSetup.profClk);
}

TimeType RawProfilingM2IRecord::getFinishTime(const FrequenciesSetup& frequenciesSetup) const {
    return convertTicksToNs(_data.finishTime, frequenciesSetup.profClk);
}

size_t RawProfilingM2IRecord::getDebugDataSize() const {
    return sizeof(M2IData_t);
}

void RawProfilingM2IRecord::printDebugInfo(std::ostream& outStream) const {
    const auto cols = getColDesc();

    outStream << std::setw(cols[0].second) << _data.fetchTime << std::setw(cols[1].second) << _data.readyTime
              << std::setw(cols[2].second) << _data.startTime << std::setw(cols[3].second) << _data.doneTime
              << std::setw(cols[4].second) << _data.finishTime << std::setw(cols[5].second) << _data.linkAgentID
              << std::setw(cols[6].second) << _data.parentID << std::setw(cols[7].second) << _data.RStallCount
              << std::setw(cols[8].second) << _data.WStallCount << std::setw(cols[9].second) << _data.WRCycleCount
              << std::setw(cols[10].second) << _data.RDCycleCount;
}

DebugFormattableRecordMixin::ColDesc RawProfilingM2IRecord::getColDesc() const {
    return {
            {"Fetch tstamp", COL_WIDTH_64}, {"Ready tstamp", COL_WIDTH_64},  {"Start tstamp", COL_WIDTH_64},
            {"Done tstamp", COL_WIDTH_64},  {"Finish tstamp", COL_WIDTH_64}, {"LA", COL_WIDTH_32},
            {"Parent id", COL_WIDTH_32},    {"RStall cnt", COL_WIDTH_32},    {"WStall cnt", COL_WIDTH_32},
            {"WRCycle cnt", COL_WIDTH_32},  {"RDCycle cnt", COL_WIDTH_32},
    };
}

//
// FakeInvariantRecord
//

FakeInvariantRecord::FakeInvariantRecord(const std::string& name, const RawProfilingRecords& records,
                                         const FrequenciesSetup& frequenciesSetup,
                                         std::shared_ptr<const TensorInfo> tensorInfo)
        : RawProfilingRecord(name, records.front()->getLayerType()),
          _firstVariant(records.front()),
          _tensorInfo(std::move(tensorInfo)),
          _startTime(std::accumulate(records.cbegin(), records.cend(), std::numeric_limits<TimeType>::max(),
                                     [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                         return std::min(a, variant->getStartTime(frequenciesSetup));
                                     })),
          _finishTime(std::accumulate(records.cbegin(), records.cend(), std::numeric_limits<TimeType>::min(),
                                      [&](TimeType a, RawProfilingRecordPtr variant) -> TimeType {
                                          return std::max(a, variant->getFinishTime(frequenciesSetup));
                                      })) {
}

TaskInfo FakeInvariantRecord::getTaskInfo(const FrequenciesSetup& frequenciesSetup) const {
    auto taskInfo = _firstVariant->getTaskInfo(frequenciesSetup);
    taskInfo.name = getTaskName();
    taskInfo.isSubtask = false;

    taskInfo.start_time_ns = getStartTime(frequenciesSetup);
    taskInfo.duration_ns = getDuration(frequenciesSetup);
    taskInfo.customArgs = getCustomArgs(frequenciesSetup);
    //  Ignore cycle information. Not used for DPU
    return taskInfo;
}

ExecutorType FakeInvariantRecord::getExecutorType() const {
    return ExecutorType::DPU;
}

TimeType FakeInvariantRecord::getStartTime(const FrequenciesSetup&) const {
    return _startTime;
}

TimeType FakeInvariantRecord::getFinishTime(const FrequenciesSetup&) const {
    return _finishTime;
}

CustomArgsVector FakeInvariantRecord::getCustomArgs(const FrequenciesSetup&) const {
    if (_tensorInfo == nullptr) {  // This check ensures backward compatibility
        return {};
    }
    return to_custom_args(*_tensorInfo);
}

}  // namespace vpux::profiling
