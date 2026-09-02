//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "parser.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"
#include "vpux/utils/profiling/reports/ted.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"
#include "vpux/utils/profiling/tasknames.hpp"

#include "schema/profiling_generated.h"

#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace vpux::profiling {

using TimeType = double;

template <typename... Args>
void warnOrFail(bool failOnError, Logger& log, bool condition, llvm::StringLiteral format, Args&&... params) {
    if (condition) {
        if (failOnError) {
            VPUX_THROW(format, std::forward<Args>(params)...);
        } else {
            log.warning(format, std::forward<Args>(params)...);
        }
    }
}

class DebugFormattableRecordMixin {
public:
    using ColDesc = std::vector<std::pair<std::string, int>>;

protected:
    DebugFormattableRecordMixin(size_t inMemoryOffset, size_t inClusterIndex = 0)
            : _inMemoryOffset(inMemoryOffset), _inClusterIndex(inClusterIndex) {
    }
    virtual ColDesc getColDesc() const = 0;

public:
    void printDebugHeader(std::ostream& os);
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
    using BarriersSet = std::set<BarrierIdType>;

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

protected:
    template <typename RawMetadata>
    explicit RawProfilingRecord(const RawMetadata* metadata)
            : RawProfilingRecord(metadata, getWaitBarriersFromTask(metadata), getUpdateBarriersFromTask(metadata)) {
    }

    // Exposed to manually extract DMA barriers metadata on NPU37XX
    template <typename RawMetadata>
    explicit RawProfilingRecord(const RawMetadata* metadata, const BarriersSet& wBarriers, const BarriersSet& uBarriers)
            : RawProfilingRecord(metadata->name()->str(), deserializeTaskName(metadata->name()->str()).layerType,
                                 wBarriers, uBarriers) {
    }

    // Used in FakeInvariantRecord
    explicit RawProfilingRecord(const std::string& name, const std::string& layerType,
                                const BarriersSet& wBarriers = {}, const BarriersSet& uBarriers = {});
    virtual ~RawProfilingRecord() = default;

public:
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
    virtual TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const;
    virtual void checkData(bool failOnError, Logger& log) const;
    virtual void sanitize(vpux::Logger&, const FrequenciesSetup&) const;
    virtual TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const = 0;
    virtual TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const = 0;
    virtual TimeType getDuration(const FrequenciesSetup& frequenciesSetup) const;
    virtual CustomArgsVector getCustomArgs(const FrequenciesSetup&) const;

private:
    const std::string _name;
    const std::string _layerType;
    BarriersSet _waitBarriers;
    BarriersSet _updateBarriers;
};

class RawProfilingDMARecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
protected:
    explicit RawProfilingDMARecord(const ProfilingFB::DMATask* metadata, size_t inMemoryOffset,
                                   const BarriersSet& wBarriers = {}, const BarriersSet& uBarriers = {});

private:
    TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const override;
    void sanitize(Logger& log, const FrequenciesSetup& frequenciesSetup) const override;

protected:
    CustomArgsVector getCustomArgs(const FrequenciesSetup&) const override;

    std::optional<unsigned short> _portId;
    std::optional<ProfilingFB::DMAChannelType> _channelType;
    std::optional<ProfilingFB::MemoryKind> _sourceMemoryKind;
    std::optional<ProfilingFB::MemoryKind> _destinationMemoryKind;
    std::unique_ptr<const TensorInfo> _tensorShapeInfo;
    std::unique_ptr<const TensorInfo> _tensorStrideInfo;
    unsigned short _gatherIndices;
    bool _dynamicStridesInput;
    bool _dynamicStridesOutput;
};

class RawProfilingDMA27Record final : public RawProfilingDMARecord {
public:
    explicit RawProfilingDMA27Record(const DMA27Data_t& record, const ProfilingFB::DMATask* metadata,
                                     size_t inMemoryOffset, const BarriersSet& wBarriers, const BarriersSet& uBarriers);

private:
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    size_t getDebugDataSize() const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

    const DMA27Data_t _record;
};

class RawProfilingDMA40Record final : public RawProfilingDMARecord {
public:
    explicit RawProfilingDMA40Record(const HwpDma40Data_t& record, const ProfilingFB::DMATask* metadata,
                                     size_t inMemoryOffset);

    void checkData(bool failOnError, Logger& log) const override;

private:
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    size_t getDebugDataSize() const override;
    CustomArgsVector getCustomArgs(const FrequenciesSetup& frequenciesSetup) const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

    const HwpDma40Data_t _record;
};

class RawProfilingDPURecord : public RawProfilingRecord, public DebugFormattableRecordMixin {
protected:
    explicit RawProfilingDPURecord(const ProfilingFB::DPUTask* metadata, uint32_t variantId, size_t inMemoryOffset,
                                   uint32_t inClusterIndex, std::shared_ptr<const TensorInfo> tensorInfo,
                                   std::unique_ptr<const DPUVariantInfo> variantInfo);

    virtual double getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const = 0;

public:
    void checkData(bool, Logger&) const override {
    }

    CustomArgsVector getCustomArgs(const FrequenciesSetup&) const override;

public:
    size_t getClusterId() {
        return _clusterId;
    }
    uint32_t getTaskId() {
        return _taskId;
    }

    const std::shared_ptr<const TensorInfo>& getTensorInfo() const;

private:
    std::string getTaskName() const override;
    TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const override;
    void sanitize(Logger& log, const FrequenciesSetup& frequenciesSetup) const override;

protected:
    const uint32_t _bufferId;
    const uint32_t _clusterId;
    const uint32_t _taskId;
    const uint32_t _variantId;

private:
    std::unique_ptr<const DPUVariantInfo> _variantInfo;
    std::shared_ptr<const TensorInfo> _tensorInfo;
};

class RawProfilingDPUHW27Record final : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW27Record(HwpDpu27Mode0Data_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterIndex,
                                       std::shared_ptr<const TensorInfo> tensorInfo,
                                       std::unique_ptr<const DPUVariantInfo> variantInfo);

    void checkData(bool failOnError, Logger& log) const override;

private:
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    double getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const override;
    size_t getDebugDataSize() const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

private:
    const HwpDpu27Mode0Data_t _timestamps;
};

class RawProfilingDPUHW40Record : public RawProfilingDPURecord {
public:
    explicit RawProfilingDPUHW40Record(HwpDpuIduOduData_t timestamps, const ProfilingFB::DPUTask* metadata,
                                       uint32_t variantId, size_t inMemoryOffset, uint32_t inClusterIndex,
                                       std::shared_ptr<const TensorInfo> tensorInfo,
                                       std::unique_ptr<const DPUVariantInfo> variantInfo);

private:
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    size_t getDebugDataSize() const override;
    double getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const override;
    ColDesc getColDesc() const override;
    void printDebugInfo(std::ostream& outStream) const override;

protected:
    const HwpDpuIduOduData_t _timestamps;
};

class RawProfilingDPUHW50Record final : public RawProfilingDPUHW40Record {
public:
    using RawProfilingDPUHW40Record::RawProfilingDPUHW40Record;

private:
    double getTaskDurationClock(const FrequenciesSetup& frequenciesSetup) const override;
};

class RawProfilingACTRecord final : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTRecord(ActShaveData_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset);

    void checkData(bool failOnError, Logger& log) const override;

private:
    TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const override;
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getDuration(const FrequenciesSetup& frequenciesSetup) const override;
    CustomArgsVector getCustomArgs(const FrequenciesSetup&) const override;
    size_t getDebugDataSize() const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

    const ActShaveData_t _data;
    const uint32_t _bufferId;
    const uint32_t _clusterId;
    const std::optional<uint8_t> _fifoId;
    std::unique_ptr<const TensorInfo> _tensorInfo;
};

class RawProfilingACTExRecord final : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingACTExRecord(ActShaveDataEx_t data, const ProfilingFB::SWTask* metadata, size_t inMemoryOffset);
    void checkData(bool failOnError, Logger& log) const override;

private:
    TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const override;

    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getDuration(const FrequenciesSetup& frequenciesSetup) const override;
    CustomArgsVector getCustomArgs(const FrequenciesSetup&) const override;
    size_t getDebugDataSize() const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

    const ActShaveDataEx_t _data;
    const uint32_t _bufferId;
    const uint32_t _clusterId;
    const std::optional<uint8_t> _fifoId;
    std::unique_ptr<const TensorInfo> _tensorInfo;
};

class RawProfilingM2IRecord final : public RawProfilingRecord, public DebugFormattableRecordMixin {
public:
    explicit RawProfilingM2IRecord(M2IData_t data, const ProfilingFB::M2ITask* metadata, size_t inMemoryOffset);

    void checkData(bool failOnError, Logger& log) const override;

private:
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;
    size_t getDebugDataSize() const override;
    void printDebugInfo(std::ostream& outStream) const override;
    ColDesc getColDesc() const override;

    const M2IData_t _data;
};

class FakeInvariantRecord final : public RawProfilingRecord {
public:
    explicit FakeInvariantRecord(const std::string& name, const RawProfilingRecords& records,
                                 const FrequenciesSetup& frequenciesSetup,
                                 std::shared_ptr<const TensorInfo> tensorInfo);

private:
    TaskInfo getTaskInfo(const FrequenciesSetup& frequenciesSetup) const override;
    ExecutorType getExecutorType() const override;
    TimeType getStartTime(const FrequenciesSetup& frequenciesSetup) const override;
    TimeType getFinishTime(const FrequenciesSetup& frequenciesSetup) const override;

    CustomArgsVector getCustomArgs(const FrequenciesSetup&) const override;

    std::shared_ptr<const RawProfilingRecord> _firstVariant;
    std::shared_ptr<const TensorInfo> _tensorInfo;
    const TimeType _startTime;
    const TimeType _finishTime;
};

}  // namespace vpux::profiling
