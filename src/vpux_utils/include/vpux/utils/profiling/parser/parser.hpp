//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//

#pragma once

#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/profiling/common.hpp"
#include "vpux/utils/profiling/parser/device.hpp"
#include "vpux/utils/profiling/parser/hw.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace vpux::profiling {

class RawProfilingRecord;
using RawProfilingRecordPtr = std::shared_ptr<RawProfilingRecord>;
using RawProfilingRecords = std::vector<RawProfilingRecordPtr>;

using WorkpointRecords = std::vector<std::pair<WorkpointConfiguration_t, size_t>>;

enum class SynchronizationPointKind { DMA_TO_DPU, DPU_TO_DMA, STRICT_DMA_TO_DPU };

// Container for conjucted storage of tasks of one format: RawProfilingRecordPtr/TaskInfo
struct RawProfilingData {
    RawProfilingRecords dmaTasks;
    RawProfilingRecords dpuTasks;
    RawProfilingRecords swTasks;
    // Pair of workpoint and offset
    WorkpointRecords workpoints;
    // Vector of [ExecutorType; offset in blob(bytes)]
    std::vector<std::pair<ExecutorType, size_t>> parseOrder;
};

// Map of exec. type to section offset and size
using RawDataLayout = std::map<ExecutorType, std::pair<uint32_t, uint32_t>>;

struct RawData {
    RawDataLayout sections;
    RawProfilingData rawRecords;
    TargetDevice device;
};

/**
 * @fn getRawProfilingTasks
 * @brief Show raw counters for debug purpose. Intended for use in prof_parser only
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param ignoreSanitizationErrors to ignore sanitization errors
 * @return RawProfilingData
 */
RawData getRawProfilingTasks(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                             bool ignoreSanitizationErrors = false);

struct FrequenciesSetup {
    static constexpr double MIN_FREQ_MHZ = 700.0;

    double vpuClk = UNINITIALIZED_FREQUENCY_VALUE;
    double dpuClk = UNINITIALIZED_FREQUENCY_VALUE;
    double profClk = UNINITIALIZED_FREQUENCY_VALUE;
    bool hasSharedDmaDpuCounter = false;
    FreqStatus clockStatus = FreqStatus::UNKNOWN;
};

// freq.cpp

FrequenciesSetup getFrequencySetup(const TargetDevice device, const WorkpointRecords& workpoints, bool highFreqPerfClk,
                                   bool fpga, vpux::Logger& log);

// sync.cpp

std::optional<int64_t> getDMA2OtherTimersShift(const RawProfilingRecords& dmaTasks,
                                               const RawProfilingRecords& otherTasks,
                                               const FrequenciesSetup& frequenciesSetup,
                                               SynchronizationPointKind pointKind, vpux::Logger& log);

}  // namespace vpux::profiling
