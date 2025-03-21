//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Profiling parser public interface

#pragma once

#include "vpux/utils/profiling/taskinfo.hpp"

#include <cstddef>
#include <ostream>
#include <vector>

namespace vpux::profiling {

/**
 * @enum VerbosityLevel
 * @brief Declares verbosity level of printing information
 */
enum VerbosityLevel {
    LOW = 0,     ///< Default, aggregated DPU timing
    MEDIUM = 1,  ///< Include individual DPU variant timing
    HIGH = 5,    ///< Same as medium
};

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param verbosity amount of DPU info to print, may be LOW|MEDIUM|HIGH
 * @param fpga whether buffer was obtained from FPGA
 * @param highFreqPerfClk use the high frequency perf_clk value (NPU40XX only)
 * @see TaskType
 * @see VerbosityLevel
 * @return std::vector of TaskInfo structures
 */
std::vector<TaskInfo> getTaskInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                                  VerbosityLevel verbosity, bool fpga = false, bool highFreqPerfClk = false);

/**
 * @fn getTaskInfo
 * @brief Parse raw profiling output to get per-tasks info.
 * @param blobData pointer to the buffer with blob binary
 * @param blobSize blob size in bytes
 * @param profData pointer to the buffer with raw profiling data
 * @param profSize raw profiling data size
 * @param verbosity amount of DPU info to print, may be LOW|MEDIUM|HIGH
 * @param fpga whether buffer was obtained from FPGA
 * @param highFreqPerfClk use the high frequency perf_clk value (NPU40XX only)
 * @see TaskType
 * @see VerbosityLevel
 * @return ProfInfo
 */
ProfInfo getProfInfo(const uint8_t* blobData, size_t blobSize, const uint8_t* profData, size_t profSize,
                     VerbosityLevel verbosity, bool fpga = false, bool highFreqPerfClk = false);

void writeDebugProfilingInfo(std::ostream& outStream, const uint8_t* blobData, size_t blobSize, const uint8_t* profData,
                             size_t profSize);

}  // namespace vpux::profiling
