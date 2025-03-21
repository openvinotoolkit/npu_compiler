//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// Profiling parser public interface

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace vpux::profiling {

struct LayerInfo {
    char name[256];
    char layer_type[50];
    enum layer_status_t { NOT_RUN, OPTIMIZED_OUT, EXECUTED };
    layer_status_t status;
    uint64_t start_time_ns;        ///< Absolute start time
    uint64_t duration_ns;          ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id = -1;        ///< Not used
    uint64_t fused_layer_id = -1;  ///< Not used
    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns = 0;
    uint64_t sw_ns = 0;
    uint64_t dma_ns = 0;
};

// Size must match L0-ext ze_profiling_layer_info
static_assert(sizeof(LayerInfo) == 368);

struct TaskInfo {
    std::string name;
    std::string layer_type;
    enum class ExecType { NONE, DPU, SW, DMA, M2I };
    ExecType exec_type;
    uint32_t clusterId;
    bool isSubtask;  // DPU variant
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t total_cycles;
    uint32_t stall_cycles;
    struct stall_counters {
        uint32_t lsu0_stalls;
        uint32_t lsu1_stalls;
        uint32_t instruction_stalls;
    } stall_counters;
};

enum class FreqStatus { UNKNOWN, VALID, INVALID, SIM };
constexpr double UNINITIALIZED_FREQUENCY_VALUE = -1;

struct FreqInfo {
    double freqMHz = UNINITIALIZED_FREQUENCY_VALUE;
    FreqStatus freqStatus = FreqStatus::UNKNOWN;
};

struct ProfInfo {
    std::vector<TaskInfo> tasks;
    std::vector<LayerInfo> layers;
    FreqInfo dpuFreq;
};

const char* to_string(FreqStatus);

template <typename T>
bool profilingTaskStartTimeComparator(const T& a, const T& b) {
    return std::forward_as_tuple(a.start_time_ns, a.name, b.duration_ns) <
           std::forward_as_tuple(b.start_time_ns, b.name, a.duration_ns);
}

}  // namespace vpux::profiling
