//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_profiling.hpp
 * @brief Define VPUXProfilingL0 which parses profiling data
 */

#pragma once

#include "vcl_common.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

namespace VPUXDriverCompiler {

// Same as defined in ze_graph_profiling_ext.h
namespace ze {

#define ZE_MAX_GRAPH_PROFILING_LAYER_NAME 256
#define ZE_MAX_GRAPH_PROFILING_LAYER_TYPE 50

typedef enum _ze_layer_status_t {
    ZE_LAYER_STATUS_NOT_RUN = 1,
    ZE_LAYER_STATUS_OPTIMIZED_OUT,
    ZE_LAYER_STATUS_EXECUTED

} ze_layer_status_t;

typedef struct _ze_profiling_layer_info {
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_layer_status_t status;
    uint64_t start_time_ns;   ///< Absolute start time
    uint64_t duration_ns;     ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id;        ///< Not used
    uint64_t fused_layer_id;  ///< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;

} ze_profiling_layer_info;

typedef enum _ze_task_execute_type_t {
    ZE_TASK_EXECUTE_NONE = 0,
    ZE_TASK_EXECUTE_DPU,
    ZE_TASK_EXECUTE_SW,
    ZE_TASK_EXECUTE_DMA

} ze_task_execute_type_t;

typedef struct _ze_profiling_task_info {
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_task_execute_type_t exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles;
    uint32_t stall_cycles;
    uint32_t task_id;
    uint32_t parent_layer_id;  ///< Not used

} ze_profiling_task_info;

static_assert(sizeof(ze_profiling_task_info) == 344);
static_assert(sizeof(ze_profiling_layer_info) == 368);
static_assert(sizeof(ze_profiling_layer_info) == sizeof(vpux::profiling::LayerInfo));

}  // namespace ze

/**
 * @brief Parse the profiling output with blob.
 *
 * Check @ref how-to-use-profiling.md about how to collect the data
 */
class VPUXProfilingL0 final {
public:
    /**
     * @brief Construct a new VPUXProfilingL0 object
     *
     * @param profInput Include the blob and correspond profiling output
     * @param vclLogger
     */
    VPUXProfilingL0(p_vcl_profiling_input_t profInput, VCLLogger* vclLogger)
            : _blobData(profInput->blobData),
              _blobSize(profInput->blobSize),
              _profData(profInput->profData),
              _profSize(profInput->profSize),
              _logger(vclLogger) {
    }

    vcl_result_t getTaskInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getLayerInfo(p_vcl_profiling_output_t profOutput);
    vcl_result_t getRawInfo(p_vcl_profiling_output_t profOutput);
    vcl_profiling_properties_t getProperties() const;
    VCLLogger* getLogger() const {
        return _logger;
    }

private:
    const uint8_t* _blobData;  ///< Pointer to the buffer with the blob
    uint64_t _blobSize;        ///< Size of the blob in bytes
    const uint8_t* _profData;  ///< Pointer to the raw profiling output
    uint64_t _profSize;        ///< Size of the raw profiling output

    std::vector<ze::ze_profiling_task_info> _taskInfo;   ///< Per-task (DPU, DMA, SW) profiling info
    std::vector<vpux::profiling::LayerInfo> _layerInfo;  ///< Per-layer profiling info
    VCLLogger* _logger;                                  ///< Internal logger
};

}  // namespace VPUXDriverCompiler
