//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/profiling.hpp"

#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/api.hpp"
#include "vpux/utils/profiling/taskinfo.hpp"

#include "vpux/utils/core/error.hpp"

#include <cstring>

namespace vpux {

namespace {

ze::ze_profiling_task_info convertTaskInfo(const profiling::TaskInfo& task) {
    ze::ze_profiling_task_info zeTask = {};

    const auto nameLen = task.name.copy(zeTask.name, sizeof(zeTask.name) - 1);
    zeTask.name[nameLen] = 0;

    const auto typeLen = task.layer_type.copy(zeTask.layer_type, sizeof(zeTask.layer_type) - 1);
    zeTask.layer_type[typeLen] = 0;

    zeTask.exec_type = static_cast<ze::ze_task_execute_type_t>(task.exec_type);
    zeTask.start_time_ns = task.start_time_ns;
    zeTask.duration_ns = task.duration_ns;
    zeTask.active_cycles = task.active_cycles;
    zeTask.stall_cycles = task.stall_cycles;
    zeTask.task_id = -1;
    zeTask.parent_layer_id = -1;

    return zeTask;
}

ze::ze_profiling_layer_info convertLayerInfo(const profiling::LayerInfo& info) {
    static_assert(sizeof(ze::ze_profiling_layer_info) == sizeof(profiling::LayerInfo));

    ze::ze_profiling_layer_info layerInfo = {};

    static_assert(sizeof(layerInfo.name) == sizeof(info.name));
    static_assert(sizeof(layerInfo.layer_type) == sizeof(info.layer_type));
    std::memcpy(static_cast<void*>(layerInfo.name), static_cast<const void*>(info.name), sizeof(info.name));
    std::memcpy(static_cast<void*>(layerInfo.layer_type), static_cast<const void*>(info.layer_type),
                sizeof(info.layer_type));

    layerInfo.status = static_cast<ze::ze_layer_status_t>(info.status);
    layerInfo.start_time_ns = info.start_time_ns;
    layerInfo.duration_ns = info.duration_ns;
    layerInfo.layer_id = info.layer_id;
    layerInfo.fused_layer_id = info.fused_layer_id;

    layerInfo.dpu_ns = info.dpu_ns;
    layerInfo.sw_ns = info.sw_ns;
    layerInfo.dma_ns = info.dma_ns;

    return layerInfo;
}

}  // namespace

std::vector<ze::ze_profiling_layer_info> getLayerInfoImpl(const uint8_t* blobData, uint64_t blobSize,
                                                          const uint8_t* profData, uint64_t profSize) {
    VPUX_THROW_WHEN(!blobData || !profData, "Null argument to get layer info");
    VPUX_THROW_WHEN(blobSize == 0 || profSize == 0, "Invalid size argument to get layer info");
    auto layerInfo = profiling::getLayerProfilingInfoHook(profData, profSize, blobData, blobSize);
    std::vector<ze::ze_profiling_layer_info> result;
    result.reserve(layerInfo.size());
    for (const auto& layer : layerInfo) {
        result.emplace_back(convertLayerInfo(layer));
    }
    return result;
}

std::vector<ze::ze_profiling_task_info> getTaskInfoImpl(const uint8_t* blobData, uint64_t blobSize,
                                                        const uint8_t* profData, uint64_t profSize) {
    VPUX_THROW_WHEN(!blobData || !profData, "Null argument to get task info");
    VPUX_THROW_WHEN(blobSize == 0 || profSize == 0, "Invalid size argument to get task info");
    auto taskInfo = profiling::getTaskInfo(blobData, blobSize, profData, profSize, profiling::VerbosityLevel::LOW);
    std::vector<ze::ze_profiling_task_info> result;
    result.reserve(taskInfo.size());
    for (const auto& task : taskInfo) {
        result.emplace_back(convertTaskInfo(task));
    }
    return result;
}

}  // namespace vpux
