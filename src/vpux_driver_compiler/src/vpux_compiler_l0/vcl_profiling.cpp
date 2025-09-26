//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_profiling.hpp"

#include "vpux/utils/profiling/parser/api.hpp"
#include "vpux/utils/profiling/reports/api.hpp"

using namespace vpux;

namespace VPUXDriverCompiler {

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

vcl_result_t VPUXProfilingL0::getTaskInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get task info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_taskInfo.empty()) {
        try {
            auto taskInfo =
                    profiling::getTaskInfo(_blobData, _blobSize, _profData, _profSize, profiling::VerbosityLevel::LOW);
            _taskInfo.reserve(taskInfo.size());
            for (auto task : taskInfo) {
                _taskInfo.push_back(convertTaskInfo(task));
            }
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_taskInfo.data());
    profOutput->size = _taskInfo.size() * sizeof(ze::ze_profiling_task_info);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getLayerInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get layer info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_layerInfo.empty()) {
        try {
            _layerInfo = profiling::getLayerProfilingInfoHook(_profData, _profSize, _blobData, _blobSize);
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<uint8_t*>(_layerInfo.data());
    profOutput->size = _layerInfo.size() * sizeof(profiling::LayerInfo);
    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXProfilingL0::getRawInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get raw info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    profOutput->data = _profData;
    profOutput->size = _profSize;
    return VCL_RESULT_SUCCESS;
}

vcl_profiling_properties_t VPUXProfilingL0::getProperties() const {
    vcl_profiling_properties_t prop;
    prop.version.major = VCL_PROFILING_VERSION_MAJOR;
    prop.version.minor = VCL_PROFILING_VERSION_MINOR;
    return prop;
}

}  // namespace VPUXDriverCompiler
