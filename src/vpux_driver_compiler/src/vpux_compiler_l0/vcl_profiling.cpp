//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_profiling.hpp"

using namespace vpux;

namespace VPUXDriverCompiler {

vcl_result_t VPUXProfilingL0::getTaskInfo(p_vcl_profiling_output_t profOutput) {
    if (!profOutput) {
        _logger->outputError("Null argument to get task info");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_taskInfo.empty()) {
        try {
            _taskInfo = _compilerLoader->getCompiler()->getTaskInfo(_blobData, _blobSize, _profData, _profSize);
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<const uint8_t*>(_taskInfo.data());
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
            _layerInfo = _compilerLoader->getCompiler()->getLayerInfo(_blobData, _blobSize, _profData, _profSize);
        } catch (const std::exception& error) {
            _logger->outputError(error.what());
            return VCL_RESULT_ERROR_UNKNOWN;
        } catch (...) {
            _logger->outputError("Internal exception! Can't parse profiling information.");
            return VCL_RESULT_ERROR_UNKNOWN;
        }
    }

    profOutput->data = reinterpret_cast<const uint8_t*>(_layerInfo.data());
    profOutput->size = _layerInfo.size() * sizeof(ze::ze_profiling_layer_info);
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
