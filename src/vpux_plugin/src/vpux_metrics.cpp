//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

// IE
#include <ie_metric_helpers.hpp>
// Plugin
#include "device_helpers.hpp"
#include "vpux_metrics.h"
#include "vpux_private_config.hpp"

namespace vpux {
namespace IE = InferenceEngine;

Metrics::Metrics(const VPUXBackends::CPtr& backends)
        : _backends(backends), _logger(vpu::Logger("VPUXMetrics", vpu::LogLevel::Error, vpu::consoleOutput())) {
    _supportedMetrics = {
            METRIC_KEY(SUPPORTED_METRICS),         METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(FULL_DEVICE_NAME),          METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES), METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
            METRIC_KEY(RANGE_FOR_STREAMS),         METRIC_KEY(IMPORT_EXPORT_SUPPORT),
            METRIC_KEY(DEVICE_ARCHITECTURE),
    };

    _supportedConfigKeys = {
            CONFIG_KEY(LOG_LEVEL),
            CONFIG_KEY(PERF_COUNT),
            CONFIG_KEY(DEVICE_ID),
            VPUX_CONFIG_KEY(THROUGHPUT_STREAMS),
            KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
            VPUX_CONFIG_KEY(INFERENCE_SHAVES),
            VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS),
    };
}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backends == nullptr ? std::vector<std::string>() : _backends->getAvailableDevicesNames();
}

// TODO each backend may support different metrics
const std::vector<std::string>& Metrics::SupportedMetrics() const {
    return _supportedMetrics;
}

// TODO: Need to add the full name
std::string Metrics::GetFullDevicesNames() const {
    return {};
}

// TODO each backend may support different configs
const std::vector<std::string>& Metrics::GetSupportedConfigKeys() const {
    return _supportedConfigKeys;
}

// TODO each backend may support different optimization capabilities
const std::vector<std::string>& Metrics::GetOptimizationCapabilities() const {
    return _optimizationCapabilities;
}

const std::tuple<uint32_t, uint32_t, uint32_t>& Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& Metrics::GetRangeForStreams() const {
    return _rangeForStreams;
}

std::string Metrics::GetDeviceArchitecture(const std::string& specifiedDeviceName) const {
    std::vector<std::string> devNames;
    if (_backends == nullptr || (devNames = _backends->getAvailableDevicesNames()).empty()) {
        IE_THROW() << "No available devices";
    }

    if (specifiedDeviceName.empty()) {
        if (devNames.size() == 1) {
            return utils::getPlatformNameByDeviceName(devNames[0]);
        } else {
            IE_THROW() << "Missing name of a specific device";
        }
    }

    return utils::getPlatformNameByDeviceName(specifiedDeviceName);
}

std::string Metrics::GetBackendName() const {
    if (_backends == nullptr) {
        return {};
    }

    return _backends->getBackendName();
}

}  // namespace vpux
