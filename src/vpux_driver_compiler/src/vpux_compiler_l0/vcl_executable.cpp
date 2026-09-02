//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vcl_executable.hpp"

#include "vpux/utils/ov/profiling_utils.hpp"

#include <memory>

using namespace vpux;

namespace VPUXDriverCompiler {
VPUXExecutableL0::VPUXExecutableL0(const std::shared_ptr<const NetworkDescription>& networkDesc,
                                   const vpux::OV::Config& config, std::shared_ptr<VCLLogger> vclLogger)
        : _networkDesc(networkDesc), _config(config), _logger(std::move(vclLogger)) {
}

VPUXExecutableL0::VPUXExecutableL0(const std::string& compatibilityString, std::shared_ptr<VCLLogger> vclLogger)
        : _config(std::nullopt), _compatibilityString(compatibilityString), _logger(std::move(vclLogger)) {
}

VPUXExecutableL0::~VPUXExecutableL0() = default;

vcl_result_t VPUXExecutableL0::serializeNetwork() {
    if (_config.has_value()) {
        [[maybe_unused]] const auto scopedStopWatch = vpux::startScopedTimer(_config.value(), [this](double deltaMs) {
            _logger->info("getCompiledNetwork time: {0} ms", deltaMs);
        });
    }

    return VCL_RESULT_SUCCESS;
}

vcl_result_t VPUXExecutableL0::getNetworkSize(uint64_t* blobSize) const {
    if (blobSize == nullptr) {
        _logger->outputError("Can not return blob size for NULL argument!");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }
    if (!_networkDesc) {
        return VCL_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    const auto& blob = _networkDesc->compiledNetwork;
    *blobSize = blob.size();
    if (*blobSize == 0) {
        // The executable handle do not contain a legal network.
        _logger->outputError("No blob created! The compiled network is empty!");
        return VCL_RESULT_ERROR_UNKNOWN;
    } else {
        return VCL_RESULT_SUCCESS;
    }
}

vcl_result_t VPUXExecutableL0::exportNetwork(uint8_t* blobOut, uint64_t blobSize) const {
    if (!_networkDesc) {
        return VCL_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    const auto& blob = _networkDesc->compiledNetwork;
    if (!blobOut || blobSize != blob.size()) {
        _logger->outputError("Invalid argument to export network");
        return VCL_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (_config.has_value()) {
        [[maybe_unused]] const auto scopedStopWatch = vpux::startScopedTimer(_config.value(), [this](double deltaMs) {
            _logger->info("exportNetwork time: {0} ms", deltaMs);
        });
    }

    memcpy(blobOut, blob.data(), blobSize);

    return VCL_RESULT_SUCCESS;
}

}  // namespace VPUXDriverCompiler
