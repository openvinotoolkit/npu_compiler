//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_executable.hpp
 * @brief Define VPUXExecutableL0 which holds compiled blob
 */

#pragma once

#include "vcl_logger.hpp"
#include "vpux/compiler/network_metadata.hpp"

#include <string>

namespace VPUXDriverCompiler {

/**
 * @brief Hold the compiled blob
 *
 */
class VPUXExecutableL0 final {
public:
    VPUXExecutableL0(const std::shared_ptr<const vpux::NetworkDescription>& networkDesc,
                     const intel_npu::Config& config, std::shared_ptr<VCLLogger> vclLogger);

    VPUXExecutableL0(const std::string& compatibilityString, std::shared_ptr<VCLLogger> vclLogger);

    VPUXExecutableL0(VPUXExecutableL0&&) = default;
    VPUXExecutableL0(const VPUXExecutableL0&) = delete;
    ~VPUXExecutableL0();
    VPUXExecutableL0& operator=(const VPUXExecutableL0&) = delete;
    VPUXExecutableL0& operator=(VPUXExecutableL0&&) = default;

    /**
     * @brief Get compiled blob from net description
     *
     * @return vcl_result_t
     */
    vcl_result_t serializeNetwork();

    /**
     * @brief Get the size of blob
     *
     * @param blobSize Store the size of blob if execute successfully
     * @return vcl_result_t
     */
    vcl_result_t getNetworkSize(uint64_t* blobSize) const;

    /**
     * @brief Copy blob data to user buffer
     *
     * @param blob Point to the buffer created by user
     * @param blobSize Need to be same with the result of getNetworkSize()
     * @return vcl_result_t
     */
    vcl_result_t exportNetwork(uint8_t* blob, uint64_t blobSize) const;

    const std::string& getCompatibilityString() const {
        if (_networkDesc) {
            return _networkDesc->metadata.compatibilityString;
        } else {
            return _compatibilityString;
        }
    }

    const std::shared_ptr<VCLLogger>& getLogger() const {
        return _logger;
    }

private:
    std::shared_ptr<const vpux::NetworkDescription> _networkDesc;  ///< The compilation result of MLIR compiler
    std::optional<intel_npu::Config> _config;                      ///< Configuration for the executable
    std::string _compatibilityString;
    std::shared_ptr<VCLLogger> _logger;
};

}  // namespace VPUXDriverCompiler
