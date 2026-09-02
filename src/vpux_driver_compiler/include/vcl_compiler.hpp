//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_compiler.hpp
 * @brief Define VPUXCompilerL0 which holds MLIR compiler
 */

#pragma once

#include "vcl_compiler_loader.hpp"
#include "vcl_logger.hpp"

#include <memory>

namespace VPUXDriverCompiler {

class VPUXExecutableL0;
class VPUXQueryNetworkL0;

/**
 * @brief Wrapper of VPUX MLIR compiler
 *
 * @details The capabilities and configs of compiler.
 * Create blob with model data and configuration.
 * Query supported layers with model data.
 */
class VPUXCompilerL0 final {
public:
    VPUXCompilerL0(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc,
                   std::shared_ptr<VCLLogger> vclLogger);

    /**
     * @brief Get the rough compiler capabilities
     *
     * @return vcl_compiler_properties_t Include compiler ID, API version, max supported opset
     */
    vcl_compiler_properties_t getCompilerProp() const {
        return _compilerProp;
    }

    /**
     * @brief Get the logger of the compiler
     *
     * @return Logger reference
     */
    const std::shared_ptr<VCLLogger>& getLogger() const {
        return _logger;
    }

    /**
     * @brief Use VPUX MLIR compiler to create blob with user info
     *
     * @param desc The description of the model and user configuration info
     * @return std::pair<VPUXExecutableL0*, vcl_result_t> Include the final blob and status
     */
    std::pair<VPUXExecutableL0*, vcl_result_t> importNetwork(const vcl_executable_desc_t& desc);

    /**
     * @brief Use VPUX MLIR compiler to create blob with user info
     * @note Blob storage is allocated via given allocator
     *
     * @param desc The description of the model and user configuration info
     * @param allocator Allocator for blob storage allocation
     * @return vpux::NetworkDescriptionView Include non-owning view into blob and metadata
     */
    vpux::NetworkDescriptionView importNetwork(const vcl_executable_desc_t& desc, vpux::BlobAllocator& allocator);

    /**
     * @brief Use VPUX MLIR compiler to create one shot weight-separated blob with user info
     * @note Blob storage is allocated via given allocator
     *
     * @param desc The description of the model and user configuration info
     * @param allocator Allocator for blob storage allocation
     * @return std::vector<std::shared_ptr<vpux::NetworkDescriptionView>> Non-owning views into
     * the blobs and metadata
     */
    std::vector<std::shared_ptr<vpux::NetworkDescriptionView>> importNetworkWSOneShot(const vcl_executable_desc_t& desc,
                                                                                      vpux::BlobAllocator& allocator);

    /**
     * @brief Check if a model can be supported by current compiler
     *
     * @param desc The description of the model and user configuration info
     * @param pQueryNetwork Output object to receive the supported layers result
     * @return vcl_result_t
     */
    vcl_result_t queryNetwork(const vcl_query_desc_t& desc, VPUXQueryNetworkL0* pQueryNetwork);

    /**
     * @brief Return the size of the compiler supported options list (string) in the provided buffer
     *
     * @param stringSize where to store the size of the string
     * @return vcl_result_t
     */
    vcl_result_t getSupportedOptionsSize(uint64_t* stringSize) const;

    /**
     * @brief Retrieve a list of configurable options the compiler supports
     *
     * @param buffer The buffer to store serialized string in.
     * @param size The size of buffer, need to be same with result of getSupportedOptionsSize().
     * @return vcl_result_t
     */
    vcl_result_t getSupportedOptions(char* buffer, uint64_t size) const;

    /**
     * @brief Verify if a compiler configuration option (if value=nullptr) or option-value pair is supported
     *
     * @param option String containing the option's name
     * @param value String containing the option value to be checked. If null, we only check if the option is supported
     * @return bool true/false
     */
    bool isOptionValueSupported(const char* option, const char* value) const;

private:
    std::shared_ptr<vpux::OV::OptionsDesc> _options;  ///< The default compilation configs
    std::unique_ptr<CompilerLoader> _compilerLoader;  ///< The handle of MLIR compiler
    vcl_compiler_properties_t _compilerProp;          ///< The capabilities of compiler
    vcl_compiler_desc_t _compilerDesc;                ///< The info of platform and debug level
    vcl_device_desc_t _deviceDesc;                    ///< The info of device
    bool _isDeviceDescEmpty;                          ///< The info of deviceDesc status
    std::shared_ptr<VCLLogger> _logger;
};

}  // namespace VPUXDriverCompiler
