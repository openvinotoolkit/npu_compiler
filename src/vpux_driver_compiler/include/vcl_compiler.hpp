//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file vcl_compiler.hpp
 * @brief Define VPUXCompilerL0 which holds MLIR compiler
 */

#pragma once

#include <map>

#include <vpux/compiler/compiler.hpp>
#include "vcl_common.hpp"

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
    VPUXCompilerL0(vcl_compiler_desc_t* compilerDesc, vcl_device_desc_t* deviceDesc, VCLLogger* vclLogger);

    /**
     * @brief Get the rough compiler capabilities
     *
     * @return vcl_compiler_properties_t Include compiler ID, API version, max supported opset
     */
    vcl_compiler_properties_t getCompilerProp() const {
        return _compilerProp;
    }

    /**
     * @brief Get the info of default platform and debug level
     *
     * @return vcl_compiler_desc_t Include current platform value, default debug level
     */
    vcl_compiler_desc_t getCompilerDesc() const {
        return _compilerDesc;
    }

    /**
     * @brief Get the device info status
     *
     * @return bool Returns device info status
     */
    bool isDeviceDescEmpty() const {
        return _isDeviceDescEmpty;
    }

    /**
     * @brief Get the info of default device info
     *
     * @return vcl_device_desc_t Include device capabilities
     */
    vcl_device_desc_t getDeviceDesc() const {
        return _deviceDesc;
    }

    /**
     * @brief Get the default compilation configs
     *
     * @details The default common option, compiler option, runtime option,
     *
     * @return std::shared_ptr<const OptionsDesc> The options can be used to do compilation
     */
    std::shared_ptr<const intel_npu::OptionsDesc> getOptions() const {
        return _options;
    }

    /**
     * @brief Get the logger of the compiler
     *
     * @return VCLLogger*  The logger is created and destroied by compiler
     */
    VCLLogger* getLogger() const {
        return _logger;
    }

    /**
     * @brief Use VPUX MLIR compiler to create blob with user info
     *
     * @param buildInfo Include the model data, ioInfo, compilation configs
     * @return std::pair<VPUXExecutableL0*, vcl_result_t>  Include the final blob and status
     */
    std::pair<VPUXExecutableL0*, vcl_result_t> importNetwork(BuildInfo& buildInfo);

    /**
     * @brief Use VPUX MLIR compiler to create blob with user info
     * @note Blob storage is allocated via given allocator
     *
     * @param buildInfo Include the model data, ioInfo, compilation configs
     * @param allocator Allocator for blob storage allocation
     * @return vpux::NetworkDescriptionView Include non-owning view into blob and metadata
     */
    vpux::NetworkDescriptionView importNetwork(BuildInfo& buildInfo, vpux::BlobAllocator& allocator);

    /**
     * @brief Use VPUX MLIR compiler to create one shot weight-separated blob with user info
     * @note Blob storage is allocated via given allocator
     *
     * @param buildInfo Include the model data, ioInfo, compilation configs
     * @param allocator Allocator for blob storage allocation
     * @return std::vector<std::shared_ptr<vpux::NetworkDescriptionView>> Include non-owning
     * views into blobs and metadatas
     */
    std::vector<std::shared_ptr<vpux::NetworkDescriptionView>> importNetworkWSOneShot(BuildInfo& buildInfo,
                                                                                      vpux::BlobAllocator& allocator);

    /**
     * @brief Check if a model can be supported by current compiler
     *
     * @param buildInfo include the model data, default compilation config
     * @param pQueryNetwork The supported layers by compiler
     * @return vcl_result_t
     */
    vcl_result_t queryNetwork(const BuildInfo& buildInfo, VPUXQueryNetworkL0* pQueryNetwork);

    /**
     * @brief Return the size of the compiler supported options list (string) in the provided buffer
     *
     * @param stringSize where to store the size of the string
     * @return vcl_result_t
     */
    vcl_result_t getSupportedOptionsSize(uint64_t* stringSize);

    /**
     * @brief Retreive a list of configurable options the compiler supports
     *
     * @param buffer The buffer to store serialized string in.
     * @param size The size of buffer, need to be same with result of getSupportedOptionsSize().
     * @return vcl_result_t
     */
    vcl_result_t getSupportedOptions(char* buffer, uint64_t size);

    /**
     * @brief Verify if a compiler configuration option (if value=nullptr) or option-value pair is supported
     *
     * @param option String containing the option's name
     * @param vlaue String containing the option value to be checked. If null, we only check if the option is supported
     * @return bool true/false
     */
    bool isOptionValueSupported(const char* option, const char* value);

private:
    std::shared_ptr<intel_npu::OptionsDesc> _options;  ///< The default compilation configs
    std::unique_ptr<vpux::CompilerImpl> _compiler;     ///< The handle of MLIR compiler
    vcl_compiler_properties_t _compilerProp;           ///< The capabilities of compiler
    vcl_compiler_desc_t _compilerDesc;                 ///< The info of platform and debug level
    vcl_device_desc_t _deviceDesc;                     ///< The info of device
    bool _isDeviceDescEmpty;                           ///< The info of deviceDesc status
    VCLLogger* _logger;
};

}  // namespace VPUXDriverCompiler
