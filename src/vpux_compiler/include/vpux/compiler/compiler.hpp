//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/icompiler.hpp"

namespace vpux {

class CompilerImpl final : public ICompiler {
public:
    CompilerImpl();
    ~CompilerImpl() override = default;

    CompilerImpl(const CompilerImpl&) = delete;
    CompilerImpl(CompilerImpl&&) = delete;
    CompilerImpl& operator=(const CompilerImpl&) = delete;
    CompilerImpl& operator=(CompilerImpl&&) = delete;

    // Mutable model variant for direct use with deserialized model in VCL
    NetworkDescription compile(const std::shared_ptr<ov::Model>& model, const vpux::OV::Config& config) const;

    NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                               const vpux::OV::Config& config) const final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const vpux::OV::Config& config) const final;

    NetworkMetadata parse(const std::vector<uint8_t>& network, const vpux::OV::Config&) const final;

    std::vector<ov::ProfilingInfo> processProfilingOutput(const std::vector<uint8_t>& profData,
                                                          const std::vector<uint8_t>& network,
                                                          const vpux::OV::Config& config) const final;

    std::vector<ze::ze_profiling_layer_info> getLayerInfo(const uint8_t* blobData, uint64_t blobSize,
                                                          const uint8_t* profData, uint64_t profSize) const final;

    std::vector<ze::ze_profiling_task_info> getTaskInfo(const uint8_t* blobData, uint64_t blobSize,
                                                        const uint8_t* profData, uint64_t profSize) const final;

    // CiD-specific methods

    NetworkDescriptionView compile(const std::shared_ptr<ov::Model>& model, const vpux::OV::Config& config,
                                   BlobAllocator& allocator) const override;

    NetworkDescriptionView compile(const std::shared_ptr<const ov::Model>& model, const vpux::OV::Config& config,
                                   BlobAllocator& allocator) const override;

    // WS CiP-specific methods

    /// @brief Returns Init schedules and Main in a single call. There is always exactly one Main schedule, placed at
    /// the back of the vector.
    std::vector<std::shared_ptr<NetworkDescription>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                      const vpux::OV::Config& config) const final;

    /// @brief Sequentially compiles Init and Main schedules. The Main schedule is always last.
    NetworkDescription compileWsIterative(const std::shared_ptr<ov::Model>& model, const vpux::OV::Config& config,
                                          size_t callIdx) const override;

    // VCL methods
    ov::SupportedOpsMap queryFromDesc(const vcl_query_desc_t& desc, vcl_compiler_desc_t& compilerDesc,
                                      vcl_compiler_properties_t& compilerProp, vcl_device_desc_t& deviceDesc,
                                      vpux::OV::Config& config, bool isDeviceDescEmpty) const final;

    NetworkDescription compileFromDesc(const vcl_executable_desc_t& desc, const vcl_compiler_properties_t& compilerProp,
                                       vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                       vpux::OV::Config& config, bool isDeviceDescEmpty) const final;

    NetworkDescriptionView compileFromDesc(const vcl_executable_desc_t& desc,
                                           const vcl_compiler_properties_t& compilerProp,
                                           vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc,
                                           vpux::OV::Config& config, bool isDeviceDescEmpty,
                                           BlobAllocator& allocator) const final;

    std::vector<std::shared_ptr<NetworkDescriptionView>> compileFromDescWsOneShot(
            const vcl_executable_desc_t& desc, const vcl_compiler_properties_t& compilerProp,
            vcl_compiler_desc_t& compilerDesc, vcl_device_desc_t& deviceDesc, vpux::OV::Config& config,
            bool isDeviceDescEmpty, BlobAllocator& allocator) const final;

private:
    /// @brief Returns Init schedules and Main in a single call. The blobs are allocated using the provided allocator.
    /// There is always exactly one Main schedule, placed at the back of the vector.
    std::vector<std::shared_ptr<NetworkDescriptionView>> compileWsOneShot(const std::shared_ptr<ov::Model>& model,
                                                                          const vpux::OV::Config& config,
                                                                          BlobAllocator& allocator) const;

    /// @brief Sequentially compiles Init and Main schedules. The blob is allocated using the provided allocator. The
    /// Main schedule is always last.
    NetworkDescriptionView compileWsIterative(const std::shared_ptr<ov::Model>& model, const vpux::OV::Config& config,
                                              size_t callIdx, BlobAllocator& allocator) const;
};

}  // namespace vpux
