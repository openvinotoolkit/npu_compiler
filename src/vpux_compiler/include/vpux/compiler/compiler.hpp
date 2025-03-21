//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "intel_npu/icompiler.hpp"

#include "vpux/compiler/version.hpp"
#include "vpux/utils/core/mem_size.hpp"

namespace vpux {

class BlobAllocator {
public:
    virtual ~BlobAllocator() = default;
    virtual uint8_t* allocate(vpux::Byte) = 0;
    virtual void deallocate(uint8_t*) = 0;
};

// Non-owning view into a memory occupied by a blob. Used by AllocatedCompiledNetwork
// to store compiled model allocated via BlobAllocator implementation.
struct BlobView final {
    // E#-140887: ptr left mutable to be compatible with initial version of VCL
    // interface; make BlobView immutable and reuse it in CompiledNetwork
    uint8_t* ptr = nullptr;
    uint64_t size = 0;

    BlobView(uint8_t*, uint64_t);
    // E#-140887: enable implicit conversion from std::vector<uint8_t>
    // currently it'll fail due to blob.data() being const uint8_t* that
    // can't be converted to uint8_t*
    // /* implicit */ BlobView(const std::vector<uint8_t>& blob);
};

// The object returned by the compiler to provide such information about a network
// as description of inputs and outputs, name and compiled network in a format
// executable by device
// The difference between NetworkDescriptionView and NetworkDescription is
// compiled network is represented via BlobView. Blob in this case is allocated by
// compiler via provided BlobAllocator implementation.
struct NetworkDescriptionView {
    NetworkDescriptionView(BlobView, intel_npu::NetworkMetadata&&);

    NetworkDescriptionView(const NetworkDescriptionView&) = delete;
    NetworkDescriptionView& operator=(const NetworkDescriptionView&) = delete;

    NetworkDescriptionView(NetworkDescriptionView&&) = default;
    NetworkDescriptionView& operator=(NetworkDescriptionView&&) = default;

    ~NetworkDescriptionView() = default;

    BlobView compiledNetwork;
    intel_npu::NetworkMetadata metadata;
};

class CompilerImpl final : public intel_npu::ICompiler {
public:
    CompilerImpl();

    uint32_t getSupportedOpsetVersion() const;

    // Mutable model variant for direct use with deserialized model in VCL
    intel_npu::NetworkDescription compile(const std::shared_ptr<ov::Model>& model,
                                          const intel_npu::Config& config) const;

    intel_npu::NetworkDescription compile(const std::shared_ptr<const ov::Model>& model,
                                          const intel_npu::Config& config) const final;

    ov::SupportedOpsMap query(const std::shared_ptr<const ov::Model>& model,
                              const intel_npu::Config& config) const final;

    intel_npu::NetworkMetadata parse(const std::vector<uint8_t>& network, const intel_npu::Config&) const final;

    std::vector<ov::ProfilingInfo> process_profiling_output(const std::vector<uint8_t>& profData,
                                                            const std::vector<uint8_t>& network,
                                                            const intel_npu::Config& config) const final;

    uint32_t get_version() const final {
        return ICOMPILER_MAKE_VERSION(NPU_COMPILER_VERSION_MAJOR, NPU_COMPILER_VERSION_MINOR);
    };

    // CiD-specific methods

    NetworkDescriptionView compile(const std::shared_ptr<ov::Model>& model, const intel_npu::Config& config,
                                   BlobAllocator& allocator) const;

    NetworkDescriptionView compile(const std::shared_ptr<const ov::Model>& model, const intel_npu::Config& config,
                                   BlobAllocator& allocator) const;
};

/**
 * @enum IRPrintingOrder
 * @brief VPUX IR pass printing before/after or before and after
 */
enum class IRPrintingOrder {
    BEFORE,
    AFTER,
    BEFORE_AFTER,
};

}  // namespace vpux
