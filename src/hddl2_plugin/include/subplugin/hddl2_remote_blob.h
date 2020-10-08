//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

// System
#include <memory>
#include <string>
// IE
#include "ie_blob.h"
#include "ie_remote_context.hpp"
// Plugin
#include "vpux.hpp"
#include "vpux_remote_context.h"
// Low-level
#include "RemoteMemory.h"

namespace vpu {
namespace HDDL2Plugin {

//------------------------------------------------------------------------------
class HDDL2BlobParams {
public:
    explicit HDDL2BlobParams(const InferenceEngine::ParamMap& paramMap, const vpu::LogLevel& logLevel);

    InferenceEngine::ParamMap getParamMap() const { return _paramMap; }
    HddlUnite::RemoteMemory::Ptr getRemoteMemory() const { return _remoteMemory; }
    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; }

protected:
    InferenceEngine::ParamMap _paramMap;
    HddlUnite::RemoteMemory::Ptr _remoteMemory;
    InferenceEngine::ColorFormat _colorFormat;
    const Logger::Ptr _logger;
};

//------------------------------------------------------------------------------
class HDDL2RemoteBlob : public InferenceEngine::RemoteBlob {
public:
    using Ptr = std::shared_ptr<HDDL2RemoteBlob>;
    using CPtr = std::shared_ptr<const HDDL2RemoteBlob>;

    explicit HDDL2RemoteBlob(const InferenceEngine::TensorDesc& tensorDesc,
        const vpux::VPUXRemoteContext::Ptr& contextPtr, const std::shared_ptr<vpux::Allocator>& allocator,
        const InferenceEngine::ParamMap& params, const LogLevel logLevel = LogLevel::None);
    ~HDDL2RemoteBlob() override { HDDL2RemoteBlob::deallocate(); }

    /**
     * @details Since Remote blob just wrap remote memory, allocation is not required
     */
    void allocate() noexcept override {}

    /**
     * @brief Deallocate local memory
     * @return True if allocation is done, False if deallocation is failed.
     */
    bool deallocate() noexcept override;

    InferenceEngine::LockedMemory<void> buffer() noexcept override;

    InferenceEngine::LockedMemory<const void> cbuffer() const noexcept override;

    InferenceEngine::LockedMemory<void> rwmap() noexcept override;

    InferenceEngine::LockedMemory<const void> rmap() const noexcept override;

    InferenceEngine::LockedMemory<void> wmap() noexcept override;

    std::shared_ptr<InferenceEngine::RemoteContext> getContext() const noexcept override;

    InferenceEngine::ParamMap getParams() const override { return _params.getParamMap(); }

    std::string getDeviceName() const noexcept override;

    HddlUnite::RemoteMemory::Ptr getRemoteMemory() const { return _remoteMemory; }

    InferenceEngine::ColorFormat getColorFormat() const { return _colorFormat; }

    std::shared_ptr<InferenceEngine::ROI> getROIPtr() const { return _roiPtr; }

    size_t size() const noexcept override;

    size_t byteSize() const noexcept override;

    InferenceEngine::Blob::Ptr createROI(const InferenceEngine::ROI& regionOfInterest) const override;

protected:
    void* _memoryHandle = nullptr;

    const HDDL2BlobParams _params;
    std::weak_ptr<vpux::VPUXRemoteContext> _remoteContextPtr;
    std::shared_ptr<InferenceEngine::IAllocator> _allocatorPtr = nullptr;

    const HddlUnite::RemoteMemory::Ptr _remoteMemory;
    const InferenceEngine::ColorFormat _colorFormat;
    std::shared_ptr<InferenceEngine::ROI> _roiPtr;
    const Logger::Ptr _logger;

    explicit HDDL2RemoteBlob(const HDDL2RemoteBlob& origBlob, const InferenceEngine::ROI& regionOfInterest);

    void* getHandle() const noexcept override;
    const std::shared_ptr<InferenceEngine::IAllocator>& getAllocator() const noexcept override;
};

}  // namespace HDDL2Plugin
}  // namespace vpu
