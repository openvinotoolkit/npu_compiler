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

#include "vpual_device.hpp"

#include <memory>

#include "vpual_core_nn_executor.hpp"
#include "vpusmm_allocator.hpp"

namespace vpux {

VpualDevice::VpualDevice(const std::string& name,
    const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform): _name(name), _platform(platform) {
    const auto id = extractIdFromDeviceName(name);
    _allocator = std::make_shared<VpusmmAllocator>(id);
}

std::shared_ptr<Executor> VpualDevice::createExecutor(
    const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) {
    const auto& vpusmmAllocator = std::dynamic_pointer_cast<VpusmmAllocator>(_allocator);
    if (vpusmmAllocator == nullptr) {
        THROW_IE_EXCEPTION << "Incompatible allocator passed into vpual_backend";
    }
    _config.parseFrom(config);

    const auto id = extractIdFromDeviceName(_name);
    const auto& executor = std::make_shared<VpualCoreNNExecutor>(networkDescription, vpusmmAllocator, id, _platform, _config);

    return executor;
}

std::shared_ptr<Allocator> VpualDevice::getAllocator() const { return _allocator; }
std::shared_ptr<Allocator> VpualDevice::getAllocator(const InferenceEngine::ParamMap&) const {
    // TODO Add validation that input param map can be handled by allocator
    return getAllocator();
}

std::string VpualDevice::getName() const { return _name; }
}  // namespace vpux
