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

#include <map>
#include <memory>
#include <vpux.hpp>

namespace vpux {

class EmulatorBackend final : public vpux::IEngineBackend {
public:
    EmulatorBackend();
    const std::string getName() const override {
        return "EMULATOR";
    }
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;

private:
    std::unique_ptr<vpu::Logger> _logger;
    std::shared_ptr<IDevice> _device;
};

}  // namespace vpux
