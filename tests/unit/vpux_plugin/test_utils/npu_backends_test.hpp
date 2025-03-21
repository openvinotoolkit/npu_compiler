//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// Plugin
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/common.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "npu/utils/logger/logger.hpp"
#include "vpux/utils/IE/private_properties.hpp"

#include "no_devices_test_backend.hpp"
#include "npu3720_test_backend.hpp"
#include "throw_test_backend.hpp"

#include "openvino/runtime/so_ptr.hpp"

namespace vpux {
class NPUBackendsTest final {
public:
    using Ptr = std::shared_ptr<NPUBackendsTest>;
    using CPtr = std::shared_ptr<const NPUBackendsTest>;

    explicit NPUBackendsTest(const std::vector<std::string>& backendRegistry)
            : _logger("NPUBackendsTest", intel_npu::Logger::global().level()) {
        std::vector<ov::SoPtr<IEngineBackend>> registeredBackends;
        const auto registerBackend = [&](const ov::SoPtr<IEngineBackend>& backend, const std::string& name) {
            const auto backendDevices = backend->getDeviceNames();
            if (!backendDevices.empty()) {
                std::stringstream deviceNames;
                for (const auto& device : backendDevices) {
                    deviceNames << device << " ";
                }
                _logger.debug("Register '%s' with devices '%s'", name.c_str(), deviceNames.str().c_str());
                registeredBackends.emplace_back(backend);
            }
        };

        for (const auto& name : backendRegistry) {
            _logger.debug("Try '%s' backend", name.c_str());

            try {
                if (name == "npu3720_test_backend") {
                    const auto backend = ov::SoPtr<IEngineBackend>(std::make_shared<NPU3720TestBackend>());
                    registerBackend(backend, name);
                }
                if (name == "no_device_test_backend") {
                    const auto backend = ov::SoPtr<IEngineBackend>(std::make_shared<NoDevicesTestBackend>());
                    registerBackend(backend, name);
                }
                if (name == "throw_test_backend") {
                    const auto backend = ov::SoPtr<IEngineBackend>(std::make_shared<ThrowTestBackend>());
                    registerBackend(backend, name);
                }
            } catch (const std::exception& ex) {
                _logger.error("Got an error during backend '%s' loading : %s", name.c_str(), ex.what());
            } catch (...) {
                _logger.error("Got an unknown error during backend '%s' loading", name.c_str());
            }
        };

        if (registeredBackends.empty()) {
            registeredBackends.emplace_back(nullptr);
        }

        _backend = *registeredBackends.begin();

        if (_backend != nullptr) {
            _logger.info("Use '%s' backend for inference", _backend->getName().c_str());
        } else {
            _logger.error("Cannot find backend for inference. Make sure the device is available.");
        }
    };

private:
    intel_npu::Logger _logger;
    ov::SoPtr<IEngineBackend> _backend;
};

}  // namespace vpux
